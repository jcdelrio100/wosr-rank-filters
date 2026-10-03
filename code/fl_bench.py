"""
fl_bench.py — Byzantine / fault-robust distributed SGD benchmark on Fashion-MNIST.

n=20 nodes, Dirichlet(alpha) label partition, small CNN, worker momentum 0.9 (computed by honest nodes).
Attacks (persistent: nodes 16..19; intermittent: nodes 8..15):
  none | signflip | alie | ipm | burst8 (8 nodes send -50*mu one step in five) |
  sdc (each node-step w.p. 5 %: 1 % of coords set to ±1e3, 'silent data corruption') |
  gauss+burst8 (persistent N(0,30^2) on 4 nodes + burst8 on 8 others)
Aggregators: mean, CW-median, CW-trim(f), NNM+CW-trim(f), CClip, AMO, Med_t3->mean, Med_t3->AMO.
"""
import sys, os, gzip, json, math, time, itertools
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from scipy.stats import norm
from amo import AMO, make_temporal_then, build_library

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fmnist")


def load_fmnist():
    rd = lambda f, off: np.frombuffer(gzip.open(os.path.join(D, f)).read(), np.uint8, offset=off)
    Xtr = rd("train-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28).astype(np.float32) / 255.
    ytr = rd("train-labels-idx1-ubyte.gz", 8).astype(np.int64)
    Xte = rd("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28).astype(np.float32) / 255.
    yte = rd("t10k-labels-idx1-ubyte.gz", 8).astype(np.int64)
    m, s = Xtr.mean(), Xtr.std()
    return [torch.tensor(a) for a in ((Xtr - m) / s, ytr, (Xte - m) / s, yte)]


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 5); self.c2 = nn.Conv2d(8, 16, 5); self.fc = nn.Linear(16 * 4 * 4, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


def dirichlet_partition(y, n, alpha, seed):
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(n)]
    for c in range(10):
        idx = rng.permutation(np.nonzero(y == c)[0])
        p = rng.dirichlet(alpha * np.ones(n))
        cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
        for i, ch in enumerate(np.split(idx, cuts)):
            parts[i] += ch.tolist()
    return [torch.tensor(p if len(p) else rng.integers(0, len(y), 8).tolist()) for p in parts]


# ------------------------------------------------------------------ aggregators (M: n x d)
def agg_mean(M, st): return M.mean(0)
def agg_cwmed(M, st):
    S = M.sort(0).values; n = M.shape[0]
    return S[n // 2] if n % 2 else 0.5 * (S[n // 2 - 1] + S[n // 2])
def make_cwtm(t):
    return lambda M, st: M.sort(0).values[t:M.shape[0] - t].mean(0)
def make_nnm_cwtm(f):
    tm = make_cwtm(f)
    def agg(M, st):
        Dm = torch.cdist(M, M)
        nn_idx = Dm.topk(M.shape[0] - f, largest=False).indices          # includes self
        return tm(M[nn_idx].mean(1), st)
    return agg
def nnm_mix(M, f):
    Dm = torch.cdist(M, M)
    return M[Dm.topk(M.shape[0] - f, largest=False).indices].mean(1)
def make_cascade3(final, f, T=3):
    """Med_t(T) -> NNM -> final  (three-stage space-time cascade)."""
    def agg(M, st):
        H = st.setdefault("hist3", [])
        H.append(M.clone())
        if len(H) > T: H.pop(0)
        Mt = torch.stack(H, 0).median(0).values if len(H) >= 2 else M
        return final(nnm_mix(Mt, f), st)
    return agg
def agg_cclip(M, st):
    v = st.get("v", torch.zeros(M.shape[1]))
    Dl = M - v; nr = Dl.norm(dim=1, keepdim=True); tau = nr.median()
    v = v + (Dl * torch.clamp(tau / (nr + 1e-12), max=1.0)).mean(0)
    st["v"] = v
    return v


# ------------------------------------------------------------------ attacks
def corrupt(M, mom, att, t, n, f, gen):
    """M: submissions (n,d) (copy of honest momenta). Returns corrupted M."""
    honest = list(range(n - f)) if att in ("signflip", "alie", "ipm", "gauss+burst8") else list(range(n))
    mu, sd = mom[honest].mean(0), mom[honest].std(0)
    P = list(range(n - f, n)); B = list(range(8, 16))
    if att == "signflip":
        M[P] = -mom[P]
    elif att == "alie":
        s = math.floor(n / 2 + 1) - f; z = norm.ppf((n - s) / n)
        M[P] = mu - z * sd
    elif att == "ipm":
        M[P] = -0.5 * mu
    elif att in ("burst8", "gauss+burst8"):
        if att == "gauss+burst8":
            M[P] = 30.0 * torch.randn(len(P), M.shape[1], generator=gen)
        if t % 5 == 0:
            M[B] = -50.0 * mu
    elif att == "sdc":
        hit = torch.rand(n, generator=gen) < 0.05
        for i in torch.nonzero(hit).flatten().tolist():
            k = max(1, M.shape[1] // 100)
            c = torch.randperm(M.shape[1], generator=gen)[:k]
            M[i, c] = 1e3 * torch.sign(torch.randn(k, generator=gen))
    return M


EXTRA = {}          # name -> factory(lib, f) for externally defined aggregators


def run(agg_name, att, alpha, seed, steps=300, n=20, f=4, lr=0.08, beta=0.9, bs=32, data=None, lib=None,
        record=False):
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte = data
    parts = dirichlet_partition(ytr.numpy(), n, alpha, seed)
    model = CNN()
    names = [k for k, _ in model.named_parameters()]
    params = {k: v.detach().clone() for k, v in model.named_parameters()}
    d = sum(v.numel() for v in params.values())
    def loss_fn(p, xb, yb): return F.cross_entropy(functional_call(model, p, (xb,)), yb)
    gfn = vmap(grad(loss_fn), in_dims=(None, 0, 0))
    AG = {"mean": agg_mean, "CW-median": agg_cwmed, "CW-trim(f)": make_cwtm(f), "NNM+CW-trim(f)": make_nnm_cwtm(f),
          "CClip": agg_cclip, "AMO": AMO(lib), "Med_t3→mean": make_temporal_then(agg_mean, 3),
          "Med_t3→AMO": AMO(lib, temporal=3),
          "Med_t3→NNM→AMO": make_cascade3(AMO(lib), f), "Med_t3→NNM→CW-trim(f)": make_cascade3(make_cwtm(f), f)}
    for k, fac in EXTRA.items():
        AG[k] = fac(lib, f)
    agg = AG[agg_name]
    mom = torch.zeros(n, d); st = {}
    gen = torch.Generator().manual_seed(seed + 1)
    curve = []
    for t in range(steps):
        idx = torch.stack([parts[i][torch.randint(0, len(parts[i]), (bs,), generator=gen)] for i in range(n)])
        G = gfn(params, Xtr[idx], ytr[idx])
        G = torch.cat([G[k].reshape(n, -1) for k in names], 1)
        mom = beta * mom + (1 - beta) * G
        M = corrupt(mom.clone(), mom, att, t, n, f, gen)
        u = agg(M, st)
        if not torch.isfinite(u).all():
            return dict(acc=float("nan"), curve=curve)
        u = u.clamp(-1e4, 1e4)
        i = 0
        for k in names:
            m = params[k].numel(); params[k] = params[k] - lr * u[i:i + m].view_as(params[k]); i += m
        if record and (t + 1) % 50 == 0:
            curve.append(evaluate(model, params, Xte, yte))
    acc = evaluate(model, params, Xte, yte)
    out = dict(acc=acc, curve=curve)
    if "trace" in st:
        out["amo_models"] = {k: st["trace"].count(k) for k in set(st["trace"])}
    return out


@torch.no_grad()
def evaluate(model, params, Xte, yte):
    logits = functional_call(model, params, (Xte,))
    a = (logits.argmax(1) == yte).float().mean().item()
    return a if math.isfinite(logits.sum().item()) else float("nan")


AGGS = ["mean", "CW-median", "CW-trim(f)", "NNM+CW-trim(f)", "CClip", "AMO", "Med_t3→mean", "Med_t3→AMO"]
ATTS = ["none", "signflip", "alie", "ipm", "burst8", "sdc", "gauss+burst8"]

_DATA = None; _LIB = None
def _job(c):
    global _DATA, _LIB
    torch.set_num_threads(1)
    if _DATA is None:
        _DATA = load_fmnist(); _LIB = build_library()
    a, att, alpha, seed = c
    t0 = time.time()
    r = run(a, att, alpha, seed, data=_DATA, lib=_LIB)
    return dict(agg=a, att=att, alpha=alpha, seed=seed, sec=round(time.time() - t0, 1), **r)


if __name__ == "__main__":
    from multiprocessing import Pool
    out = sys.argv[1] if len(sys.argv) > 1 else "fl_results.json"
    seeds = [0, 1]
    aggs = AGGS if len(sys.argv) <= 2 else sys.argv[2].split(",")
    cfg = list(itertools.product(aggs, ATTS, [1.0, 0.1], seeds))
    done = []
    if os.path.exists(out):
        done = json.load(open(out))
    have = {(r["agg"], r["att"], r["alpha"], r["seed"]) for r in done}
    cfg = [c for c in cfg if c not in have]
    print("pending", len(cfg), flush=True)
    with Pool(2) as p:
        for r in p.imap_unordered(_job, cfg):
            done.append(r)
            json.dump(done, open(out, "w"))
            print(len(done), r["agg"], r["att"], r["alpha"], r["seed"], round(r["acc"], 4), r["sec"], flush=True)
