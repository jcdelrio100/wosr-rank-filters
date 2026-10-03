"""B5 — Byzantine/fault-robust FL: learned ML aggregator vs rank cascades.
ML baseline: coordinate-wise MLP on the SORTED vector of the n=20 submissions (permutation invariant, DeepSets-style),
robustly standardised (median / MAD), trained offline on synthetic contamination (ALIE-like shifts, sign-flip-like
shifts, large Gaussian, bursts of 8, no attack) to predict the honest mean.  Also plugged into the space–time cascade.
Bench: Fashion-MNIST, n=20, Dirichlet alpha in {1, 0.1}, 7 attacks, 2 seeds (same as note 3)."""
import sys, os, json, itertools, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import fl_bench
from fl_bench import nnm_mix

N, FB = 20, 4

class SortedMLP(nn.Module):
    def __init__(self, n=N, h=64):
        super().__init__(); self.net = nn.Sequential(nn.Linear(n, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, V):                       # V (..., n) unsorted
        S = V.sort(-1).values
        med = S[..., N // 2 - 1:N // 2 + 1].mean(-1, keepdim=True)
        mad = (S - med).abs().median(-1, keepdim=True).values + 1e-12
        return (med + mad * self.net((S - med) / mad)).squeeze(-1)

def synth(B, g):
    """benign 16 (or 20) values ~ N(0,1) or t3; attackers per family; target = honest mean."""
    heavy = torch.rand(B, 1, generator=g) < 0.3
    Z = torch.randn(B, N, generator=g)
    T3 = torch.distributions.StudentT(3.0).sample((B, N)) / math.sqrt(3)
    V = torch.where(heavy, T3, Z)
    fam = torch.randint(0, 5, (B,), generator=g)
    hon = torch.ones(B, N, dtype=torch.bool)
    mu, sd = V[:, :N - FB].mean(1), V[:, :N - FB].std(1)
    for b in range(B):
        k = int(fam[b])
        if k == 0: continue                                            # no attack
        if k == 1: V[b, N - FB:] = mu[b] + float(torch.empty(1).uniform_(-3, 3, generator=g)) * sd[b]       # ALIE-like
        if k == 2: V[b, N - FB:] = -float(torch.empty(1).uniform_(1, 20, generator=g)) * (mu[b].abs() + sd[b])  # sign-flip-like
        if k == 3: V[b, N - FB:] = 30 * torch.randn(FB, generator=g)                                       # gaussian
        if k == 4: V[b, N - 8:] = -float(torch.empty(1).uniform_(5, 50, generator=g)) * (mu[b].abs() + sd[b]); hon[b, N - 8:] = False  # burst-8
        if k in (1, 2, 3): hon[b, N - FB:] = False
    tgt = (V * hon).sum(1) / hon.sum(1)
    perm = torch.argsort(torch.rand(B, N, generator=g), 1)
    return torch.gather(V, 1, perm), tgt

def train_mlp_agg(steps=4000, seed=0):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    m = SortedMLP(); opt = torch.optim.Adam(m.parameters(), 2e-3)
    for s in range(steps):
        V, t = synth(512, g)
        loss = F.mse_loss(m(V), t)
        opt.zero_grad(); loss.backward(); opt.step()
    return m

def make_mlp_agg(m):
    @torch.no_grad()
    def agg(M, st): return m(M.T.contiguous())          # (d, n) -> (d,)
    return agg

def make_cascade_mlp(m, f, T=3):
    a = make_mlp_agg(m)
    def agg(M, st):
        H = st.setdefault("hist_mlp", []); H.append(M.clone())
        if len(H) > T: H.pop(0)
        Mt = torch.stack(H, 0).median(0).values if len(H) >= 2 else M
        return a(nnm_mix(Mt, f), st)
    return agg

if __name__ == "__main__":
    from multiprocessing import Pool
    t0 = time.time()
    m = train_mlp_agg(); torch.save(m.state_dict(), "bench/b5_mlp_agg.pt")
    print("MLP aggregator trained %.0fs, params %d" % (time.time() - t0, sum(p.numel() for p in m.parameters())), flush=True)
    fl_bench.EXTRA["MLP agregador"] = lambda lib, f, m=m: make_mlp_agg(m)
    fl_bench.EXTRA["Med_t3→NNM→MLP"] = lambda lib, f, m=m: make_cascade_mlp(m, f)
    cfg = list(itertools.product(["MLP agregador", "Med_t3→NNM→MLP"], fl_bench.ATTS, [1.0, 0.1], [0, 1]))
    outp = "bench/b5_results.json"
    R = json.load(open(outp)) if os.path.exists(outp) else []
    have = {(r["agg"], r["att"], r["alpha"], r["seed"]) for r in R}
    cfg = [c for c in cfg if c not in have]
    with Pool(2) as p:
        for r in p.imap_unordered(fl_bench._job, cfg):
            R.append(r); json.dump(R, open(outp, "w"))
            print(len(R), r["agg"], r["att"], r["alpha"], r["seed"], round(r["acc"], 4), flush=True)
