"""
robust.py — Rank-order (OWA / L-) aggregators for Byzantine-robust distributed SGD.

1) Minimax-optimal OWA aggregator with guaranteed breakdown (convex program, cvxpy):
       min_w  max_{attack in A}  E[(w^T s_attack)^2]     s.t.  w >= 0, 1^T w = 1, w_k = 0 for k<=m or k>n-m
   s_attack = sorted vector of n values: n-f benign (centred, unit scale) + f Byzantine placed by the attacker.
   For each attack the objective is the quadratic form w^T S w, S = E[s s^T]  ->  max of convex quadratics
   over a polytope: a convex SOCP. Sanity check: for contaminated Gaussians the solution should resemble
   the Huber/Jaeckel trimmed-mean family.
2) End-to-end Byzantine-robust training (sklearn digits, MLP 64-64-10, n=20 workers, f=4 Byzantine,
   worker momentum 0.9), attacks: none / sign-flip / Gaussian / ALIE / IPM;
   aggregators: mean, CW-median, CW-trimmed mean, Krum, centered clipping, OWA*, Med_t(3)->OWA* (NEst∘NEst).
"""
import sys, json, math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import norm
import cvxpy as cp

# ----------------------------------------------------------------------------------------------
# 1. Minimax OWA
# ----------------------------------------------------------------------------------------------
def benign_sampler(kind, rng, emp=None):
    if kind == "gauss":
        return lambda size: rng.standard_normal(size)
    if kind == "t3":
        return lambda size: rng.standard_t(3, size) / math.sqrt(3.0)          # unit variance
    if kind == "empirical":
        return lambda size: rng.choice(emp, size)
    raise ValueError(kind)


def second_moments(kind, n=20, f=4, mc=6000, seed=0, emp=None,
                   z1=np.linspace(-25, 25, 101), z2c=np.linspace(-8, 8, 17)):
    """Return list of (label, S) with S = E[s s^T] for every attack scenario (+ benign scenario)."""
    rng = np.random.default_rng(seed)
    draw = benign_sampler(kind, rng, emp)
    B = draw((mc, n - f))
    mu, sd = B.mean(1, keepdims=True), B.std(1, keepdims=True)
    out = []
    # benign: the f nodes are honest
    s = np.sort(np.concatenate([B, draw((mc, f))], 1), 1)
    out.append(("benign", s.T @ s / mc))
    # single-point attacks (ALIE-like, omniscient): all f at mu + z*sd
    for z in z1:
        s = np.sort(np.concatenate([B, np.repeat(mu + z * sd, f, 1)], 1), 1)
        out.append((f"z={z:.1f}", s.T @ s / mc))
    # split attacks: k at +z, f-k at -z'
    for k in range(1, f):
        for za in z2c:
            for zb in z2c:
                A = np.concatenate([np.repeat(mu + za * sd, k, 1), np.repeat(mu + zb * sd, f - k, 1)], 1)
                s = np.sort(np.concatenate([B, A], 1), 1)
                out.append((f"k={k},{za:.0f},{zb:.0f}", s.T @ s / mc))
    return out


def minimax_owa(Ss, n=20, trim=4, benign_weight=None, symmetric=True):
    """Solve min_w max_i w^T S_i w over the interior-supported simplex (SOCP via Cholesky factors)."""
    w = cp.Variable(n)
    t = cp.Variable()
    cons = [w >= 0, cp.sum(w) == 1]
    if trim > 0:
        cons += [w[:trim] == 0, w[n - trim:] == 0]
    if symmetric:
        cons += [w == w[::-1]]
    for _, S in Ss:
        L = np.linalg.cholesky(S + 1e-10 * np.eye(n))
        cons.append(cp.sum_squares(L.T @ w) <= t)
    obj = t
    if benign_weight:                     # optional: trade worst-case for benign efficiency
        obj = t + benign_weight * cp.quad_form(w, Ss[0][1])
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver="CLARABEL")
    return np.maximum(w.value, 0) / np.maximum(w.value, 0).sum(), prob.value


def worst_case(w, Ss):
    vals = np.array([w @ S @ w for _, S in Ss])
    return vals.max(), vals[0], Ss[int(vals.argmax())][0]


def owa_baselines(n=20, trim=4):
    W = {}
    W["mean"] = np.full(n, 1 / n)
    med = np.zeros(n); med[n // 2 - 1] = med[n // 2] = 0.5; W["median"] = med
    tm = np.zeros(n); tm[trim:n - trim] = 1 / (n - 2 * trim); W[f"trimmed({trim})"] = tm
    return W


# ----------------------------------------------------------------------------------------------
# 2. Distributed training
# ----------------------------------------------------------------------------------------------
def load_digits_tensors(seed=0):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    X, y = load_digits(return_X_y=True)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    T = lambda a, d=torch.float32: torch.tensor(a, dtype=d)
    return T(Xtr), T(ytr, torch.long), T(Xte), T(yte, torch.long)


def make_model():
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_flat(model, v):
    i = 0
    for p in model.parameters():
        k = p.numel(); p.data.copy_(v[i:i + k].view_as(p)); i += k


def worker_grads(model, X, y, idx_list):
    from torch.func import functional_call, grad, vmap
    params = {k: v.detach() for k, v in model.named_parameters()}
    def loss_fn(p, xb, yb):
        return F.cross_entropy(functional_call(model, p, (xb,)), yb)
    xb = torch.stack([X[i] for i in idx_list]); yb = torch.stack([y[i] for i in idx_list])
    G = vmap(grad(loss_fn), in_dims=(None, 0, 0))(params, xb, yb)
    return torch.cat([G[k].reshape(len(idx_list), -1) for k in params], 1)


# ---- aggregators: M is (n, d) ----
def agg_mean(M, st): return M.mean(0)
def agg_cwmed(M, st): return M.median(0).values if M.shape[0] % 2 else M.sort(0).values[M.shape[0] // 2 - 1:M.shape[0] // 2 + 1].mean(0)
def make_owa(w):
    wt = torch.tensor(w, dtype=torch.float32)
    return lambda M, st: wt @ M.sort(0).values
def make_krum(f):
    def agg(M, st):
        D = torch.cdist(M, M) ** 2
        n = M.shape[0]; k = n - f - 2
        sc = D.sort(1).values[:, 1:k + 1].sum(1)
        return M[sc.argmin()]
    return agg
def agg_cclip(M, st):
    v = st.get("v", torch.zeros(M.shape[1]))
    for _ in range(1):
        Dl = M - v
        nr = Dl.norm(dim=1, keepdim=True)
        tau = nr.median()
        v = v + (Dl * torch.clamp(tau / (nr + 1e-12), max=1.0)).mean(0)
    st["v"] = v
    return v
def make_cascade(w, T=3):
    owa = make_owa(w)
    def agg(M, st):
        H = st.setdefault("hist", [])
        H.append(M.clone())
        if len(H) > T: H.pop(0)
        Mt = torch.stack(H, 0).median(0).values if len(H) >= 2 else M     # temporal rank filter per worker
        return owa(Mt, st)
    return agg


def attack(Mb, kind, f, n, own=None, z=None):
    mu, sd = Mb.mean(0), Mb.std(0)
    if kind == "none":
        return own
    if kind == "signflip":
        return -own
    if kind == "gauss":
        return 30.0 * torch.randn(f, Mb.shape[1])
    if kind == "alie":
        if z is None:
            s = math.floor(n / 2 + 1) - f
            z = norm.ppf((n - s) / n)
        return (mu - z * sd).expand(f, -1)
    if kind == "ipm":
        return (-0.5 * mu).expand(f, -1)
    raise ValueError(kind)


def train_distributed(agg, att, n=20, f=4, steps=300, lr=0.1, beta=0.9, bs=16, seed=0, data=None,
                      noniid=0.0, z_alie=None):
    """noniid in [0,1]: fraction of each worker's batch drawn from its own label-sorted shard."""
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte = data
    order = torch.argsort(ytr)
    shards = torch.chunk(order, n)
    model = make_model()
    mom = torch.zeros(n, flat_params(model).numel())
    st = {}
    errs = []
    g = torch.Generator().manual_seed(seed)
    for t in range(steps):
        idx = []
        for i in range(n):
            k = int(round(noniid * bs))
            loc = shards[i][torch.randint(0, len(shards[i]), (k,), generator=g)]
            glo = torch.randint(0, len(Xtr), (bs - k,), generator=g)
            idx.append(torch.cat([loc, glo]))
        G = worker_grads(model, Xtr, ytr, idx)
        mom = beta * mom + (1 - beta) * G
        M = mom.clone()
        if f > 0:
            M[n - f:] = attack(mom[:n - f], att, f, n, own=mom[n - f:], z=z_alie)
        upd = agg(M, st)
        if not torch.isfinite(upd).all():
            return dict(acc=float("nan"), err=float("nan"))
        Mb = mom[:n - f] if f > 0 else mom
        errs.append(((upd - Mb.mean(0)) ** 2).sum().item() / (Mb.var(0).sum().item() / Mb.shape[0] + 1e-30))
        set_flat(model, flat_params(model) - lr * upd)
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    return dict(acc=acc, err=float(np.median(errs[20:])))


def collect_benign_coords(steps=150, n=20, seed=0, data=None):
    """Clean run; pool coordinate-wise standardized worker momenta -> empirical benign distribution."""
    torch.manual_seed(seed)
    Xtr, ytr, _, _ = data
    model = make_model(); mom = torch.zeros(n, flat_params(model).numel())
    pool = []
    g = torch.Generator().manual_seed(seed)
    for t in range(steps):
        idx = [torch.randint(0, len(Xtr), (16,), generator=g) for _ in range(n)]
        G = worker_grads(model, Xtr, ytr, idx)
        mom = 0.9 * mom + 0.1 * G
        if t > 30 and t % 10 == 0:
            Z = (mom - mom.mean(0)) / (mom.std(0) + 1e-12)
            keep = mom.std(0) > 1e-6
            pool.append(Z[:, keep].reshape(-1).numpy())
        set_flat(model, flat_params(model) - 0.05 * mom.mean(0))
    return np.concatenate(pool)
