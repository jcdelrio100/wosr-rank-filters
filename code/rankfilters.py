"""
rankfilters.py — Cascaded rank-order (NEst) structures: 1993 LMS vs. 2026 differentiable training.

Reproduces the NEst cascade of J.C. del Río (UPC, 1993), "Adaptation of Structures and Order Filters":
    y_i = sum_k |h_k| (x + b)_(k)        (eq. 2.27),  cascade F' o F  (eqs. 3.39-3.46)
and compares it with a modern formulation:
    * exact reverse-mode gradients through the sorting permutation (autograd), no j'=0 truncation;
    * order weights on the simplex  w = softmax(theta)  (OWA operator; removes the h·h'=1 scale ambiguity);
    * optional NeuralSort relaxation with temperature annealing (continuation from smooth to hard sort);
    * Adam on mini-batches + multi-start.

Experiments
    E1  thesis example (max -> min, masks {0,2,4,5}, {0,1,4,6}, search region 7, U[-5,5], SNR 35 dB)
    E2  identification success rate over random two-stage rank targets
    E3  practical: impulsive-noise denoising vs. median, linear Wiener and 1-D CNN baselines (+ OOD test)
"""
import math, time, json, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
DEV = "cpu"

# ----------------------------------------------------------------------------------------------
# Common helpers
# ----------------------------------------------------------------------------------------------
def causal_windows(x, m):
    """x (B,T) -> (B,T,m) with W[..., i, j] = x[i-j] (zero-padded on the left)."""
    xp = F.pad(x, (m - 1, 0))
    return xp.unfold(-1, m, 1).flip(-1)


def centered_windows(x, m):
    """x (B,T) -> (B,T,m), W[..., i, j] = x[i - j + m//2] (replicate padding)."""
    p = m // 2
    xp = F.pad(x.unsqueeze(1), (p, p), mode="replicate").squeeze(1)
    return xp.unfold(-1, m, 1).flip(-1)


def neuralsort(z, tau):
    """NeuralSort relaxation (Grover et al., ICLR 2019). z (...,n) -> soft-sorted ascending (...,n)."""
    n = z.shape[-1]
    A = (z.unsqueeze(-1) - z.unsqueeze(-2)).abs()                      # (...,n,n)
    B = A.sum(-1)                                                       # (...,n)
    scal = (n + 1 - 2 * torch.arange(1, n + 1, dtype=z.dtype, device=z.device))  # descending order
    C = z.unsqueeze(-2) * scal.view(-1, 1)                               # (...,n,n)
    P = torch.softmax((C - B.unsqueeze(-2)) / tau, dim=-1)               # rows -> descending
    return (P @ z.unsqueeze(-1)).squeeze(-1).flip(-1)                    # ascending


def hard_rank_filter(x, mask, rank, windows=causal_windows):
    """Ground-truth rank filter. mask: list of offsets j, rank: 1..|mask| (1=min)."""
    m = max(mask) + 1
    W = windows(x, m)[..., mask]
    return W.sort(-1).values[..., rank - 1]


# ----------------------------------------------------------------------------------------------
# Modern NEst layer
# ----------------------------------------------------------------------------------------------
class NEst(nn.Module):
    def __init__(self, m, windows=causal_windows, init_b_std=0.1):
        super().__init__()
        self.m, self.windows = m, windows
        self.b = nn.Parameter(init_b_std * torch.randn(m))
        self.theta = nn.Parameter(0.01 * torch.randn(m))

    def weights(self, hard=False):
        w = torch.softmax(self.theta, -1)
        if hard:
            w = F.one_hot(w.argmax(), self.m).to(w.dtype)
        return w

    def forward(self, x, tau=None, hard=False):
        Z = self.windows(x, self.m) + self.b
        Zs = Z.sort(-1).values if (tau is None or hard) else neuralsort(Z, tau)
        return Zs @ self.weights(hard)


class SoftWOS(nn.Module):
    """ROF revisited (thesis §2.3.1) with continuous mask weights = Weighted Order Statistic filter.

    Implicit definition (generalises eq. 2.8, sgn(m_j) -> a_j >= 0, sgn(.) -> tanh(./tau)):
        G(y) = sum_j a_j * sigmoid((y - z_j)/tau) - r * sum_j a_j = 0 ,   z_j = x_{i-j} + b_j
    y is the r-quantile of the a-weighted empirical CDF of {z_j}. Solved by bisection (no grad), then
    gradients through the implicit-function theorem  dy/dθ = -G_θ / G_y  (exactly the thesis' eq. 2.11,
    but with a smooth surrogate and exact chain rule through the cascade).
    a_j -> 0 removes tap j regardless of signal amplitude (no 'peak-to-peak gap' needed, unlike NEst).
    """
    def __init__(self, m, windows=causal_windows, nonflat=True):
        super().__init__()
        self.m, self.windows = m, windows
        self.alpha = nn.Parameter(0.5 + 0.05 * torch.randn(m))    # mask logits  a = sigmoid(alpha)
        self.rho = nn.Parameter(torch.zeros(()))                   # order  r = sigmoid(rho)  (0=min, 1=max)
        self.b = nn.Parameter(torch.zeros(m)) if nonflat else None

    def mask(self):
        return torch.sigmoid(4 * self.alpha)

    def order(self):
        return torch.sigmoid(self.rho)

    def forward(self, x, tau=None, hard=False):
        tau = 1e-3 if (tau is None or hard) else tau
        Z = self.windows(x, self.m)
        if self.b is not None:
            Z = Z + self.b
        a = self.mask()
        if hard:
            a = (a > 0.5).to(a.dtype) + 1e-6
        r = self.order().clamp(1e-3, 1 - 1e-3)
        A = a.sum()

        def G(y, a=a, r=r, Z=Z):
            return (a * torch.sigmoid((y.unsqueeze(-1) - Z) / tau)).sum(-1) - r * A

        with torch.no_grad():
            lo = Z.min(-1).values - 10 * tau
            hi = Z.max(-1).values + 10 * tau
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                g = G(mid)
                lo = torch.where(g < 0, mid, lo)
                hi = torch.where(g < 0, hi, mid)
            ys = 0.5 * (lo + hi)
            s = torch.sigmoid((ys.unsqueeze(-1) - Z) / tau)
            Gy = (a * s * (1 - s)).sum(-1) / tau + 1e-12
        if not torch.is_grad_enabled():
            return ys
        return ys - G(ys) / Gy            # value = ys (G≈0); gradient = -G_θ/G_y (implicit function thm)


class Cascade(nn.Module):
    def __init__(self, ms, windows=causal_windows, kind="nest"):
        super().__init__()
        Layer = NEst if kind == "nest" else SoftWOS
        self.layers = nn.ModuleList([Layer(m, windows) for m in ms])

    def forward(self, x, tau=None, hard=False):
        for L in self.layers:
            x = L(x, tau, hard)
        return x


def train_modern(xtr, dtr, ms, steps=1500, lr=0.05, anneal=True, tau0=2.0, tau1=0.02,
                 batch=8, seed=0, windows=causal_windows, model=None, kind="nest", crop=None):
    torch.manual_seed(seed)
    net = model if model is not None else Cascade(ms, windows, kind)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    nseq = xtr.shape[0]
    for s in range(steps):
        idx = torch.randint(0, nseq, (batch,))
        xb, db = xtr[idx], dtr[idx]
        if crop and crop < xb.shape[-1]:
            o = int(torch.randint(0, xb.shape[-1] - crop, (1,)))
            xb, db = xb[:, o:o + crop], db[:, o:o + crop]
        if anneal and s < int(0.8 * steps):
            tau = tau0 * (tau1 / tau0) ** (s / (0.8 * steps))
        else:
            tau = None
        y = net(xb, tau)
        loss = F.mse_loss(y[:, 16:], db[:, 16:])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net


# ----------------------------------------------------------------------------------------------
# 1993 NEst cascade, sample-by-sample LMS exactly as in eqs. (3.43)-(3.46)
# ----------------------------------------------------------------------------------------------
def train_lms_thesis(x_in, d, m1, m2, mu1=0.2, lam1=15e-6, mu2=0.05, lam2=10e-6, seed=0):
    rng = np.random.default_rng(seed)
    b1 = np.zeros(m1); h1 = np.full(m1, 1.0 / m1) + 1e-3 * rng.standard_normal(m1)
    b2 = np.zeros(m2); h2 = np.full(m2, 1.0 / m2) + 1e-3 * rng.standard_normal(m2)
    T = len(x_in)
    xbuf = np.zeros(m1)                  # x_{i-j}, j=0..m1-1
    x1buf = np.zeros(m2)                 # x'_{i-j'}
    zs1buf = np.zeros((m2, m1))          # sorted z of layer 1 at i-j'
    rank1_cur = np.zeros(m1, int)
    for i in range(T):
        xbuf = np.roll(xbuf, 1); xbuf[0] = x_in[i]
        z1 = xbuf + b1
        p1 = np.argsort(z1, kind="stable"); zs1 = z1[p1]
        rank1_cur[p1] = np.arange(m1)                       # rank position of each tap j
        x1 = np.abs(h1) @ zs1
        x1buf = np.roll(x1buf, 1); x1buf[0] = x1
        zs1buf = np.roll(zs1buf, 1, axis=0); zs1buf[0] = zs1
        z2 = x1buf + b2
        p2 = np.argsort(z2, kind="stable"); zs2 = z2[p2]
        rank2 = np.empty(m2, int); rank2[p2] = np.arange(m2)
        y = np.abs(h2) @ zs2
        e = d[i] - y
        ah2_of_tap = np.abs(h2)[rank2]                      # |h'_(j')| for each tap j'
        # (3.44)  dy/db'_k = |h'_(k)|
        g_b2 = ah2_of_tap
        # (3.46)  dy/dh'_k = (x'+b')_(k) sgn(h'_k)
        g_h2 = zs2 * np.sign(h2)
        # (3.43)  dy/db_k ≈ |h'_(0)| |h_(k)|   (truncated to j'=0)
        g_b1 = ah2_of_tap[0] * np.abs(h1)[rank1_cur]
        # (3.45)  dy/dh_k = sum_j' |h'_(j')| (x+b)_(k)(i-j') sgn(h_k)
        g_h1 = (ah2_of_tap @ zs1buf) * np.sign(h1)
        b2 += 2 * mu2 * e * g_b2; h2 += 2 * lam2 * e * g_h2
        b1 += 2 * mu1 * e * g_b1; h1 += 2 * lam1 * e * g_h1
    return dict(b1=b1, h1=h1, b2=b2, h2=h2)


def apply_thesis_params(x, P, hard=False):
    def layer(x, b, h):
        Z = causal_windows(x, len(b)) + torch.tensor(b, dtype=x.dtype)
        Zs = Z.sort(-1).values
        w = np.abs(h)
        if hard:
            w = np.eye(len(h))[np.argmax(w)]
        return Zs @ torch.tensor(w, dtype=x.dtype)
    return layer(layer(x, P["b1"], P["h1"]), P["b2"], P["h2"])


# ----------------------------------------------------------------------------------------------
# Data generation for system-identification experiments (thesis test bench, §2.1 / §3.1)
# ----------------------------------------------------------------------------------------------
def make_ident_data(target, n_seq, T, A=5.0, snr_db=35.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(n_seq, T, generator=g) * 2 - 1) * A
    d = target(x)
    pn = (A ** 2 / 3) / 10 ** (snr_db / 10)
    xn = x + math.sqrt(3 * pn) * (torch.rand(n_seq, T, generator=g) * 2 - 1)   # uniform noise
    return xn, d


def two_stage_target(mask1, r1, mask2, r2):
    return lambda x: hard_rank_filter(hard_rank_filter(x, mask1, r1), mask2, r2)


def nmse_db(y, d, skip=20):
    y, d = y[..., skip:], d[..., skip:]
    return 10 * torch.log10(((y - d) ** 2).mean() / (d ** 2).mean()).item()


# ----------------------------------------------------------------------------------------------
# E1: thesis example
# ----------------------------------------------------------------------------------------------
def run_E1(n_train=250_000, seed=0):
    tgt = two_stage_target([0, 2, 4, 5], 4, [0, 1, 4, 6], 1)    # max then min (thesis §3.2.3.4)
    m = 7
    xtr, dtr = make_ident_data(tgt, 1, n_train, seed=seed)
    xte, dte = make_ident_data(tgt, 8, 4096, seed=seed + 100)
    floor = nmse_db(tgt(xte), dte)                                # optimum: true structure on noisy input
    t0 = time.time()
    P = train_lms_thesis(xtr[0].numpy(), dtr[0].numpy(), m, m, seed=seed)
    t_lms = time.time() - t0
    res = dict(floor=floor,
               lms=nmse_db(apply_thesis_params(xte, P), dte),
               lms_hard=nmse_db(apply_thesis_params(xte, P, hard=True), dte), t_lms=t_lms,
               lms_h1=np.round(np.abs(P["h1"]) / np.abs(P["h1"]).sum(), 3).tolist(),
               lms_h2=np.round(np.abs(P["h2"]) / np.abs(P["h2"]).sum(), 3).tolist())
    # modern: same data budget (reshape the single 250k sequence into 61 x 4096)
    L = 4096; k = n_train // L
    xs, ds = xtr[0, :k * L].view(k, L), dtr[0, :k * L].view(k, L)
    t0 = time.time()
    net = train_modern(xs, ds, [m, m], steps=1500, seed=seed)
    res["t_modern"] = time.time() - t0
    with torch.no_grad():
        res["modern"] = nmse_db(net(xte), dte)
        res["modern_hard"] = nmse_db(net(xte, hard=True), dte)
        res["mod_w1"] = np.round(net.layers[0].weights().numpy(), 3).tolist()
        res["mod_w2"] = np.round(net.layers[1].weights().numpy(), 3).tolist()
        res["mod_b1"] = np.round(net.layers[0].b.numpy(), 2).tolist()
        res["mod_b2"] = np.round(net.layers[1].b.numpy(), 2).tolist()
    return res


# ----------------------------------------------------------------------------------------------
# E2: random targets, success rates
# ----------------------------------------------------------------------------------------------
def random_target(rng, m=7):
    out = []
    for _ in range(2):
        k = int(rng.integers(3, 6))
        mask = sorted(rng.choice(m, k, replace=False).tolist())
        r = int(rng.integers(1, k + 1))
        out += [mask, r]
    return out


def e2_trial(t, n_train=150_000):
    torch.set_num_threads(1)
    rng = np.random.default_rng(1000 + t)
    mask1, r1, mask2, r2 = random_target(rng)
    tgt = two_stage_target(mask1, r1, mask2, r2)
    xtr, dtr = make_ident_data(tgt, 1, n_train, seed=t)
    xte, dte = make_ident_data(tgt, 4, 4096, seed=t + 500)
    floor = nmse_db(tgt(xte), dte)
    P = train_lms_thesis(xtr[0].numpy(), dtr[0].numpy(), 7, 7, seed=t)
    r = dict(t=t, target=[mask1, r1, mask2, r2], floor=floor,
             lms=nmse_db(apply_thesis_params(xte, P), dte),
             lms_hard=nmse_db(apply_thesis_params(xte, P, hard=True), dte))
    L = 4096; k = n_train // L
    xs, ds = xtr[0, :k * L].view(k, L), dtr[0, :k * L].view(k, L)
    best = None
    for s in range(4):
        net = train_modern(xs, ds, [7, 7], steps=3000, lr=0.1, anneal=False, batch=4, crop=512,
                           seed=10 * t + s)
        with torch.no_grad():
            tr = F.mse_loss(net(xs[:4]), ds[:4]).item()
            cur = (tr, nmse_db(net(xte), dte), nmse_db(net(xte, hard=True), dte))
        if s == 0:
            r["adam"], r["adam_hard"] = cur[1], cur[2]
        if best is None or cur[0] < best[0]:
            best = cur
    r["adam_x4"], r["adam_x4_hard"] = best[1], best[2]
    return r


# ----------------------------------------------------------------------------------------------
# E3: practical impulsive-noise denoising
# ----------------------------------------------------------------------------------------------
def make_denoise_data(n_seq, T, p_imp=0.10, sigma=0.05, seed=0):
    g = np.random.default_rng(seed)
    S = np.zeros((n_seq, T))
    t = np.arange(T)
    for s in range(n_seq):
        nseg = g.integers(4, 12)
        cuts = np.sort(g.choice(np.arange(1, T), nseg, replace=False))
        lev = g.uniform(-1, 1, nseg + 1)
        S[s] = np.repeat(lev, np.diff(np.r_[0, cuts, T]))
        S[s] += 0.3 * np.sin(2 * np.pi * g.uniform(0.002, 0.02) * t + g.uniform(0, 2 * np.pi))
    X = S + sigma * g.standard_normal(S.shape)
    M = g.random(S.shape) < p_imp
    X[M] += g.choice([-1, 1], M.sum()) * g.uniform(1.5, 4.0, M.sum())
    return torch.tensor(X, dtype=torch.float32), torch.tensor(S, dtype=torch.float32)


class RankBank(nn.Module):
    """C parallel 2-stage NEst chains (centered windows) + learned affine mix: a 'rank layer' network."""
    def __init__(self, C=4, m=7):
        super().__init__()
        self.chains = nn.ModuleList([Cascade([m, m], centered_windows) for _ in range(C)])
        self.mix = nn.Linear(C, 1)
        nn.init.constant_(self.mix.weight, 1.0 / C); nn.init.zeros_(self.mix.bias)

    def forward(self, x, tau=None, hard=False):
        Y = torch.stack([c(x, tau, hard) for c in self.chains], -1)
        return self.mix(Y).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self, ch=8, k=7, depth=3):
        super().__init__()
        L = [nn.Conv1d(1, ch, k, padding=k // 2), nn.ReLU()]
        for _ in range(depth - 2):
            L += [nn.Conv1d(ch, ch, k, padding=k // 2), nn.ReLU()]
        L += [nn.Conv1d(ch, 1, k, padding=k // 2)]
        self.net = nn.Sequential(*L)

    def forward(self, x, tau=None, hard=False):
        return x + self.net(x.unsqueeze(1)).squeeze(1)       # residual


def train_generic(net, X, S, steps=2000, lr=3e-3, batch=32, anneal=False, seed=0, crop=256):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    for s in range(steps):
        idx = torch.randint(0, X.shape[0], (batch,))
        o = int(torch.randint(0, X.shape[-1] - crop, (1,)))
        xb, sb = X[idx, o:o + crop], S[idx, o:o + crop]
        tau = (1.0 * (0.01 / 1.0) ** (s / (0.8 * steps))) if (anneal and s < 0.8 * steps) else None
        loss = F.mse_loss(net(xb, tau), sb)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net


def wiener_fir(X, S, taps=15):
    W = centered_windows(X, taps).reshape(-1, taps).double()
    s = S.reshape(-1).double()
    Wb = torch.cat([W, torch.ones(len(W), 1, dtype=W.dtype)], 1)
    coef = torch.linalg.lstsq(Wb, s.unsqueeze(1)).solution.squeeze(1)
    return lambda x: (torch.cat([centered_windows(x, taps).double(),
                                  torch.ones(*x.shape, 1, dtype=torch.float64)], -1) @ coef).float()


def nparams(net):
    return sum(p.numel() for p in net.parameters())


def run_E3(seed=0):
    Xtr, Str = make_denoise_data(128, 1024, seed=seed)
    tests = {"in-dist p=10%": make_denoise_data(32, 1024, 0.10, seed=seed + 1),
             "OOD p=25%": make_denoise_data(32, 1024, 0.25, seed=seed + 2),
             "OOD p=10%, impulses x2": None}
    Xo, So = make_denoise_data(32, 1024, 0.10, seed=seed + 3)
    tests["OOD p=10%, impulses x2"] = (So + 2 * (Xo - So), So)   # same clean signal, noise+impulses doubled
    models = {}
    models["median-5"] = (lambda x: hard_rank_filter(x, list(range(5)), 3, centered_windows), 0)
    models["median-5 o median-5"] = (lambda x: hard_rank_filter(hard_rank_filter(x, list(range(5)), 3, centered_windows),
                                                                list(range(5)), 3, centered_windows), 0)
    models["Wiener FIR-15"] = (wiener_fir(Xtr, Str, 15), 16)
    torch.manual_seed(seed)
    nest = Cascade([7, 7], centered_windows)
    train_generic(nest, Xtr, Str, steps=3000, lr=0.03, anneal=False, seed=seed)
    models["NEst 2-stage (learned)"] = (lambda x, n=nest: n(x, hard=True), nparams(nest))
    bank = RankBank(4, 7)
    train_generic(bank, Xtr, Str, steps=3000, lr=0.03, anneal=False, seed=seed)
    models["NEst bank C=4 (learned)"] = (lambda x, n=bank: n(x), nparams(bank))
    cnn_s = CNN1D(ch=4, k=5, depth=3); train_generic(cnn_s, Xtr, Str, steps=3000, lr=3e-3, seed=seed)
    models["CNN-ReLU small"] = (cnn_s, nparams(cnn_s))
    cnn_l = CNN1D(ch=32, k=7, depth=5); train_generic(cnn_l, Xtr, Str, steps=3000, lr=2e-3, seed=seed)
    models["CNN-ReLU large"] = (cnn_l, nparams(cnn_l))
    out = {}
    with torch.no_grad():
        for name, (f, npar) in models.items():
            row = {"params": npar}
            for tn, (X, S) in tests.items():
                y = f(X)
                row[tn] = 10 * torch.log10(((y - S) ** 2).mean() / (S ** 2).mean()).item()
            out[name] = row
        row = {"params": 0}
        for tn, (X, S) in tests.items():
            row[tn] = 10 * torch.log10(((X - S) ** 2).mean() / (S ** 2).mean()).item()
        out["(noisy input)"] = row
    return out, dict(nest=nest, bank=bank, cnn_s=cnn_s, cnn_l=cnn_l), tests


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "E1"
    if which == "E1":
        print(json.dumps(run_E1(), indent=1))
    elif which == "E2":
        from multiprocessing import Pool
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        torch.set_num_threads(1)
        with Pool(2) as p:
            R = p.map(e2_trial, range(n))
        json.dump(R, open("e2_results.json", "w"), indent=1)
        for r in R:
            print(r)
    elif which == "E3":
        out, _, _ = run_E3()
        json.dump(out, open("e3_results.json", "w"), indent=1)
        print(json.dumps(out, indent=1))
