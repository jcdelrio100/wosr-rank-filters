"""
cfar_corr.py — Decisive test of positional repetition weights: does 'boosting' cells pay off when POSITION
carries information?  K clutter with range-correlated texture (Gaussian-copula AR(1), coefficient rho):
cells near the CUT share its texture -> more informative about the CUT's local clutter level.

Detectors (log domain, homogeneous): CA, OS(24), GO-OS(12), learned WOS-1 with free per-cell repetitions,
and the same WOS with repetitions forced uniform (only the quantile r learned) -> isolates the positional effect.
Pfa exact conditional on the textures:  Pfa = E[ exp(-alpha Z / (L tau_CUT)) ]   (speckle exponential, independent).
"""
import math, json, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.special import gammaincinv, ndtr
from cfar import NH, G, W, CUT, REF, ref, Z_ca, Z_os, Z_go_os
from cfar_wos import WOSMat

NU = 2.0


def texture(n, rho, rng):
    g = np.empty((n, W)); g[:, 0] = rng.standard_normal(n)
    e = rng.standard_normal((n, W))
    for j in range(1, W):
        g[:, j] = rho * g[:, j - 1] + math.sqrt(1 - rho ** 2) * e[:, j]
    u = np.clip(ndtr(g), 1e-12, 1 - 1e-12)
    return gammaincinv(NU, u) / NU                     # Gamma(nu, 1/nu): unit mean


def simulate(n, rng, rho, n_int=0, inr_db=20.0, snr_db=None):
    tau = texture(n, rho, rng)
    L = 10 ** (rng.uniform(-20, 20, n) / 10)
    x = L[:, None] * tau * rng.exponential(size=(n, W))
    for _ in range(n_int):
        c = rng.choice(REF, n)
        x[np.arange(n), c] += L * tau[np.arange(n), c] * 10 ** (inr_db / 10) * rng.exponential(size=n)
    if snr_db is not None:
        x[:, CUT] += L * 10 ** (snr_db / 10) * rng.exponential(size=n)        # SNR w.r.t. mean clutter power
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    return T(x), T(L), T(tau[:, CUT])


class WOS1(nn.Module):
    def __init__(self, uniform=False):
        super().__init__(); self.l = WOSMat(2 * NH, 24); self.c = nn.Parameter(torch.tensor(math.log(20.0)))
        self.uniform = uniform
        if uniform: self.l.alpha.requires_grad_(False)
    def forward(self, X): return self.l(torch.log(ref(X))) + self.c


def train(model, rho, steps=800, lr=0.02, seed=0, kappa=0.25):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for s in range(steps):
        X, L, tc = simulate(6000, rng, rho)
        pf = torch.exp(-torch.exp(model(X)) / (L * tc)).mean()
        pen = F.relu(torch.log(pf + 1e-30) - math.log(1e-4)) ** 2
        pd = 0
        for ni in (0, 2, 4, 8):
            X, L, _ = simulate(1500, rng, rho, n_int=ni, snr_db=float(rng.uniform(10, 20)))
            pd = pd + torch.sigmoid((torch.log(X[:, CUT]) - model(X)) / kappa).mean() / 4
        loss = -pd + pen
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def calib(f, rho, n=400000, seed=99, target=1e-4):
    X, L, tc = simulate(n, np.random.default_rng(seed), rho)
    r = (f(X) / (L * tc)).numpy()
    lo, hi = 1e-3, 1e6
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if np.exp(-mid * r).mean() > target else (lo, mid)
    return hi


@torch.no_grad()
def pd_curve(f, a, rho, seed=11, n=100000):
    rng = np.random.default_rng(seed); out = {}
    for ni in (0, 2, 4, 8):
        out[ni] = {}
        for s in (5, 10, 15, 20, 25, 30):
            X, L, _ = simulate(n, rng, rho, n_int=ni, snr_db=float(s))
            out[ni][s] = (X[:, CUT] > a * f(X)).float().mean().item()
    return out


def snr50(row):
    xs = sorted(row); ys = [row[s] for s in xs]
    for i in range(len(xs) - 1):
        if ys[i] < 0.5 <= ys[i + 1]:
            return xs[i] + (xs[i + 1] - xs[i]) * (0.5 - ys[i]) / (ys[i + 1] - ys[i])
    return float("inf")


def job(spec):
    torch.set_num_threads(1)
    rho, uniform, seed = spec
    m = train(WOS1(uniform), rho, seed=seed)
    return spec, m


if __name__ == "__main__":
    from multiprocessing import Pool
    t0 = time.time()
    specs = [(rho, u, s) for rho in (0.0, 0.9) for u in (False, True) for s in (0, 1)]
    with Pool(2) as p:
        trained = p.map(job, specs)
    print("trained %.0fs" % (time.time() - t0), flush=True)
    torch.set_num_threads(2)
    res = {}
    for rho in (0.0, 0.9):
        dets = {"CA": Z_ca, "OS (k=24)": Z_os, "GO-OS (k=12)": Z_go_os, "GO-OS (k=11)": lambda X: Z_go_os(X, k=11)}
        for (r_, u, s), m in trained:
            if r_ == rho:
                dets[f"WOS {'uniforme (solo r)' if u else 'repeticiones libres'} s{s}"] = (lambda X, m=m: torch.exp(m(X)))
        for k, f in dets.items():
            a = calib(f, rho); pdc = pd_curve(f, a, rho)
            res[f"rho={rho} | {k}"] = dict(alpha=a, snr50={ni: snr50(pdc[ni]) for ni in pdc}, pd=pdc)
            print(rho, k, {ni: round(snr50(pdc[ni]), 2) for ni in pdc}, flush=True)
        for (r_, u, s), m in trained:
            if r_ == rho:
                w = F.softplus(m.l.alpha).detach().numpy(); res[f"rho={rho} | w {'u' if u else 'free'} s{s}"] = dict(
                    w=np.round(w / w.max(), 3).tolist(), r=float(torch.sigmoid(m.l.rho)))
    json.dump(res, open("cfar_corr_results.json", "w"), indent=1)
    print("done %.0fs" % (time.time() - t0))
