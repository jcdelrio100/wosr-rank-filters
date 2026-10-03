"""
cfar_np.py — Minimax Neyman–Pearson design and evaluation of rank-cascade CFAR detectors.

Evaluation (fair, for every detector):
  alpha_hom : Pfa = 1e-4 in homogeneous K(nu=2) clutter (classical design)
  alpha_mm  : smallest alpha with Pfa <= 1e-4 in EVERY H0 scenario: homogeneous + clutter edges of
              -20/-10/+10/+20 dB at every position of the window (145 scenarios)  -> minimax CFAR
  Pfa is computed semi-analytically: the CUT is independent of the reference cells, so
      Pfa = E_ref[ S_nu( alpha Z / L_CUT ) ],  S_nu(c) = 2 (nu c)^{nu/2} K_nu(2 sqrt(nu c)) / Gamma(nu)
  (exact survival of unit-mean K power) -> unbiased, low variance at 1e-4.
  Pd by Monte Carlo, Swerling-I target, homogeneous and with 2/4/8 interferers (INR 20 dB).
Training (learned detectors): maximise a smooth Pd surrogate subject to a penalty on log Pfa above 1e-4 in
  25 H0 groups (homogeneous + 4 edge levels x 6 position bins), Pfa estimated with 16 texture draws per sample.
"""
import math, json, time, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.special import kv, gammaln
from cfar import (simulate, NH, G, W, CUT, REF, Z_ca, Z_os, Z_go_ca, Z_so_ca, Z_go_os, NEstLog,
                  CFARNEst1, CFARNEst2, CFARMLP, init_os_like)

NU = 2.0
LOGPFA = math.log(1e-4)


def S_k(c, nu=NU):
    c = np.maximum(c, 1e-300)
    a = 2 * np.sqrt(nu * c)
    with np.errstate(over="ignore", under="ignore"):
        v = np.exp(np.log(2) + (nu / 2) * np.log(nu * c) + np.log(kv(nu, a) + 1e-300) - gammaln(nu))
    return np.where(a > 700, 0.0, v)


# --------------------------------------------------------------------------- scenario banks (H0)
def h0_scenarios():
    S = [("hom", dict())]
    for e in [-20.0, -10.0, 10.0, 20.0]:
        for pos in range(1, W):
            S.append((f"edge{e:+.0f}@{pos}", dict(edge_db=e, edge_pos=pos)))
    return S


@torch.no_grad()
def logZ_all(fn, X, chunk=200000):
    return torch.cat([torch.log(fn(X[i:i + chunk]).clamp_min(1e-30)) for i in range(0, len(X), chunk)])


@torch.no_grad()
def build_h0_bank(n_per, seed=7):
    gen = torch.Generator().manual_seed(seed)
    bank = []
    for name, kw in h0_scenarios():
        X, L = simulate(n_per, gen, nu=NU, **kw)
        bank.append((name, X, L))
    return bank


def pfa_curve(fn, bank, alphas):
    """Return array (n_scen, n_alpha) of exact-conditional Pfa."""
    out = np.zeros((len(bank), len(alphas)))
    for i, (_, X, L) in enumerate(bank):
        r = (torch.exp(logZ_all(fn, X)) / L).numpy()
        for j, a in enumerate(alphas):
            out[i, j] = S_k(a * r).mean()
    return out


def calibrate(fn, bank, target=1e-4):
    """alpha_hom (scenario 0) and alpha_mm (max over scenarios) by bisection in log-alpha."""
    rs = [(torch.exp(logZ_all(fn, X)) / L).numpy() for _, X, L in bank]
    def pfa(a, idx): return S_k(a * rs[idx]).mean()
    def bis(f):
        lo, hi = 1e-3, 1e6
        for _ in range(50):
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if f(mid) > target else (lo, mid)
        return hi
    a_hom = bis(lambda a: pfa(a, 0))
    a_mm = bis(lambda a: max(pfa(a, i) for i in range(len(rs))))
    worst = max(range(len(rs)), key=lambda i: pfa(a_mm, i))
    return a_hom, a_mm, bank[worst][0], max(pfa(a_hom, i) for i in range(len(rs)))


@torch.no_grad()
def pd_table(fn, alpha, seed=11, n=200000, snrs=(5, 10, 15, 20, 25, 30), ints=(0, 2, 4, 8)):
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for ni in ints:
        row = {}
        for s in snrs:
            X, L = simulate(n, gen, nu=NU, n_int=ni, inr_db=20.0, snr_db=float(s))
            row[s] = (X[:, CUT] > alpha * torch.exp(logZ_all(fn, X))).float().mean().item()
        out[ni] = row
    return out


# --------------------------------------------------------------------------- NP-minimax training
def h0_groups(gen, n_per=1500):
    groups = [simulate(n_per, gen, nu=NU)]
    bins = np.array_split(np.arange(1, W), 6)
    for e in [-20.0, -10.0, 10.0, 20.0]:
        for b in bins:
            X, L = [], []
            for pos in np.random.default_rng(int(torch.randint(0, 1 << 30, (1,), generator=gen))).choice(b, 3):
                x, l = simulate(n_per // 3, gen, nu=NU, edge_db=e, edge_pos=int(pos)); X.append(x); L.append(l)
            groups.append((torch.cat(X), torch.cat(L)))
    return groups


def train_np(model, steps=1000, lr=0.02, seed=0, lam=1.0, kappa=0.25, n_tau=16):
    gen = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    with torch.no_grad():                                    # start near the right threshold scale (OS alpha ~ 20)
        for mod in model.modules():
            if isinstance(mod, NEstLog):
                pass
        last = [m for m in model.modules() if isinstance(m, NEstLog)]
        if last:
            last[-1].b += math.log(20.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gamma = torch.distributions.Gamma(torch.tensor(NU), torch.tensor(NU))
    hist = []
    for s in range(steps):
        pen = 0.0; lp_max = -1e9
        for X, L in h0_groups(gen):
            logZ = model(X)
            tau = gamma.sample((X.shape[0], n_tau))
            pf = torch.exp(-torch.exp(logZ)[:, None] / (L[:, None] * tau)).mean()
            lp = torch.log(pf + 1e-30)
            pen = pen + F.relu(lp - LOGPFA) ** 2
            lp_max = max(lp_max, lp.item())
        pd = 0.0
        for ni in (0, 2, 4, 8):
            X, L = simulate(2000, gen, nu=NU, n_int=ni, inr_db=20.0, snr_db=float(torch.empty(1).uniform_(10, 20, generator=gen)))
            pd = pd + torch.sigmoid((torch.log(X[:, CUT]) - model(X)) / kappa).mean() / 4
        loss = -pd + lam * pen
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if s % 100 == 0:
            hist.append((s, float(pd.detach()), lp_max / math.log(10)))
            print(model.name, s, "Pd~%.3f" % float(pd), "max log10 Pfa %.2f" % (lp_max / math.log(10)), flush=True)
    return model, hist


class Named(nn.Module):
    def __init__(self, inner, name):
        super().__init__(); self.inner = inner; self.name = name
    def forward(self, X): return self.inner(X)


if __name__ == "__main__":
    torch.set_num_threads(2)
    t0 = time.time()
    learned = []
    for mk, nm in [(lambda: init_os_like(CFARNEst1()), "NEst-log 1 etapa (NP-minimax)"),
                   (lambda: init_os_like(CFARNEst2(2)), "NEst-log 2 etapas 2×16 (NP-minimax)"),
                   (lambda: init_os_like(CFARNEst2(4)), "NEst-log 2 etapas 4×8 (NP-minimax)"),
                   (lambda: CFARMLP(), "MLP-log equivariante (NP-minimax)")]:
        m = Named(mk(), nm)
        if "MLP" in nm:
            with torch.no_grad(): m.inner.net[-1].bias += math.log(20.0)
            m, h = train_np(m, lr=1e-3)
        else:
            m, h = train_np(m)
        learned.append(m)
    torch.save({m.name: m.state_dict() for m in learned}, "cfar_np_models.pt")
    # previously trained log-MSE models (objective ablation)
    old = torch.load("cfar_models.pt")
    prev = []
    for cls, nm in [(CFARNEst1, "NEst-log 1 etapa"), (lambda: CFARNEst2(2), "NEst-log 2 etapas (2×16)")]:
        m = cls(); m.load_state_dict(old[nm]); prev.append(Named(m, nm + " (MSE-log)"))
    D = {"CA": Z_ca, "OS (k=24)": Z_os, "GO-CA": Z_go_ca, "SO-CA": Z_so_ca, "GO-OS": Z_go_os}
    for m in prev + learned:
        D[m.name] = (lambda X, m=m: torch.exp(m(X)))
    print("training done %.0fs" % (time.time() - t0), flush=True)
    bank = build_h0_bank(20000)
    res = {}
    for k, f in D.items():
        a_hom, a_mm, worst, pfa_hom_worst = calibrate(f, bank)
        res[k] = dict(alpha_hom=a_hom, alpha_mm=a_mm, worst_scenario=worst, max_pfa_with_alpha_hom=pfa_hom_worst,
                      overhead_db=10 * math.log10(a_mm / a_hom),
                      pd_hom_cal=pd_table(f, a_hom), pd_mm_cal=pd_table(f, a_mm))
        print(k, "a_hom %.1f a_mm %.1f (+%.2f dB) worst %s maxPfa(a_hom)=%.2e" %
              (a_hom, a_mm, res[k]["overhead_db"], worst, pfa_hom_worst), flush=True)
    res["_params"] = {m.name: {k: v.detach().numpy().round(3).tolist() for k, v in m.state_dict().items()}
                      for m in learned if "MLP" not in m.name}
    json.dump(res, open("cfar_np_results.json", "w"), indent=1)
    print("done %.0fs" % (time.time() - t0))
