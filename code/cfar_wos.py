"""
cfar_wos.py — Repetition-weight (WOS) CFAR estimators in the log domain, NP-minimax trained from OS / GO-OS.

  WOS-log 1 etapa : log Z = WOS_{w,r}(log x_ref) + c                     (32 per-cell repetitions, init = OS 24/32)
  WOS-log 2 etapas: u_h = WOS_{w_h,r_h}(log x_half_h);  log Z = WOS_{v,s}(u_1,u_2) + c   (init = GO-OS 12/16 -> max)
Homogeneous of degree 1 (quantile of log samples shifts with log-scale)  ->  CFAR by construction.
Also: re-run of the note-4 NEst GO-OS refinement with another seed, and GO-OS on a second calibration bank,
to measure seed / calibration noise of the 0.1–0.8 dB gains.
"""
import math, json, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from cfar import NH, REF, ref, Z_go_os, CFARNEst2
from cfar_np import train_np, calibrate, pd_table, build_h0_bank, Named
from wos import wos_hard


class WOSMat(nn.Module):
    """WOS over the last dim of U (n,m) with STE gradient (implicit function of tau-smoothed CDF)."""
    def __init__(self, m, k_init, tau=0.3, sharp=None):
        super().__init__()
        self.tau = tau
        self.alpha = nn.Parameter(torch.full((m,), math.log(math.expm1(1.0))))
        r0 = (k_init - 0.5) / m if sharp is None else sharp
        self.rho = nn.Parameter(torch.tensor(math.log(r0 / (1 - r0))))
    def w(self): return F.softplus(self.alpha)
    def r(self): return torch.sigmoid(self.rho)
    def forward(self, U):
        w, r = self.w(), self.r()
        with torch.no_grad():
            y0 = wos_hard(U, w, r)
        if not torch.is_grad_enabled():
            return y0
        s = torch.sigmoid((y0.unsqueeze(-1) - U) / self.tau)
        G = (w * s).sum(-1) - r * w.sum()
        Gy = ((w * s * (1 - s)).sum(-1) / self.tau).detach() + 1e-6
        return y0 - (G - G.detach()) / Gy


class CFARWOS1(nn.Module):
    def __init__(self):
        super().__init__(); self.l = WOSMat(2 * NH, 24); self.c = nn.Parameter(torch.tensor(math.log(20.0)))
    def forward(self, X): return self.l(torch.log(ref(X))) + self.c


class CFARWOS2(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = nn.ModuleList([WOSMat(NH, 12), WOSMat(NH, 12)])
        self.top = WOSMat(2, 2, sharp=0.99)          # cumw >= 0.99*2 -> second (max) of 2
        self.c = nn.Parameter(torch.tensor(math.log(28.7)))
    def forward(self, X):
        U = torch.log(ref(X))
        V = torch.stack([self.h[0](U[:, :NH]), self.h[1](U[:, NH:])], 1)
        return self.top(V) + self.c


def init_go(model, k=12):
    with torch.no_grad():
        for l in model.l1: l.theta.zero_(); l.theta[k - 1] = 8.0; l.b.zero_()
        model.l2.theta.zero_(); model.l2.theta[-1] = 8.0; model.l2.b.zero_(); model.l2.b += math.log(28.7)
    return model


def job(spec):
    torch.set_num_threads(1)
    name, seed = spec
    if name == "WOS-log 1 etapa (init OS)":
        m = Named(CFARWOS1(), name)
    elif name == "WOS-log 2 etapas (init GO-OS)":
        m = Named(CFARWOS2(), name)
    else:
        m = Named(init_go(CFARNEst2(2)), name)
    with torch.no_grad():                         # train_np adds log(20) to the last NEstLog.b -> undo for NEst
        pass
    m, hist = train_np(m, steps=800, lr=0.005 if "NEst" in name else 0.02, seed=seed)
    torch.save(m.state_dict(), f"cfar_wos_{name.split()[0]}_{seed}.pt".replace("/", "_"))
    return name, seed, m


if __name__ == "__main__":
    from multiprocessing import Pool
    t0 = time.time()
    specs = [("WOS-log 2 etapas (init GO-OS)", 1), ("WOS-log 2 etapas (init GO-OS)", 2),
             ("WOS-log 1 etapa (init OS)", 1), ("NEst-log 2 etapas 2×16 (init GO-OS, NP) semilla 2", 2)]
    with Pool(2) as p:
        trained = p.map(job, specs)
    print("trained %.0fs" % (time.time() - t0), flush=True)
    torch.set_num_threads(2)
    res = {}
    for bank_seed in [7, 77]:
        bank = build_h0_bank(20000, seed=bank_seed)
        dets = [("GO-OS", Z_go_os)] + [(f"{n} s{s}", (lambda X, m=m: torch.exp(m(X)))) for n, s, m in trained]
        for k, f in dets:
            a_hom, a_mm, worst, ph = calibrate(f, bank)
            res[f"{k} | banco {bank_seed}"] = dict(alpha_hom=a_hom, alpha_mm=a_mm, worst=worst, max_pfa_hom=ph,
                                                   overhead_db=10 * math.log10(a_mm / a_hom), pd_mm=pd_table(f, a_mm))
            print(k, bank_seed, "a_mm %.3f worst %s" % (a_mm, worst), flush=True)
    res["_params"] = {f"{n} s{s}": {k: v.detach().numpy().round(3).tolist() for k, v in m.state_dict().items()} for n, s, m in trained}
    json.dump(res, open("cfar_wos_results.json", "w"), indent=1)
    print("done %.0fs" % (time.time() - t0))
