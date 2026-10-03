"""B4 — CFAR with range-correlated K clutter (rho in {0, 0.9}), Pfa = 1e-4 (homogeneous calibration, note 6 protocol).
WOS-R (ramp-trained, exact deploy, 32 per-cell repetitions, log domain) vs ML baselines trained with the SAME
Neyman–Pearson objective: equivariant MLP and equivariant Transformer over the 32 log-cells; classical OS / GO-OS."""
import sys, os, json, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from cfar_corr import train, calib, pd_curve, snr50, NH, ref
from cfar import Z_os, Z_go_os
from ramp import ramp_wos

class WOSRCFAR(nn.Module):
    def __init__(s):
        super().__init__(); s.alpha = nn.Parameter(torch.full((2 * NH,), math.log(math.expm1(1.0))))
        s.rho = nn.Parameter(torch.tensor(math.log((23.5 / 32) / (1 - 23.5 / 32)))); s.c = nn.Parameter(torch.tensor(math.log(20.0)))
    def forward(s, X):
        return ramp_wos(torch.log(ref(X)), F.softplus(s.alpha), torch.sigmoid(s.rho), 1.0 if s.training else 0.0) + s.c

class MLPCFAR(nn.Module):
    def __init__(s, h=64):
        super().__init__(); s.net = nn.Sequential(nn.Linear(2 * NH, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        with torch.no_grad(): s.net[-1].bias.fill_(math.log(20.0))
    def forward(s, X):
        U = torch.log(ref(X)); m = U.mean(1, keepdim=True); return (m + s.net(U - m)).squeeze(1)

class TrCFAR(nn.Module):
    def __init__(s, d=16, heads=2, layers=2):
        super().__init__(); s.inp = nn.Linear(1, d); s.pos = nn.Parameter(0.02 * torch.randn(2 * NH, d))
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 32, dropout=0.0, batch_first=True), layers)
        s.out = nn.Linear(d, 1)
        with torch.no_grad(): s.out.bias.fill_(math.log(20.0))
    def forward(s, X):
        U = torch.log(ref(X)); m = U.mean(1, keepdim=True)
        h = s.tr(s.inp((U - m).unsqueeze(-1)) + s.pos)
        return (m + s.out(h.mean(1))).squeeze(1)

MODELS = {"WOS-R (32 repeticiones)": WOSRCFAR, "MLP equivariante": MLPCFAR, "Transformer equivariante": TrCFAR}

def job(spec):
    torch.set_num_threads(1)
    name, rho, seed = spec
    m = MODELS[name](); lr = 0.02 if name.startswith("WOS") else 1e-3
    m.train(); m = train(m, rho, seed=seed, lr=lr); m.eval()
    return spec, m

if __name__ == "__main__":
    from multiprocessing import Pool
    t0 = time.time()
    specs = [(n, rho, s) for n in MODELS for rho in (0.0, 0.9) for s in (0, 1)]
    ck = "bench/b4_models.pt"
    if os.path.exists(ck):
        sd = torch.load(ck); T = []
        for sp in specs:
            m = MODELS[sp[0]](); m.load_state_dict(sd[str(sp)]); m.eval(); T.append((sp, m))
    else:
        with Pool(2) as p: T = p.map(job, specs)
        torch.save({str(sp): m.state_dict() for sp, m in T}, ck)
    print("trained %.0fs" % (time.time() - t0), flush=True)
    def chunked(f, B=20000):                        # bounded memory for the attention model (400k-sample calibration)
        return lambda X: torch.cat([f(X[i:i + B]) for i in range(0, X.shape[0], B)])
    torch.set_num_threads(2); res = {}
    for rho in (0.0, 0.9):
        dets = {"OS (k=24)": Z_os, "GO-OS (k=11)": lambda X: Z_go_os(X, k=11)}
        for (n, r_, s), m in T:
            if r_ == rho: dets[f"{n} s{s}"] = chunked(lambda X, m=m: torch.exp(m(X)))
        for k, f in dets.items():
            with torch.no_grad():
                a = calib(f, rho); pdc = pd_curve(f, a, rho)
            res[f"rho={rho} | {k}"] = dict(snr50={ni: snr50(pdc[ni]) for ni in pdc})
            print(rho, k, {ni: round(snr50(pdc[ni]), 2) for ni in pdc}, flush=True)
    for (n, r_, s), m in T:
        res[f"params | {n}"] = sum(p.numel() for p in m.parameters())
    json.dump(res, open("bench/b4_results.json", "w"), indent=1)
    print("done %.0fs" % (time.time() - t0))
