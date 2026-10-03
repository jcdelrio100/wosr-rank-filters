"""B2 — Operator coverage matrix: which operator classes can each model learn (system identification, thesis bench:
i.i.d. U[-5,5] input, SNR 35 dB). Metric: NMSE above the floor of the true operator on the noisy input (dB)."""
import sys, os, json, time, itertools
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wosnet import *
from rankfilters import hard_rank_filter, centered_windows, random_target, RankBank
from wos import wos_hard

def cw(x, m): return centered_windows(x, m)
def rank_c(x, mask, r):
    # centred window of odd length (an even max(mask)+1 would give T+1 outputs); unchanged for odd lengths
    m = max(mask) + 1; m += 1 - m % 2
    return centered_windows(x, m)[..., mask].sort(-1).values[..., r - 1]
def fir(x, h):
    h = torch.tensor(h, dtype=x.dtype); m = len(h)
    return (cw(x, m) * h).sum(-1)

def e2(t):
    m1, r1, m2, r2 = random_target(np.random.default_rng(1000 + t))
    return lambda x: rank_c(rank_c(x, m1, r1), m2, r2)

rng = np.random.default_rng(77); wt = torch.tensor([1., 3., 0., 2., 4., 1., 2.]); WT = wt.sum()
TARGETS = {
    "Mediana 7": ("rango", lambda x: rank_c(x, list(range(7)), 4)),
    "Erosión plana 5": ("morfológico", lambda x: rank_c(x, list(range(5)), 1)),
    "Apertura 5 (erosión→dilatación)": ("morfológico 2 etapas", lambda x: rank_c(rank_c(x, list(range(5)), 1), list(range(5)), 5)),
    "WOS ponderado (1,3,0,2,4,1,2)": ("WOS", lambda x: wos_hard(cw(x, 7), wt, torch.tensor((7 - 0.5) / WT))),
    "Cascada E2 t1": ("cascada de rango", e2(1)),
    "Cascada E2 t12": ("cascada de rango", e2(12)),
    "Cascada E2 t5": ("cascada de rango", e2(5)),
    "Media móvil 5": ("lineal, pesos ≥0", lambda x: fir(x, [0.2] * 5)),
    "Paso alto [-1,2,-1]/2": ("lineal con signo", lambda x: fir(x, [-0.5, 1.0, -0.5])),
    "Top-hat x − apertura": ("no creciente", lambda x: x - rank_c(rank_c(x, list(range(5)), 1), list(range(5)), 5)),
    "No linealidad puntual x²/5": ("no invariante en nivel", lambda x: x ** 2 / 5),
}

class NEstBank(torch.nn.Module):
    def __init__(s): super().__init__(); s.b = RankBank(4, 7)
    def forward(s, x): return s.b(x)

MODELS = {
    "WOS-R cadena": lambda: WOSChain(2, 1, 7),
    "WOS-R banco C=4": lambda: WOSBank(4, 2, 1, 7, anneal=2100),
    "NEst banco C=4": lambda: NEstBank(),
    "MLP": lambda: MLP1D(15, 64),
    "CNN-M": lambda: CNN1D(16, 7, 4),
    "Transformer": lambda: Transformer1D(),
}

def data(f, n, T, seed, snr_db=35.0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(n, T, generator=g) * 2 - 1) * 5
    d = f(x)
    pn = (25 / 3) / 10 ** (snr_db / 10)
    return x + (3 * pn) ** 0.5 * (torch.rand(n, T, generator=g) * 2 - 1), d

def nmse(y, d): return 10 * torch.log10(((y - d)[:, 8:-8] ** 2).mean() / (d[:, 8:-8] ** 2).mean()).item()

def job(spec):
    torch.set_num_threads(1)
    tname, mname, seed = spec
    cls, f = TARGETS[tname]
    X, D = data(f, 64, 2048, seed); Xt, Dt = data(f, 8, 2048, seed + 100)
    floor = nmse(f(Xt), Dt)
    torch.manual_seed(seed); model = MODELS[mname]()
    lr = 0.03 if (mname.startswith("WOS") or mname.startswith("NEst")) else 2e-3
    opt = torch.optim.Adam(model.parameters(), lr=lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000, eta_min=lr * 0.05)
    model.train()
    for s in range(3000):
        i = torch.randint(0, 64, (16,)); o = int(torch.randint(0, 2048 - 256, (1,)))
        loss = F.mse_loss(model(X[i, o:o + 256])[:, 8:-8], D[i, o:o + 256][:, 8:-8])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    set_deploy(model)
    with torch.no_grad(): g = nmse(model(Xt), Dt) - floor
    out = dict(target=tname, cls=cls, model=mname, seed=seed, gap=g, floor=floor, params=n_params(model))
    if mname.startswith("WOS"):
        with torch.no_grad():
            set_deploy(model, ramp=True); out["gap_ramp_deploy"] = nmse(model(Xt), Dt) - floor
    return out

if __name__ == "__main__":
    from multiprocessing import Pool
    outp = "b2_results.json"
    jobs = list(itertools.product(TARGETS, MODELS, [0, 1]))
    R = json.load(open(outp)) if os.path.exists(outp) else []
    have = {(r["target"], r["model"], r["seed"]) for r in R}
    jobs = [j for j in jobs if j not in have]
    print("pending", len(jobs), flush=True)
    with Pool(2) as p:
        for r in p.imap_unordered(job, jobs):
            R.append(r); json.dump(R, open(outp, "w"), indent=0)
            print(r["target"], "|", r["model"], r["seed"], round(r["gap"], 1), flush=True)
