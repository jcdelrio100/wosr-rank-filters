"""B1 — 1D signal restoration: impulsive (train p=10 %, tests in-dist / OOD p=25 % / OOD impulses x2) and Gaussian
(sigma=0.2). WOS-R (ramp-trained, exact deploy) vs residual CNN S/M/L, MLP, Transformer, NEst bank, median."""
import sys, os, json, time, itertools
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wosnet import *
from rankfilters import make_denoise_data, RankBank, centered_windows

def tests(kind, seed):
    if kind == "imp":
        T = {"in-dist": make_denoise_data(32, 1024, 0.10, seed=seed + 1), "OOD p=25%": make_denoise_data(32, 1024, 0.25, seed=seed + 2)}
        Xo, So = make_denoise_data(32, 1024, 0.10, seed=seed + 3); T["OOD x2"] = (So + 2 * (Xo - So), So)
    else:
        T = {"gauss s=0.2": make_denoise_data(32, 1024, 0.0, sigma=0.2, seed=seed + 1),
             "gauss s=0.4 (OOD)": make_denoise_data(32, 1024, 0.0, sigma=0.4, seed=seed + 2)}
    return T

def nmse(y, S): return 10 * torch.log10(((y - S) ** 2).mean() / (S ** 2).mean()).item()

def train(model, X, S, steps=3000, lr=2e-3, batch=32, crop=256, seed=0):
    torch.manual_seed(seed); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    for s in range(steps):
        i = torch.randint(0, X.shape[0], (batch,)); o = int(torch.randint(0, X.shape[-1] - crop, (1,)))
        out = model(X[i, o:o + crop])
        loss = F.mse_loss(out, S[i, o:o + crop])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return model

class NEstBank(torch.nn.Module):
    def __init__(s): super().__init__(); s.b = RankBank(4, 7)
    def forward(s, x): return s.b(x)

MODELS = {
    "Mediana 5": None,
    "WOS-R cadena (2×7)": lambda: WOSChain(2, 1, 7),
    "WOS-R banco C=4": lambda: WOSBank(4, 2, 1, 7, anneal=2100),
    "WOS-R banco C=8": lambda: WOSBank(8, 2, 1, 7, anneal=2100),
    "NEst banco C=4": lambda: NEstBank(),
    "CNN-S": lambda: CNN1D(4, 5, 3),
    "CNN-M": lambda: CNN1D(16, 7, 4),
    "CNN-L": lambda: CNN1D(32, 7, 5),
    "MLP (ventana 15)": lambda: MLP1D(15, 64),
    "Transformer": lambda: Transformer1D(),
}
LR = {"WOS": 0.03, "NEst": 0.03}

def job(spec):
    torch.set_num_threads(1)
    kind, name, seed = spec
    if kind == "imp": X, S = make_denoise_data(128, 1024, 0.10, seed=seed)
    else: X, S = make_denoise_data(128, 1024, 0.0, sigma=0.2, seed=seed)
    T = tests(kind, seed); out = dict(kind=kind, model=name, seed=seed)
    t0 = time.time()
    if MODELS[name] is None:
        f = lambda x: median_nd(x, 5, 1); model = None
    else:
        model = MODELS[name]()
        lr = next((v for k, v in LR.items() if name.startswith(k)), 2e-3)
        model = train(model, X, S, lr=lr, seed=seed)
        set_deploy(model, ramp=False); f = model
    out["train_s"] = round(time.time() - t0, 1)
    with torch.no_grad():
        for k, (Xt, St) in T.items(): out[k] = nmse(f(Xt), St)
        if model is not None and name.startswith("WOS"):
            set_deploy(model, ramp=True)
            out["ramp_deploy"] = {k: nmse(model(Xt), St) for k, (Xt, St) in T.items()}
            set_deploy(model, ramp=False)
        x = torch.randn(1, 4096)
        out["latency_ms_4k"] = 1e3 * latency(f, x)
    out["params"] = n_params(model) if model is not None else 0
    out["mults"] = mults_per_output(model) if model is not None else 0
    out["compares"] = compares_per_output(model) if model is not None else 5 * 3
    return out

if __name__ == "__main__":
    from multiprocessing import Pool
    outp = sys.argv[1] if len(sys.argv) > 1 else "b1_results.json"
    jobs = list(itertools.product(["imp", "gauss"], MODELS, [0, 1, 2]))
    R = json.load(open(outp)) if os.path.exists(outp) else []
    have = {(r["kind"], r["model"], r["seed"]) for r in R}
    jobs = [j for j in jobs if j not in have]
    print("pending", len(jobs), flush=True)
    with Pool(2) as p:
        for r in p.imap_unordered(job, jobs):
            R.append(r); json.dump(R, open(outp, "w"), indent=0)
            print(r["kind"], r["model"], r["seed"], {k: round(v, 1) for k, v in r.items() if isinstance(v, float) and k not in ("train_s",)}, flush=True)
