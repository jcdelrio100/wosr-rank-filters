"""B6 — training curves for the B1 impulsive task: deployed in-distribution NMSE every 100 steps vs wall-clock training
time (evaluation time excluded). Same models, data, optimiser and steps as B1. 3 seeds."""
import sys, os, json, time, itertools
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wosnet import *
import b1_signal1d as b1
from rankfilters import make_denoise_data

def job(spec):
    torch.set_num_threads(1)
    name, seed = spec
    X, S = make_denoise_data(128, 1024, 0.10, seed=seed)
    Xt, St = make_denoise_data(32, 1024, 0.10, seed=seed + 1)
    torch.manual_seed(seed); model = b1.MODELS[name]()
    lr = next((v for k, v in b1.LR.items() if name.startswith(k)), 2e-3)
    steps, batch, crop = 3000, 32, 256
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    curve, t_train = [], 0.0
    for s in range(steps + 1):
        if s % 100 == 0:
            set_deploy(model)
            with torch.no_grad(): curve.append((s, t_train, b1.nmse(model(Xt), St)))
            model.train()
        if s == steps: break
        t0 = time.perf_counter()
        i = torch.randint(0, X.shape[0], (batch,)); o = int(torch.randint(0, X.shape[-1] - crop, (1,)))
        loss = F.mse_loss(model(X[i, o:o + crop]), S[i, o:o + crop])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        t_train += time.perf_counter() - t0
    return dict(model=name, seed=seed, curve=curve, params=n_params(model))

if __name__ == "__main__":
    from multiprocessing import Pool
    names = [n for n in b1.MODELS if b1.MODELS[n] is not None]
    jobs = list(itertools.product(names, [0, 1, 2]))
    outp = "b6_results.json"
    R = json.load(open(outp)) if os.path.exists(outp) else []
    have = {(r["model"], r["seed"]) for r in R}
    jobs = [j for j in jobs if j not in have]
    jobs.sort(key=lambda j: 0 if ("Transformer" in j[0] or "C=8" in j[0]) else 1)
    with Pool(2) as p:
        for r in p.imap_unordered(job, jobs):
            R.append(r); json.dump(R, open(outp, "w"))
            print(r["model"], r["seed"], "final %.2f dB in %.0f s" % (r["curve"][-1][2], r["curve"][-1][1]), flush=True)
