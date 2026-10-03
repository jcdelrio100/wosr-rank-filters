"""B3 — Image restoration on Set12 / BSD68 (grayscale, [0,1]).
Regime 'imp': blind salt-and-pepper training, density ~ U[0.1, 0.6]; tests SP 10/30/50/70 % and RVIN 20/40 % (OOD type).
Regime 'gauss': Gaussian sigma = 25/255 training; tests sigma 25 and 50 (OOD).  Coverage: WOS is expected to be weak here.
Metrics: PSNR (dB), SSIM; parameters, multiplications and comparisons per pixel, CPU latency (512x512)."""
import sys, os, json, time, glob, itertools
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wosnet import *

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "img")

def load(dirname):
    out = []
    for f in sorted(glob.glob(os.path.join(D, dirname, "*.png"))):
        try:
            out.append(torch.tensor(np.asarray(Image.open(f).convert("L"), dtype=np.float32) / 255.0))
        except Exception:
            pass
    return out

def sp(x, p, g):
    u = torch.rand(x.shape, generator=g)
    y = x.clone(); y[u < p / 2] = 0.0; y[(u >= p / 2) & (u < p)] = 1.0; return y
def rvin(x, p, g):
    u = torch.rand(x.shape, generator=g); return torch.where(u < p, torch.rand(x.shape, generator=g), x)
def gauss(x, s, g): return x + s * torch.randn(x.shape, generator=g)

def patches(imgs, n, P, g):
    out = []
    for _ in range(n):
        im = imgs[int(torch.randint(0, len(imgs), (1,), generator=g))]
        i = int(torch.randint(0, im.shape[0] - P, (1,), generator=g)); j = int(torch.randint(0, im.shape[1] - P, (1,), generator=g))
        pt = im[i:i + P, j:j + P]
        k = int(torch.randint(0, 4, (1,), generator=g)); pt = torch.rot90(pt, k)
        if torch.rand(1, generator=g) < 0.5: pt = pt.flip(0)
        out.append(pt)
    return torch.stack(out)

def noisy_batch(C, regime, g):
    if regime == "imp":
        p = 0.1 + 0.5 * torch.rand(C.shape[0], generator=g)
        return torch.stack([sp(c, float(pp), g) for c, pp in zip(C, p)])
    return gauss(C, 25 / 255, g)

MODELS_IMP = {
    "Mediana 3×3": None, "Mediana 5×5": None, "Mediana adaptativa (7)": None,
    "WOS-R 3×3 ×2": lambda: WOSChain(2, 2, 3),
    "WOS-R 5×5 ×2": lambda: WOSChain(2, 2, 5),
    "WOS-R banco C=4 (3×3)": lambda: WOSBank(4, 2, 2, 3, anneal=1400),
    "WOS-R banco C=4 (5×5)": lambda: WOSBank(4, 2, 2, 5, anneal=1400),
    "MLP 5×5": lambda: MLP2D(5, 64),
    "DnCNN-S (6×16)": lambda: DnCNN(6, 16),
    "DnCNN-M (10×32)": lambda: DnCNN(10, 32),
    "DnCNN-17 (17×64)": lambda: DnCNN(17, 64),
    "Transformer ventana 8×8": lambda: WinTransformer2D(),
}
MODELS_GAUSS = {k: MODELS_IMP[k] for k in ["Mediana 3×3", "WOS-R banco C=4 (3×3)", "WOS-R banco C=4 (5×5)", "MLP 5×5",
                                            "DnCNN-S (6×16)", "DnCNN-M (10×32)", "Transformer ventana 8×8"]}
STEPS = {"WOS": 2000, "DnCNN-17": 2000}

def classical(name):
    if name == "Mediana 3×3": return lambda x: median_nd(x, 3, 2)
    if name == "Mediana 5×5": return lambda x: median_nd(x, 5, 2)
    return lambda x: adaptive_median_2d(x, 7)

def psnr(y, x): return float(10 * torch.log10(1.0 / ((y.clamp(0, 1) - x) ** 2).mean()))

def evaluate(f, sets, regime, seed=1234):
    g = torch.Generator().manual_seed(seed)
    conds = {"SP 10%": lambda x: sp(x, .1, g), "SP 30%": lambda x: sp(x, .3, g), "SP 50%": lambda x: sp(x, .5, g),
             "SP 70% (OOD)": lambda x: sp(x, .7, g), "RVIN 20% (OOD)": lambda x: rvin(x, .2, g), "RVIN 40% (OOD)": lambda x: rvin(x, .4, g)} \
        if regime == "imp" else {"Gauss σ=25": lambda x: gauss(x, 25 / 255, g), "Gauss σ=50 (OOD)": lambda x: gauss(x, 50 / 255, g)}
    res = {}
    with torch.no_grad():
        for sname, imgs in sets.items():
            for cname, nf in conds.items():
                ps, ss = [], []
                for im in imgs:
                    y = f(nf(im).unsqueeze(0))[0].clamp(0, 1)
                    ps.append(psnr(y, im)); ss.append(structural_similarity(y.numpy(), im.numpy(), data_range=1.0))
                res[f"{sname} | {cname}"] = dict(psnr=float(np.mean(ps)), ssim=float(np.mean(ss)))
    return res

def job(spec):
    torch.set_num_threads(1)
    regime, name, seed = spec
    train_imgs = load("Train400"); sets = {"Set12": load("Set12"), "BSD68": load("BSD68")}
    MODELS = MODELS_IMP if regime == "imp" else MODELS_GAUSS
    out = dict(regime=regime, model=name, seed=seed); t0 = time.time()
    if MODELS[name] is None:
        f = classical(name); model = None
    else:
        torch.manual_seed(seed); model = MODELS[name]()
        steps = next((v for k, v in STEPS.items() if name.startswith(k)), 3000)
        lr = 0.03 if name.startswith("WOS") else 1e-3
        opt = torch.optim.Adam(model.parameters(), lr=lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
        g = torch.Generator().manual_seed(seed); model.train()
        for s in range(steps):
            C = patches(train_imgs, 16, 48, g); N = noisy_batch(C, regime, g)
            loss = F.mse_loss(model(N), C)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        set_deploy(model); f = model
    out["train_s"] = round(time.time() - t0, 1)
    out["metrics"] = evaluate(f, sets, regime)
    if model is not None and name.startswith("WOS"):
        set_deploy(model, ramp=True); out["metrics_ramp_deploy"] = evaluate(model, {"Set12": sets["Set12"]}, regime); set_deploy(model)
        core = getattr(model, "base", model)
        out["weights"] = [[np.round((l.w() / l.w().max()).detach().numpy(), 2).tolist(), round(float(l.r()), 3)]
                          for c in (core.chains if isinstance(core, WOSBank) else [core]) for l in c.layers]
        if isinstance(core, WOSBank): out["mix"] = core.mix.detach().numpy().round(3).tolist()
    x = torch.rand(1, 512, 512)
    out["latency_ms_512"] = 1e3 * latency(f, x, reps=2)
    out["params"] = n_params(model) if model is not None else 0
    core = getattr(model, "base", model)
    out["mults_per_px"] = mults_per_output(core) if model is not None else 0
    out["compares_per_px"] = (compares_per_output(core) + (9 * 2 if core is not model else 0)) if model is not None else 0
    if model is not None and not name.startswith("WOS"):
        torch.save(model.state_dict(), f"b3_{regime}_{name.split()[0]}_{seed}.pt")
    elif model is not None:
        import re
        torch.save(model.state_dict(), "b3_%s_%s_%d.pt" % (regime, re.sub(r"[^A-Za-z0-9]+", "_", name), seed))
    return out

if __name__ == "__main__":
    from multiprocessing import Pool
    outp = "b3_results.json"
    jobs = [("imp", m, 0) for m in MODELS_IMP] + [("gauss", m, 0) for m in MODELS_GAUSS]
    R = json.load(open(outp)) if os.path.exists(outp) else []
    have = {(r["regime"], r["model"], r["seed"]) for r in R}
    jobs = [j for j in jobs if j not in have]
    jobs.sort(key=lambda j: 0 if "DnCNN-17" in j[1] else 1)       # longest first
    print("pending", len(jobs), flush=True)
    with Pool(2) as p:
        for r in p.imap_unordered(job, jobs):
            R.append(r); json.dump(R, open(outp, "w"), indent=0)
            print(r["regime"], r["model"], r["train_s"], {k: round(v["psnr"], 2) for k, v in r["metrics"].items() if k.startswith("Set12")}, flush=True)
