"""
cfar.py — Learned CFAR detectors built from the thesis' rank structures (NEst) in the log domain.

Window: [lead 16 | guard 2 | CUT | guard 2 | lag 16]  (square-law power cells).
Clutter: K-distributed power  x = L * tau * s,  tau~Gamma(nu,1/nu) (nu=inf -> exponential/Rayleigh),
         or Weibull-amplitude power; level L may have an edge (+-10/20 dB) anywhere in the window;
         interfering Swerling-I targets (INR 10..30 dB) in reference cells.
Detector: declare target if x_CUT > alpha * Z(ref),  alpha calibrated for Pfa = 1e-4 on homogeneous K(nu=2).

Estimators Z:
  CA, OS(k=24 of 32), GO-CA, SO-CA, GO-OS  (classical; GO/SO = max/min of the two halves: a 2-stage rank cascade)
  NEst-log 1-stage :  log Z = OWA_w(log x_j + b_j)                    (multiplicative mask e^{b_j}, unbounded)
  NEst-log 2-stage :  per half h: u_h = OWA_{w_h}(log x_j + b_j);  log Z = OWA_{w2}(u_h + c_h)   (learned GO/SO-OS)
  NEst-log 2-stage (4x8): same with 4 sub-windows
  NEst-lin 1-stage :  Z = OWA_w(x_j + b_j)  (additive in power domain; NOT scale-homogeneous)
  MLP-log (equivariant): log Z = mean(u) + MLP(u - mean(u))
All log-domain models satisfy Z(lambda x) = lambda Z(x)  =>  Pfa independent of clutter power (CFAR by construction).
"""
import math, json, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

NH, G = 16, 2
W = 2 * NH + 2 * G + 1
CUT = NH + G
REF = list(range(NH)) + list(range(NH + 2 * G + 1, W))
LEAD, LAG = list(range(NH)), list(range(NH, 2 * NH))      # indices inside the 32 reference cells


# ------------------------------------------------------------------------------------------- simulator
def clutter(shape, n, gen, nu=None, wc=None):
    """Unit-mean clutter power samples. nu: K shape (None -> exponential); wc: Weibull amplitude shape."""
    shape = (n, W)
    s = torch.empty(shape).exponential_(generator=gen)
    if wc is not None:
        u = torch.rand(shape, generator=gen).clamp_min(1e-12)
        a = (-torch.log(u)) ** (1.0 / wc)                          # Weibull amplitude, scale 1
        x = a ** 2
        return x / math.gamma(1 + 2.0 / wc)                        # unit-mean power
    if nu is not None:
        tau = torch.distributions.Gamma(torch.full(shape, float(nu)), torch.full(shape, float(nu))).sample()
        return tau * s
    return s


def simulate(n, gen, nu=2.0, wc=None, edge_db=0.0, edge_pos=None, n_int=0, inr_db=(10, 30),
             snr_db=None, level_db=(-20, 20)):
    """Returns cells (n,W) and true clutter level at the CUT (n,)."""
    lvl_db = torch.empty(n).uniform_(*level_db, generator=gen) if isinstance(level_db, tuple) else torch.full((n,), float(level_db))
    L = 10 ** (lvl_db / 10)
    prof = L[:, None].expand(n, W).clone()
    if edge_db != 0.0:
        pos = torch.randint(1, W, (n,), generator=gen) if edge_pos is None else torch.full((n,), int(edge_pos))
        mask = torch.arange(W)[None, :] >= pos[:, None]
        prof = torch.where(mask, prof * 10 ** (edge_db / 10), prof)
    x = prof * clutter(None, n, gen, nu=nu, wc=wc)
    if n_int > 0:
        for _ in range(n_int):
            c = torch.tensor(REF)[torch.randint(0, len(REF), (n,), generator=gen)]
            inr = 10 ** (torch.empty(n).uniform_(*inr_db, generator=gen) / 10) if isinstance(inr_db, tuple) else torch.full((n,), 10 ** (inr_db / 10))
            amp = prof[torch.arange(n), c] * inr * torch.empty(n).exponential_(generator=gen)
            x[torch.arange(n), c] += amp
    lev = prof[:, CUT].clone()
    if snr_db is not None:
        x[:, CUT] += lev * 10 ** (snr_db / 10) * torch.empty(n).exponential_(generator=gen)   # Swerling I
    return x, lev


def training_batch(n, gen):
    """Mixture: homogeneous / edges / interferers, random K shape or Weibull."""
    parts = []
    k = n // 6
    for nu in [0.3, 0.5, 1.0, 2.0, 5.0, None]:
        typ = int(torch.randint(0, 3, (1,), generator=gen))
        if typ == 0:
            parts.append(simulate(k, gen, nu=nu))
        elif typ == 1:
            e = float([-20, -10, 10, 20][int(torch.randint(0, 4, (1,), generator=gen))])
            parts.append(simulate(k, gen, nu=nu, edge_db=e))
        else:
            parts.append(simulate(k, gen, nu=nu, n_int=int(torch.randint(1, 5, (1,), generator=gen))))
    X = torch.cat([p[0] for p in parts]); L = torch.cat([p[1] for p in parts])
    return X, L


# ------------------------------------------------------------------------------------------- estimators
def ref(X): return X[:, REF]

def Z_ca(X): return ref(X).mean(1)
def Z_os(X, k=24): return ref(X).sort(1).values[:, k - 1]
def Z_go_ca(X): R = ref(X); return torch.maximum(R[:, LEAD].mean(1), R[:, LAG].mean(1))
def Z_so_ca(X): R = ref(X); return torch.minimum(R[:, LEAD].mean(1), R[:, LAG].mean(1))
def Z_go_os(X, k=12): R = ref(X); return torch.maximum(R[:, LEAD].sort(1).values[:, k - 1], R[:, LAG].sort(1).values[:, k - 1])


class NEstLog(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(m)); self.theta = nn.Parameter(torch.zeros(m))
    def forward(self, U):                                   # U: (n,m) log-domain
        return (U + self.b).sort(1).values @ torch.softmax(self.theta, 0)


class CFARNEst1(nn.Module):
    name = "NEst-log 1 etapa"
    def __init__(self):
        super().__init__(); self.l = NEstLog(2 * NH)
    def forward(self, X): return self.l(torch.log(ref(X)))


class CFARNEst2(nn.Module):
    def __init__(self, parts=2):
        super().__init__()
        self.parts = parts; m = 2 * NH // parts
        self.idx = [list(range(i * m, (i + 1) * m)) for i in range(parts)]
        self.l1 = nn.ModuleList([NEstLog(m) for _ in range(parts)]); self.l2 = NEstLog(parts)
        self.name = f"NEst-log 2 etapas ({parts}×{m})"
    def forward(self, X):
        U = torch.log(ref(X))
        V = torch.stack([l(U[:, ix]) for l, ix in zip(self.l1, self.idx)], 1)
        return self.l2(V)


class CFARNEstLin(nn.Module):
    name = "NEst-lin 1 etapa (no homogéneo)"
    def __init__(self):
        super().__init__(); self.b = nn.Parameter(torch.zeros(2 * NH)); self.theta = nn.Parameter(torch.zeros(2 * NH))
    def forward(self, X):
        Z = (ref(X) + self.b).sort(1).values @ torch.softmax(self.theta, 0)
        return torch.log(Z.clamp_min(1e-12))


class CFARMLP(nn.Module):
    name = "MLP-log equivariante"
    def __init__(self, h=64):
        super().__init__(); self.net = nn.Sequential(nn.Linear(2 * NH, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, X):
        U = torch.log(ref(X)); m = U.mean(1, keepdim=True)
        return (m + self.net(U - m)).squeeze(1)


def init_os_like(model):
    """Start every OWA at the OS-CFAR solution (k=24/32 ≈ 75 %), b=0."""
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, NEstLog):
                m = mod.theta.numel(); k = max(0, int(round(0.75 * m)) - 1) if m > 2 else 0
                mod.theta.zero_(); mod.theta[k] = 3.0
                if m == 2: mod.theta.zero_()
    return model


def train(model, steps=3000, lr=0.02, batch=6000, seed=0, log_target=True):
    gen = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        X, L = training_batch(batch, gen)
        out = model(X)
        loss = F.mse_loss(out, torch.log(L))                # estimate log clutter level at the CUT
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return model


# ------------------------------------------------------------------------------------------- evaluation
def detector_fns(models):
    D = {"CA": lambda X: Z_ca(X), "OS (k=24)": lambda X: Z_os(X), "GO-CA": Z_go_ca, "SO-CA": Z_so_ca, "GO-OS": Z_go_os}
    for m in models:
        D[m.name] = (lambda X, m=m: torch.exp(m(X)))
    return D


@torch.no_grad()
def stat(fn, X, chunk=200000):
    return torch.cat([X[i:i + chunk, CUT] / fn(X[i:i + chunk]).clamp_min(1e-30) for i in range(0, len(X), chunk)])


@torch.no_grad()
def calibrate(D, gen, n=2_000_000, pfa=1e-4, nu=2.0):
    X, _ = simulate(n, gen, nu=nu)
    return {k: float(np.quantile(stat(f, X).numpy(), 1 - pfa)) for k, f in D.items()}


@torch.no_grad()
def pfa_of(D, alpha, gen, n, **kw):
    X, _ = simulate(n, gen, **kw)
    return {k: (stat(f, X) > alpha[k]).float().mean().item() for k, f in D.items()}


@torch.no_grad()
def pd_of(D, alpha, gen, n, **kw):
    X, _ = simulate(n, gen, **kw)
    return {k: (stat(f, X) > alpha[k]).float().mean().item() for k, f in D.items()}


if __name__ == "__main__":
    torch.set_num_threads(2)
    t0 = time.time()
    models = []
    for M in [CFARNEst1, lambda: CFARNEst2(2), lambda: CFARNEst2(4)]:
        models.append(train(init_os_like(M())))
    models.append(train(CFARNEstLin(), lr=0.02))
    models.append(train(CFARMLP(), lr=1e-3))
    print("trained %.0fs" % (time.time() - t0), flush=True)
    torch.save({m.name: m.state_dict() for m in models}, "cfar_models.pt")
    D = detector_fns(models)
    gen = torch.Generator().manual_seed(123)
    res = {}
    res["alpha"] = alpha = calibrate(D, gen)
    # T1: CFAR w.r.t. clutter power (homogeneous K nu=2) at fixed levels
    res["T1_pfa_vs_level"] = {lv: pfa_of(D, alpha, gen, 1_000_000, nu=2.0, level_db=float(lv)) for lv in [-30, 0, 30]}
    # T2: clutter shape mismatch
    res["T2_pfa_vs_shape"] = {name: pfa_of(D, alpha, gen, 1_000_000, **kw) for name, kw in
                              {"K nu=0.5": dict(nu=0.5), "K nu=5": dict(nu=5.0), "Rayleigh": dict(nu=None),
                               "Weibull c=1.2": dict(nu=None, wc=1.2)}.items()}
    # T3: clutter edge +20 dB, sweep edge position across the window (CUT at index CUT)
    res["T3_edge"] = {}
    for e in [10.0, 20.0]:
        rows = {}
        for pos in range(2, W, 2):
            rows[pos] = pfa_of(D, alpha, gen, 200_000, nu=2.0, edge_db=e, edge_pos=pos)
        res["T3_edge"][e] = rows
    # T4: detection with interferers (homogeneous K nu=2), SNR sweep
    res["T4_pd"] = {}
    for ni in [0, 2, 4, 8]:
        res["T4_pd"][ni] = {snr: pd_of(D, alpha, gen, 200_000, nu=2.0, n_int=ni, inr_db=20.0, snr_db=float(snr))
                            for snr in [5, 10, 15, 20, 25, 30]}
    # learned parameters for interpretability
    res["params"] = {m.name: {k: v.detach().numpy().round(3).tolist() for k, v in m.state_dict().items()}
                     for m in models if "MLP" not in m.name}
    json.dump(res, open("cfar_results.json", "w"), indent=1)
    print("done %.0fs" % (time.time() - t0))
