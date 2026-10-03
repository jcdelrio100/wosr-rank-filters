"""B7 — Are the number of stages, the window size and the bank size hyper-parameters, or can they be learned?

(a) Grid (reference): WOS-R chains with K in {1,2,3,4} stages and windows m in {3,5,7,9} on the B1 impulsive task.
(b) Learned structure: an over-provisioned chain (K_max=4 stages, m_max=9 taps) trained with two penalties that keep
    the model inside the WOS family:
      - identity penalty: a stage is the identity iff its centre weight p_c = w_c/W satisfies p_c >= r and 1-p_c < r,
        i.e. p_c > max(r, 1-r).  lam * relu(max(r,1-r) - p_c + delta) pushes unneeded stages to the identity;
      - tap sparsity: mu * sum_{j!=c} sqrt(p_j) pushes unneeded taps to zero weight.
    After training, taps with p_j < 0.02 are pruned, identity stages are removed, and the deployed model is re-evaluated.
    Tasks: B1 impulsive denoising and three B2 targets with known structure (median 7: 1 stage / 7 taps;
    opening 5: 2 stages / 5 taps; rank cascade #1: 2 stages).
(c) Bank size: bank with C_max=8 chains and an L1 penalty on the mix weights; chains with |a_c| < 2 % of max are removed."""
import sys, os, json, time, math, itertools
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wosnet import *
from ramp import ramp_wos
from rankfilters import make_denoise_data
import b2_coverage as b2


class FlexStage(nn.Module):
    def __init__(self, m, identity_init=False):
        super().__init__()
        self.m, self.c = m, m // 2
        a = torch.full((m,), math.log(math.expm1(1.0))) + 0.05 * torch.randn(m)
        if identity_init: a[self.c] = math.log(math.expm1(float(2 * m)))       # p_c ~ 2m/(3m-1) ~ 0.67
        self.alpha = nn.Parameter(a); self.rho = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("mask", torch.ones(m)); self.bypass = False
    def w(self): return F.softplus(self.alpha) * self.mask
    def r(self): return torch.sigmoid(self.rho)
    def p(self): w = self.w(); return w / w.sum()
    def forward(self, x):
        if self.bypass: return x
        return ramp_wos(nb1d(x, self.m), self.w(), self.r(), 1.0 if self.training else 0.0)
    def is_identity(self):
        p, r = self.p().detach(), float(self.r()); pc = float(p[self.c])
        return pc >= r - 1e-9 and (1 - pc) < r

class FlexChain(nn.Module):
    def __init__(self, K, m, identity_init_from=1):
        super().__init__()
        self.stages = nn.ModuleList([FlexStage(m, identity_init=(k >= identity_init_from)) for k in range(K)])
    def forward(self, x):
        for s in self.stages: x = s(x)
        return x
    def penalty(self, lam, mu, delta=0.05):
        pen = 0.0
        for s in self.stages:
            p, r = s.p(), s.r()
            pen = pen + lam * F.relu(torch.maximum(r, 1 - r) - p[s.c] + delta)
            pen = pen + mu * torch.sqrt(p[torch.arange(s.m) != s.c] + 1e-4).sum()
        return pen
    @torch.no_grad()
    def prune(self, err, tol=0.05):
        """Functional pruning on a validation set: drop identity stages, then any stage and any tap whose removal
        does not raise the validation error by more than tol dB (smallest weights first)."""
        for s in self.stages:
            if s.is_identity(): s.bypass = True
        base = err(self)
        for s in self.stages:
            if s.bypass: continue
            s.bypass = True
            if err(self) > base + tol: s.bypass = False
            else: base = min(base, err(self))
        for s in self.stages:
            if s.bypass: continue
            for j in s.p().argsort().tolist():
                if j == s.c or s.mask.sum() <= 1: continue
                s.mask[j] = 0
                if err(self) > base + tol: s.mask[j] = 1
    def structure(self):
        act = [s for s in self.stages if not s.bypass and not s.is_identity()]
        return dict(stages=len(act), taps=[int(s.mask.sum()) for s in act],
                    span=[int((torch.nonzero(s.mask).max() - torch.nonzero(s.mask).min() + 1)) for s in act],
                    params=sum(int(s.mask.sum()) + 1 for s in act))

def task_data(task, seed):
    if task == "B1 impulsive":
        X, S = make_denoise_data(128, 1024, 0.10, seed=seed); Xt, St = make_denoise_data(32, 1024, 0.10, seed=seed + 1)
        nm = lambda y, d: 10 * torch.log10(((y - d) ** 2).mean() / (d ** 2).mean()).item()
        return X, S, Xt, St, nm, 0.0
    cls, f = b2.TARGETS[task]
    X, D = b2.data(f, 64, 2048, seed); Xt, Dt = b2.data(f, 8, 2048, seed + 100)
    return X, D, Xt, Dt, b2.nmse, b2.nmse(f(Xt), Dt)

def train(model, X, S, steps=3000, lr=0.03, pen=None, seed=0, crop=256, batch=32):
    torch.manual_seed(seed); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    var = float((S ** 2).mean()); t0 = time.perf_counter()
    for s in range(steps):
        i = torch.randint(0, X.shape[0], (batch,)); o = int(torch.randint(0, X.shape[-1] - crop, (1,)))
        loss = F.mse_loss(model(X[i, o:o + crop])[:, 8:-8], S[i, o:o + crop][:, 8:-8]) / var
        if pen is not None: loss = loss + pen(model)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return time.perf_counter() - t0

def job(spec):
    torch.set_num_threads(1)
    kind = spec[0]
    if kind == "grid":
        _, K, m, seed = spec
        X, S, Xt, St, nm, floor = task_data("B1 impulsive", seed)
        torch.manual_seed(seed); model = FlexChain(K, m, identity_init_from=99)
        ts = train(model, X, S, seed=seed); model.eval()
        with torch.no_grad(): e = nm(model(Xt), St)
        return dict(kind=kind, K=K, m=m, seed=seed, nmse=e, params=K * (m + 1), train_s=ts)
    if kind == "learn":
        _, task, lam, mu, seed = spec
        X, S, Xt, St, nm, floor = task_data(task, seed)
        Xv, Sv = task_data(task, seed + 50)[2:4]                  # validation set for pruning (not the test set)
        torch.manual_seed(seed); model = FlexChain(4, 9, identity_init_from=1)
        pen = (lambda mod: mod.penalty(lam, mu)) if lam > 0 else None
        ts = train(model, X, S, pen=pen, seed=seed); model.eval()
        with torch.no_grad():
            e_full = nm(model(Xt), St) - floor; model.prune(lambda mod: nm(mod(Xv), Sv)); e_pruned = nm(model(Xt), St) - floor
        st = model.structure()
        wts = [dict(p=s.p().detach().numpy().round(3).tolist(), r=round(float(s.r()), 3), identity=s.is_identity(), bypass=s.bypass, mask=s.mask.tolist()) for s in model.stages]
        return dict(kind=kind, task=task, lam=lam, mu=mu, seed=seed, err_full=e_full, err_pruned=e_pruned, floor=floor,
                    structure=st, stages=wts, train_s=ts)
    if kind == "bank":
        _, nu, seed = spec
        X, S, Xt, St, nm, floor = task_data("B1 impulsive", seed)
        torch.manual_seed(seed); model = WOSBank(8, 2, 1, 7, anneal=2100)
        pen = (lambda mod: nu * mod.mix.abs().sum()) if nu > 0 else None
        ts = train(model, X, S, pen=pen, seed=seed); set_deploy(model)
        with torch.no_grad():
            e_full = nm(model(Xt), St); a = model.mix.detach().abs(); keep = a >= 0.02 * a.max()
            model.mix.mul_(keep.float()); e_pruned = nm(model(Xt), St)
        return dict(kind=kind, nu=nu, seed=seed, err_full=e_full, err_pruned=e_pruned, chains=int(keep.sum()),
                    mix=model.mix.detach().numpy().round(3).tolist(), train_s=ts)

if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = [("grid", K, m, s) for K in (1, 2, 3, 4) for m in (3, 5, 7, 9) for s in (0, 1)]
    for task in ["B1 impulsive", "Mediana 7", "Apertura 5 (erosión→dilatación)", "Cascada E2 t1"]:
        for lam, mu in [(0.0, 0.0), (1e-3, 2e-4)]:
            for s in ((0, 1, 2) if task == "B1 impulsive" else (0, 1)):
                jobs.append(("learn", task, lam, mu, s))
    jobs += [("bank", nu, s) for nu in (0.0, 1e-3) for s in (0, 1, 2)]
    outp = "b7_results.json"
    R = json.load(open(outp)) if os.path.exists(outp) else []
    key = lambda j: json.dumps(j)
    have = {r["spec"] for r in R}
    jobs = [j for j in jobs if key(j) not in have]
    jobs.sort(key=lambda j: 0 if j[0] == "bank" else (1 if j[0] == "learn" else 2))
    print("pending", len(jobs), flush=True)
    with Pool(2) as p:
        for j, r in zip(jobs, p.imap(job, jobs)):
            r["spec"] = key(j); R.append(r); json.dump(R, open(outp, "w"))
            print({k: v for k, v in r.items() if k not in ("stages", "spec", "mix")}, flush=True)
