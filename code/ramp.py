"""
ramp.py — Repetitions spread as a RAMP between ranked neighbours instead of exact replicas (proposal of J.C. del Río).

Discrete (user's definition, integer weights): sample z_(k) with weight n_k is replaced by n_k points evenly spaced
from the midpoint with the previous ranked sample to the midpoint with the next one (endpoints included; n_k = 1 keeps
the sample itself); extremes are not extended outwards. Example: (0,w1), (0.5,w3), (1,w1) -> [0, .25, .5, .75, 1].
Output = element ceil(r * W) of the (already ordered) replicated list.

Continuous (real weights, differentiable): each sample carries mass w_k spread UNIFORMLY over its cell
[m_k^-, m_k^+],  m_k^- = z_(k) - beta (z_(k) - z_(k-1))/2,  m_k^+ = z_(k) + beta (z_(k+1) - z_(k))/2
(beta in [0,1]; beta = 0 -> exact WOS; beta = 1 -> cells tile [z_(1), z_(m)] and the output is continuous in r and w).
Output = r-quantile of that piecewise-linear CDF:
     y = m_k*^- + (rW - C_{k*-1}) / w_k* * (m_k*^+ - m_k*^-),   k* = first k with C_k >= rW.
"""
import math, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from rankfilters import causal_windows, centered_windows


def ramp_cells(Zs, beta):
    d = Zs[..., 1:] - Zs[..., :-1]
    lo = Zs.clone(); hi = Zs.clone()
    lo[..., 1:] = Zs[..., 1:] - beta * d / 2
    hi[..., :-1] = Zs[..., :-1] + beta * d / 2
    return lo, hi


def ramp_wos(Z, w, r, beta=1.0):
    """Continuous ramp-WOS. Z (...,m), w (m,) >= 0, r in (0,1). Differentiable in Z, w, r (a.e.)."""
    Zs, idx = Z.sort(-1)
    ws = w[idx]
    C = ws.cumsum(-1)
    q = r * w.sum()
    k = (C < q - 1e-9).sum(-1, keepdim=True).clamp(max=Z.shape[-1] - 1)
    Cprev = torch.where(k > 0, C.gather(-1, (k - 1).clamp(min=0)), torch.zeros_like(k, dtype=C.dtype))
    wk = ws.gather(-1, k)
    frac = ((q - Cprev) / wk.clamp_min(1e-12)).clamp(0, 1)
    lo, hi = ramp_cells(Zs, beta)
    lo_k, hi_k = lo.gather(-1, k), hi.gather(-1, k)
    return (lo_k + frac * (hi_k - lo_k)).squeeze(-1)


def ramp_wos_discrete(Z, n, r):
    """User's exact discrete version. n (m,) non-negative integers. Endpoint-inclusive ramps between midpoints."""
    Zs, idx = Z.sort(-1)
    ns = n[idx].long()
    W = int(n.sum().item())
    target = max(1, math.ceil(float(r) * W - 1e-9))
    C = ns.cumsum(-1)
    k = (C < target).sum(-1, keepdim=True).clamp(max=Z.shape[-1] - 1)
    Cprev = torch.where(k > 0, C.gather(-1, (k - 1).clamp(min=0)), torch.zeros_like(k))
    nk = ns.gather(-1, k)
    j = (target - Cprev).clamp(min=1)                         # 1..nk position within the ramp
    lo, hi = ramp_cells(Zs, 1.0)
    x = Zs.gather(-1, k); lo_k = lo.gather(-1, k); hi_k = hi.gather(-1, k)
    t = torch.where(nk > 1, -1 + 2 * (j - 1).float() / (nk - 1).clamp(min=1).float(), torch.zeros_like(x))
    y = torch.where(t < 0, x + t * (x - lo_k), x + t * (hi_k - x))
    return y.squeeze(-1)


class RampWOS(nn.Module):
    """Learnable ramp-WOS layer. beta: float or callable schedule set externally via .beta"""
    def __init__(self, m, windows=causal_windows, beta=1.0, offsets=False, init_r=0.5):
        super().__init__()
        self.m, self.windows, self.beta = m, windows, beta
        self.alpha = nn.Parameter(torch.full((m,), math.log(math.expm1(1.0))) + 0.05 * torch.randn(m))
        self.rho = nn.Parameter(torch.tensor(math.log(init_r / (1 - init_r))))
        self.b = nn.Parameter(torch.zeros(m)) if offsets else None
    def weights(self): return F.softplus(self.alpha)
    def r(self): return torch.sigmoid(self.rho)
    def forward(self, x, tau=None, hard=False):
        Z = self.windows(x, self.m)
        if self.b is not None: Z = Z + self.b
        beta = 0.0 if hard == "selection" else self.beta
        return ramp_wos(Z, self.weights(), self.r(), beta)


class RampCascade(nn.Module):
    def __init__(self, ms, **kw):
        super().__init__(); self.layers = nn.ModuleList([RampWOS(m, **kw) for m in ms])
    def set_beta(self, b):
        for l in self.layers: l.beta = b
    def forward(self, x, tau=None, hard=False):
        for l in self.layers: x = l(x, tau, hard)
        return x


def train_ramp(net, xs, ds, steps=3000, lr=0.05, batch=4, crop=512, seed=0, schedule="fixed", skip=16):
    """schedule: 'fixed' (beta as constructed) or 'anneal' (beta 1 -> 0 over 80 % of steps, then 0)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    for s in range(steps):
        if schedule == "anneal":
            net.set_beta(max(0.0, 1.0 - s / (0.8 * steps)))
        i = torch.randint(0, xs.shape[0], (batch,))
        o = int(torch.randint(0, xs.shape[-1] - crop, (1,)))
        xb, db = xs[i, o:o + crop], ds[i, o:o + crop]
        loss = F.mse_loss(net(xb)[:, skip:], db[:, skip:])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if schedule == "anneal":
        net.set_beta(0.0)
    return net
