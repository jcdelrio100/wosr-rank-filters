"""
wos.py — Mask weights as REPETITION COUNTS before sorting  ==  Weighted Order Statistic (WOS) filter.

    y = smallest z_(k) such that  sum_{j: z_j <= z_(k)} w_j  >=  r * sum_j w_j          (w_j >= 0 real)
Integer w_j  <=> sample j replicated w_j times (the proposal);  w_j = 0 -> tap removed (mask);
real w_j need no rounding: only the induced threshold Boolean function matters, and every real-weight WOS
equals an integer-weight one (finite set of threshold functions).

E-A  expressiveness: which of the 20 flat cascades of the project are a single WOS? (LP on the truth table;
     MILP for the minimal integer repetitions)
E-B  learnability: identify random WOS (1 and 2 stages) with (i) straight-through estimator (hard forward,
     implicit-function backward at finite tau), (ii) annealed soft implicit (previous SoftWOS), (iii) NEst baseline
E-C  practical: impulsive-noise denoising (bench E3), learned WOS / CWM / median / NEst / CNN, real vs integer weights
"""
import itertools, math, json, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from rankfilters import (causal_windows, centered_windows, hard_rank_filter, two_stage_target, make_ident_data,
                         nmse_db, train_modern, random_target, Cascade, make_denoise_data)
from matheron import boolean_cascade, minimal_true_sets


# ------------------------------------------------------------------ E-A: threshold-function test
def truth_table(U, pred):
    rows, ys = [], []
    for bits in itertools.product([0, 1], repeat=len(U)):
        S = frozenset(u for u, b in zip(U, bits) if b)
        rows.append(bits); ys.append(pred(S))
    return np.array(rows, float), np.array(ys, bool)


def wos_realization(U, pred):
    """Return (is_WOS, integer weights, threshold) with minimal sum of integer weights (MILP)."""
    A, y = truth_table(U, pred)
    n = len(U)
    # variables: w_1..w_n, T ; constraints  A_true w - T >= 0 ;  A_false w - T <= -1
    At = np.hstack([A[y], -np.ones((y.sum(), 1))])
    Af = np.hstack([A[~y], -np.ones(((~y).sum(), 1))])
    cons = [LinearConstraint(At, 0, np.inf), LinearConstraint(Af, -np.inf, -1)]
    c = np.r_[np.ones(n), 0.0]
    res = milp(c, constraints=cons, integrality=np.ones(n + 1), bounds=Bounds(0, 200))
    if not res.success:
        return False, None, None
    w = np.round(res.x[:n]).astype(int); T = int(round(res.x[n]))
    return True, w.tolist(), T


# ------------------------------------------------------------------ hard WOS + STE
def wos_hard(Z, w, r):
    """Z (...,m), w (m,) >=0, r scalar in (0,1). Weighted r-quantile (lower), per row."""
    Zs, idx = Z.sort(-1)
    ws = w[idx]
    cw = ws.cumsum(-1)
    k = (cw < r * w.sum() - 1e-9).sum(-1, keepdim=True).clamp(max=Z.shape[-1] - 1)
    return Zs.gather(-1, k).squeeze(-1)


class WOS(nn.Module):
    """Repetition-weight order filter. w = softplus(alpha) (or exp), r = sigmoid(rho), optional offsets b.
    mode='ste': forward exact WOS; backward = implicit-function gradient of the tau-smoothed CDF at y_hard.
    mode='soft': forward = tau-smoothed quantile (bisection), tau annealed externally."""
    def __init__(self, m, windows=causal_windows, offsets=False, tau=0.5, mode="ste", init_w=1.0, init_r=0.5):
        super().__init__()
        self.m, self.windows, self.tau, self.mode = m, windows, tau, mode
        self.alpha = nn.Parameter(torch.full((m,), math.log(math.expm1(init_w))) + 0.05 * torch.randn(m))
        self.rho = nn.Parameter(torch.tensor(math.log(init_r / (1 - init_r))))
        self.b = nn.Parameter(torch.zeros(m)) if offsets else None

    def weights(self): return F.softplus(self.alpha)
    def r(self): return torch.sigmoid(self.rho)

    def forward(self, x, tau=None, hard=False):
        Z = self.windows(x, self.m)
        if self.b is not None:
            Z = Z + self.b
        w, r = self.weights(), self.r()
        with torch.no_grad():
            y0 = wos_hard(Z, w, r)
        if not torch.is_grad_enabled() or hard:
            return y0
        t = self.tau if (self.mode == "ste" or tau is None) else tau
        if self.mode == "soft" and tau is not None:          # smoothed forward: refine y0 by bisection
            with torch.no_grad():
                lo, hi = Z.min(-1).values - 5 * t, Z.max(-1).values + 5 * t
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    g = (w * torch.sigmoid((mid.unsqueeze(-1) - Z) / t)).sum(-1) - r * w.sum()
                    lo, hi = torch.where(g < 0, mid, lo), torch.where(g < 0, hi, mid)
                y0 = 0.5 * (lo + hi)
        s = torch.sigmoid((y0.unsqueeze(-1) - Z) / t)
        G = (w * s).sum(-1) - r * w.sum()
        Gy = ((w * s * (1 - s)).sum(-1) / t).detach() + 1e-6
        return y0 + (-(G - G.detach()) / Gy)               # value y0, gradient -G_theta/G_y


class WOSCascade(nn.Module):
    def __init__(self, ms, **kw):
        super().__init__(); self.layers = nn.ModuleList([WOS(m, **kw) for m in ms])
    def forward(self, x, tau=None, hard=False):
        for L in self.layers: x = L(x, tau, hard)
        return x


def round_weights(model, K=8):
    """Integer repetitions: scale so that max weight = K, round to nearest integer (0 removes the tap)."""
    with torch.no_grad():
        for L in model.modules():
            if isinstance(L, WOS):
                w = L.weights(); q = torch.round(w / w.max() * K).clamp_min(0)
                L.alpha.copy_(torch.where(q > 0, torch.log(torch.expm1(q.clamp_min(1e-3))), torch.full_like(q, -30.0)))
    return model


# ------------------------------------------------------------------ targets for E-B
def random_wos_target(rng, m=7):
    w = rng.integers(0, 5, m); w[rng.integers(0, m)] = max(1, w.max())        # weights 0..4, at least one > 0
    W = w.sum(); T = int(rng.integers(1, W + 1))                                # selects order where cumw >= T
    wt = torch.tensor(w, dtype=torch.float32); r = (T - 0.5) / W
    def f(x):
        return wos_hard(causal_windows(x, m), wt, torch.tensor(r))
    return f, w.tolist(), T
