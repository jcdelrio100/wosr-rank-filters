"""
matheron.py — Matheron/Maragos kernel-basis representation of rank-filter cascades, and learnability.

Theory used (flat case): an operator that commutes with thresholding (stack filter) and is increasing and
translation-invariant is  Psi(f)(i) = max_{B in Bas(Psi)} min_{b in B} f(i-b),
where Bas(Psi) = minimal true sets of the positive Boolean function obtained by threshold decomposition.
Rank-r (1=min) of a window of N samples  <->  threshold function "at least N-r+1 ones".

B1  exact basis of the thesis cascades + numerical verification of the representation
B2  learnability: 'Matheron layer' (C non-flat erosions + max) vs NEst cascade on the same targets
B3  beyond increasing operators: top-hat x - opening(x) needs a difference of two Matheron layers
    (tropical rational function), as Maragos/Banon-Barrera and the tropical view of ReLU nets predict.
"""
import itertools, math, json, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from rankfilters import (causal_windows, hard_rank_filter, two_stage_target, make_ident_data, nmse_db,
                         train_modern, random_target, Cascade)


# ---------------------------------------------------------------- B1: exact basis
def boolean_cascade(mask1, r1, mask2, r2):
    """Return support U (offsets) and predicate on a set S ⊆ U (ones) for the flat cascade."""
    U = sorted({j + jp for j in mask1 for jp in mask2})
    k1, k2 = len(mask1) - r1 + 1, len(mask2) - r2 + 1
    def pred(S):
        ones2 = sum(1 for jp in mask2 if sum(1 for j in mask1 if (j + jp) in S) >= k1)
        return ones2 >= k2
    return U, pred


def minimal_true_sets(U, pred):
    true_sets = []
    for r in range(len(U) + 1):
        for S in itertools.combinations(U, r):
            Sset = frozenset(S)
            if any(T <= Sset for T in true_sets):
                continue
            if pred(Sset):
                true_sets.append(Sset)
    return true_sets


def basis_operator(basis, x):
    """max_{B} min_{b in B} x[i-b] (causal)."""
    m = max(max(B) for B in basis) + 1
    W = causal_windows(x, m)
    vals = torch.stack([W[..., sorted(B)].min(-1).values for B in basis], -1)
    return vals.max(-1).values


# ---------------------------------------------------------------- B2: Matheron layer
class MatheronLayer(nn.Module):
    """y = max_c min_j (x_{i-j} - g_{c,j})   — sup of C non-flat erosions (Maragos kernel form).
    Optional smooth version (temperature tau) uses -tau*logsumexp(-./tau) and tau*logsumexp(./tau)."""
    def __init__(self, C, m, init_spread=1.0, seed=0, subset_init=None):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.m = m
        G = init_spread * torch.rand(C, m, generator=g)
        if subset_init is not None:            # flat erosions of random subsets, others pushed out by +big
            big, kmax = subset_init
            G = torch.full((C, m), float(big))
            for c in range(C):
                k = int(torch.randint(1, kmax + 1, (1,), generator=g))
                G[c, torch.randperm(m, generator=g)[:k]] = 0.1 * torch.rand(k, generator=g)
        self.g = nn.Parameter(G)
        self.c = nn.Parameter(torch.zeros(()))

    def forward(self, x, tau=None, hard=False):
        W = causal_windows(x, self.m).unsqueeze(-2) - self.g          # (B,T,C,m)
        if tau is None or hard:
            e = W.min(-1).values
            return e.max(-1).values + self.c
        e = -tau * torch.logsumexp(-W / tau, -1)
        return tau * torch.logsumexp(e / tau, -1) + self.c


class TropicalRational(nn.Module):
    """Difference of two Matheron layers  (max-plus 'polynomial' minus another): non-increasing operators."""
    def __init__(self, C, m, seed=0):
        super().__init__()
        self.p = MatheronLayer(C, m, seed=seed)
        self.q = MatheronLayer(C, m, seed=seed + 1)

    def forward(self, x, tau=None, hard=False):
        return self.p(x, tau, hard) - self.q(x, tau, hard)


def fit(model, xs, ds, steps=3000, lr=0.05, batch=4, crop=512, anneal=False, seed=0):
    return train_modern(xs, ds, None, steps=steps, lr=lr, anneal=anneal, batch=batch, crop=crop,
                        seed=seed, model=model, tau0=1.0, tau1=0.02)


def recovered_basis(layer, thr):
    """Taps with g below thr (relative to row minimum) form the recovered basis element of each channel."""
    G = layer.g.detach()
    G = G - G.min(1, keepdim=True).values
    return {frozenset(np.nonzero((row < thr).numpy())[0].tolist()) for row in G}
