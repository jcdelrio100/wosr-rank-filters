"""
wosnet.py — Common infrastructure for the paper benchmark.

Model under test ("WOS-R"): cascades / banks of flat Weighted-Order-Statistic filters whose repetition weights w and
quantile r are TRAINED through the continuous ramp relaxation (beta = 1, exact gradients) and DEPLOYED as exact WOS
(beta = 0, pure selection). 1D and 2D neighbourhoods.

Baselines: residual CNN (1D/2D, DnCNN-like, several sizes), neighbourhood MLP, small Transformer (1D full attention /
2D window attention), classical median / adaptive median.
Cost accounting: parameters, multiplications per output sample, comparisons per output sample, CPU latency.
"""
import math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ramp import ramp_wos


# ------------------------------------------------------------------ neighbourhoods
def nb1d(x, m):
    """x (B,T) -> (B,T,m) centred, replicate padding."""
    p = m // 2
    xp = F.pad(x.unsqueeze(1), (p, m - 1 - p), mode="replicate").squeeze(1)
    return xp.unfold(-1, m, 1)


def nb2d(x, k):
    """x (B,H,W) -> (B,H,W,k*k) centred k x k neighbourhood, replicate padding."""
    p = k // 2
    xp = F.pad(x.unsqueeze(1), (p, p, p, p), mode="replicate")
    U = F.unfold(xp, k)                                   # (B, k*k, H*W)
    B, _, H, W = x.shape[0], None, x.shape[1], x.shape[2]
    return U.transpose(1, 2).reshape(x.shape[0], x.shape[1], x.shape[2], k * k)


# ------------------------------------------------------------------ WOS-R
class WOSR(nn.Module):
    """One flat WOS stage; training: ramp (beta=1); eval: exact selection (beta=0) unless self.deploy_ramp."""
    def __init__(self, m, dim=1, k=None, init_r=0.5):
        super().__init__()
        self.dim, self.m, self.k = dim, m, k
        self.alpha = nn.Parameter(torch.full((m,), math.log(math.expm1(1.0))) + 0.05 * torch.randn(m))
        self.rho = nn.Parameter(torch.tensor(math.log(init_r / (1 - init_r))))
        self.deploy_ramp = False
    def nb(self, x): return nb1d(x, self.m) if self.dim == 1 else nb2d(x, self.k)
    def w(self): return F.softplus(self.alpha)
    def r(self): return torch.sigmoid(self.rho)
    def forward(self, x):
        beta = getattr(self, "beta_train", 1.0) if self.training else (1.0 if self.deploy_ramp else 0.0)
        return ramp_wos(self.nb(x), self.w(), self.r(), beta)


class WOSChain(nn.Module):
    def __init__(self, stages, dim=1, k=3):
        super().__init__()
        m = k * k if dim == 2 else k
        self.layers = nn.ModuleList([WOSR(m, dim, k) for _ in range(stages)])
    def forward(self, x):
        for l in self.layers: x = l(x)
        return x


class WOSBank(nn.Module):
    """C parallel chains + affine mix (the only multiplications: C per output sample)."""
    def __init__(self, C=4, stages=2, dim=1, k=7, anneal=None):
        super().__init__()
        self.chains = nn.ModuleList([WOSChain(stages, dim, k) for _ in range(C)])
        self.mix = nn.Parameter(torch.full((C,), 1.0 / C)); self.bias = nn.Parameter(torch.zeros(()))
        self.C = C; self.anneal = anneal; self.step = 0
    def forward(self, x):
        # ramp continuation: with a linear mix after the selections, beta is annealed 1 -> 0 over `anneal` training
        # steps (one forward per step), so the mix cannot co-adapt to the ramp's interpolation (see paper, Sec. 6).
        if self.training and self.anneal:
            b = max(0.0, 1.0 - self.step / self.anneal); self.step += 1
            for c in self.chains:
                for l in c.layers: l.beta_train = b
        Y = torch.stack([c(x) for c in self.chains], -1)
        return Y @ self.mix + self.bias if self.C > 1 else Y[..., 0]


# ------------------------------------------------------------------ baselines 1D
class CNN1D(nn.Module):
    def __init__(self, ch=16, k=7, depth=4):
        super().__init__()
        L = [nn.Conv1d(1, ch, k, padding=k // 2), nn.ReLU()]
        for _ in range(depth - 2): L += [nn.Conv1d(ch, ch, k, padding=k // 2), nn.ReLU()]
        L += [nn.Conv1d(ch, 1, k, padding=k // 2)]
        self.net = nn.Sequential(*L)
    def forward(self, x): return x + self.net(x.unsqueeze(1)).squeeze(1)


class MLP1D(nn.Module):
    def __init__(self, m=15, h=64):
        super().__init__(); self.m = m
        self.net = nn.Sequential(nn.Linear(m, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x): return x + self.net(nb1d(x, self.m)).squeeze(-1)


class Transformer1D(nn.Module):
    def __init__(self, d=16, heads=2, layers=2, ff=32, k=5):
        super().__init__()
        self.emb = nn.Conv1d(1, d, k, padding=k // 2)
        enc = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, 1)
    def forward(self, x):
        h = self.emb(x.unsqueeze(1)).transpose(1, 2)
        return x + self.head(self.tr(h)).squeeze(-1)


# ------------------------------------------------------------------ baselines 2D
class DnCNN(nn.Module):
    def __init__(self, depth=10, ch=32):
        super().__init__()
        L = [nn.Conv2d(1, ch, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 2): L += [nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU()]
        L += [nn.Conv2d(ch, 1, 3, padding=1)]
        self.net = nn.Sequential(*L)
    def forward(self, x): return x - self.net(x.unsqueeze(1)).squeeze(1)      # residual learning of the noise


class MLP2D(nn.Module):
    def __init__(self, k=5, h=64):
        super().__init__(); self.k = k
        self.net = nn.Sequential(nn.Linear(k * k, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x): return x + self.net(nb2d(x, self.k)).squeeze(-1)


class WinTransformer2D(nn.Module):
    """Conv embed -> 2 blocks of non-overlapping 8x8 window self-attention (second block shifted) -> conv head."""
    def __init__(self, d=32, heads=4, win=8, blocks=2, ff=64):
        super().__init__()
        self.win = win
        self.emb = nn.Conv2d(1, d, 3, padding=1)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True) for _ in range(blocks)])
        self.head = nn.Conv2d(d, 1, 3, padding=1)
    def forward(self, x):
        B, H, W = x.shape
        h = self.emb(x.unsqueeze(1))                            # (B,d,H,W)
        w = self.win; ph, pw = (-H) % w, (-W) % w
        h = F.pad(h, (0, pw, 0, ph), mode="replicate")
        Hp, Wp = h.shape[2], h.shape[3]
        for i, blk in enumerate(self.blocks):
            s = (w // 2) if i % 2 else 0
            if s: h = torch.roll(h, (-s, -s), (2, 3))
            t = h.reshape(B, -1, Hp // w, w, Wp // w, w).permute(0, 2, 4, 3, 5, 1).reshape(-1, w * w, h.shape[1])
            t = blk(t)
            h = t.reshape(B, Hp // w, Wp // w, w, w, -1).permute(0, 5, 1, 3, 2, 4).reshape(B, -1, Hp, Wp)
            if s: h = torch.roll(h, (s, s), (2, 3))
        h = h[:, :, :H, :W]
        return x - self.head(h).squeeze(1)


# ------------------------------------------------------------------ classical
def median_nd(x, k, dim):
    N = nb1d(x, k) if dim == 1 else nb2d(x, k)
    return N.median(-1).values


def adaptive_median_2d(x, smax=7):
    """Hwang–Haddad adaptive median filter (vectorised): windows 3,5,...,smax."""
    out = x.clone(); done = torch.zeros_like(x, dtype=torch.bool)
    for k in range(3, smax + 1, 2):
        N = nb2d(x, k); zmin, zmax, zmed = N.min(-1).values, N.max(-1).values, N.median(-1).values
        levelA = (zmed > zmin) & (zmed < zmax)
        levelB = (x > zmin) & (x < zmax)
        sel = levelA & ~done
        out = torch.where(sel & levelB, x, torch.where(sel, zmed, out))
        done = done | levelA
    return torch.where(done, out, zmed)


# ------------------------------------------------------------------ cost accounting
def n_params(model): return sum(p.numel() for p in model.parameters()) if isinstance(model, nn.Module) else 0


def mults_per_output(model):
    """Multiply–accumulates per output sample (WOS stages: 0; bank mix: C; conv/linear: weights applied per output)."""
    if not isinstance(model, nn.Module): return 0
    if isinstance(model, WOSBank): return model.C if model.C > 1 else 0
    if isinstance(model, WOSChain): return 0
    total = 0
    for mod in model.modules():
        if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
            total += mod.weight.numel()
        elif isinstance(mod, nn.Linear):
            total += mod.weight.numel()
        elif isinstance(mod, nn.MultiheadAttention):
            total += mod.in_proj_weight.numel()
    if isinstance(model, Transformer1D):
        total += 2 * 2 * 16 * 1024 // 1                          # attention QK^T and AV over L=1024 (d=16, 2 layers): per token
    if isinstance(model, WinTransformer2D):
        total += 2 * 2 * 32 * 64                                   # window 64 tokens, d=32, 2 blocks
    return total


def compares_per_output(model, m_override=None):
    """Comparisons for sorting each window (merge-sort bound m log2 m) per WOS stage."""
    if isinstance(model, WOSBank):
        return sum(sum(l.m * math.ceil(math.log2(l.m)) for l in c.layers) for c in model.chains)
    if isinstance(model, WOSChain):
        return sum(l.m * math.ceil(math.log2(l.m)) for l in model.layers)
    return 0


@torch.no_grad()
def latency(fn, x, reps=5):
    fn(x)
    t = time.perf_counter()
    for _ in range(reps): fn(x)
    return (time.perf_counter() - t) / reps


def set_deploy(model, ramp=False):
    model.eval()
    for m in model.modules():
        if isinstance(m, WOSR): m.deploy_ramp = ramp
    return model
