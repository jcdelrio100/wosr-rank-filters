"""
amo.py — Adaptive Minimax OWA (AMO) aggregator and space-time rank cascade Med_t(T) -> OWA.

AMO:
  * offline: library {w*(model)} solved with the SOCP of robust.minimax_owa for benign models
    Student-t nu in {2.5, 3, 5, 10} and Gaussian, for a design trim t (= assumed f), benign-efficiency weight lambda;
  * online (every K steps): robust tail statistic from INTERIOR order statistics only (immune to <= t outliers),
        rho = median_coords[(Q_{.70}-Q_{.30}) / (Q_{.60}-Q_{.40})]   (per-coordinate quantiles across the n nodes)
    compared with its clean-sample expectation under each model -> pick nearest model's w*.
Space-time cascade (thesis' NEst∘NEst, axes time x nodes):
    u_i = median over the last T submissions of node i (per coordinate);   y = OWA_w(u_1..u_n).
"""
import math, json, os
import numpy as np
import torch
from robust import second_moments, minimax_owa

MODELS = {"t2.5": ("t", 2.5), "t3": ("t", 3.0), "t5": ("t", 5.0), "t10": ("t", 10.0), "gauss": ("g", None)}


def _draw(model, rng, size):
    kind, nu = MODELS[model]
    if kind == "g":
        return rng.standard_normal(size)
    x = rng.standard_t(nu, size)
    return x / math.sqrt(nu / (nu - 2))


def interior_tail_stat(V):
    """V (..., n) -> rho per row, using only interior quantiles (30/40/60/70 %)."""
    q = torch.quantile(V, torch.tensor([0.30, 0.40, 0.60, 0.70], dtype=V.dtype), dim=-1)
    return (q[3] - q[0]) / (q[2] - q[1] + 1e-12)


def build_library(n=20, t=4, lam=0.1, mc=10000, path="amo_library.json", seed=0):
    if os.path.exists(path):
        return json.load(open(path))
    lib = {}
    rng = np.random.default_rng(seed)
    for name in MODELS:
        emp = _draw(name, rng, 400000)                       # sample pool for second_moments('empirical')
        Ss = second_moments("empirical", n=n, f=t, mc=mc, emp=emp,
                            z1=np.linspace(-25, 25, 51), z2c=np.linspace(-8, 8, 9))
        w, _ = minimax_owa(Ss, n=n, trim=t, benign_weight=lam)
        clean = torch.tensor(_draw(name, rng, (20000, n)))
        rho = interior_tail_stat(clean).median().item()
        lib[name] = dict(w=np.round(w, 6).tolist(), rho=rho)
        print(name, "rho=%.4f" % rho, "support", np.nonzero(w > 1e-3)[0].tolist(), flush=True)
    json.dump(lib, open(path, "w"), indent=1)
    return lib


class AMO:
    """Callable aggregator agg(M, st); M (n, d)."""
    def __init__(self, lib, K=20, temporal=0, max_coords=4096):
        self.lib, self.K, self.T, self.maxc = lib, K, temporal, max_coords
        self.names = list(lib)
        self.W = {k: torch.tensor(v["w"], dtype=torch.float32) for k, v in lib.items()}
        self.rho = torch.tensor([lib[k]["rho"] for k in self.names])

    def __call__(self, M, st):
        if self.T > 1:
            H = st.setdefault("hist", [])
            H.append(M.clone())
            if len(H) > self.T:
                H.pop(0)
            if len(H) >= 2:
                M = torch.stack(H, 0).median(0).values
        st["t"] = st.get("t", 0) + 1
        if "model" not in st or st["t"] % self.K == 1:
            idx = torch.randperm(M.shape[1])[: self.maxc]
            r = interior_tail_stat(M[:, idx].T.double()).median().item()
            st["model"] = self.names[int((self.rho - r).abs().argmin())]
            st.setdefault("trace", []).append(st["model"])
        return self.W[st["model"]] @ M.sort(0).values


def make_temporal_then(agg, T=3):
    def f(M, st):
        H = st.setdefault("hist", [])
        H.append(M.clone())
        if len(H) > T:
            H.pop(0)
        Mt = torch.stack(H, 0).median(0).values if len(H) >= 2 else M
        return agg(Mt, st)
    return f


if __name__ == "__main__":
    build_library()
