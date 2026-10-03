"""Figure: output of one WOS stage as a function of the quantile r, exact (deployed) vs ramp (training).
Example of the text: samples 0, 0.5, 1 with repetition weights 1, 3, 1."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _ROOT = _os.path.dirname(_HERE)
_CODE = _os.path.join(_ROOT, "code")
import sys, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, _CODE)
from ramp import ramp_wos
Z = torch.tensor([0.0, 0.5, 1.0]); w = torch.tensor([1.0, 3.0, 1.0])
rs = np.linspace(0.001, 0.999, 999)
ex = [float(ramp_wos(Z, w, torch.tensor(r), 0.0)) for r in rs]
rp = [float(ramp_wos(Z, w, torch.tensor(r), 1.0)) for r in rs]
fig, ax = plt.subplots(figsize=(3.6, 2.5))
ax.plot(rs, ex, color="#1f4e79", lw=2.2, label="deployed (exact selection)")
ax.plot(rs, rp, color="#c0392b", lw=1.6, ls="--", label="training (ramp)")
ax.set_xlabel("quantile $r$ (learned parameter)"); ax.set_ylabel("stage output $y$")
ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left")
ax.text(0.52, 0.53, "flat: zero gradient", fontsize=7, color="#1f4e79", ha="center", va="bottom")
fig.tight_layout(); fig.savefig(_os.path.join(_HERE, "figs", "ramp_curve.pdf")); print("ok")
