"""Uniform CPU latency (1 thread, idle machine): 1D models on 4096 samples, 2D models on a 512x512 image.
Latency does not depend on the trained values, so freshly initialised models (deploy mode) are timed."""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wosnet import *
import b1_signal1d as b1, b3_image as b3
from b3x_switch import SwitchWOS
from b3fix_dncnn17 import DnCNNBN
torch.set_num_threads(1); torch.manual_seed(0)

def t(f, x, reps):
    with torch.no_grad():
        f(x); return 1e3 * min(latency(f, x, reps=1) for _ in range(reps))

out = {"1d": {}, "2d": {}}
x = torch.randn(1, 4096)
for n, c in b1.MODELS.items():
    f = (lambda z: median_nd(z, 5, 1)) if c is None else c()
    if c is not None: set_deploy(f)
    out["1d"][n] = t(f, x, 7); print(n, round(out["1d"][n], 2), flush=True)
X = torch.rand(1, 512, 512)
M2 = dict(b3.MODELS_IMP); M2["DnCNN-17 BN (17×64)"] = lambda: DnCNNBN()
M2["WOS-R conmutado 3×3 ×2"] = lambda: SwitchWOS(WOSChain(2, 2, 3)); M2["WOS-R conmutado 5×5 ×2"] = lambda: SwitchWOS(WOSChain(2, 2, 5))
M2["WOS-R conmutado banco C=4 (3×3)"] = lambda: SwitchWOS(WOSBank(4, 2, 2, 3))
for n, c in M2.items():
    f = b3.classical(n) if c is None else c()
    if c is not None: set_deploy(f)
    out["2d"][n] = t(f, X, 3); print(n, round(out["2d"][n], 1), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "latency.json"), "w"), indent=1)
