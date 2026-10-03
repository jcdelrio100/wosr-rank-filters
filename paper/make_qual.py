"""Qualitative figure: Set12 crop, SP 50 % and RVIN 40 %, from the saved B3 models (no retraining)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _ROOT = _os.path.dirname(_HERE)
_CODE = _os.path.join(_ROOT, "code")
import sys, os, re, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = _os.path.join(_CODE, "bench"); sys.path.insert(0, B); sys.path.insert(0, _CODE)
import b3_image as b
from b3x_switch import SwitchWOS
from b3fix_dncnn17 import DnCNNBN
from wosnet import *
torch.set_num_threads(2)

def load(name, ctor):
    if ctor is None: return b.classical(name)
    m = ctor()
    f = os.path.join(B, "b3_imp_%s_0.pt" % (re.sub(r"[^A-Za-z0-9]+", "_", name) if name.startswith("WOS") else name.split()[0]))
    m.load_state_dict(torch.load(f)); set_deploy(m); return m

MODELS = [("Mediana adaptativa (7)", None, "AMF"),
          ("WOS-R conmutado 5×5 ×2", lambda: SwitchWOS(WOSChain(2, 2, 5)), "Switching WOS-R 5x5 (50 par.)"),
          ("WOS-R banco C=4 (5×5)", lambda: WOSBank(4, 2, 2, 5, anneal=1400), "WOS-R bank 5x5"),
          ("DnCNN-M (10×32)", lambda: DnCNN(10, 32), "DnCNN-M (75k par.)"),
          ("DnCNN-17 BN (17×64)", lambda: DnCNNBN(), "DnCNN-17 BN (556k par.)")]

if __name__ == "__main__":
    im = b.load("Set12")[9]                                # 10.png (boat)
    crop = (slice(128, 384), slice(128, 384))
    g = torch.Generator().manual_seed(5)
    noisy = {"SP 50%": b.sp(im, .5, g), "RVIN 40% (OOD)": b.rvin(im, .4, g)}
    import json
    PR = {r["model"]: r["params"] for r in json.load(open(os.path.join(B, "b3_results.json"))) if r["regime"] == "imp"}
    fs = [((lab.split(" (")[0] + (" (%s par.)" % format(PR[n], ",") if PR.get(n) else "")), load(n, c)) for n, c, lab in MODELS]
    fig, axs = plt.subplots(2, 2 + len(fs), figsize=(2.1 * (2 + len(fs)), 5.2))
    for i, (cn, y) in enumerate(noisy.items()):
        cols = [("clean", im), (cn, y)]
        with torch.no_grad():
            for lab, f in fs:
                out = f(y.unsqueeze(0))[0].clamp(0, 1); cols.append(("%s\n%.2f dB" % (lab, b.psnr(out, im)), out))
        for j, (t, z) in enumerate(cols):
            axs[i, j].imshow(z[crop].numpy(), cmap="gray", vmin=0, vmax=1); axs[i, j].set_title(t, fontsize=7); axs[i, j].axis("off")
    fig.tight_layout(); fig.savefig(_os.path.join(_HERE, "figs", "qual.pdf"), dpi=200); print("ok")
