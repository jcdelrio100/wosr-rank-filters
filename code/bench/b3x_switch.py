"""B3 extra: switching WOS-R — apply the learned WOS only where the centre pixel is an extreme of its 3x3 window
(generic impulse detector, no knowledge of the 0/1 values); elsewhere pass the input through. Same protocol as B3."""
import sys, os, json
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b3_image as b
from wosnet import WOSChain, WOSBank, nb2d

class SwitchWOS(nn.Module):
    def __init__(s, base, eps=1e-6):
        super().__init__(); s.base = base; s.eps = eps
    def forward(s, x):
        N = nb2d(x, 3); mn, mx = N.min(-1).values, N.max(-1).values
        imp = (x <= mn + s.eps) | (x >= mx - s.eps)
        return torch.where(imp, s.base(x), x)

b.MODELS_IMP["WOS-R conmutado 3×3 ×2"] = lambda: SwitchWOS(WOSChain(2, 2, 3))
b.MODELS_IMP["WOS-R conmutado 5×5 ×2"] = lambda: SwitchWOS(WOSChain(2, 2, 5))
b.MODELS_IMP["WOS-R conmutado banco C=4 (3×3)"] = lambda: SwitchWOS(WOSBank(4, 2, 2, 3, anneal=1400))
b.STEPS["WOS"] = 2000

if __name__ == "__main__":
    from multiprocessing import Pool
    outp = "b3_results.json"
    jobs = [("imp", m, 0) for m in ["WOS-R conmutado 3×3 ×2", "WOS-R conmutado 5×5 ×2", "WOS-R conmutado banco C=4 (3×3)"]]
    R = json.load(open(outp)) if os.path.exists(outp) else []
    have = {(r["regime"], r["model"], r["seed"]) for r in R}
    jobs = [j for j in jobs if j not in have]
    with Pool(2) as p:
        for r in p.imap_unordered(b.job, jobs):
            R = json.load(open(outp)) if os.path.exists(outp) else []
            R.append(r); json.dump(R, open(outp, "w"), indent=0)
            print(r["model"], {k: round(v["psnr"], 2) for k, v in r["metrics"].items() if k.startswith("Set12")}, flush=True)
