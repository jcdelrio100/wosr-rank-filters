"""B3 fix: the 17-layer DnCNN without batch norm (lr 1e-3) failed to train on CPU (17 dB at SP 10 %).
Re-train the original DnCNN design with batch normalisation (Zhang et al. 2017), 2 threads."""
import sys, os, json
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b3_image as b

class DnCNNBN(nn.Module):
    def __init__(s, depth=17, ch=64):
        super().__init__()
        L = [nn.Conv2d(1, ch, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 2): L += [nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU()]
        L += [nn.Conv2d(ch, 1, 3, padding=1)]
        s.net = nn.Sequential(*L)
    def forward(s, x): return x - s.net(x.unsqueeze(1)).squeeze(1)

b.MODELS_IMP["DnCNN-17 BN (17×64)"] = lambda: DnCNNBN()
b.STEPS["DnCNN-17"] = 2000
_set = torch.set_num_threads
b.torch.set_num_threads = lambda n: _set(2)          # job() forces 1 thread; use 2 here (runs alone)

if __name__ == "__main__":
    r = b.job(("imp", "DnCNN-17 BN (17×64)", 0))
    R = json.load(open("b3_results.json")); R.append(r); json.dump(R, open("b3_results.json", "w"), indent=0)
    print(r["model"], {k: round(v["psnr"], 2) for k, v in r["metrics"].items() if k.startswith("Set12")}, flush=True)
