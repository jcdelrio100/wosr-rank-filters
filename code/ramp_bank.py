import numpy as np, json, torch, torch.nn as nn
from ramp import RampCascade, train_ramp
from wos_ec import tests, nm
from rankfilters import make_denoise_data, centered_windows
from multiprocessing import Pool
class RampBank(nn.Module):
    def __init__(s,C=4):
        super().__init__(); s.ch=nn.ModuleList([RampCascade([7,7],windows=centered_windows,beta=1.0) for _ in range(C)]); s.mix=nn.Linear(C,1)
        nn.init.constant_(s.mix.weight,1/C); nn.init.zeros_(s.mix.bias)
    def set_beta(s,b): [c.set_beta(b) for c in s.ch]
    def forward(s,x): return s.mix(torch.stack([c(x) for c in s.ch],-1)).squeeze(-1)
def run(seed):
    torch.set_num_threads(1); Xtr,Str=make_denoise_data(128,1024,seed=seed); TT=tests(seed); torch.manual_seed(seed)
    net=train_ramp(RampBank(4),Xtr,Str,steps=3000,lr=0.03,batch=32,crop=256,seed=seed,skip=0)
    with torch.no_grad(): r={k:nm(net(X),S) for k,(X,S) in TT.items()}
    r["params"]=sum(p.numel() for p in net.parameters()); r["seed"]=seed; return r
if __name__=="__main__":
    with Pool(2) as p: R=p.map(run,[0,1])
    json.dump(R,open("ramp_bank.json","w"),indent=1)
    for k in ["in-dist p=10%","OOD p=25%","OOD impulsos ×2"]: print(k, round(np.mean([r[k] for r in R]),1), [round(r[k],1) for r in R])
    print("params",R[0]["params"])
