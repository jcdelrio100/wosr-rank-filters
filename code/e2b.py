import json, torch, numpy as np
from rankfilters import *
from multiprocessing import Pool
class Bank(nn.Module):
    def __init__(s, C=4):
        super().__init__(); s.ch=nn.ModuleList([Cascade([7,7]) for _ in range(C)]); s.mix=nn.Linear(C,1)
        nn.init.constant_(s.mix.weight,1/C); nn.init.zeros_(s.mix.bias)
    def forward(s,x,tau=None,hard=False): return s.mix(torch.stack([c(x,tau,hard) for c in s.ch],-1)).squeeze(-1)
def run(t):
    torch.set_num_threads(1)
    rng=np.random.default_rng(1000+t); m1,r1,m2,r2=random_target(rng); tgt=two_stage_target(m1,r1,m2,r2)
    xtr,dtr=make_ident_data(tgt,1,150_000,seed=t); xte,dte=make_ident_data(tgt,4,4096,seed=t+500)
    L=4096;k=150_000//L; xs,ds=xtr[0,:k*L].view(k,L),dtr[0,:k*L].view(k,L)
    out=dict(t=t)
    net=train_modern(xs,ds,[7,7],steps=3000,lr=0.1,anneal=True,tau0=1.0,batch=4,crop=512,seed=t)
    with torch.no_grad(): out["anneal"]=nmse_db(net(xte),dte)
    torch.manual_seed(t); bank=Bank(4)
    bank=train_modern(xs,ds,None,steps=3000,lr=0.1,anneal=False,batch=4,crop=512,seed=t,model=bank)
    with torch.no_grad(): out["bank4"]=nmse_db(bank(xte),dte)
    return out
with Pool(2) as p: R=p.map(run,[5,6,7,8,10,11,14])
json.dump(R,open("e2b_results.json","w"),indent=1); print(*R,sep="\n")
