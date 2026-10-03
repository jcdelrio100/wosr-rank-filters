"""Additive (NEst, z=x+b) vs multiplicative mask (z=a·x, a=min(|b|,1)) vs log-domain NEst, signed and positive signals."""
import math, json, sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from multiprocessing import Pool
from rankfilters import causal_windows, hard_rank_filter, two_stage_target, nmse_db, train_modern, random_target, NEst

LO=float(os.environ.get('POS_LO','0'))
class MulNEst(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m=m
        self.b=nn.Parameter(0.95+0.02*torch.randn(m)); self.theta=nn.Parameter(0.01*torch.randn(m))
    def forward(self, x, tau=None, hard=False):
        with torch.no_grad(): self.b.data.clamp_(0.0,1.0)        # projected gradient onto 0<=a<=1
        a=self.b
        Z=causal_windows(x,self.m)*a
        w=torch.softmax(self.theta,-1)
        if hard: w=F.one_hot(w.argmax(),self.m).float()
        return Z.sort(-1).values@w
class Casc(nn.Module):
    def __init__(self, L): super().__init__(); self.layers=nn.ModuleList(L)
    def forward(self,x,tau=None,hard=False):
        for l in self.layers: x=l(x,tau,hard)
        return x
class LogWrap(nn.Module):
    def __init__(self, inner): super().__init__(); self.inner=inner
    def forward(self,x,tau=None,hard=False): return torch.exp(self.inner(torch.log(x),tau,hard))

def targets():
    T={"E1 max→min":([0,2,4,5],4,[0,1,4,6],1),"min∘id (erosión {0,2,5})":([0,2,5],1,[0],1),
       "max∘id (dilatación {1,3,6})":([1,3,6],3,[0],1),"mediana5∘id":([0,1,2,3,4],3,[0],1),
       "med∘med":([0,1,2,3,4],3,[0,2,4],2)}
    for t in [1,8,9,12,14]: T[f"E2 t={t}"]=random_target(np.random.default_rng(1000+t))
    return T

def data(tgt, positive, n_seq, T, seed, snr_db=35.0):
    g=torch.Generator().manual_seed(seed)
    x=(LO+torch.rand(n_seq,T,generator=g)*(10-LO)) if positive else (torch.rand(n_seq,T,generator=g)*2-1)*5
    d=tgt(x); P=(x**2).mean().item() if positive else 25/3
    pn=P/10**(snr_db/10); x=x+math.sqrt(3*pn)*(torch.rand(n_seq,T,generator=g)*2-1)
    if positive: x=x.clamp(min=1e-3); d=d.clamp(min=1e-3)
    return x,d

def run(job):
    torch.set_num_threads(1)
    name,spec,variant,seed=job
    m1,r1,m2,r2=spec; tgt=two_stage_target(m1,r1,m2,r2)
    positive=variant.startswith("pos")
    xtr,dtr=data(tgt,positive,36,4096,seed); xte,dte=data(tgt,positive,4,4096,seed+500)
    floor=nmse_db(tgt(xte.clamp(min=0) if positive else xte),dte)
    torch.manual_seed(seed)
    if "log" in variant: net=LogWrap(Casc([NEst(7),NEst(7)]))
    elif variant.endswith("aditiva"): net=Casc([NEst(7),NEst(7)])
    elif variant.endswith("multiplicativa"): net=Casc([MulNEst(7),MulNEst(7)])
    else: net=LogWrap(Casc([NEst(7),NEst(7)]))
    net=train_modern(xtr,dtr,None,steps=3000,lr=0.1,anneal=False,batch=4,crop=512,seed=seed,model=net)
    with torch.no_grad():
        g=nmse_db(net(xte),dte)-floor
    return dict(target=name,variant=variant,seed=seed,gap=g,floor=floor)

if __name__=="__main__":
    V=sys.argv[1].split(",") if len(sys.argv)>1 else ["signed · aditiva","signed · multiplicativa","pos · aditiva","pos · multiplicativa","pos · log-aditiva"]
    jobs=[(n,s,v,sd) for n,s in targets().items() for v in V for sd in [0,1]]
    R=[]
    with Pool(2) as p:
        for r in p.imap_unordered(run,jobs): R.append(r); print(r,flush=True)
    json.dump(R,open(sys.argv[2] if len(sys.argv)>2 else "mulmask_results.json","w"),indent=1)
