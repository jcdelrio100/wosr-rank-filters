"""Arce-type signed WOS vs difference of two WOS cascades on the non-increasing top-hat (note 2, B3)."""
import numpy as np, json, torch, torch.nn as nn, torch.nn.functional as F, math
from wos import *
from matheron import TropicalRational
torch.set_num_threads(1)
def tophat(x):
    op=hard_rank_filter(hard_rank_filter(x,[0,1,2,3,4],1),[0,1,2,3,4],5); return x-op
def data(n,T,seed):
    g=torch.Generator().manual_seed(seed)
    x=torch.cumsum(0.05*torch.randn(n,T,generator=g),1)+(torch.rand(n,T,generator=g)<0.05).float()*torch.rand(n,T,generator=g)*3
    return x,tophat(x)
class SignedWOS(WOS):
    """Arce (1998): weight w_j real; sample enters as sgn(w_j) x_j with repetition |w_j|."""
    def __init__(self,m,**kw):
        super().__init__(m,**kw); self.alpha=nn.Parameter(1.0+0.3*torch.randn(m))
    def weights(self): return self.alpha.abs()+1e-6
    def forward(self,x,tau=None,hard=False):
        Z=self.windows(x,self.m)*torch.sign(self.alpha).detach()
        w,r=self.weights(),self.r()
        with torch.no_grad(): y0=wos_hard(Z,w,r)
        if not torch.is_grad_enabled() or hard: return y0
        s=torch.sigmoid((y0.unsqueeze(-1)-Z)/self.tau); G=(w*s).sum(-1)-r*w.sum()
        Gy=((w*s*(1-s)).sum(-1)/self.tau).detach()+1e-6
        return y0-(G-G.detach())/Gy
class SignedCascade(nn.Module):
    def __init__(s): super().__init__(); s.layers=nn.ModuleList([SignedWOS(13,tau=0.3),SignedWOS(13,tau=0.3)])
    def forward(s,x,tau=None,hard=False):
        for l in s.layers: x=l(x,tau,hard)
        return x
class WOSDiff(nn.Module):
    def __init__(s): super().__init__(); s.p=WOSCascade([7,7],tau=0.3); s.q=WOSCascade([7,7],tau=0.3)
    def forward(s,x,tau=None,hard=False): return s.p(x,tau,hard)-s.q(x,tau,hard)
xs,ds=data(32,4096,0); xt,dt=data(8,4096,1); res={}
for name,mk in [("WOS 2 etapas (creciente)",lambda: WOSCascade([7,7],tau=0.3)),("WOS con pesos negativos (Arce), 2 etapas",SignedCascade),
                ("Diferencia de dos WOS 2 etapas",WOSDiff)]:
    vals=[]
    for seed in [0,1]:
        torch.manual_seed(seed); net=train_modern(xs,ds,None,steps=1500,lr=0.05,anneal=False,batch=4,crop=512,seed=seed,model=mk())
        with torch.no_grad(): vals.append(nmse_db(net(xt,hard=True),dt))
    res[name]=vals; print(name,[round(v,1) for v in vals],flush=True)
json.dump(res,open("wos_signed.json","w"),indent=1)
