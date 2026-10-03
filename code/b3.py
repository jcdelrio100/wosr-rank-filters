"""B3: top-hat y = x - opening(x) (non-increasing). Increasing models (NEst cascade, Matheron layer) cannot
represent it; a difference of two Matheron layers (tropical rational map) can."""
import numpy as np, torch, json
from matheron import *
torch.set_num_threads(2)
def tophat(x):
    op = hard_rank_filter(hard_rank_filter(x,[0,1,2,3,4],1),[0,1,2,3,4],5)   # flat opening (causal), support 9
    return x - op
# signal: smooth background + narrow positive peaks (what a top-hat extracts)
def data(n,T,seed):
    g=torch.Generator().manual_seed(seed)
    x=torch.cumsum(0.05*torch.randn(n,T,generator=g),1)+ (torch.rand(n,T,generator=g)<0.05).float()*torch.rand(n,T,generator=g)*3
    return x, tophat(x)
xs,ds=data(32,4096,0); xt,dt=data(8,4096,1)
res={}
for name,mk in [("NEst cascade (crec.)",lambda: Cascade([7,7])),
                ("Matheron C=8 (crec.)",lambda: MatheronLayer(8,13,1.0,0)),
                ("Diferencia de 2 Matheron C=8 (tropical racional)",lambda: TropicalRational(8,13,0))]:
    torch.manual_seed(0); net=mk(); fit(net,xs,ds,steps=1500,lr=0.05,seed=0)
    with torch.no_grad(): res[name]=nmse_db(net(xt),dt)
    print(name,res[name],flush=True)
# certificate: any increasing TI model has |Psi(x+c)-Psi(x)| = c ; top-hat is TI but not increasing
json.dump(res,open("b3_results.json","w"),indent=1)
