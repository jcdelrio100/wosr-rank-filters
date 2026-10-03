import numpy as np, json, torch
from wos import *
cases=[("E1 tesis max→min",[0,2,4,5],4,[0,1,4,6],1),("min→max (apertura 5)",[0,1,2,3,4],1,[0,1,2,3,4],5),
       ("mediana5∘mediana5",[0,1,2,3,4],3,[0,1,2,3,4],3),("mediana7",[0,1,2,3,4,5,6],4,[0],1),
       ("CWM 5 (centro×3)",None,None,None,None)]
for t in range(16):
    m1,r1,m2,r2=random_target(np.random.default_rng(1000+t)); cases.append((f"E2 t={t}",m1,r1,m2,r2))
out=[]
for name,m1,r1,m2,r2 in cases:
    if m1 is None: continue
    U,pred=boolean_cascade(m1,r1,m2,r2)
    ok,w,T=wos_realization(U,pred)
    row=dict(name=name,support=U,is_wos=ok,weights=w,T=T)
    if ok:   # numerical verification: WOS with integer repetitions == cascade
        x=torch.randn(4,3000); m=max(U)+1
        wt=torch.zeros(m); wt[U]=torch.tensor(w,dtype=torch.float32)
        # f(S)=1 iff sum_{S} w >= T  -> y = largest t with sum_{x_j>=t} w >= T  -> lower quantile at r=(W-T+0.5)/W
        W=wt.sum().item(); r=(W-T+0.5)/W
        y=wos_hard(causal_windows(x,m),wt,torch.tensor(r)); ref=hard_rank_filter(hard_rank_filter(x,m1,r1),m2,r2)
        row["maxerr"]=float((y-ref)[:,20:].abs().max())
    out.append(row); print(row)
json.dump(out,open("wos_ea.json","w"),indent=1)
print("WOS exactos:",sum(r["is_wos"] for r in out),"de",len(out))
