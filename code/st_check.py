"""Numerical check of the space-time breakdown proposition for Med_t(3) -> interior OWA (t=4, n=20)."""
import numpy as np, json
rng=np.random.default_rng(0); n,t,T,d,steps=20,4,3,200,400
w=np.zeros(n); w[t:n-t]=1/(n-2*t)
res={}
for scen,(nP,nI) in {"P=4,I=0":(4,0),"P=4,I=12":(4,12),"P=0,I=16":(0,16),"P=5,I=0":(5,0)}.items():
    viol={"cascade":0,"spatial":0,"temporal":0}; worst={k:0.0 for k in viol}
    H=[]; Bh=[]
    for s in range(steps):
        B=rng.standard_normal((n,d))                      # benign submissions
        V=B.copy()
        V[n-nP:]=rng.choice([-1e6,1e6])*np.ones(d)        # persistent: arbitrary every step
        I=list(range(n-nP-nI,n-nP))
        if s%3==0: V[I]=rng.choice([-1e6,1e6])*np.ones(d) # intermittent: 1 step in 3
        H.append(V); Bh.append(B)
        if len(H)>T: H.pop(0); Bh.pop(0)
        if len(H)<T: continue
        good=[i for i in range(n) if i < n-nP]            # nodes not persistently corrupt
        lo=np.min(np.stack(Bh)[:,good],axis=(0,1)); hi=np.max(np.stack(Bh)[:,good],axis=(0,1))
        U=np.median(np.stack(H),0)
        outs={"cascade":w@np.sort(U,0),"spatial":w@np.sort(V,0),"temporal":U.mean(0)}
        for k,y in outs.items():
            ex=np.maximum(y-hi,lo-y).max()
            worst[k]=max(worst[k],ex); viol[k]+=int(ex>1e-9)
    res[scen]={k:dict(violations=viol[k],max_excess=float(f"{worst[k]:.3g}")) for k in viol}
    print(scen,res[scen])
json.dump(res,open("st_check.json","w"),indent=1)
