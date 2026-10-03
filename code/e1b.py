import time, json, torch, numpy as np
from rankfilters import *
tgt = two_stage_target([0,2,4,5],4,[0,1,4,6],1)
xtr,dtr = make_ident_data(tgt,1,250_000,seed=0); xte,dte = make_ident_data(tgt,8,4096,seed=100)
L=4096;k=250_000//L; xs,ds=xtr[0,:k*L].view(k,L),dtr[0,:k*L].view(k,L)
print("floor",nmse_db(tgt(xte),dte))
for kind,steps,lr in [("wos",600,0.05),("nest",600,0.05)]:
    t0=time.time(); net=train_modern(xs,ds,[7,7],steps=steps,lr=lr,kind=kind,seed=0)
    with torch.no_grad():
        print(kind, "t=%.0fs"%(time.time()-t0), "soft",nmse_db(net(xte),dte),"hard",nmse_db(net(xte,hard=True),dte))
        if kind=="wos":
            for L_ in net.layers: print(" mask",np.round(L_.mask().numpy(),2)," r=%.3f"%L_.order().item()," b",np.round(L_.b.numpy(),2))
