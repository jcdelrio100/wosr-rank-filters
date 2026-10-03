import numpy as np, json, torch, time
from ramp import *
from wos import random_wos_target, WOSCascade
from rankfilters import two_stage_target, make_ident_data, nmse_db, random_target, train_modern
from multiprocessing import Pool
E2={eval(l)["t"]:eval(l) for l in open("e2.log") if l.startswith("{")}
def target(kind,t):
    if kind=="E2":
        m1,r1,m2,r2=random_target(np.random.default_rng(1000+t)); return two_stage_target(m1,r1,m2,r2),2,t
    f,w,T=random_wos_target(np.random.default_rng(5000+t)); return f,1,t
L=["rampa β=1 fija","rampa β 1→0 (continuación)"]
def run(job):
    torch.set_num_threads(1); kind,t,s,name=job
    f,st,dseed=target(kind,t)
    xtr,dtr=make_ident_data(f,1,150_000,seed=dseed); xte,dte=make_ident_data(f,4,4096,seed=dseed+500)
    k=150_000//4096; xs,ds=xtr[0,:k*4096].view(k,4096),dtr[0,:k*4096].view(k,4096)
    floor=E2[t]["floor"] if kind=="E2" else nmse_db(f(xte),dte)
    torch.manual_seed(100*s+t); net=RampCascade([7]*st,beta=1.0)
    net=train_ramp(net,xs,ds,steps=3000,lr=0.05,seed=100*s+t,schedule="fixed" if "fija" in name else "anneal")
    with torch.no_grad():
        out=dict(kind=kind,t=t,seed=s,learner=name,gap=nmse_db(net(xte),dte)-floor)
        net.set_beta(0.0); out["gap_sel"]=nmse_db(net(xte),dte)-floor   # same weights used as exact WOS (selection)
    return out
if __name__=="__main__":
    jobs=[("WOS",t,s,n) for t in range(12) for s in [0,1,2] for n in L]+[("E2",t,s,n) for t in range(16) for s in [0,1,2] for n in L]
    R=[]; t0=time.time()
    with Pool(2) as p:
        for r in p.imap_unordered(run,jobs): R.append(r); json.dump(R,open("ramp_ident.json","w"))
    print("done",len(R),"%.0fs"%(time.time()-t0))
