import numpy as np, json, torch, sys, copy
from wos import *
from rankfilters import NEst
from multiprocessing import Pool
def mk_target(t):
    rng=np.random.default_rng(5000+t)
    if t<12:
        f,w,T=random_wos_target(rng); return f,dict(kind="WOS 1 etapa",w=w,T=T),1
    f1,w1,T1=random_wos_target(rng); f2,w2,T2=random_wos_target(rng)
    return (lambda x: f2(f1(x))),dict(kind="WOS 2 etapas",w=[w1,w2],T=[T1,T2]),2
def learner(name,stages):
    if name=="WOS-STE τ=0.5": return WOSCascade([7]*stages,tau=0.5)
    if name=="WOS-STE τ=2": return WOSCascade([7]*stages,tau=2.0)
    if name=="WOS-soft recocido": return WOSCascade([7]*stages,mode="soft")
    if name=="NEst (aditiva)": return Cascade([7]*stages)
L=["WOS-STE τ=0.5","WOS-STE τ=2","WOS-soft recocido","NEst (aditiva)"]
def run(job):
    torch.set_num_threads(1); t,name=job
    f,info,st=mk_target(t)
    xtr,dtr=make_ident_data(f,1,150_000,seed=t); xte,dte=make_ident_data(f,4,4096,seed=t+500)
    k=150_000//4096; xs,ds=xtr[0,:k*4096].view(k,4096),dtr[0,:k*4096].view(k,4096)
    floor=nmse_db(f(xte),dte)
    torch.manual_seed(t); net=learner(name,st)
    ann = name=="WOS-soft recocido"
    net=train_modern(xs,ds,None,steps=2000,lr=0.05,anneal=ann,tau0=2.0,tau1=0.02,batch=4,crop=512,seed=t,model=net)
    out=dict(t=t,learner=name,**info,floor=floor)
    with torch.no_grad():
        out["gap"]=nmse_db(net(xte,hard=True),dte)-floor
        if name.startswith("WOS"):
            out["w_learned"]=[np.round((l.weights()/l.weights().max()).numpy(),2).tolist() for l in net.layers]
            out["r_learned"]=[round(l.r().item(),3) for l in net.layers]
            for K in [4,8,16]:
                n2=round_weights(copy.deepcopy(net),K); out[f"gap_int{K}"]=nmse_db(n2(xte,hard=True),dte)-floor
    return out
if __name__=="__main__":
    jobs=[(t,n) for t in range(18) for n in L]
    R=[]
    with Pool(2) as p:
        for r in p.imap_unordered(run,jobs): R.append(r); print({k:(round(v,1) if isinstance(v,float) else v) for k,v in r.items() if k not in ("w_learned",)},flush=True)
    json.dump(R,open("wos_eb.json","w"),indent=1)
