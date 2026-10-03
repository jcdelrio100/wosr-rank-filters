import numpy as np, json, torch, copy
from wos import *
from rankfilters import train_generic, CNN1D, Cascade
from multiprocessing import Pool
def tests(seed):
    T={"in-dist p=10%":make_denoise_data(32,1024,0.10,seed=seed+1),"OOD p=25%":make_denoise_data(32,1024,0.25,seed=seed+2)}
    Xo,So=make_denoise_data(32,1024,0.10,seed=seed+3); T["OOD impulsos ×2"]=(So+2*(Xo-So),So); return T
def nm(y,S): return 10*torch.log10(((y-S)**2).mean()/(S**2).mean()).item()
def cwm(x,c=3,m=5):
    w=torch.ones(m); w[m//2]=c; W=w.sum(); return wos_hard(centered_windows(x,m),w,torch.tensor(0.5))
def run(job):
    torch.set_num_threads(1); seed,name=job
    Xtr,Str=make_denoise_data(128,1024,seed=seed); TT=tests(seed); torch.manual_seed(seed)
    if name=="mediana 5": f=lambda x: hard_rank_filter(x,list(range(5)),3,centered_windows); npar=0
    elif name=="CWM 5 (centro×3)": f=lambda x: cwm(x); npar=0
    else:
        if name=="WOS 1 etapa (STE)": net=WOSCascade([7],windows=centered_windows,tau=0.1)
        elif name=="WOS 2 etapas (STE)": net=WOSCascade([7,7],windows=centered_windows,tau=0.1)
        elif name=="WOS 2 etapas (recocido)": net=WOSCascade([7,7],windows=centered_windows,mode="soft")
        elif name=="WOS 2 etapas + offsets (STE)": net=WOSCascade([7,7],windows=centered_windows,tau=0.1,offsets=True)
        elif name=="NEst 2 etapas": net=Cascade([7,7],centered_windows)
        elif name=="CNN-ReLU grande": net=CNN1D(ch=32,k=7,depth=5)
        lr=2e-3 if "CNN" in name else 0.03
        net=train_generic(net,Xtr,Str,steps=3000,lr=lr,anneal="recocido" in name,seed=seed)
        npar=sum(p.numel() for p in net.parameters())
        f=(lambda x,n=net: n(x,hard=True)) if "CNN" not in name else net
    out=dict(seed=seed,model=name,params=npar)
    with torch.no_grad():
        for k,(X,S) in TT.items(): out[k]=nm(f(X),S)
        if name.startswith("WOS"):
            for K in [4,8]:
                n2=round_weights(copy.deepcopy(net),K)
                out[f"int{K}"]={k:nm(n2(X,hard=True),S) for k,(X,S) in TT.items()}
            out["w"]=[np.round((l.weights()/l.weights().max()).numpy(),2).tolist() for l in net.layers]
            out["r"]=[round(l.r().item(),3) for l in net.layers]
    return out
M=["mediana 5","CWM 5 (centro×3)","WOS 1 etapa (STE)","WOS 2 etapas (STE)","WOS 2 etapas (recocido)","WOS 2 etapas + offsets (STE)","NEst 2 etapas","CNN-ReLU grande"]
if __name__=="__main__":
    with Pool(2) as p: R=p.map(run,[(s,m) for s in [0,1] for m in M])
    json.dump(R,open("wos_ec.json","w"),indent=1)
    keys=["in-dist p=10%","OOD p=25%","OOD impulsos ×2"]
    for m in M:
        rs=[r for r in R if r["model"]==m]
        line="%-30s %6d "%(m,rs[0]["params"])+" ".join("%7.1f"%np.mean([r[k] for r in rs]) for k in keys)
        if m.startswith("WOS"): line+="  | int8: "+" ".join("%6.1f"%np.mean([r["int8"][k] for r in rs]) for k in keys)+"  int4: "+" ".join("%6.1f"%np.mean([r["int4"][k] for r in rs]) for k in keys)
        print(line)
