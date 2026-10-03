"""E3 impulsive-noise bench with ramp-WOS (continuous, learnable beta-fixed / annealed) and the user's discrete integer
ramp (weights from the learned model rounded to integers <= K, r refit)."""
import numpy as np, json, torch, copy, math
from ramp import *
from wos import WOSCascade, round_weights, wos_hard
from wos_ec import tests, nm
from rankfilters import train_generic, make_denoise_data, centered_windows, hard_rank_filter
from multiprocessing import Pool
class Wrap(torch.nn.Module):
    """adapter so train_generic (net(x,tau)) works with RampCascade"""
    def __init__(s,net): super().__init__(); s.net=net
    def forward(s,x,tau=None,hard=False): return s.net(x)
def discrete_version(net,K,X,S):
    """user's integer ramp: per layer n = round(w/max*K), refit r on grid; returns callable"""
    Ls=[(torch.round(l.weights()/l.weights().max()*K).clamp_min(0).long(), float(l.r())) for l in net.layers]
    def f(x,Ls):
        for (n,r),l in zip(Ls,net.layers): x=ramp_wos_discrete(centered_windows(x,l.m),n,r)
        return x
    best=list(Ls)
    for i in range(len(Ls)):
        cand=[]
        for r in np.linspace(0.02,0.98,49):
            L2=list(best); L2[i]=(best[i][0],float(r)); cand.append((((f(X,L2)-S)**2).mean().item(),float(r)))
        best[i]=(best[i][0],min(cand)[1])
    return (lambda x: f(x,best)), [ (b[0].tolist(),round(b[1],3)) for b in best]
def run(job):
    torch.set_num_threads(1); seed,name=job
    Xtr,Str=make_denoise_data(128,1024,seed=seed); TT=tests(seed); torch.manual_seed(seed)
    out=dict(seed=seed,model=name)
    if name=="mediana-rampa 5 (L 1/4,1/2,1/4)":
        f=lambda x: ramp_wos(centered_windows(x,5),torch.ones(5),torch.tensor(0.5),1.0)
    elif name=="mediana 5":
        f=lambda x: hard_rank_filter(x,list(range(5)),3,centered_windows)
    else:
        st=1 if "1 etapa" in name else 2
        net=RampCascade([7]*st,windows=centered_windows,beta=1.0)
        if "continuación" in name:
            # train with annealed beta through manual loop on random crops
            net=train_ramp(net,Xtr,Str,steps=3000,lr=0.03,batch=32,crop=256,seed=seed,schedule="anneal",skip=0)
        else:
            net=train_ramp(net,Xtr,Str,steps=3000,lr=0.03,batch=32,crop=256,seed=seed,schedule="fixed",skip=0)
        f=lambda x,n=net: n(x)
        if "fija" in name:
            with torch.no_grad():
                for K in [2,4]:
                    g,Ls=discrete_version(net,K,Xtr[:32],Str[:32]); out[f"discreta_int{K}"]={k:nm(g(X),S) for k,(X,S) in TT.items()}; out[f"n_int{K}"]=Ls
                out["w"]=[np.round((l.weights()/l.weights().max()).numpy(),2).tolist() for l in net.layers]; out["r"]=[round(float(l.r()),3) for l in net.layers]
    with torch.no_grad():
        for k,(X,S) in TT.items(): out[k]=nm(f(X),S)
    return out
M=["mediana 5","mediana-rampa 5 (L 1/4,1/2,1/4)","rampa 1 etapa β=1 fija","rampa 2 etapas β=1 fija","rampa 2 etapas β 1→0 (continuación)"]
if __name__=="__main__":
    with Pool(2) as p: R=p.map(run,[(s,m) for s in [0,1] for m in M])
    json.dump(R,open("ramp_ec.json","w"),indent=1)
    keys=["in-dist p=10%","OOD p=25%","OOD impulsos ×2"]
    for m in M:
        rs=[r for r in R if r["model"]==m]
        line="%-40s "%m+" ".join("%7.1f"%np.mean([r[k] for r in rs]) for k in keys)
        for K in [2,4]:
            if f"discreta_int{K}" in rs[0]: line+="  | discreta K=%d: "%K+" ".join("%6.1f"%np.mean([r[f"discreta_int{K}"][k] for r in rs]) for k in keys)
        print(line)
