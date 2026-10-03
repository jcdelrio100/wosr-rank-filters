import numpy as np, json, torch, copy
from wos import *
from wos_ec import tests, nm
from rankfilters import train_generic
from multiprocessing import Pool
def refit_r(net, X, S, grid=np.linspace(0.02,0.98,49)):
    with torch.no_grad():
        for _ in range(2):
            for L in net.layers:
                best=(1e9,None)
                for r in grid:
                    L.rho.fill_(float(np.log(r/(1-r)))); e=((net(X,hard=True)-S)**2).mean().item()
                    if e<best[0]: best=(e,r)
                L.rho.fill_(float(np.log(best[1]/(1-best[1]))))
    return net
def run(job):
    torch.set_num_threads(1); seed,st=job
    Xtr,Str=make_denoise_data(128,1024,seed=seed); TT=tests(seed); torch.manual_seed(seed)
    net=train_generic(WOSCascade([7]*st,windows=centered_windows,tau=0.1),Xtr,Str,steps=3000,lr=0.03,seed=seed)
    out=dict(seed=seed,stages=st)
    with torch.no_grad():
        out["real"]={k:nm(net(X,hard=True),S) for k,(X,S) in TT.items()}
        for K in [2,3,4,6,8,16]:
            n2=round_weights(copy.deepcopy(net),K); out[f"int{K}"]={k:nm(n2(X,hard=True),S) for k,(X,S) in TT.items()}
            n3=refit_r(n2,Xtr[:32],Str[:32]); out[f"int{K}+r"]={k:nm(n3(X,hard=True),S) for k,(X,S) in TT.items()}
            out[f"w_int{K}"]=[torch.round(l.weights()).int().tolist() for l in n3.layers]
    return out
if __name__=="__main__":
    with Pool(2) as p: R=p.map(run,[(s,st) for s in [0,1] for st in [1,2]])
    json.dump(R,open("wos_round.json","w"),indent=1)
    for st in [1,2]:
        rs=[r for r in R if r["stages"]==st]
        for key in ["real"]+[f"int{K}{s}" for K in [2,3,4,6,8,16] for s in ["","+r"]]:
            print(st,"%-8s"%key," ".join("%6.1f"%np.mean([r[key][k] for r in rs]) for k in rs[0]["real"]))
        print([r["w_int4"] for r in rs])
