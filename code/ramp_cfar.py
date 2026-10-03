"""Ramp-WOS CFAR (log domain, 32 cells, free repetitions) vs WOS (note 6) under correlated / independent texture."""
import json, math, time, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from cfar_corr import train, calib, pd_curve, snr50, NH, ref
from ramp import ramp_wos
class RampCFAR(nn.Module):
    def __init__(s, beta=1.0):
        super().__init__(); s.alpha=nn.Parameter(torch.full((2*NH,),math.log(math.expm1(1.0)))); s.rho=nn.Parameter(torch.tensor(math.log((23.5/32)/(1-23.5/32))))
        s.c=nn.Parameter(torch.tensor(math.log(20.0))); s.beta=beta
    def forward(s,X): return ramp_wos(torch.log(ref(X)),F.softplus(s.alpha),torch.sigmoid(s.rho),s.beta)+s.c
def job(spec):
    torch.set_num_threads(1); rho,seed=spec
    return spec, train(RampCFAR(1.0),rho,seed=seed)
if __name__=="__main__":
    from multiprocessing import Pool
    t0=time.time()
    with Pool(2) as p: T=p.map(job,[(0.0,0),(0.9,0),(0.9,1)])
    torch.set_num_threads(2); res={}
    for (rho,s),m in T:
        f=lambda X,m=m: torch.exp(m(X)); a=calib(f,rho); pdc=pd_curve(f,a,rho)
        w=F.softplus(m.alpha).detach().numpy()
        res[f"rho={rho} | rampa s{s}"]=dict(snr50={ni:snr50(pdc[ni]) for ni in pdc},w=np.round(w/w.max(),3).tolist(),r=float(torch.sigmoid(m.rho)))
        print(rho,s,{ni:round(snr50(pdc[ni]),2) for ni in pdc},flush=True)
    json.dump(res,open("ramp_cfar.json","w"),indent=1); print("done %.0fs"%(time.time()-t0))
