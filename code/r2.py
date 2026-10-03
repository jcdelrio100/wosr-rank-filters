import torch, numpy as np, json, itertools
from robust import *
from multiprocessing import Pool
d=load_digits_tensors()
W=json.load(open("owa_minimax.json")); wstar=np.array(W["empirical"]["w*"])
B=owa_baselines(); tm7=np.zeros(20); tm7[7:13]=1/6
A={"mean":agg_mean,"cw-median":agg_cwmed,"cw-trim(4)":make_owa(B["trimmed(4)"]),"cw-trim(7)":make_owa(tm7),
   "OWA*":make_owa(wstar),"Krum":make_krum(4),"CClip":agg_cclip,"Med_t3→OWA*":make_cascade(wstar,3)}
ATT=[("none",None),("signflip",None),("gauss",None),("alie",None),("alie",2.0),("ipm",None)]
def run(c):
    torch.set_num_threads(1); a,(att,z),ni,seed=c
    r=train_distributed(A[a],att,data=d,noniid=ni,z_alie=z,seed=seed)
    return dict(agg=a,att=att+("" if z is None else f"(z={z})"),noniid=ni,seed=seed,**r)
cfg=list(itertools.product(A,ATT,[0.0,0.5],[0,1,2]))
R=[]
with Pool(2) as p:
    for r in p.imap_unordered(run,cfg):
        R.append(r); 
        if len(R)%24==0: print(len(R),flush=True); json.dump(R,open("r2_results.json","w"))
json.dump(R,open("r2_results.json","w"),indent=0); print("done")
