import sys, time, itertools, torch, numpy as np
from rankfilters import *
from multiprocessing import Pool
tgt = two_stage_target([0,2,4,5],4,[0,1,4,6],1)
xtr,dtr = make_ident_data(tgt,1,250_000,seed=0); xte,dte = make_ident_data(tgt,4,4096,seed=100)
L=4096;k=250_000//L; xs,ds=xtr[0,:k*L].view(k,L),dtr[0,:k*L].view(k,L)
def run(cfg):
    torch.set_num_threads(1)
    kind,lr,anneal,tau0,seed=cfg
    t0=time.time()
    net=train_modern(xs,ds,[7,7],steps=3000,lr=lr,kind=kind,anneal=anneal,tau0=tau0,seed=seed,batch=4,crop=512)
    with torch.no_grad():
        return cfg, round(time.time()-t0), round(nmse_db(net(xte),dte),1), round(nmse_db(net(xte,hard=True),dte),1)
cfgs=list(itertools.product(["nest","wos"],[0.02,0.1],[True,False],[1.0],[0,1]))
with Pool(2) as p:
    for r in p.imap_unordered(run,cfgs): print(r, flush=True)
