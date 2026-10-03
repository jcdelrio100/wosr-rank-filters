import numpy as np, torch, json, sys
from matheron import *
from multiprocessing import Pool
B1={r["name"]:r for r in json.load(open("b1_results.json"))}
E2={eval(l)["t"]:eval(l) for l in open("e2.log") if l.startswith("{")}
VARIANTS=[("C=|Bas|, init 10",1,10.0,False,None),("C=4|Bas|, init 10",4,10.0,False,None),("C=4|Bas|, init 1 (all taps)",4,1.0,False,None),("C=4|Bas|, LSE-anneal",4,1.0,True,None),("C=4|Bas|, random-subset init",4,0,False,(20.0,6))]
def run(args):
    t,(vn,mult,spread,ann,sub)=args
    torch.set_num_threads(1)
    m1,r1,m2,r2=random_target(np.random.default_rng(1000+t)); tgt=two_stage_target(m1,r1,m2,r2)
    xtr,dtr=make_ident_data(tgt,1,150_000,seed=t); xte,dte=make_ident_data(tgt,4,4096,seed=t+500)
    L=4096;k=150_000//L; xs,ds=xtr[0,:k*L].view(k,L),dtr[0,:k*L].view(k,L)
    nb=B1[f"E2 t={t}"]["basis"]; C=min(max(1,nb*mult),512)
    torch.manual_seed(t); lay=MatheronLayer(C,13,init_spread=spread,seed=t,subset_init=sub)
    fit(lay,xs,ds,steps=STEPS,lr=0.05,anneal=ann,seed=t)
    with torch.no_grad(): v=nmse_db(lay(xte,hard=True),dte)
    return dict(t=t,variant=vn,C=C,basis=nb,gap=v-E2[t]["floor"],nest_gap=E2[t]["adam"]-E2[t]["floor"])
STEPS=1500
if __name__=="__main__":
    ts=[int(a) for a in sys.argv[1].split(",")] if len(sys.argv)>1 else list(range(16))
    vs=[VARIANTS[int(a)] for a in sys.argv[2].split(",")] if len(sys.argv)>2 else VARIANTS
    cfg=[(t,v) for v in vs for t in ts]
    R=[]
    with Pool(2) as p:
        for r in p.imap_unordered(run,cfg): R.append(r); print(r,flush=True)
    json.dump(R,open(sys.argv[3] if len(sys.argv)>3 else "b2_results.json","w"),indent=1)
