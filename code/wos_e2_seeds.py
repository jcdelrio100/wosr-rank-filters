import numpy as np, json, torch
from wos import *
from multiprocessing import Pool
E2={eval(l)["t"]:eval(l) for l in open("e2.log") if l.startswith("{")}
L=["WOS-STE τ=0.5","WOS recocido","NEst (Adam)"]
def run(job):
    torch.set_num_threads(1); t,s,name=job
    m1,r1,m2,r2=random_target(np.random.default_rng(1000+t)); tgt=two_stage_target(m1,r1,m2,r2)
    xtr,dtr=make_ident_data(tgt,1,150_000,seed=t); xte,dte=make_ident_data(tgt,4,4096,seed=t+500)
    k=150_000//4096; xs,ds=xtr[0,:k*4096].view(k,4096),dtr[0,:k*4096].view(k,4096)
    torch.manual_seed(100*s+t)
    if name=="NEst (Adam)":
        net=train_modern(xs,ds,[7,7],steps=3000,lr=0.1,anneal=False,batch=4,crop=512,seed=100*s+t)
    else:
        net=WOSCascade([7,7],mode="soft") if "recocido" in name else WOSCascade([7,7],tau=0.5)
        net=train_modern(xs,ds,None,steps=3000,lr=0.05,anneal="recocido" in name,tau0=2.0,tau1=0.02,batch=4,crop=512,seed=100*s+t,model=net)
    with torch.no_grad(): g=nmse_db(net(xte,hard=True),dte)-E2[t]["floor"]
    return dict(t=t,seed=s,learner=name,gap=g)
if __name__=="__main__":
    jobs=[(t,s,n) for s in [0,1,2] for t in range(16) for n in L]
    R=[]
    with Pool(2) as p:
        for r in p.imap_unordered(run,jobs): R.append(r); json.dump(R,open("wos_e2_seeds.json","w"))
    print("done",len(R))
