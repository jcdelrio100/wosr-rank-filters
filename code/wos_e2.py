import numpy as np, json, torch
from wos import *
from multiprocessing import Pool
E2={eval(l)["t"]:eval(l) for l in open("e2.log") if l.startswith("{")}
def run(job):
    torch.set_num_threads(1); t,name=job
    m1,r1,m2,r2=random_target(np.random.default_rng(1000+t)); tgt=two_stage_target(m1,r1,m2,r2)
    xtr,dtr=make_ident_data(tgt,1,150_000,seed=t); xte,dte=make_ident_data(tgt,4,4096,seed=t+500)
    k=150_000//4096; xs,ds=xtr[0,:k*4096].view(k,4096),dtr[0,:k*4096].view(k,4096)
    floor=E2[t]["floor"]; torch.manual_seed(t)
    net=WOSCascade([7,7],mode="soft") if "soft" in name else WOSCascade([7,7],tau=0.5)
    net=train_modern(xs,ds,None,steps=3000,lr=0.05,anneal="soft" in name,tau0=2.0,tau1=0.02,batch=4,crop=512,seed=t,model=net)
    with torch.no_grad(): g=nmse_db(net(xte,hard=True),dte)-floor
    return dict(t=t,learner=name,gap=g,nest_gap=E2[t]["adam"]-floor,nest_x4=E2[t]["adam_x4"]-floor)
if __name__=="__main__":
    jobs=[(t,n) for t in range(16) for n in ["WOS-soft recocido","WOS-STE τ=0.5"]]
    with Pool(2) as p: R=p.map(run,jobs)
    json.dump(R,open("wos_e2.json","w"),indent=1)
    for n in ["WOS-soft recocido","WOS-STE τ=0.5"]:
        g=[r["gap"] for r in R if r["learner"]==n]; print(n,"éxitos",sum(x<=3 for x in g),"/16 mediana %.1f"%np.median(g))
    g=[r["nest_gap"] for r in R if r["learner"]=="WOS-STE τ=0.5"]; print("NEst Adam (E2)",sum(x<=3 for x in g),"/16 mediana %.1f"%np.median(g))
    for r in sorted(R,key=lambda r:(r["t"],r["learner"])): print(r["t"],r["learner"],round(r["gap"],1),round(r["nest_gap"],1))
