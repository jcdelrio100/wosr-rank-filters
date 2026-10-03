import numpy as np, json, time
from robust import *
emp=np.load("emp_pool.npy"); res={}
for kind in ["gauss","empirical","t3"]:
    Ss = second_moments(kind, emp=emp, mc=20000)
    sig2=1/16; row={}
    w,_ = minimax_owa(Ss); row["w*"]=np.round(w,4).tolist()
    pts={}
    for k in range(0,10):
        tm=np.zeros(20); tm[k:20-k]=1/(20-2*k); wc,ben,_=worst_case(tm,Ss); pts[f"trim{k}"]=(wc/sig2,ben/sig2)
    wc,ben,_=worst_case(owa_baselines()["median"],Ss); pts["median"]=(wc/sig2,ben/sig2)
    wc,ben,arg=worst_case(w,Ss); pts["OWA*"]=(wc/sig2,ben/sig2)
    front=[]
    for lam in [0.03,0.1,0.3,1,3,10]:
        wl,_=minimax_owa(Ss,benign_weight=lam); wc,ben,_=worst_case(wl,Ss); front.append((lam,wc/sig2,ben/sig2,np.round(wl,3).tolist()))
    row["points"]=pts; row["front"]=front; res[kind]=row
    print(kind, row["w*"]); print({k:(round(a,3),round(b,3)) for k,(a,b) in pts.items()}); print([(l,round(a,3),round(b,3)) for l,a,b,_ in front])
json.dump(res,open("owa_minimax.json","w"),indent=1)
