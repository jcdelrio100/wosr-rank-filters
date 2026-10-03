import numpy as np, json
from robust import *
emp=np.load("emp_pool.npy"); out={}
for n,f in [(20,4),(50,10),(100,20)]:
    Ss=second_moments("gauss",n=n,f=f,mc=20000,z1=np.linspace(-25,25,51),z2c=np.linspace(-8,8,9))
    w,_=minimax_owa(Ss,n=n,trim=f)
    sup=np.nonzero(w>1e-3)[0]; trim_eff=sup.min()/n
    best=min(range(f,n//2),key=lambda k: worst_case(np.r_[np.zeros(k),np.full(n-2*k,1/(n-2*k)),np.zeros(k)],Ss)[0])
    out[f"{n},{f}"]=dict(support=[int(sup.min())+1,int(sup.max())+1],trim_frac=round(trim_eff,3),best_trim_frac=round(best/n,3))
    print(n,f,out[f"{n},{f}"],flush=True)
json.dump(out,open("owa_scaling.json","w"),indent=1)
