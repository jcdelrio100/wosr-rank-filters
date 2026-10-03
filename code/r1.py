import numpy as np, json, time
from robust import *
data = load_digits_tensors()
t0=time.time(); emp = collect_benign_coords(data=data); print("empirical pool", emp.shape, "kurtosis", ((emp-emp.mean())**4).mean()/emp.var()**2, "%.0fs"%(time.time()-t0))
np.save("emp_pool.npy", emp[np.random.default_rng(0).choice(len(emp), 200000)])
res={}
for kind in ["gauss","t3","empirical"]:
    t0=time.time()
    Ss = second_moments(kind, emp=np.load("emp_pool.npy"))
    w, val = minimax_owa(Ss)
    row={"w*":np.round(w,4).tolist()}
    sig2 = 1/16   # MSE of the oracle mean of the 16 benign values (unit variance)
    for name, wb in {**owa_baselines(), "OWA*": w}.items():
        wc, ben, arg = worst_case(wb, Ss)
        row[name]={"worst/oracle":round(wc/sig2,3),"benign/oracle":round(ben/sig2,3),"argmax":arg}
    res[kind]=row; print(kind, "%.0fs"%(time.time()-t0), json.dumps(row))
json.dump(res, open("owa_minimax.json","w"), indent=1)
