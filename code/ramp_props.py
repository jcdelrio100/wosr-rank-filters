import torch, numpy as np, json, math
from ramp import *
from wos import wos_hard
torch.manual_seed(0); out={}
# 1) user's example
Z=torch.tensor([[0.0,0.5,1.0]]); n=torch.tensor([1,3,1])
lst=[ramp_wos_discrete(Z,n,(j-0.5)/5).item() for j in range(1,6)]; out["ejemplo_usuario"]=lst; print("lista ramp discreta:",lst)
# 2) continuous beta=1, uniform weights, median -> 1/4,1/2,1/4 of central order stats
X=torch.randn(100000,7); y=ramp_wos(X,torch.ones(7),torch.tensor(0.5),1.0); S=X.sort(-1).values
lf=0.25*S[:,2]+0.5*S[:,3]+0.25*S[:,4]; out["mediana_rampa_vs_L(1/4,1/2,1/4)_maxerr"]=float((y-lf).abs().max()); print("max|ramp-median - L|",out["mediana_rampa_vs_L(1/4,1/2,1/4)_maxerr"])
# 3) properties: monotone, V-invariance, scale, Lipschitz linf, continuity in r and w
def props(f,name):
    X=torch.randn(20000,7); d=torch.rand(20000,7)*0.5
    mono=bool((f(X+d)>=f(X)-1e-6).all())
    vinv=float((f(X+3.7)-(f(X)+3.7)).abs().max()); sc=float((f(2.5*X)-2.5*f(X)).abs().max())
    E=torch.randn(20000,7)*0.1; lip=float(((f(X+E)-f(X)).abs()/E.abs().max(-1).values).max())
    return dict(creciente=mono,V_inv_err=vinv,escala_err=sc,lipschitz_inf=lip)
w=torch.tensor([1.,2.,3.,1.,0.5,2.,1.]); r=torch.tensor(0.4)
for name,f in {"WOS (seleccion)":lambda X: wos_hard(X,w,r),"rampa continua beta=1":lambda X: ramp_wos(X,w,r,1.0),
               "rampa continua beta=0.5":lambda X: ramp_wos(X,w,r,0.5),
               "rampa discreta (enteros)":lambda X: ramp_wos_discrete(X,torch.tensor([1,2,3,1,1,2,1]),r)}.items():
    out[name]=props(f,name); print(name,out[name])
# continuity in r: max jump of y when r moves by 1e-4 across [0.05,0.95]
X=torch.randn(2000,7); rs=torch.linspace(0.05,0.95,9001)
for name,f in {"WOS":lambda r_: wos_hard(X,w,r_),"rampa beta=1":lambda r_: ramp_wos(X,w,r_,1.0),"rampa beta=0.5":lambda r_: ramp_wos(X,w,r_,0.5)}.items():
    Y=torch.stack([f(rr) for rr in rs[::50]],0); jumps=(Y[1:]-Y[:-1]).abs().max().item()
    out[f"salto_max_en_r ({name}, dr=5e-3)"]=jumps; print(name,"max jump in r (dr=5e-3):",jumps)
# gradient availability
wv=w.clone().requires_grad_(True); rv=r.clone().requires_grad_(True)
yv=ramp_wos(X,wv,rv,1.0).sum(); yv.backward(); out["grad_no_nulo_w"]=float((wv.grad.abs()>0).float().mean()); out["grad_r"]=float(rv.grad)
print("frac grad w != 0:",out["grad_no_nulo_w"],"grad r",out["grad_r"])
# 4) breakdown: median-5 uniform with k impulses of +1e6
for k_imp in [1,2,3]:
    X=torch.randn(10000,5); X[:,:k_imp]=1e6
    med=wos_hard(X,torch.ones(5),torch.tensor(0.5)); rmp=ramp_wos(X,torch.ones(5),torch.tensor(0.5),1.0)
    rd=ramp_wos_discrete(X,torch.tensor([1,1,1,1,1]),0.5)
    out[f"ruptura mediana5, {k_imp} impulsos"]=dict(mediana=float(med.abs().max()),rampa_continua=float(rmp.abs().max()),rampa_discreta=float(rd.abs().max()))
    print(k_imp,out[f"ruptura mediana5, {k_imp} impulsos"])
json.dump(out,open("ramp_props.json","w"),indent=1)
