"""Refinement from the GO-OS solution: learned 2-stage log-NEst initialised exactly at GO-OS (k=12/16 -> max),
and 4x8 initialised at GO of four OS(k=6/8). NP-minimax objective, lower lr. Can learning improve on GO-OS?"""
import math, json, time, torch
from cfar_np import *
torch.set_num_threads(2)
def init_go(model, k):
    with torch.no_grad():
        for l in model.l1: l.theta.zero_(); l.theta[k-1] = 8.0; l.b.zero_()
        model.l2.theta.zero_(); model.l2.theta[-1] = 8.0; model.l2.b.zero_()
    return model
t0=time.time(); res={}
gen_models=[]
for parts,k,nm in [(2,12,"NEst-log 2 etapas 2×16 (init GO-OS, NP-minimax)"),(4,6,"NEst-log 2 etapas 4×8 (init GO-OS4, NP-minimax)")]:
    m=Named(init_go(CFARNEst2(parts),k),nm)
    base=Named(init_go(CFARNEst2(parts),k),nm.replace("NP-minimax","sin entrenar"))
    # put threshold scale near GO-OS alpha (~29) before training
    with torch.no_grad(): m.inner.l2.b += math.log(28.7)
    m,_=train_np(m, steps=800, lr=0.005, seed=1)
    gen_models += [base, m]
bank=build_h0_bank(20000)
for m in gen_models:
    f=(lambda X,m=m: torch.exp(m(X)))
    a_hom,a_mm,worst,ph=calibrate(f,bank)
    res[m.name]=dict(alpha_hom=a_hom,alpha_mm=a_mm,worst_scenario=worst,max_pfa_with_alpha_hom=ph,
                     overhead_db=10*math.log10(a_mm/a_hom),pd_hom_cal=pd_table(f,a_hom),pd_mm_cal=pd_table(f,a_mm))
    print(m.name,"a_mm %.2f worst %s"%(a_mm,worst),flush=True)
res["_params"]={m.name:{k:v.detach().numpy().round(3).tolist() for k,v in m.state_dict().items()} for m in gen_models}
json.dump(res,open("cfar_np2_results.json","w"),indent=1); print("done %.0fs"%(time.time()-t0))
