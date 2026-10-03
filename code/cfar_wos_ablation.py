"""Ablation: is the 8-interferer gain from the quantile shift (r) or from positional repetitions (w)?"""
import json, math, torch, numpy as np
from cfar import ref, NH, Z_go_os
from cfar_np import calibrate, pd_table, build_h0_bank
from cfar_wos import CFARWOS2
torch.set_num_threads(1)
def go_os_k(k): return lambda X: Z_go_os(X, k=k)
m=CFARWOS2(); m.load_state_dict(torch.load("cfar_wos_WOS-log_1.pt"), strict=False) if False else None
P=json.load(open("cfar_wos_results.json"))["_params"]["WOS-log 2 etapas (init GO-OS) s1"]
def make(uniform):
    m=CFARWOS2()
    sd={k.replace("inner.",""):torch.tensor(v) for k,v in P.items()}
    if uniform:
        for h in ["h.0","h.1"]: sd[h+".alpha"]=torch.full_like(sd[h+".alpha"], float(sd[h+".alpha"].mean()))
    m.load_state_dict(sd); m.eval(); return lambda X,m=m: torch.exp(m(X))
dets={"GO-OS k=10":go_os_k(10),"GO-OS k=11":go_os_k(11),"WOS aprendido, pesos uniformes (solo r)":make(True),"WOS aprendido completo":make(False)}
bank=build_h0_bank(20000,seed=7); res={}
for k,f in dets.items():
    with torch.no_grad():
        a_hom,a_mm,worst,ph=calibrate(f,bank); res[k]=dict(overhead_db=10*math.log10(a_mm/a_hom),pd_mm=pd_table(f,a_mm))
    print(k,"ovh %.2f"%res[k]["overhead_db"],flush=True)
json.dump(res,open("cfar_wos_ablation.json","w"),indent=1)
