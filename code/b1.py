import numpy as np, torch, json, math
from matheron import *
torch.set_num_threads(1)
out=[]
cases=[("tesis E1: max{0,2,4,5}→min{0,1,4,6}",[0,2,4,5],4,[0,1,4,6],1),
       ("min→max (apertura plana 5)",[0,1,2,3,4],1,[0,1,2,3,4],5),
       ("mediana5→mediana5",[0,1,2,3,4],3,[0,1,2,3,4],3),
       ("mediana7 sola",[0,1,2,3,4,5,6],4,[0],1)]
for t in range(16):
    m1,r1,m2,r2=random_target(np.random.default_rng(1000+t)); cases.append((f"E2 t={t}",m1,r1,m2,r2))
for name,m1,r1,m2,r2 in cases:
    U,pred=boolean_cascade(m1,r1,m2,r2); bas=minimal_true_sets(U,pred)
    x=torch.randn(4,3000)
    ref=hard_rank_filter(hard_rank_filter(x,m1,r1),m2,r2)
    rep=basis_operator(bas,x)
    err=(ref-rep)[:,20:].abs().max().item()
    sizes=sorted({len(B) for B in bas})
    row=dict(name=name,support=len(U),basis=len(bas),sizes=sizes,maxerr=err,nest_params=2*(len(m1)+len(m2)) if False else 28,
             erosion_bank_params=len(bas)*(max(U)+1))
    out.append(row); print(row)
json.dump(out,open("b1_results.json","w"),indent=1)
