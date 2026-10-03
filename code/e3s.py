import json, torch, numpy as np
from rankfilters import *
out, models, tests = run_E3(seed=1)
json.dump(out, open("e3_results_seed1.json","w"), indent=1)
for k,v in out.items(): print(k, {kk:(round(vv,1) if isinstance(vv,float) else vv) for kk,vv in v.items()})
nest=models["nest"]
with torch.no_grad():
    for L in nest.layers: print("w",np.round(L.weights().numpy(),2),"b",np.round(L.b.numpy(),2))
torch.save({k:m.state_dict() for k,m in models.items()},"e3_models_seed1.pt")
X,S=tests["in-dist p=10%"]
with torch.no_grad():
    np.savez("e3_trace.npz", x=X[0].numpy(), s=S[0].numpy(),
             med=hard_rank_filter(X[:1],list(range(5)),3,centered_windows)[0].numpy(),
             nest=nest(X[:1],hard=True)[0].numpy(), cnn=models["cnn_l"](X[:1])[0].numpy())
