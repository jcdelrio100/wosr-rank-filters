import sys, os, json, torch, torch.nn.functional as F, time
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import b2_coverage as b2
from wosnet import *
torch.set_num_threads(1)
def setbeta(m, beta):
    for l in m.modules():
        if isinstance(l, WOSR): l.beta_train = beta
def run(tname, variant, seed=0, steps=3000):
    cls, f = b2.TARGETS[tname]
    X, D = b2.data(f, 64, 2048, seed); Xt, Dt = b2.data(f, 8, 2048, seed + 100); floor = b2.nmse(f(Xt), Dt)
    torch.manual_seed(seed); model = WOSBank(4, 2, 1, 7)
    opt = torch.optim.Adam(model.parameters(), lr=0.03); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=0.0015)
    model.train()
    for s in range(steps):
        if variant == "anneal": setbeta(model, max(0.0, 1 - s / (0.7 * steps)))
        if variant == "refit" and s == int(0.8 * steps):
            setbeta(model, 0.0)
        i = torch.randint(0, 64, (16,)); o = int(torch.randint(0, 2048 - 256, (1,)))
        loss = F.mse_loss(model(X[i, o:o + 256])[:, 8:-8], D[i, o:o + 256][:, 8:-8])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    set_deploy(model)
    if variant in ("ls", "refit"):
        with torch.no_grad():
            Y = torch.stack([c(X) for c in model.chains], -1)[:, 8:-8].reshape(-1, 4)
            A = torch.cat([Y, torch.ones(len(Y), 1)], 1); sol = torch.linalg.lstsq(A, D[:, 8:-8].reshape(-1, 1)).solution[:, 0]
            model.mix.copy_(sol[:4]); model.bias.copy_(sol[4])
    with torch.no_grad(): return b2.nmse(model(Xt), Dt) - floor, model.mix.detach().numpy().round(2).tolist(), [round(float(l.r()),2) for l in model.modules() if isinstance(l, WOSR)]
for t in ["Mediana 7", "Apertura 5 (erosión→dilatación)"]:
    for v in ["ls", "anneal", "refit"]:
        t0=time.time(); print(t[:10], v, run(t, v), round(time.time()-t0), flush=True)
