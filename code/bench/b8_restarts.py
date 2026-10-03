"""B8 — sensitivity of the WOS-R cascade (2 stages x 7 taps, B1 impulsive) to the random initialisation, and the
remedy: N cheap restarts, keep the one with the best VALIDATION error, report its TEST error.
12 independent runs (init seeds 100..111) on the seed-0 data; validation = seed 50, test = seed 1 (as in B1)."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wosnet import *
from rankfilters import make_denoise_data
import b7_structure as b7

def job(init_seed):
    torch.set_num_threads(1)
    X, S = make_denoise_data(128, 1024, 0.10, seed=0)
    Xv, Sv = make_denoise_data(32, 1024, 0.10, seed=50); Xt, St = make_denoise_data(32, 1024, 0.10, seed=1)
    nm = lambda y, d: 10 * torch.log10(((y - d) ** 2).mean() / (d ** 2).mean()).item()
    torch.manual_seed(init_seed); m = b7.FlexChain(2, 7, identity_init_from=99)
    ts = b7.train(m, X, S, seed=init_seed); m.eval()
    with torch.no_grad(): return dict(init=init_seed, val=nm(m(Xv), Sv), test=nm(m(Xt), St), train_s=ts)

if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(2) as p: R = p.map(job, range(100, 112))
    json.dump(R, open("b8_results.json", "w"), indent=1)
    t = np.array([r["test"] for r in R]); v = np.array([r["val"] for r in R])
    print("single run: test %.2f +- %.2f  [%.2f, %.2f]" % (t.mean(), t.std(), t.min(), t.max()))
    for N in (2, 3, 4):
        best = [t[i:i + N][np.argmin(v[i:i + N])] for i in range(0, 12, N)]
        print("best of %d (by validation): test mean %.2f  values %s" % (N, np.mean(best), np.round(best, 2)))
