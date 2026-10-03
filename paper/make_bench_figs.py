"""Benchmark dashboards for the Zenodo preprint (all numbers read from bench/*.json).
Colour = model family (fixed order): rank-based (blue), neural networks (orange), classical filters (aqua)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _ROOT = _os.path.dirname(_HERE)
_CODE = _os.path.join(_ROOT, "code")
import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
B = _os.path.join(_CODE, "bench"); FG = lambda n: os.path.join(_os.path.join(_HERE, "figs"), n)
RANK, NET, CLS = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})
L = lambda f: json.load(open(os.path.join(B, f)))

def fam(name):
    if name.startswith(("WOS", "NEst")): return RANK
    if name.startswith(("Mediana", "Median", "AMF", "OS", "GO")): return CLS
    return NET

def hbar(ax, labels, vals, colors, log=False, fmt="%.0f", xlabel=""):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=colors, height=0.62, edgecolor="white", linewidth=1.0)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    if log: ax.set_xscale("log")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_xlabel(xlabel)
    for yi, v in zip(y, vals):
        t = (("%.0f" % v) if v >= 10 else ("%.2g" % v)) if fmt == "auto" else fmt % v
        ax.text(v * (1.12 if log else 1) + (0 if log else 0.01 * max(vals)), yi, t, va="center", fontsize=6.5, color=MUTED)
    ax.set_xlim(right=(max(vals) * (6 if log else 1.22)))

def dumbbell(ax, labels, a, b, colors, la, lb, xlabel, better="lower"):
    y = np.arange(len(labels))[::-1]
    for yi, u, v, c in zip(y, a, b, colors):
        ax.plot([u, v], [yi, yi], color=c, lw=1.6, alpha=0.55, solid_capstyle="round")
        ax.plot(u, yi, "o", color=c, ms=6, mec="white", mew=1.0)
        ax.plot(v, yi, "X", color=c, ms=6.5, mec="white", mew=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel(xlabel + ("  (← better)" if better == "lower" else "  (better →)"))
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED, label=la), Line2D([], [], marker="X", ls="", color=MUTED, label=lb)],
              loc="upper center", bbox_to_anchor=(0.45, -0.17), fontsize=7, ncol=1)

def family_legend(fig, extra_cls=True):
    h = [Line2D([], [], marker="s", ls="", ms=8, color=RANK, label="rank-based (WOS-R, NEst)"),
         Line2D([], [], marker="s", ls="", ms=8, color=NET, label="neural networks")]
    if extra_cls: h.append(Line2D([], [], marker="s", ls="", ms=8, color=CLS, label="classical filter"))
    fig.legend(handles=h, loc="upper center", ncol=len(h), fontsize=8, bbox_to_anchor=(0.5, 1.0))

# ------------------------------------------------------------------------------------------------ 1D (B1 + B6)
M1 = [("WOS-R cadena (2×7)", "WOS-R cascade"), ("WOS-R banco C=4", "WOS-R bank C=4"), ("WOS-R banco C=8", "WOS-R bank C=8"),
      ("NEst banco C=4", "NEst bank C=4 (1993)"), ("CNN-S", "CNN small"), ("CNN-M", "CNN medium"), ("CNN-L", "CNN large"),
      ("MLP (ventana 15)", "MLP"), ("Transformer", "Transformer")]

def conv_time(curve, tol=0.5):
    e = np.array([c[2] for c in curve]); t = np.array([c[1] for c in curve]); fin = e[-1]
    ok = e <= fin + tol
    for i in range(len(e)):
        if ok[i:].all(): return t[i]
    return t[-1]

def fig_1d():
    R1, R6, LAT = L("b1_results.json"), L("b6_results.json"), L("latency.json")["1d"]
    rows = []
    for k, lab in [("Mediana 5", "Median 5 (classical)")] + M1:
        ri = [r for r in R1 if r["model"] == k and r["kind"] == "imp"]
        c6 = [r for r in R6 if r["model"] == k]
        mults = {"NEst banco C=4": 60}.get(k, ri[0]["mults"])
        rows.append(dict(k=k, lab=lab, ind=np.mean([r["in-dist"] for r in ri]), ood=np.mean([r["OOD x2"] for r in ri]),
                         params=ri[0]["params"], mults=mults, lat=LAT[k], col=fam(k),
                         tconv=np.median([conv_time(r["curve"]) for r in c6]) if c6 else None,
                         ttot=np.median([r["curve"][-1][1] for r in c6]) if c6 else None, curves=[r["curve"] for r in c6]))
    fig, axs = plt.subplots(2, 3, figsize=(12, 7.6)); fig.subplots_adjust(top=0.91, hspace=0.62, wspace=0.7, left=0.13, right=0.98, bottom=0.08)
    labs = [r["lab"] for r in rows]; cols = [r["col"] for r in rows]
    dumbbell(axs[0, 0], labs, [r["ind"] for r in rows], [r["ood"] for r in rows], cols, "training conditions",
             "impulses ×2 (unseen)", "residual error NMSE (dB)")
    axs[0, 0].set_title("(a) Residual error")
    nz = [r for r in rows if r["params"] > 0]
    hbar(axs[0, 1], [r["lab"] for r in nz], [r["params"] for r in nz], [r["col"] for r in nz], log=True, fmt="%d", xlabel="parameters (log)")
    axs[0, 1].set_title("(b) Model size")
    hbar(axs[0, 2], [r["lab"] for r in nz], [max(r["mults"], 0.5) for r in nz], [r["col"] for r in nz], log=True, fmt="%d",
         xlabel="multiplications per output sample (log)"); axs[0, 2].set_title("(c) Arithmetic cost")
    hbar(axs[1, 0], labs, [r["lat"] for r in rows], cols, log=True, fmt="auto", xlabel="CPU latency, ms per 4096 samples (log)")
    axs[1, 0].set_title("(d) Latency (PyTorch, 1 thread)")
    ax = axs[1, 1]
    SHOW = {"WOS-R cascade": "-", "WOS-R bank C=8": "--", "NEst bank C=4 (1993)": ":", "CNN medium": "-", "CNN large": "--", "Transformer": ":"}
    for r in nz:
        if r["lab"] not in SHOW: continue
        cs = r["curves"]; t = np.median([[c[1] for c in cu] for cu in cs], 0); e = np.mean([[c[2] for c in cu] for cu in cs], 0)
        ax.plot(np.maximum(t, 0.05), e, color=r["col"], lw=1.6, ls=SHOW[r["lab"]], label=r["lab"].replace(" (1993)", ""))
    ax.set_xscale("log"); ax.set_ylim(-23, -5); ax.set_xlim(0.05, 400); ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=6.5, loc="upper right"); ax.text(0.06, -18.3, "WOS-R starts as a median filter", fontsize=6.5, color=MUTED)
    ax.set_xlabel("training time (s, log)"); ax.set_ylabel("deployed NMSE (dB, mean of 3 runs)"); ax.set_title("(e) Convergence")
    tc = [r for r in nz if r["tconv"] is not None]
    hbar(axs[1, 2], [r["lab"] for r in tc], [r["tconv"] for r in tc], [r["col"] for r in tc], log=True, fmt="%.0f s",
         xlabel="time to reach final error ±0.5 dB (s, log)"); axs[1, 2].set_title("(f) Time to converge")
    family_legend(fig); fig.savefig(FG("bench_1d.pdf")); plt.close(fig)
    return {r["lab"]: dict(ind=round(r["ind"], 2), ood=round(r["ood"], 2), params=r["params"], mults=r["mults"],
                           lat=round(r["lat"], 2), tconv=None if r["tconv"] is None else round(float(r["tconv"]), 1),
                           ttot=None if r["ttot"] is None else round(float(r["ttot"]), 1)) for r in rows}

# ------------------------------------------------------------------------------------------------ images (B3)
M3 = [("Mediana 3×3", "Median 3×3"), ("Mediana adaptativa (7)", "Adaptive median (AMF)"),
      ("WOS-R 3×3 ×2", "WOS-R cascade 3×3"), ("WOS-R 5×5 ×2", "WOS-R cascade 5×5"), ("WOS-R banco C=4 (5×5)", "WOS-R bank 5×5"),
      ("WOS-R conmutado 5×5 ×2", "Switching WOS-R 5×5"), ("MLP 5×5", "MLP"), ("DnCNN-S (6×16)", "DnCNN small"),
      ("DnCNN-M (10×32)", "DnCNN medium"), ("DnCNN-17 BN (17×64)", "DnCNN-17"), ("Transformer ventana 8×8", "Window Transformer")]

def fig_img():
    R3, LAT = L("b3_results.json"), L("latency.json")["2d"]
    rows = []
    for k, lab in M3:
        r = next(r for r in R3 if r["model"] == k and r["regime"] == "imp")
        rows.append(dict(lab=lab, sp=r["metrics"]["BSD68 | SP 50%"]["psnr"], rv=r["metrics"]["BSD68 | RVIN 40% (OOD)"]["psnr"],
                         params=r["params"], lat=LAT[k], ts=r["train_s"], col=fam(k)))
    fig, axs = plt.subplots(1, 4, figsize=(13.5, 4.4)); fig.subplots_adjust(top=0.84, wspace=0.8, left=0.115, right=0.99, bottom=0.3)
    labs = [r["lab"] for r in rows]; cols = [r["col"] for r in rows]
    dumbbell(axs[0], labs, [r["sp"] for r in rows], [r["rv"] for r in rows], cols, "salt & pepper 50 %",
             "random impulses 40 % (unseen)", "PSNR (dB)", better="higher"); axs[0].set_title("(a) Image quality, BSD68")
    nz = [r for r in rows if r["params"] > 0]
    hbar(axs[1], [r["lab"] for r in nz], [r["params"] for r in nz], [r["col"] for r in nz], log=True, fmt="%d", xlabel="parameters (log)")
    axs[1].set_title("(b) Model size")
    hbar(axs[2], labs, [r["lat"] for r in rows], cols, log=True, fmt="%.0f", xlabel="ms per 512×512 image (log)")
    axs[2].set_title("(c) Latency (PyTorch, 1 thread)")
    hbar(axs[3], [r["lab"] for r in nz], [r["ts"] for r in nz], [r["col"] for r in nz], log=True, fmt="%.0f s",
         xlabel="training time, 2000–3000 steps (s, log)"); axs[3].set_title("(d) Training time")
    family_legend(fig); fig.savefig(FG("bench_img.pdf")); plt.close(fig)
    return {r["lab"]: dict(sp=round(r["sp"], 2), rv=round(r["rv"], 2), params=r["params"], lat=round(r["lat"]), ts=r["ts"]) for r in rows}

# ------------------------------------------------------------------------------------------------ accuracy (B4 + B5)
def fig_acc():
    R4 = L("b4_results.json")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.7)); fig.subplots_adjust(top=0.84, wspace=0.55, left=0.07, right=0.98, bottom=0.15)
    ax = axs[0]; out = {}
    for k, lab, c, mk in [("OS (k=24)", "OS-CFAR (classical)", CLS, "s"), ("WOS-R (32 repeticiones)", "WOS-R (34 par.)", RANK, "o"),
                          ("MLP equivariante", "MLP (6 337 par.)", NET, "^"), ("Transformer equivariante", "Transformer (5 009 par.)", NET, "v")]:
        keys = [kk for kk in R4 if kk.startswith("rho=0.9 | " + k)]
        v = [np.mean([min(R4[kk]["snr50"][n], 31.0) for kk in keys]) for n in ("0", "2", "4", "8")]
        ls = "--" if "Transformer" in k else "-"
        ax.plot([0, 2, 4, 8], v, ls, marker=mk, color=c, lw=1.8, ms=6, mec="white", label=lab); out[lab] = v
    ax.axhline(30, color=MUTED, lw=0.6, ls=":"); ax.text(8.1, 30.3, "never detects (>30 dB)", fontsize=6.5, color=MUTED, ha="right", va="bottom")
    ax.set_xticks([0, 2, 4, 8]); ax.set_xlabel("interfering targets in the reference cells")
    ax.set_ylabel("SNR needed for 50 % detection (dB)  (← better)"); ax.grid(color=GRID, lw=0.6); ax.set_ylim(13, 32)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(0.3, 0.6)); ax.set_title("(a) Radar detection accuracy (correlated clutter)")
    R5 = L("../fl_results.json") + L("../fl_results_extra.json") + L("b5_results.json")
    A = [("mean", "Mean", CLS), ("CW-median", "Coordinate median", CLS), ("CClip", "Centred clipping", CLS),
         ("MLP agregador", "Learned aggregator (MLP)", NET), ("Med_t3→NNM→CW-trim(f)", "Rank cascade", RANK),
         ("Med_t3→NNM→MLP", "Rank cascade + MLP", RANK)]
    ax = axs[1]; y = np.arange(len(A))[::-1]
    for al, off, alpha in [(1.0, 0.18, 1.0), (0.1, -0.18, 0.55)]:
        w = []
        for a, lab, c in A:
            accs = {}
            for r in R5:
                if r["agg"] == a and r["alpha"] == al:
                    accs.setdefault(r["att"], []).append(r["acc"] if np.isfinite(r["acc"]) else 0.1)
            w.append(min(np.mean(v) for v in accs.values()))
        ax.barh(y + off, w, height=0.34, color=[c for _, _, c in A], alpha=alpha, edgecolor="white")
        for yi, v in zip(y + off, w): ax.text(v + 0.01, yi, "%.2f" % v, va="center", fontsize=6.5, color=MUTED)
        out["FL worst alpha=%s" % al] = dict(zip([l for _, l, _ in A], [round(x, 3) for x in w]))
    ax.set_yticks(y); ax.set_yticklabels([l for _, l, _ in A]); ax.set_xlim(0, 0.95); ax.axvline(0.1, color=MUTED, lw=0.6, ls=":")
    ax.text(0.105, y[-1] - 0.55, "chance", fontsize=6.5, color=MUTED)
    ax.set_xlabel("worst-case test accuracy over 7 attacks  (better →)"); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(handles=[matplotlib.patches.Patch(color=MUTED, alpha=1.0, label="similar data on all devices"),
                       matplotlib.patches.Patch(color=MUTED, alpha=0.55, label="very different data per device")], fontsize=7, loc="upper right")
    ax.set_title("(b) Federated learning with 4 of 20 faulty devices")
    family_legend(fig); fig.savefig(FG("bench_acc.pdf")); plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ structure (B7, B8)
def fig_struct():
    R = L("b7_results.json"); G = [r for r in R if r["kind"] == "grid"]
    Ks, ms = (1, 2, 3, 4), (3, 5, 7, 9)
    M = np.array([[np.mean([r["nmse"] for r in G if r["K"] == K and r["m"] == m]) for m in ms] for K in Ks])
    fig, ax = plt.subplots(figsize=(4.0, 3.0)); fig.subplots_adjust(left=0.2, bottom=0.18, right=0.86, top=0.86)
    im = ax.imshow(-M, cmap="Blues", vmin=5, vmax=21, aspect="auto", origin="lower")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, "%.1f" % M[i, j], ha="center", va="center", fontsize=8, color="white" if -M[i, j] > 15 else INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(ms); ax.set_yticks(range(4)); ax.set_yticklabels(Ks)
    ax.set_xlabel("window size (taps)"); ax.set_ylabel("number of stages")
    cb = fig.colorbar(im, ax=ax, fraction=0.05); cb.set_label("−NMSE (dB), darker = better", fontsize=7)
    ax.set_title("WOS-R cascade, 1D impulses (mean of 2 runs)", fontsize=8)
    fig.savefig(FG("struct_grid.pdf")); plt.close(fig)
    return M.round(2).tolist()

if __name__ == "__main__":
    import sys
    for w in sys.argv[1:] or ["img", "acc"]:
        print(w, json.dumps(globals()["fig_" + w](), indent=0)[:1500])
