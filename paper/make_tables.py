"""Generate all paper tables/figures from benchmark JSON files (no hand transcription)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _ROOT = _os.path.dirname(_HERE)
_CODE = _os.path.join(_ROOT, "code")
import json, os, math, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
B = _os.path.join(_CODE, "bench"); P = _HERE
T = lambda n: os.path.join(P, "tables", n); FG = lambda n: os.path.join(P, "figs", n)
EN = {"Mediana 5": "Median 5", "WOS-R cadena (2×7)": "WOS-R chain $2{\\times}7$", "WOS-R banco C=4": "WOS-R bank $C{=}4$",
      "WOS-R banco C=8": "WOS-R bank $C{=}8$", "WOS-R banco C=4 [solo rampa]": "\\quad ablation: bank $C{=}4$, ramp only", "WOS-R banco C=8 [solo rampa]": "\\quad ablation: bank $C{=}8$, ramp only", "NEst banco C=4": "NEst bank $C{=}4$", "CNN-S": "CNN-S", "CNN-M": "CNN-M",
      "CNN-L": "CNN-L", "MLP (ventana 15)": "MLP (window 15)", "Transformer": "Transformer"}
LAT = json.load(open(os.path.join(B, "latency.json"))) if os.path.exists(os.path.join(B, "latency.json")) else {}
def ms(v): return "$%.1f\\pm%.1f$" % (np.mean(v), np.std(v)) if len(v) > 1 else "$%.1f$" % v[0]
def fmt_int(n): return "{:,}".format(int(n)).replace(",", "\\,")

def table_b1():
    R = json.load(open(os.path.join(B, "b1_results.json")))
    models = list(EN)
    cols_imp = ["in-dist", "OOD p=25%", "OOD x2"]; cols_g = ["gauss s=0.2", "gauss s=0.4 (OOD)"]
    best = {}
    for c in cols_imp + cols_g:
        kind = "imp" if c in cols_imp else "gauss"
        best[c] = min(np.mean([r[c] for r in R if r["model"] == m and r["kind"] == kind]) for m in models
                      if any(r["model"] == m and r["kind"] == kind for r in R))
    L = [r"\begin{table*}[t]\centering\small",
         r"\caption{B1: 1D restoration, normalised mean squared error (NMSE) in dB (mean $\pm$ std over 3 seeds; lower is better). Impulsive models are trained with 10\,\% impulses; OOD (out-of-distribution) columns change impulse rate or amplitude. NEst: 1993 thesis structure; CNN-S/M/L: small/medium/large residual CNN. Gaussian models are trained with $\sigma=0.2$. Params, multiply--accumulates and comparisons per output sample; ms: CPU latency for 4096 samples (1 thread). Best per column in bold.}\label{tab:b1}",
         r"\begin{tabular}{lrrrr|ccc|cc}\toprule",
         r"& & & & & \multicolumn{3}{c|}{Impulsive} & \multicolumn{2}{c}{Gaussian}\\",
         r"Model & Params & Mults & Comps & ms & in-dist & OOD $p{=}25\%$ & OOD $\times2$ & $\sigma{=}0.2$ & $\sigma{=}0.4$ (OOD)\\\midrule"]
    summ = {}
    for m in models:
        ri = [r for r in R if r["model"] == m and r["kind"] == "imp"]; rg = [r for r in R if r["model"] == m and r["kind"] == "gauss"]
        if not ri: continue
        cells = []
        for c, rr in [(c, ri) for c in cols_imp] + [(c, rg) for c in cols_g]:
            v = [r[c] for r in rr]
            s = ms(v) if v else "--"
            if v and abs(np.mean(v) - best[c]) < 0.05: s = s.replace("$", "$\\mathbf{", 1)[:-1] + "}$"
            cells.append(s)
            summ.setdefault(m, {})[c] = float(np.mean(v)) if v else None
        if m.startswith("NEst"):          # NEst stage y = sum_k |h_k| (z+b)_(k): 7 mults + sort per stage; 4 chains x 2 stages + mix
            for r in ri: r["mults"], r["compares"] = 4 * 2 * 7 + 4, 4 * 2 * 7 * 3
        summ[m]["params"] = ri[0]["params"]; summ[m]["mults"] = ri[0]["mults"]; summ[m]["comps"] = ri[0]["compares"]
        lat = LAT.get("1d", {}).get(m.replace(" [solo rampa]", ""))
        summ[m]["ms"] = lat
        L.append("%s & %s & %s & %s & %s & %s\\\\" % (EN[m], fmt_int(ri[0]["params"]), fmt_int(ri[0]["mults"]), fmt_int(ri[0]["compares"]), "--" if lat is None else "%.1f" % lat, " & ".join(cells)))
    L.append(r"\bottomrule\end{tabular}\end{table*}")
    open(T("b1.tex"), "w").write("\n".join(L)); json.dump(summ, open(T("b1_summary.json"), "w"), indent=1)
    return summ


EN3 = {"Mediana 3×3": "Median $3{\\times}3$", "Mediana 5×5": "Median $5{\\times}5$", "Mediana adaptativa (7)": "Adaptive median (AMF, 7)",
       "WOS-R 3×3 ×2": "WOS-R $3{\\times}3$, 2 stages", "WOS-R 5×5 ×2": "WOS-R $5{\\times}5$, 2 stages",
       "WOS-R banco C=4 (3×3)": "WOS-R bank $C{=}4$ ($3{\\times}3$)", "WOS-R banco C=4 (5×5)": "WOS-R bank $C{=}4$ ($5{\\times}5$)",
       "WOS-R conmutado 3×3 ×2": "Switching WOS-R $3{\\times}3$", "WOS-R conmutado 5×5 ×2": "Switching WOS-R $5{\\times}5$",
       "WOS-R conmutado banco C=4 (3×3)": "Switching WOS-R bank $C{=}4$ ($3{\\times}3$)",
       "WOS-R banco C=4 (3×3) [solo rampa]": "\\quad ablation: bank $C{=}4$ ($3{\\times}3$), ramp only", "WOS-R banco C=4 (5×5) [solo rampa]": "\\quad ablation: bank $C{=}4$ ($5{\\times}5$), ramp only",
       "MLP 5×5": "MLP $5{\\times}5$", "DnCNN-S (6×16)": "DnCNN-S (6 layers, 16 ch)", "DnCNN-M (10×32)": "DnCNN-M (10 layers, 32 ch)",
       "DnCNN-17 BN (17×64)": "DnCNN-17 (BN, 64 ch)", "Transformer ventana 8×8": "Window Transformer"}
ORDER3 = ['Mediana 3×3', 'Mediana 5×5', 'Mediana adaptativa (7)', 'WOS-R 3×3 ×2', 'WOS-R 5×5 ×2', 'WOS-R banco C=4 (3×3)', 'WOS-R banco C=4 (5×5)', 'WOS-R banco C=4 (3×3) [solo rampa]', 'WOS-R banco C=4 (5×5) [solo rampa]', 'WOS-R conmutado 3×3 ×2', 'WOS-R conmutado 5×5 ×2', 'WOS-R conmutado banco C=4 (3×3)', 'MLP 5×5', 'DnCNN-S (6×16)', 'DnCNN-M (10×32)', 'DnCNN-17 BN (17×64)', 'Transformer ventana 8×8']
def table_b3():
    R = json.load(open(os.path.join(B, "b3_results.json")))
    def get(reg, m): return next((r for r in R if r["regime"] == reg and r["model"] == m), None)
    conds = ["SP 10%", "SP 30%", "SP 50%", "SP 70% (OOD)", "RVIN 20% (OOD)", "RVIN 40% (OOD)"]
    out = {}
    for sname in ["BSD68", "Set12"]:
        rows = [(m, get("imp", m)) for m in ORDER3 if get("imp", m)]
        best = {c: max(r["metrics"][f"{sname} | {c}"]["psnr"] for _, r in rows) for c in conds}
        L = [r"\begin{table*}[t]\centering\small",
             r"\caption{B3: blind impulse denoising on %s, peak signal-to-noise ratio (PSNR, dB; higher is better) / structural similarity (SSIM). Training: salt-and-pepper (SP) with density $\sim U[0.1,0.6]$. Out of distribution (OOD): SP 70\,\%%, random-valued impulse noise (RVIN). AMF: adaptive median filter; switching: WOS-R applied only to pixels that are an extreme of their $3{\times}3$ window. Mults and comparisons per pixel; latency in ms for a $512{\times}512$ image (1 CPU thread). Best per column in bold.}\label{tab:b3%s}" % (sname, sname.lower()),
             r"\begin{tabular}{lrrrr|cccc|cc}\toprule",
             r"Model & Params & Mults & Comps & ms & SP 10\% & SP 30\% & SP 50\% & SP 70\% & RVIN 20\% & RVIN 40\%\\\midrule"]
        for m, r in rows:
            cells = []
            for c in conds:
                v = r["metrics"][f"{sname} | {c}"]; s = "%.2f/%.3f" % (v["psnr"], v["ssim"])
                if abs(v["psnr"] - best[c]) < 0.01: s = "\\textbf{%.2f}/%.3f" % (v["psnr"], v["ssim"])
                cells.append(s)
                out.setdefault(sname, {}).setdefault(m, {})[c] = v
            out[sname][m]["params"] = r["params"]
            CMP = {"Mediana 3×3": "36", "Mediana 5×5": "125", "Mediana adaptativa (7)": "var."}       # m*ceil(log2 m); AMF adapts its window
            L.append("%s & %s & %s & %s & %.0f & %s\\\\" % (EN3[m], fmt_int(r["params"]), fmt_int(r["mults_per_px"]), CMP.get(m, fmt_int(r["compares_per_px"])),
                                                          LAT.get("2d", {}).get(m.replace(" [solo rampa]", ""), r["latency_ms_512"]), " & ".join(cells)))
        L.append(r"\bottomrule\end{tabular}\end{table*}")
        open(T("b3_%s.tex" % sname.lower()), "w").write("\n".join(L).replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}").replace("\\end{tabular}", "\\end{tabular}}"))
    # gaussian
    rows = [(m, get("gauss", m)) for m in ORDER3 if get("gauss", m) and "solo rampa" not in m]
    gc = ["Gauss σ=25", "Gauss σ=50 (OOD)"]
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{B3 coverage check: Gaussian denoising (trained at $\sigma=25$), PSNR in dB, Set12 / BSD68. Order statistics are not the right tool here (Prop.~\ref{prop:lip}: no averaging).}\label{tab:b3g}",
         r"\begin{tabular}{lrcc}\toprule Model & Params & $\sigma=25$ & $\sigma=50$ (OOD)\\\midrule"]
    for m, r in rows:
        L.append("%s & %s & %s\\\\" % (EN3[m], fmt_int(r["params"]), " & ".join("%.2f / %.2f" % (r["metrics"][f"Set12 | {c}"]["psnr"], r["metrics"][f"BSD68 | {c}"]["psnr"]) for c in gc)))
        out.setdefault("gauss", {})[m] = {c: (r["metrics"][f"Set12 | {c}"]["psnr"], r["metrics"][f"BSD68 | {c}"]["psnr"]) for c in gc}
    L.append(r"\bottomrule\end{tabular}\end{table}")
    open(T("b3_gauss.tex"), "w").write("\n".join(L))
    json.dump(out, open(T("b3_summary.json"), "w"), indent=1)
    return {k: list(v)[:3] for k, v in out.items()}


# ---------------------------------------------------------------- B2 coverage
EN2T = {"Mediana 7": ("Median 7", "rank"), "Erosión plana 5": ("Flat erosion 5", "morphological"),
        "Apertura 5 (erosión→dilatación)": ("Opening 5", "morph., 2 stages"),
        "WOS ponderado (1,3,0,2,4,1,2)": ("WOS $(1,3,0,2,4,1,2)$", "WOS"),
        "Cascada E2 t1": ("Rank cascade \\#1", "rank cascade"), "Cascada E2 t12": ("Rank cascade \\#12", "rank cascade"),
        "Cascada E2 t5": ("Rank cascade \\#5", "rank cascade"), "Media móvil 5": ("Moving average 5", "linear, $h\\ge0$"),
        "Paso alto [-1,2,-1]/2": ("High-pass $[-1,2,-1]/2$", "linear, signed"),
        "Top-hat x − apertura": ("Top-hat $x-$opening", "non-increasing"),
        "No linealidad puntual x²/5": ("Point-wise $x^2/5$", "not translation-inv.")}
EN2M = {"WOS-R cadena": "WOS-R chain", "WOS-R banco C=4": "WOS-R bank 4", "WOS-R banco C=4 [solo rampa]": "bank 4, ramp only", "NEst banco C=4": "NEst bank 4",
        "MLP": "MLP", "CNN-M": "CNN-M", "Transformer": "Transformer"}
def table_b2():
    R = json.load(open(os.path.join(B, "b2_results.json")))
    tg = [t for t in EN2T if any(r["target"] == t for r in R)]; md = [m for m in EN2M if any(r["model"] == m for r in R)]
    G = np.full((len(tg), len(md)), np.nan); GR = np.full_like(G, np.nan); PR = {}
    for i, t in enumerate(tg):
        for j, m in enumerate(md):
            v = [r["gap"] for r in R if r["target"] == t and r["model"] == m]
            if v: G[i, j] = np.mean(v)
            vr = [r["gap_ramp_deploy"] for r in R if r["target"] == t and r["model"] == m and "gap_ramp_deploy" in r]
            if vr: GR[i, j] = np.mean(vr)
            pp = [r["params"] for r in R if r["model"] == m]
            if pp: PR[m] = pp[0]
    def cell(v):
        if np.isnan(v): return "--"
        s = "%.1f" % v
        return "\\textbf{%s}" % s if v <= 1.0 else (s if v <= 6.0 else "\\textcolor{red!70!black}{%s}" % s)
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{B2 coverage: excess normalised mean squared error (NMSE, dB) above the noise floor of the true operator (input $U[-5,5]$, SNR 35\,dB, mean of 2 seeds). Bold: $\le1$\,dB (operator represented); red: $>6$\,dB (not represented). Parameters in the last row.}\label{tab:b2}",
         r"\setlength{\tabcolsep}{4pt}\begin{tabular}{ll" + "r" * len(md) + r"}\toprule",
         "Target & Class & " + " & ".join(EN2M[m] for m in md) + r"\\\midrule"]
    for i, t in enumerate(tg):
        L.append("%s & %s & %s\\\\" % (EN2T[t][0], EN2T[t][1], " & ".join(cell(v) for v in G[i])))
    L.append(r"\midrule Params & & " + " & ".join(fmt_int(PR[m]) for m in md) + r"\\")
    L.append(r"\bottomrule\end{tabular}\end{table}")
    open(T("b2.tex"), "w").write("\n".join(L))
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    im = ax.imshow(np.clip(G, 0, 20), cmap="RdYlGn_r", vmin=0, vmax=20, aspect="auto")
    ax.set_xticks(range(len(md))); ax.set_xticklabels([EN2M[m] for m in md], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(tg))); ax.set_yticklabels([EN2T[t][0].replace("\\#", "#").replace("$", "").replace("\\ge", ">=") for t in tg], fontsize=8)
    for i in range(len(tg)):
        for j in range(len(md)):
            if not np.isnan(G[i, j]): ax.text(j, i, "%.1f" % G[i, j], ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="excess NMSE (dB)"); fig.tight_layout(); fig.savefig(FG("b2_coverage.pdf")); plt.close(fig)
    out = {"gap": {t: {m: (None if np.isnan(G[i, j]) else float(G[i, j])) for j, m in enumerate(md)} for i, t in enumerate(tg)},
           "gap_ramp_deploy": {t: {m: (None if np.isnan(GR[i, j]) else float(GR[i, j])) for j, m in enumerate(md)} for i, t in enumerate(tg)},
           "params": PR}
    json.dump(out, open(T("b2_summary.json"), "w"), indent=1)
    return out["gap"]

# ---------------------------------------------------------------- B4 CFAR
def table_b4():
    R = json.load(open(os.path.join(B, "b4_results.json")))
    EN4 = {"OS (k=24)": "OS-CFAR ($k{=}24$)", "GO-OS (k=11)": "GO-OS ($k{=}11$)", "WOS-R (32 repeticiones)": "WOS-R (32 repetitions)",
           "MLP equivariante": "MLP (log-equivariant)", "Transformer equivariante": "Transformer (log-equivariant)"}
    out = {}
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{B4: constant-false-alarm-rate (CFAR) detection in K-distributed clutter with range-correlated texture (AR(1) coefficient $\rho$), false-alarm probability $P_{fa}=10^{-4}$ calibrated per detector. Signal-to-noise ratio (SNR, dB) for detection probability $P_d=0.5$ with 0/2/4/8 interfering targets in the reference window (lower is better); learned detectors: mean of 2 seeds, trained with the same Neyman--Pearson objective.}\label{tab:b4}",
         r"\setlength{\tabcolsep}{4pt}\begin{tabular}{lr|cccc|cccc}\toprule",
         r"& & \multicolumn{4}{c|}{$\rho=0$} & \multicolumn{4}{c}{$\rho=0.9$}\\",
         r"Detector & Params & 0 & 2 & 4 & 8 & 0 & 2 & 4 & 8\\\midrule"]
    for k, nm in EN4.items():
        row = []
        for rho in ("0.0", "0.9"):
            keys = [kk for kk in R if kk.startswith(f"rho={rho} | {k}")]
            if not keys: row += ["--"] * 4; continue
            for ni in ("0", "2", "4", "8"):
                v = [R[kk]["snr50"][ni] for kk in keys]
                vv = [x for x in v if x is not None and np.isfinite(x)]
                row.append("%.2f" % np.mean(vv) if len(vv) == len(v) else ("$>30$" if not vv else "%.2f$^*$" % np.mean(vv)))
                out.setdefault(k, {})[f"rho={rho} ni={ni}"] = v
        p = R.get(f"params | {k}", 0)
        out.setdefault(k, {})["params"] = p
        L.append("%s & %s & %s\\\\" % (nm, fmt_int(p), " & ".join(row)))
    # bold best per column
    import re as _re
    rows = L[5:]; cols = [[c.strip() for c in r.rstrip("\\").split("&")] for r in rows]
    for j in range(2, 10):
        vals = [float(c[j]) if _re.fullmatch(r"[0-9.]+", c[j]) else 1e9 for c in cols]; b_ = min(vals)
        for i, c in enumerate(cols):
            if abs(vals[i] - b_) < 0.005: c[j] = "\\textbf{%s}" % c[j]
    L = L[:5] + [" & ".join(c) + "\\\\" for c in cols]
    L.append(r"\bottomrule\end{tabular}\end{table}")
    open(T("b4.tex"), "w").write("\n".join(L)); json.dump(out, open(T("b4_summary.json"), "w"), indent=1)
    return out

# ---------------------------------------------------------------- B5 FL
EN5 = {"mean": ("Mean", "0"), "CW-median": ("CW median", "0"), "CW-trim(f)": ("CW trimmed mean", "0"),
       "NNM+CW-trim(f)": ("NNM + CW trim", "0"), "CClip": ("Centred clipping", "0"), "AMO": ("AMO (minimax OWA)", "0$^\\dagger$"),
       "Med_t3→AMO": ("Med$_t$3 $\\to$ AMO", "0$^\\dagger$"), "Med_t3→NNM→CW-trim(f)": ("Med$_t$3 $\\to$ NNM $\\to$ CW trim", "0"),
       "MLP agregador": ("Sorted-MLP aggregator", None), "Med_t3→NNM→MLP": ("Med$_t$3 $\\to$ NNM $\\to$ sorted-MLP", None)}
ATT = ["none", "signflip", "alie", "ipm", "burst8", "sdc", "gauss+burst8"]
ATTEN = ["none", "sign-flip", "ALIE", "IPM", "burst-8", "SDC", "Gauss+burst"]
def table_b5():
    R = json.load(open(_os.path.join(_CODE, "fl_results.json"))) + json.load(open(_os.path.join(_CODE, "fl_results_extra.json")))
    if os.path.exists(os.path.join(B, "b5_results.json")): R += json.load(open(os.path.join(B, "b5_results.json")))
    def acc(a, att, al):
        v = [r["acc"] for r in R if r["agg"] == a and r["att"] == att and r["alpha"] == al]
        v = [0.1 if not np.isfinite(x) else x for x in v]
        return float(np.mean(v)) if v else None
    out = {}
    for a in EN5:
        for al in (1.0, 0.1):
            row = {att: acc(a, att, al) for att in ATT}
            if None in row.values(): continue
            out.setdefault(a, {})[str(al)] = dict(row=row, worst=min(row.values()), mean=float(np.mean(list(row.values()))))
    L = [r"\begin{table*}[t]\centering\small",
         r"\caption{B5: Byzantine/fault-robust aggregation (Fashion-MNIST, $n{=}20$ nodes, $f{=}4$ faulty, 300 steps, mean of 2 seeds). Test accuracy under each attack with heterogeneous data (Dirichlet $\alpha{=}0.1$), and worst case over attacks for $\alpha\in\{1,0.1\}$. CW: coordinate-wise; Med$_t$3: per-node temporal median of the last 3 submissions; NNM: nearest-neighbour mixing \cite{allouah2023}; AMO: adaptive minimax ordered weighted average (OWA). Attacks: ALIE (``a little is enough''), IPM (inner-product manipulation), burst-8 (faults persisting 8 steps), SDC (silent data corruption). $^\dagger$AMO uses a precomputed library of 5 OWA weight vectors (minimax second-order cone program), no training.}\label{tab:b5}",
         r"\begin{tabular}{lr|ccccccc|cc}\toprule",
         r"& & \multicolumn{7}{c|}{$\alpha=0.1$} & \multicolumn{2}{c}{worst case}\\",
         r"Aggregator & Params & " + " & ".join(ATTEN) + r" & $\alpha{=}1$ & $\alpha{=}0.1$\\\midrule"]
    bw = {al: max(out[a][al]["worst"] for a in out if al in out[a]) for al in ("1.0", "0.1")}
    pm = json.load(open(os.path.join(B, "b5_params.json"))) if os.path.exists(os.path.join(B, "b5_params.json")) else {}
    for a, (nm, p) in EN5.items():
        if a not in out or "0.1" not in out[a]: continue
        r = out[a]["0.1"]["row"]
        ws = []
        for al in ("1.0", "0.1"):
            w = out[a].get(al, {}).get("worst")
            ws.append("--" if w is None else ("\\textbf{%.3f}" % w if abs(w - bw[al]) < 1e-3 else "%.3f" % w))
        L.append("%s & %s & %s & %s\\\\" % (nm, p if p is not None else fmt_int(pm.get("mlp", 5569)), " & ".join("%.3f" % r[att] for att in ATT), " & ".join(ws)))
    L.append(r"\bottomrule\end{tabular}\end{table*}")
    open(T("b5.tex"), "w").write("\n".join(L)); json.dump(out, open(T("b5_summary.json"), "w"), indent=1)
    return {a: {al: round(v["worst"], 3) for al, v in d.items()} for a, d in out.items()}

# ---------------------------------------------------------------- Pareto figures
def fig_pareto():
    S1 = {"WOS-R cadena (2×7)": "WOS-R chain", "WOS-R banco C=4": "WOS-R bank 4", "WOS-R banco C=8": "WOS-R bank 8", "NEst banco C=4": "NEst bank 4",
          "CNN-S": "CNN-S", "CNN-M": "CNN-M", "CNN-L": "CNN-L", "MLP (ventana 15)": "MLP", "Transformer": "Transformer"}
    S3 = {"WOS-R 3×3 ×2": "WOS-R 3x3", "WOS-R 5×5 ×2": "WOS-R 5x5", "WOS-R banco C=4 (5×5)": "WOS-R bank 5x5",
          "WOS-R conmutado 3×3 ×2": "Sw. WOS-R 3x3", "WOS-R conmutado 5×5 ×2": "Sw. WOS-R 5x5", "MLP 5×5": "MLP",
          "DnCNN-S (6×16)": "DnCNN-S", "DnCNN-M (10×32)": "DnCNN-M", "DnCNN-17 BN (17×64)": "DnCNN-17", "Transformer ventana 8×8": "Win. Transformer"}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    R1 = json.load(open(os.path.join(B, "b1_results.json")))
    for m, lab in S1.items():
        ri = [r for r in R1 if r["model"] == m and r["kind"] == "imp"]
        x = ri[0]["params"]; y = np.mean([r["in-dist"] for r in ri]); yo = np.mean([r["OOD x2"] for r in ri])
        c = "C3" if m.startswith(("WOS", "NEst")) else "C0"
        axs[0].plot([x, x], [y, yo], c=c, lw=0.6, alpha=0.5); axs[0].scatter(x, y, c=c, marker="o", s=18); axs[0].scatter(x, yo, c=c, marker="x", s=18)
        off = {"WOS-R bank 4": (-30, 6), "WOS-R bank 8": (-12, 12), "NEst bank 4": (5, 4), "WOS-R chain": (4, 4)}.get(lab, (4, -3))
        axs[0].annotate(lab, (x, y), fontsize=7, xytext=off, textcoords="offset points")
    axs[0].set_xscale("log"); axs[0].set_xlabel("parameters"); axs[0].set_ylabel("NMSE (dB), lower is better"); axs[0].invert_yaxis()
    axs[0].set_title("B1 1D impulsive: o in-distribution, x OOD amplitude x2", fontsize=9)
    R3 = json.load(open(os.path.join(B, "b3_results.json")))
    for m, lab in S3.items():
        r = next(r for r in R3 if r["regime"] == "imp" and r["model"] == m)
        x = r["params"]; y = r["metrics"]["BSD68 | SP 50%"]["psnr"]; yo = r["metrics"]["BSD68 | RVIN 40% (OOD)"]["psnr"]
        c = "C3" if "WOS" in m else "C0"
        axs[1].plot([x, x], [y, yo], c=c, lw=0.6, alpha=0.5); axs[1].scatter(x, y, c=c, marker="o", s=18); axs[1].scatter(x, yo, c=c, marker="x", s=18)
        off = {"WOS-R 5x5": (4, -8), "WOS-R 3x3": (4, -8)}.get(lab, (4, 2))
        axs[1].annotate(lab, (x, y), fontsize=7, xytext=off, textcoords="offset points")
    axs[1].set_xscale("log"); axs[1].set_xlabel("parameters"); axs[1].set_ylabel("PSNR (dB)")
    axs[1].set_title("B3 BSD68: o SP 50%, x RVIN 40% (OOD)", fontsize=9)
    fig.tight_layout(); fig.savefig(FG("pareto.pdf")); plt.close(fig)
    return "ok"

def fig_cfarw():
    import torch, torch.nn.functional as F
    sd = torch.load(os.path.join(B, "b4_models.pt"))
    fig, ax = plt.subplots(figsize=(5.2, 2.4)); out = {}
    for k, v in sd.items():
        if "WOS" not in k: continue
        w = F.softplus(v["alpha"]); w = (w / w.max()).numpy(); rho = k.split(",")[1].strip(); seed = k.split(",")[2].strip(" )")
        ax.plot(range(1, 17), w[:16], "C0-" if rho == "0.0" else "C3-", alpha=0.8, label=("rho=%s" % rho) if seed == "0" else None)
        ax.plot(range(22, 38), w[16:], "C0-" if rho == "0.0" else "C3-", alpha=0.8)
        out[k] = w.tolist()
    ax.axvspan(16.5, 21.5, color="0.85"); ax.text(19, 1.02, "guard+CUT", ha="center", fontsize=7)
    ax.set_xlabel("range cell"); ax.set_ylabel("repetition weight (norm.)"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(FG("cfar_weights.pdf")); plt.close(fig); json.dump(out, open(T("cfarw.json"), "w")); return "ok"


def fit(name):
    """Wrap the tabular of tables/<name> in a \\resizebox to the line width (idempotent)."""
    f = T(name); t = open(f).read()
    if "resizebox" in t: return
    t = t.replace("\\begin{tabular}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}", 1).replace("\\end{tabular}", "\\end{tabular}}", 1)
    open(f, "w").write(t)
def table_fit():
    for n in ["props.tex", "b1.tex", "b2.tex", "b5.tex"]: fit(n)
    return "ok"


def table_cost():
    R1 = json.load(open(os.path.join(B, "b1_results.json"))); R3 = json.load(open(os.path.join(B, "b3_results.json")))
    R4 = json.load(open(os.path.join(B, "b4_results.json")))
    def b1(m, c): rr = [r for r in R1 if r["model"] == m and r["kind"] == "imp"]; return np.mean([r[c] for r in rr]), rr[0]
    def b3(m, c): r = next(r for r in R3 if r["model"] == m and r["regime"] == "imp"); return r["metrics"]["BSD68 | " + c]["psnr"], r
    def b4(k): return np.mean([R4[kk]["snr50"]["8"] for kk in R4 if kk.startswith("rho=0.9 | " + k)])
    rows = []
    for task, w, n, unit in [("B1 impulsive (NMSE dB)", "WOS-R banco C=8", "CNN-L", "in-dist"), ("B1 OOD amplitude (NMSE dB)", "WOS-R banco C=8", "CNN-L", "OOD x2")]:
        a, ra = b1(w, unit); b_, rb = b1(n, unit)
        rows.append((task, EN[w], ra["params"], ra["mults"], ra["compares"], LAT["1d"][w], "%.1f" % a, EN[n], rb["params"], rb["mults"], LAT["1d"][n], "%.1f" % b_))
    for task, w, n, c in [("B3 BSD68 SP 50\\% (PSNR)", "WOS-R conmutado 5×5 ×2", "DnCNN-M (10×32)", "SP 50%"), ("B3 BSD68 RVIN 40\\% (PSNR)", "WOS-R 5×5 ×2", "DnCNN-M (10×32)", "RVIN 40% (OOD)")]:
        a, ra = b3(w, c); b_, rb = b3(n, c)
        rows.append((task, EN3[w], ra["params"], ra["mults_per_px"], ra["compares_per_px"], LAT["2d"][w], "%.2f" % a, EN3[n], rb["params"], rb["mults_per_px"], LAT["2d"][n], "%.2f" % b_))
    mlp_m = 32 * 64 + 64 * 64 + 64
    rows.append(("B4 CFAR, $\\rho{=}0.9$, 8 interf.\\ (SNR$_{50}$ dB)", "WOS-R", R4["params | WOS-R (32 repeticiones)"], 1, 32 * 5, None, "%.2f" % b4("WOS-R (32 repeticiones)"),
                 "MLP", R4["params | MLP equivariante"], mlp_m, None, "%.2f" % b4("MLP equivariante")))
    L = [r"\begin{table*}[t]\centering\small",
         r"\caption{Cost summary: best WOS-R vs best network per task. Mults/Comps per output sample or pixel; ms: CPU latency (1 thread; 4096 samples in 1D, $512{\times}512$ in 2D) with generic PyTorch kernels.}\label{tab:cost}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{l|lrrrrr|lrrrr}\toprule",
         r"Task & WOS-R model & Params & Mults & Comps & ms & Score & Network & Params & Mults & ms & Score\\\midrule"]
    for r in rows:
        L.append("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s\\\\" % (r[0], r[1], fmt_int(r[2]), fmt_int(r[3]), fmt_int(r[4]), "--" if r[5] is None else "%.1f" % r[5], r[6],
                                                                         r[7], fmt_int(r[8]), fmt_int(r[9]), "--" if r[10] is None else "%.1f" % r[10], r[11]))
    L.append(r"\bottomrule\end{tabular}}\end{table*}")
    open(T("cost.tex"), "w").write("\n".join(L)); return rows

if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["b1"]
    for w in which:
        print(w, (globals().get("table_" + w) or globals()["fig_" + w])())
