import json
P=json.load(open('ramp_props.json'))
rows=[("Exact WOS ($\\beta=0$)","WOS (seleccion)"),("Ramp, $\\beta=1$","rampa continua beta=1"),("Ramp, $\\beta=0.5$","rampa continua beta=0.5"),("Ramp, integer version","rampa discreta (enteros)")]
jumps={"WOS (seleccion)":"salto_max_en_r (WOS, dr=5e-3)","rampa continua beta=1":"salto_max_en_r (rampa beta=1, dr=5e-3)","rampa continua beta=0.5":"salto_max_en_r (rampa beta=0.5, dr=5e-3)"}
bkey={"WOS (seleccion)":"mediana","rampa continua beta=1":"rampa_continua","rampa continua beta=0.5":None,"rampa discreta (enteros)":"rampa_discreta"}
bd={k:P["ruptura mediana5, %d impulsos"%k] for k in [2,3]}
def fmt(v):
    if v<1e4: return "%.1f"%v
    e=len(str(int(v)))-1; return "$%.1f\\cdot10^{%d}$"%(v/10**e,e)
NO="\\textbf{no}"
L=[r"\begin{table}[t]\centering\small",
   r"\caption{Measured properties with non-uniform weights $w=(1,2,3,1,0.5,2,1)$, $r=0.4$, on $2\cdot10^4$ random windows of 7 samples. ``Jump in $r$'': largest output change when $r$ moves by $5\cdot10^{-3}$. Last two columns: largest output magnitude of the 5-point (ramp) median with uniform weights under $k$ impulses of $10^6$.}\label{tab:props}",
   r"\begin{tabular}{lcccccc}\toprule",
   r"Operator & Increasing & $\Psi(z{+}c){=}\Psi(z){+}c$ & Lipschitz $\ell_\infty$ & Jump in $r$ & 2 impulses & 3 impulses\\\midrule"]
for lab,k in rows:
    p=P[k]; j=("%.2f"%P[jumps[k]]) if k in jumps else "--"
    b=bkey[k]; b2=fmt(bd[2][b]) if b else "--"; b3=fmt(bd[3][b]) if b else "--"
    inc="yes" if p["creciente"] else NO
    L.append("%s & %s & yes & %.2f & %s & %s & %s\\\\"%(lab,inc,p["lipschitz_inf"],j,b2,b3))
L.append(r"\bottomrule\end{tabular}\end{table}")
open('paper/tables/props.tex','w').write("\n".join(L)); print("\n".join(L))
