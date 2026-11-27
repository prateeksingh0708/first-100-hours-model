"""
Robustness of the simulation result to its own settings (one setting changed at a time).
Answers: does the multi-band model still beat the carbonyl index when the chosen kinetic settings are halved or doubled?
usage: python3 robustness.py <sim_out_dir> <fig_dir>
"""
import sys, json
import numpy as np
SIMDIR, FIG = sys.argv[1], sys.argv[2]
src = open(__file__.replace("robustness.py", "sim.py")).read().split("# ------------------------------------------------------------------ main experiment")[0]
G = {"__name__": "simdefs"}; sys.argv = ["sim.py", SIMDIR]; exec(compile(src, "sim.py", "exec"), G)
BASE = dict(kscale=1.0, K_STAB=4.0, TAU_STAB=400.0, A_AUTO=3.0)
CASES = [("Baseline settings", {}), ("Oxidation rate halved", dict(kscale=0.5)), ("Oxidation rate doubled", dict(kscale=2.0)),
         ("No stabiliser effect", dict(K_STAB=0.0)), ("Stabiliser effect halved", dict(K_STAB=2.0)), ("Stabiliser effect doubled", dict(K_STAB=8.0)),
         ("Stabiliser fades in 200 h", dict(TAU_STAB=200.0)), ("Stabiliser fades in 800 h", dict(TAU_STAB=800.0)),
         ("Self-acceleration halved", dict(A_AUTO=1.5)), ("Self-acceleration doubled", dict(A_AUTO=6.0))]
import os
out = json.load(open(f"{SIMDIR}/robustness.json")) if os.path.exists(f"{SIMDIR}/robustness.json") and "--figure-only" in sys.argv else []
for name, ch in ([] if out else CASES):
    s = dict(BASE, **ch); G["RNG"] = np.random.default_rng(20261115); G["K_STAB"], G["TAU_STAB"], G["A_AUTO"] = s["K_STAB"], s["TAU_STAB"], s["A_AUTO"]
    pop = G["make_population"](120); pop["k"] = pop["k"] * s["kscale"]; sim = G["simulate"](pop)
    assert not np.isnan(sim["tf"]).any(); y = np.log10(sim["tf"]); f = G["features"](sim, pop, 100)
    full = G["cv_eval"](np.column_stack([f["F"], f["desc"]]), y, G["enet"])[:2]; ci = G["cv_eval"](np.log10(np.clip(f["ci"], 1e-5, None)), y, G["lin"])[:2]
    q = np.percentile(sim["tf"], [5, 50, 95])
    out.append(dict(case=name, settings=s, model_error_pct=full[0], model_r2=full[1], carbonyl_error_pct=ci[0], carbonyl_r2=ci[1], failure_h_p5_p50_p95=[float(v) for v in q]))
    print(f"{name:28s} model {full[0]:5.1f}%  carbonyl {ci[0]:5.1f}%  tf {q.round(0)}", flush=True)
json.dump(out, open(f"{SIMDIR}/robustness.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
P, GOLD, ORG, DK, GRY = "#502B8B", "#F2A01E", "#C8601A", "#241A3D", "#6B6480"
fam = "TeX Gyre Adventor" if any("Adventor" in f_.name for f_ in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": fam, "font.size": 10.5, "axes.edgecolor": GRY, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 260, "legend.frameon": False})
fig, ax = plt.subplots(figsize=(6.4, 3.35)); yy = np.arange(len(out))[::-1]
for o, y_ in zip(out, yy):
    ax.plot([o["model_error_pct"], o["carbonyl_error_pct"]], [y_, y_], color="#D9CCF0", lw=3, zorder=1)
ax.scatter([o["carbonyl_error_pct"] for o in out], yy, s=55, color=GOLD, zorder=3, label="Carbonyl index (today)"); ax.scatter([o["model_error_pct"] for o in out], yy, s=55, color=P, zorder=3, label="Our model")
ax.set_yticks(yy); ax.set_yticklabels([o["case"] for o in out], fontsize=10); ax.set_xlabel("Median forecast error at 100 h (%)"); ax.set_xlim(0, max(o["carbonyl_error_pct"] for o in out) + 4)
ax.legend(fontsize=10, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2, handletextpad=0.2); ax.text(1.0, 1.01, "SIMULATED", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=ORG, fontweight="bold")
fig.savefig(f"{FIG}/fig_robust.png", facecolor="white", bbox_inches="tight", pad_inches=0.03)
