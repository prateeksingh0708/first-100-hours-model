"""Compact, slide-sized charts for the v7 deck (shown at native size, so chart text stays readable).
usage: python3 figs_v7.py <sim_out_dir> <fig_dir>
Reads sim_arrays.npz, robustness.json and open_model_results.json; writes fig_insight_c, fig_robust_c, fig_decide_c."""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

SIM, FIG = sys.argv[1], sys.argv[2]
P, GOLD, ORG, MID, LAV, DK, GRY, GRN, RED = "#502B8B", "#F2A01E", "#C8601A", "#7A5BB5", "#D9CCF0", "#241A3D", "#6B6480", "#2E9E5B", "#D2453B"
fam = "TeX Gyre Adventor" if any("Adventor" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": fam, "font.size": 9, "axes.edgecolor": GRY, "axes.labelcolor": DK, "text.color": DK, "xtick.color": DK, "ytick.color": DK,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "savefig.dpi": 300, "legend.frameon": False})
def save(fig, name): fig.savefig(f"{FIG}/{name}.png", facecolor="white", bbox_inches="tight", pad_inches=0.03); plt.close(fig); print("fig", name)
def tag(ax, txt, col): ax.text(1.0, 1.02, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=col, fontweight="bold")

# ---- 1. the insight: chemistry moves first, gloss last (one simulated recipe)
Z = np.load(f"{SIM}/sim_arrays.npz", allow_pickle=True); t = Z["t"]; tf = int(Z["tf_ab"][0]); G = Z["Ga"] / Z["Ga"][0]; X = Z["Xa"] / Z["Xa"][tf]
fig, ax = plt.subplots(figsize=(3.1, 1.52)); m = t <= 2000
ax.axvspan(0, 100, color=GOLD, alpha=0.4, lw=0)
ax.plot(t[m], np.clip(G[m], 0, 1.02), color=P, lw=2.2); ax.plot(t[m & (t <= tf)], X[m & (t <= tf)], color=ORG, lw=2.0, ls=(0, (4, 2)))
ax.axhline(0.5, color=GRY, lw=0.7, ls=":"); ax.plot([tf], [0.5], "o", color=P, ms=5, zorder=5)
ax.text(tf + 60, 0.53, f"gloss fails\nat {tf:,} h", fontsize=8.5, color=P, va="bottom", fontweight="bold", linespacing=1.0)
ax.text(170, 0.84, "gloss (the eye)", fontsize=8.5, color=P, va="bottom")
ax.text(640, 0.06, "oxidation (infrared)", fontsize=8.5, color=ORG, va="bottom")
ax.text(125, 0.3, "first\n100 h", fontsize=8.5, color=ORG, fontweight="bold", va="bottom", linespacing=1.0)
ax.set_xlim(0, 2000); ax.set_ylim(0, 1.08); ax.set_xticks([0, 500, 1000, 1500, 2000]); ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0", "50%", "100%"])
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")); ax.set_xlabel("Hours in the UV test", labelpad=1)
tag(ax, "SIMULATED", ORG); save(fig, "fig_insight_c")

# ---- 2. robustness: our model against the carbonyl index in all ten simulation settings
R = json.load(open(f"{SIM}/robustness.json")); a = np.array([o["carbonyl_error_pct"] for o in R]); b = np.array([o["model_error_pct"] for o in R])
fig, ax = plt.subplots(figsize=(2.9, 1.72)); lim = [0, 32]
ax.fill_between(lim, [0, 0], lim, color=LAV, alpha=0.45, lw=0); ax.plot(lim, lim, color=GRY, lw=0.8)
ax.scatter(a, b, s=26, color=P, zorder=3, lw=0)
ax.text(4.5, 25.5, "equal error", fontsize=8.5, color=GRY, rotation=0); ax.text(31.5, 1.2, "below the line:\nour model wins", fontsize=8.5, color=P, ha="right", va="bottom", linespacing=1.0)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xticks([0, 10, 20, 30]); ax.set_yticks([0, 10, 20, 30])
ax.set_xlabel("Carbonyl index error (%)", labelpad=1); ax.set_ylabel("Our model error (%)", labelpad=1)
tag(ax, "SIMULATED", ORG); save(fig, "fig_robust_c")

# ---- 3. decisions at 100 h for four target lives (80 held-out simulated recipes)
D = json.load(open(f"{SIM}/open_model_results.json"))["decision_analysis"]["by_target_hours"]; ks = sorted(D, key=int)
fig, ax = plt.subplots(figsize=(2.9, 1.72)); yy = np.arange(len(ks))[::-1]
for k, y in zip(ks, yy):
    d = D[k]; n = d["n"]; x0 = 0
    for key, col, tc in (("advance", GRN, "white"), ("full_test", GOLD, DK), ("drop", RED, "white")):
        w = 100 * d[key] / n; ax.barh(y, w, left=x0, color=col, height=0.62, lw=0)
        if w >= 9: ax.text(x0 + w / 2, y, f"{w:.0f}%", ha="center", va="center", fontsize=8.5, color=tc, fontweight="bold")
        x0 += w
ax.set_yticks(yy); ax.set_yticklabels([f"{int(k):,} h" for k in ks]); ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.set_xticklabels(["0", "50%", "100%"])
ax.set_ylabel("Target life", labelpad=1); ax.set_xlabel("Share of 80 unseen recipes", labelpad=1)
for i, (lab, col) in enumerate((("advance", GRN), ("full test", GOLD), ("drop", RED))):
    ax.text(0.0 + i * 0.36, 1.02, "■ ", transform=ax.transAxes, color=col, fontsize=8.5, va="bottom"); ax.text(0.06 + i * 0.36, 1.02, lab, transform=ax.transAxes, fontsize=8.5, va="bottom")
save(fig, "fig_decide_c")
