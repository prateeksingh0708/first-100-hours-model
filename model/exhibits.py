"""Extra exhibits: 3D waterfall of simulated difference spectra, 3D life surface, literature timeline.
usage: python3 exhibits.py <figdir>"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
FIG = sys.argv[1]
P, GOLD, ORG, MID, LAV, DK, GRY, TINT = "#502B8B", "#F2A01E", "#C8601A", "#7A5BB5", "#D9CCF0", "#241A3D", "#6B6480", "#F1ECF9"
fam = "TeX Gyre Adventor" if any("Adventor" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": fam, "font.size": 10, "axes.edgecolor": GRY, "axes.labelcolor": DK, "text.color": DK, "xtick.color": GRY, "ytick.color": GRY,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 240, "savefig.bbox": "tight", "savefig.pad_inches": 0.04})
NU = np.arange(4000.0, 648.0, -2.0)
lor = lambda c, w: (0.5 * w) ** 2 / ((NU - c) ** 2 + (0.5 * w) ** 2)
gau = lambda c, w: np.exp(-4 * np.log(2) * (NU - c) ** 2 / w ** 2)
PROD = 0.5 * (0.5 * lor(1708, 30) + 0.15 * lor(1640, 30)) + 0.5 * (0.35 * gau(3400, 300) + 0.30 * lor(1778, 28)) - 0.3 * (lor(2955, 20) * .18 + lor(2930, 25) * .14 + lor(2873, 20) * .10)

def X_of_t(k, s, tmax):
    t = np.arange(0, tmax + 1.0); X = np.zeros_like(t)
    for j in range(1, len(t)): X[j] = X[j - 1] + k * (1 + 3 * X[j - 1]) / (1 + 4 * s * np.exp(-t[j - 1] / 400))
    return t, X

# ---------- 1. 3D waterfall of difference spectra (simulated, noise-free, one formulation)
t, X = X_of_t(2.6e-4, 0.3, 600); times = [24, 48, 100, 200, 300, 400, 500]
fig = plt.figure(figsize=(6.4, 3.9)); ax = fig.add_subplot(projection="3d")
cm = plt.get_cmap("plasma")
for i, T in enumerate(times):
    y = X[T] * PROD
    ax.plot(NU, np.full_like(NU, T), y, color=cm(0.1 + 0.75 * i / len(times)), lw=1.4)
    ax.fill_between(NU, np.full_like(NU, T), y, NU, np.full_like(NU, T), 0 * y, color=cm(0.1 + 0.75 * i / len(times)), alpha=0.18, lw=0)
ax.set_xlim(4000, 650); ax.set_ylim(0, 520); ax.set_xlabel("Wavenumber (cm$^{-1}$)", labelpad=6); ax.set_ylabel("Exposure (h)", labelpad=6); ax.set_zlabel("ΔA vs 0 h", labelpad=4)
ax.view_init(elev=24, azim=-62); ax.set_box_aspect((2.2, 1.3, 0.8))
for a in (ax.xaxis, ax.yaxis, ax.zaxis): a.pane.set_facecolor((1, 1, 1, 0)); a.pane.set_edgecolor("#DDDDDD")
ax.tick_params(labelsize=8); ax.zaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
for x, lab in [(3400, "O–H"), (1745, "C=O"), (2930, "C–H")]:
    ax.text(x, 520, X[500] * PROD[np.argmin(abs(NU - x))] + 0.0006, lab, color=P, fontsize=9, fontweight="bold")
ax.text2D(0.99, 0.97, "SIMULATED", transform=ax.transAxes, ha="right", fontsize=7.5, color=ORG, fontweight="bold")
fig.savefig(f"{FIG}/fig3d_waterfall.png", facecolor="white"); plt.close(fig); print("waterfall")

# ---------- 2. 3D life surface: hours to 50% gloss vs initiation rate and stabiliser
ks = np.geomspace(1.0e-4, 4.0e-4, 22); ss = np.linspace(0, 1, 22); TF = np.zeros((len(ss), len(ks)))
for i, s_ in enumerate(ss):
    for j, k_ in enumerate(ks):
        t, X = X_of_t(k_, s_, 12000); w = 1.0; Xc = 0.5; r = w * X / Xc
        TF[i, j] = t[np.argmax(r >= 1)] if (r >= 1).any() else np.nan
K, S = np.meshgrid(ks * 1e4, ss)
fig = plt.figure(figsize=(5.0, 3.9)); ax = fig.add_subplot(projection="3d")
surf = ax.plot_surface(K, S, TF, cmap="plasma_r", edgecolor="white", lw=0.25, alpha=0.95, rstride=1, cstride=1)
ax.set_xlabel("Initiation rate k (10$^{-4}$ h$^{-1}$)", labelpad=6); ax.set_ylabel("Stabiliser loading s", labelpad=6); ax.set_zlabel("Hours to failure", labelpad=8)
ax.view_init(elev=26, azim=-130); ax.tick_params(labelsize=8)
for a in (ax.xaxis, ax.yaxis, ax.zaxis): a.pane.set_facecolor((1, 1, 1, 0)); a.pane.set_edgecolor("#DDDDDD")
ax.zaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.text2D(0.99, 0.97, "SIMULATED", transform=ax.transAxes, ha="right", fontsize=7.5, color=ORG, fontweight="bold")
fig.savefig(f"{FIG}/fig3d_surface.png", facecolor="white"); plt.close(fig); print("surface", np.nanmin(TF), np.nanmax(TF))

# ---------- 3. Literature timeline on a log-hour axis (published or computed from published constants)
fig, ax = plt.subplots(figsize=(8.6, 2.35))
ax.set_xscale("log"); ax.set_xlim(15, 60000); ax.set_ylim(-1.6, 2.3); ax.spines["left"].set_visible(False); ax.set_yticks([])
ax.axhline(0, color=GRY, lw=1)
ax.axvspan(15, 100, color=GOLD, alpha=0.25, lw=0); ax.text(39, 2.12, "FIRST 100 H", ha="center", fontsize=8.5, color=ORG, fontweight="bold")
ev = [(24, 0.95, "24 h: earliest QUV pull in the\ncarbonyl-index study (EPS)", P), (168, -0.95, "168 h: carbonyl change fits gloss-\nfailure onset, R² 0.9977 (EPS)", P),
      (500, 0.95, "500 h xenon: no visible change,\nspectra already differ (Kyoto)", P), (1556, -0.95, "1,556 h: one South Florida\nyear of 340 nm UV dose", GRY),
      (2000, 1.75, "2,000 h: a typical\naccelerated test", GRY), (15560, -0.95, "15,560 h: dose of AAMA\n2605's 10 Florida years", ORG), (23340, 0.95, "23,340 h: dose of a\n15-year warranty", ORG)]
for x, y, txt, c in ev:
    ax.plot([x, x], [0, y * 0.55], color=c, lw=0.9); ax.plot([x], [0], "o", color=c, ms=6, zorder=3)
    ax.text(x, y, txt, ha="center", va="center", fontsize=8.5, color=c if c != GRY else DK, linespacing=1.15)
ax.set_xticks([24, 100, 500, 2000, 10000]); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f} h")); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.tick_params(labelsize=9)
fig.savefig(f"{FIG}/fig_timeline.png", facecolor="white"); plt.close(fig); print("timeline")
