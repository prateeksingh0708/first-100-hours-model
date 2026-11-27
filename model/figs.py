"""Figures for the First 100 Hours deck. usage: python3 figs.py <sim_out_dir> <fig_dir>"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle, Circle
from matplotlib import font_manager

SIM, FIG = sys.argv[1], sys.argv[2]
P, GOLD, ORG, MID, LAV, DK, GRY, TINT = "#502B8B", "#F2A01E", "#C8601A", "#7A5BB5", "#D9CCF0", "#241A3D", "#6B6480", "#F1ECF9"
fam = "TeX Gyre Adventor" if any("Adventor" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": fam, "font.size": 10.5, "axes.edgecolor": GRY, "axes.labelcolor": DK, "text.color": DK,
                     "xtick.color": GRY, "ytick.color": GRY, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "mathtext.fontset": "custom", "mathtext.rm": fam, "mathtext.it": fam + ":italic",
                     "savefig.dpi": 240, "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "legend.frameon": False})
R = json.load(open(SIM + "/sim_results.json")); Z = np.load(SIM + "/sim_arrays.npz", allow_pickle=True)
def tag(ax, txt="SIMULATED SPECTRA, NOT COATING DATA", loc=(0.99, 1.02)):
    ax.text(loc[0], loc[1], txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=ORG, fontweight="bold")
def save(fig, name): fig.savefig(f"{FIG}/{name}.png", facecolor="white"); plt.close(fig); print(name)

# ---------- 1. ATR sampling depth (physics; n values are standard optical constants)
nu = np.linspace(650, 4000, 600); lam_um = 1e4 / nu; th = np.radians(45); n2 = 1.5
dp = lambda n1: lam_um / (2 * np.pi * n1 * np.sqrt(np.sin(th) ** 2 - (n2 / n1) ** 2))
fig, ax = plt.subplots(figsize=(6.3, 3.3))
for lo, hi, lab in [(3100, 3600, "O–H"), (2800, 3000, "C–H"), (1650, 1800, "C=O")]:
    ax.axvspan(lo, hi, color=TINT, lw=0); ax.text((lo + hi) / 2, 3.72, lab, ha="center", fontsize=9, color=P, fontweight="bold")
ax.plot(nu, dp(2.4), color=P, lw=2.2, label="Diamond or ZnSe crystal (n = 2.4)")
ax.plot(nu, dp(4.0), color=GOLD, lw=2.2, label="Germanium crystal (n = 4.0)")
for v in (3400, 1730):
    d = 1e4 / v / (2 * np.pi * 2.4 * np.sqrt(np.sin(th) ** 2 - (n2 / 2.4) ** 2))
    ax.plot([v], [d], "o", color=P, ms=5); ax.annotate(f"{d:.2f} µm", (v, d), textcoords="offset points", xytext=(-8, 12), ha="right", fontsize=9.5, color=P, fontweight="bold")
ax.set_xlim(4000, 650); ax.set_ylim(0, 4); ax.set_xlabel("Wavenumber (cm$^{-1}$)"); ax.set_ylabel("Penetration depth $d_p$ (µm)")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=9)
save(fig, "fig_atr_depth")
print("dp diamond 3400/1730/1000:", [round(float(1e4 / v / (2 * np.pi * 2.4 * np.sqrt(np.sin(th) ** 2 - (n2 / 2.4) ** 2))), 2) for v in (3400, 1730, 1000)],
      "Ge 1730:", round(float(1e4 / 1730 / (2 * np.pi * 4.0 * np.sqrt(np.sin(th) ** 2 - (n2 / 4.0) ** 2))), 2))

# ---------- 2. UV dose: xenon light hours vs Florida years (Atlas: 280 MJ/m2 TUV per year, 1.1% at 340 nm)
FL = 0.011 * 280e3   # kJ/m2 at 340 nm per Florida year = 3,080
h = np.linspace(0, 2200, 200); dose = 0.55 * 3.6 * h
fig, ax = plt.subplots(figsize=(6.3, 3.3))
ax.plot(h, dose, color=P, lw=2.2)
ax.axhline(FL, color=GRY, lw=0.8, ls=(0, (4, 3))); ax.text(30, FL + 90, "1 Florida year = 3,080 kJ/m$^2$ at 340 nm", fontsize=9, color=GRY)
for hh, lab, dx, dy in [(100, "100 h = 198 kJ/m$^2$ = 23 days of Florida UV", 16, -4), (FL / (0.55 * 3.6), "1,556 h = 1 Florida year", -150, 14), (2000, "2,000 h = 1.29 Florida years", -165, 12)]:
    ax.plot([hh], [0.55 * 3.6 * hh], "o", color=GOLD if hh == 100 else P, ms=6.5, zorder=5)
    ax.annotate(lab, (hh, 0.55 * 3.6 * hh), textcoords="offset points", xytext=(dx, dy), fontsize=9.5, color=ORG if hh == 100 else P, fontweight="bold")
ax.set_xlim(0, 2200); ax.set_ylim(0, 4700); ax.set_xlabel("Xenon-arc light hours at 0.55 W/m$^2$/nm (340 nm)"); ax.set_ylabel("Radiant exposure at 340 nm (kJ/m$^2$)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
save(fig, "fig_dose")
print("dose check: 100h", 0.55 * 3.6 * 100, "kJ ->", 0.55 * 3.6 * 100 / FL * 365, "days; 1yr h", FL / (0.55 * 3.6), "; 2000h yrs", 0.55 * 3.6 * 2000 / FL)

# ---------- 3. Cross-section schematic
rng = np.random.default_rng(3)
fig, ax = plt.subplots(figsize=(6.3, 3.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
ax.add_patch(Rectangle((0, 0.2), 10, 1.0, fc="#D5D2DC", ec="none")); ax.text(0.2, 0.62, "Substrate and primer", fontsize=9.5, color=DK, va="center")
ax.add_patch(Rectangle((0, 1.2), 10, 3.0, fc=TINT, ec="none"))
grad = np.linspace(1, 0, 60).reshape(-1, 1) ** 1.6
ax.imshow(np.dstack([np.full_like(grad, 0.784), np.full_like(grad, 0.376), np.full_like(grad, 0.102), 0.75 * grad]), extent=(0, 10, 3.25, 4.2), aspect="auto", zorder=2)
for _ in range(70):
    x, y, r = rng.uniform(0.2, 9.8), rng.uniform(1.4, 4.0), rng.uniform(0.07, 0.15)
    ax.add_patch(Circle((x, y), r, fc="white", ec=MID, lw=0.6, zorder=3))
ax.text(0.2, 2.2, "Paint film: binder, TiO$_2$, extenders", fontsize=9.5, color=DK, va="center", zorder=4, bbox=dict(fc=TINT, ec="none", pad=1.5))
ax.text(0.2, 3.72, "Oxidised surface layer", fontsize=9.5, color="white", fontweight="bold", va="center", zorder=4)
for x in (0.9, 1.9, 2.9, 3.9):
    ax.add_patch(FancyArrowPatch((x - 0.5, 6.0), (x, 4.3), arrowstyle="-|>", mutation_scale=11, color=GOLD, lw=1.8, zorder=5))
ax.text(4.3, 5.55, "UV + O$_2$ + moisture", fontsize=10, color=ORG, fontweight="bold", va="center")
ax.add_patch(Polygon([(6.1, 4.2), (9.7, 4.2), (9.2, 5.5), (6.6, 5.5)], closed=True, fc=P, ec="none", zorder=5))
ax.text(7.9, 5.12, "ATR crystal", color="white", fontsize=9.5, ha="center", va="center", fontweight="bold", zorder=6)
ax.plot([6.75, 7.9, 9.05], [5.5, 4.2, 5.5], color=GOLD, lw=1.6, zorder=6); ax.text(7.9, 5.78, "IR beam", color=ORG, fontsize=9, ha="center")
for k, a in enumerate([0.9, 0.6, 0.35, 0.18]):
    ax.plot([7.45 + 0.1 * k, 8.35 - 0.1 * k], [4.2 - 0.13 * (k + 1)] * 2, color=GOLD, lw=1.5, alpha=a, zorder=6)
ax.annotate("", xy=(9.35, 3.62), xytext=(9.35, 4.2), arrowprops=dict(arrowstyle="<->", color=DK, lw=1), zorder=7)
ax.text(9.45, 3.25, "Sampling depth\n0.6 to 2 µm", fontsize=9, color=DK, ha="right", va="top", zorder=7, bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.85))
ax.text(9.95, 0.02, "Schematic, not to scale", fontsize=7.5, color=GRY, ha="right")
save(fig, "fig_xsec")

# ---------- 4. Photo-oxidation cycle schematic
fig, ax = plt.subplots(figsize=(8.4, 3.6)); ax.set_xlim(0, 14.8); ax.set_ylim(0, 7); ax.axis("off")
def box(x, y, w, h_, txt, fc=TINT, tc=DK, bold=True, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h_, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec="none"))
    ax.text(x + w / 2, y + h_ / 2, txt, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal")
def arr(a, b, txt="", off=(0, 0.25), color=P, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=color, lw=1.5, ls=ls, connectionstyle=f"arc3,rad={rad}"))
    if txt: ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], txt, ha="center", va="center", fontsize=8.5, color=GRY)
box(0.2, 4.6, 1.9, 1.0, "Polymer\nP–H"); box(3.3, 4.6, 1.5, 1.0, "P•"); box(6.0, 4.6, 1.7, 1.0, "POO•"); box(8.9, 4.6, 1.9, 1.0, "POOH"); box(12.0, 4.6, 2.5, 1.0, "PO• + •OH")
arr((2.1, 5.1), (3.3, 5.1), "hν", (0, 0.3)); arr((4.8, 5.1), (6.0, 5.1), "+ O$_2$", (0, 0.3)); arr((7.7, 5.1), (8.9, 5.1), "+ P–H", (0, 0.3)); arr((10.8, 5.1), (12.0, 5.1), "hν", (0, 0.3))
arr((13.25, 5.6), (4.05, 5.6), "radicals abstract H and restart the chain", (0, 0.95), color=MID, rad=0.22)
box(8.15, 2.25, 3.2, 1.15, "Hydroperoxide, alcohol\nO–H", fc=P, tc="white", fs=9)
box(11.55, 2.25, 3.15, 1.15, "Acid, ketone, lactone\nC=O", fc=P, tc="white", fs=9)
box(4.75, 2.25, 3.2, 1.15, "Chain scission\nC–H and C–O loss", fc=P, tc="white", fs=9)
arr((9.85, 4.6), (9.85, 3.4)); arr((13.15, 4.6), (13.15, 3.4), "β-scission", (0.85, 0)); arr((12.6, 4.6), (7.0, 3.4), rad=0.0)
for x, t in [(9.85, "3100–3600 cm$^{-1}$ rises"), (13.15, "1650–1800 cm$^{-1}$ broadens"), (6.45, "2800–3000 cm$^{-1}$ falls")]:
    ax.text(x, 1.85, t, ha="center", va="center", fontsize=9, color=ORG, fontweight="bold")
box(4.9, 0.25, 9.7, 0.95, "Later: surface erosion, roughness, pigment exposure  →  gloss loss, colour shift, chalking", fc="#E9E6EE", fs=9.5, bold=False)
arr((6.45, 1.6), (6.45, 1.2), color=GRY); arr((9.85, 1.6), (9.85, 1.2), color=GRY); arr((13.15, 1.6), (13.15, 1.2), color=GRY)
box(0.2, 2.25, 3.9, 1.15, "HALS traps P• and POO•\nUV absorber cuts hν", fc=GOLD, tc=DK, fs=9)
arr((2.0, 3.4), (3.9, 4.6), color=ORG, ls=(0, (3, 2))); arr((2.6, 3.4), (6.6, 4.6), color=ORG, ls=(0, (3, 2)))
ax.text(0.2, 1.5, "What ATR-FTIR sees\nin the first 100 hours", fontsize=9.5, color=ORG, fontweight="bold", va="center")
save(fig, "fig_mechanism")

# ---------- 5. Simulated: chemistry leads, gloss lags
t = Z["t"]; fig, ax = plt.subplots(figsize=(6.3, 3.3))
ax.axvspan(0, 100, color=GOLD, alpha=0.35, lw=0); ax.text(130, 1.02, "first 100 h", fontsize=9, color=ORG, fontweight="bold")
for G, X, c, lab in [(Z["Ga"], Z["Xa"], P, f"fails at {Z['tf_ab'][0]:,.0f} h"), (Z["Gb"], Z["Xb"], MID, f"fails at {Z['tf_ab'][1]:,.0f} h")]:
    ax.plot(t, G, color=c, lw=2.2, label="Gloss retention, formulation that " + lab)
    tfi = int(Z["tf_ab"][0] if c == P else Z["tf_ab"][1]); m_ = t <= tfi
    ax.plot(t[m_], X[m_] / X[tfi], color=c, lw=1.6, ls=(0, (4, 2)), label="Oxidation, as a fraction of the extent at failure")
ax.axhline(0.5, color=GRY, lw=0.8, ls=":"); ax.text(4900, 0.52, "failure threshold: 50% gloss", fontsize=8.5, color=GRY, ha="right")
ax.set_xlim(0, 5000); ax.set_ylim(0, 1.1); ax.set_xlabel("Exposure (h)"); ax.set_ylabel("Fraction")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2); tag(ax)
save(fig, "fig_sim_concept")

# ---------- 6. Simulated spectra and difference spectrum
NU = Z["NU"]; fig, axs = plt.subplots(2, 1, figsize=(6.6, 3.9), sharex=True, gridspec_kw=dict(height_ratios=[1, 1], hspace=0.12))
axs[0].plot(NU, Z["s0_a"], color=GRY, lw=1.2, label="0 h"); axs[0].plot(NU, Z["s100_a"], color=P, lw=1.0, label="100 h"); axs[0].legend(fontsize=8.5, loc="upper left", ncol=2)
axs[0].set_ylabel("Normalised\nabsorbance"); tag(axs[0])
for lo, hi in [(3100, 3650), (2840, 3000), (1600, 1850)]: axs[1].axvspan(lo, hi, color=TINT, lw=0)
sm = lambda v: np.convolve(v, np.ones(41) / 41, mode="same")
da, db = Z["s100_a"] - Z["s0_a"], Z["s100_b"] - Z["s0_b"]
axs[1].plot(NU, da, color=LAV, lw=0.6); axs[1].plot(NU, db, color="#F7D9A3", lw=0.6)
axs[1].plot(NU, sm(da), color=P, lw=1.8, label=f"fails at {Z['tf_ab'][0]:,.0f} h (smoothed)"); axs[1].plot(NU, sm(db), color=ORG, lw=1.8, label=f"fails at {Z['tf_ab'][1]:,.0f} h (smoothed)")
axs[1].axhline(0, color=GRY, lw=0.6); axs[1].set_ylabel("Difference\n100 h − 0 h"); axs[1].set_xlabel("Wavenumber (cm$^{-1}$)"); axs[1].set_xlim(4000, 650); axs[1].legend(fontsize=8.5, loc="upper left", ncol=2)
save(fig, "fig_sim_spectra")

# ---------- 7. Model comparison at 100 h
M = R["models_100h"]; names = list(M.keys()); vals = [M[k]["mdape"] for k in names]
fig, ax = plt.subplots(figsize=(6.3, 3.0)); yy = np.arange(len(names))[::-1]
ax.barh(yy, vals, color=[LAV, LAV, MID, MID, P], height=0.62)
for y_, v, k in zip(yy, vals, names): ax.text(v + 0.6, y_, f"{v:.0f}%   (R² {M[k]['r2']:.2f})", va="center", fontsize=9.5, color=DK, fontweight="bold" if k == names[-1] else "normal")
ax.axvline(R["mean_baseline_mdape"], color=GRY, lw=0.8, ls=(0, (4, 3))); ax.text(R["mean_baseline_mdape"] + 0.5, -0.38, "guessing the mean", fontsize=8.5, color=GRY)
ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=9.5, color=DK); ax.set_xlim(0, 44); ax.set_xlabel("Median error in forecast hours to failure (%)"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0); tag(ax)
save(fig, "fig_sim_models")

# ---------- 8. Parity on 80 held-out formulations with conformal band
yT, yh, q = 10 ** Z["yT"], 10 ** Z["yhT"], float(Z["q"]); fig, ax = plt.subplots(figsize=(3.9, 3.6))
lim = [600, 12000]; xx = np.array(lim)
ax.fill_between(xx, xx / 10 ** q, xx * 10 ** q, color=TINT, lw=0, label="90% conformal band"); ax.plot(xx, xx, color=GRY, lw=0.8)
ax.scatter(yT, yh, s=16, color=P, alpha=0.85, lw=0, zorder=3, label="Held-out formulation")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
for a in (ax.xaxis, ax.yaxis): a.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")); a.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([1000, 2000, 5000, 10000]); ax.set_yticks([1000, 2000, 5000, 10000])
ax.set_xlabel("Actual hours to failure"); ax.set_ylabel("Forecast at 100 h"); ax.legend(fontsize=8.5, loc="upper left"); tag(ax, "SIMULATED")
save(fig, "fig_sim_parity")

# ---------- 9. Window, learning curve, noise
W = R["window"]; T = [int(k) for k in W]; fig, ax = plt.subplots(figsize=(4.1, 3.0))
ax.plot(T, [W[str(k)]["gloss"][0] for k in T], "o-", color=LAV, lw=1.8, ms=5, label="Early gloss only")
ax.plot(T, [W[str(k)]["ci"][0] for k in T], "o-", color=MID, lw=1.8, ms=5, label="Single carbonyl index")
ax.plot(T, [W[str(k)]["full"][0] for k in T], "o-", color=P, lw=2.2, ms=6, label="Band features + descriptors")
ax.set_xscale("log"); ax.set_xticks(T); ax.set_xticklabels([str(k) for k in T]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_ylim(0, 36)
ax.set_xlabel("Observation window (h)"); ax.set_ylabel("Median error (%)"); ax.legend(fontsize=8, loc="lower left"); tag(ax, "SIMULATED")
save(fig, "fig_sim_window")
L = R["learning"]; n = [int(k) for k in L]; mu = np.array([L[str(k)][0] for k in n]); sd = np.array([L[str(k)][1] for k in n])
fig, ax = plt.subplots(figsize=(4.1, 3.0)); ax.fill_between(n, mu - sd, mu + sd, color=TINT, lw=0); ax.plot(n, mu, "o-", color=P, lw=2.2, ms=6)
ax.set_ylim(0, 22); ax.set_xlabel("Formulations in training archive"); ax.set_ylabel("Median error on 80 held-out (%)"); tag(ax, "SIMULATED")
save(fig, "fig_sim_learning")
Nz = R["noise"]; sg = [float(k) for k in Nz]; fig, ax = plt.subplots(figsize=(4.1, 3.0))
ax.plot(np.array(sg) * 1e3, [Nz[k]["ci"][0] for k in Nz], "o-", color=MID, lw=1.8, ms=5, label="Single carbonyl index")
ax.plot(np.array(sg) * 1e3, [Nz[k]["full"][0] for k in Nz], "o-", color=P, lw=2.2, ms=6, label="Band features + descriptors")
ax.set_xscale("log"); ax.set_xticks(np.array(sg) * 1e3); ax.set_xticklabels([f"{s * 1e3:g}" for s in sg]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_ylim(0, 36)
ax.set_xlabel("Spectral noise per point (milli-absorbance)"); ax.set_ylabel("Median error (%)"); ax.legend(fontsize=8, loc="lower right"); tag(ax, "SIMULATED")
save(fig, "fig_sim_noise")

# ---------- 10. Elastic-net standardised coefficients
E = R["enet_coef"]; lab = {"rate_OH": "O–H growth rate (3100–3650)", "rate_LAC": "Lactone C=O rate (1760–1800)", "rate_ACID": "Acid C=O rate (1690–1720)", "rate_CC": "C=C rate (1620–1660)",
       "rate_CH": "C–H loss rate (2840–3000)", "logvar_carbonyl": "log variance, carbonyl window", "stabiliser_loading": "Stabiliser loading (descriptor)", "binder_Tg": "Binder Tg (descriptor)", "styrene_fraction": "Styrene fraction (descriptor)"}
ks = sorted(E, key=lambda k: abs(E[k])); fig, ax = plt.subplots(figsize=(4.6, 3.3)); yy = np.arange(len(ks))
ax.barh(yy, [E[k] for k in ks], color=[P if E[k] < 0 else GOLD for k in ks], height=0.62); ax.axvline(0, color=GRY, lw=0.8)
ax.set_yticks(yy); ax.set_yticklabels([lab[k] for k in ks], fontsize=10, color=DK); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.set_xlabel("Coefficient on log$_{10}$ hours to failure\n(negative = shortens life)"); tag(ax, "SIMULATED", (0.99, 1.02))
save(fig, "fig_sim_coef")

# ---------- equations (mathtext)
plt.rcParams.update({"mathtext.fontset": "cm"})
EQ = {
 "eq_diff": r"$\Delta A_i(\tilde{\nu},t)=\tilde{A}_i(\tilde{\nu},t)-\tilde{A}_i(\tilde{\nu},0)$",
 "eq_rate": r"$r_{i,b}=\dfrac{\sum_{t} t\int_{b}\Delta A_i(\tilde{\nu},t)\,d\tilde{\nu}}{\sum_{t} t^{2}}$",
 "eq_var": r"$v_i=\log_{10}\,\mathrm{Var}_{\tilde{\nu}\in[1600,1850]}\left[\Delta A_i(\tilde{\nu},100\,\mathrm{h})\right]$",
 "eq_enet": r"$\hat{\beta}=\arg\min_{\beta}\ \dfrac{1}{2n}\Vert y-X\beta\Vert_2^2+\lambda\left[\alpha\Vert\beta\Vert_1+\dfrac{1-\alpha}{2}\Vert\beta\Vert_2^2\right]$",
 "eq_conf": r"$\hat{q}=Q_{\lceil (n+1)(0.9)\rceil/n}\left(|y_j-\hat{y}_j|\right)$",
 "eq_conf2": r"$t_f\in\left[\,\hat{t}_f\,10^{-\hat{q}},\ \ \hat{t}_f\,10^{+\hat{q}}\,\right]$",
 "eq_maha": r"$d_M(x)=\sqrt{(x-\mu)^{\top}\Sigma^{-1}(x-\mu)}\ >\ d_{0.95}$",
 "eq_dose": r"$D_{\mathrm{eff}}(t)=\int_0^{t}\!\!\int_{\lambda_{min}}^{\lambda_{max}}E_0(\lambda,t')\left(1-10^{-A(\lambda,t')}\right)\phi(\lambda)\,d\lambda\,dt'$",
 "eq_recip": r"$I\,t^{\,p}=\mathrm{const},\qquad p=1\ \Rightarrow\ \mathrm{reciprocity\ holds}$",
 "eq_h340": r"$H_{340}=E_{340}\times 3.6\times t_{\mathrm{light}}\quad[\mathrm{kJ\,m^{-2}\,nm^{-1}}]$",
 "eq_atr": r"$d_p=\dfrac{\lambda}{2\pi n_1\sqrt{\sin^{2}\theta-(n_2/n_1)^{2}}}$",
 "eq_kin": r"$\dfrac{dX}{dt}=\dfrac{k\,(1+aX)}{1+K\,s\,e^{-t/\tau}},\qquad \dfrac{G(t)}{G_0}=\dfrac{1}{1+\left(wX/X_c\right)^{m}}$",
 "eq_spec": r"$A(\tilde{\nu},t)=A_0(\tilde{\nu})+X(t)\left[p\,B_{sc}(\tilde{\nu})+(1-p)\,B_{ox}(\tilde{\nu})\right]-0.3\,X(t)\,B_{CH}(\tilde{\nu})$",
 "eq_meas": r"$A_{meas}=c\,A+b_0+b_1\tilde{\nu}'+\varepsilon,\quad \ln c\sim N(0,0.03^2),\ \ b_0,b_1\sim N(0,(2{\times}10^{-3})^2),\ \ \varepsilon\sim N(0,\sigma^2)$",
 "eq_ci": r"$\mathrm{CI}=\dfrac{A_{1600-1800}}{A_{2800-3050}},\qquad \Delta\mathrm{CI}(t)=\mathrm{CI}(t)-\mathrm{CI}(0)$",
 "eq_poi": r"$\mathrm{POI}(t)=\left[\dfrac{A_{\mathrm{OH,NH}}}{A_{\mathrm{CH}}}\right]_{t}-\left[\dfrac{A_{\mathrm{OH,NH}}}{A_{\mathrm{CH}}}\right]_{0}$",
}
for k, tex in EQ.items():
    fig = plt.figure(figsize=(0.1, 0.1)); fig.text(0, 0, tex, fontsize=15, color=DK)
    fig.savefig(f"{FIG}/{k}.png", dpi=300, bbox_inches="tight", pad_inches=0.03, transparent=True); plt.close(fig)
print("equations", len(EQ))
