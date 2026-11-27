"""v3 figures, drawn at their placed size so 1 pt in the figure = 1 pt on the slide.
usage: python3 figs_v3.py <sim_out_dir> <fig_dir>"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

SIM, FIG = sys.argv[1], sys.argv[2]
P, GOLD, ORG, MID, LAV, DK, GRY, TINT = "#502B8B", "#F2A01E", "#C8601A", "#7A5BB5", "#B9A6E0", "#241A3D", "#6B6480", "#F1ECF9"
fam = "TeX Gyre Adventor" if any("Adventor" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": fam, "font.size": 11, "axes.edgecolor": GRY, "axes.labelcolor": DK, "text.color": DK,
                     "xtick.color": DK, "ytick.color": DK, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "axes.labelsize": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "mathtext.fontset": "custom", "mathtext.rm": fam, "mathtext.it": fam + ":italic", "mathtext.bf": fam + ":bold",
                     "savefig.dpi": 260, "legend.frameon": False})
R = json.load(open(SIM + "/sim_results.json")); Z = np.load(SIM + "/sim_arrays.npz", allow_pickle=True)
def save(fig, name, tight=True):
    fig.savefig(f"{FIG}/{name}.png", facecolor="white", bbox_inches="tight" if tight else None, pad_inches=0.03); plt.close(fig); print(name)
def tag(ax, txt="SIMULATED", loc=(1.0, 1.01)):
    ax.text(loc[0], loc[1], txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=ORG, fontweight="bold")
fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")

# ======================================================================  1. Photo-oxidation mechanism (8.6 x 4.1 in)
W_, H_ = 8.6, 4.22
fig = plt.figure(figsize=(W_, H_)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W_); ax.set_ylim(0, H_); ax.axis("off")
def box(x0, x1, y0, y1, txt, fc=TINT, tc=DK, fs=11.5, bold=True):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec="none"))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, txt, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal", linespacing=1.25)
def arrow(a, b, color=P, ls="-", lw=1.6, ms=13):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))
y0, y1 = 2.95, 3.5
CH = [("Binder\nP–H", 0.0, 1.05), ("P•", 1.65, 2.35), ("POO•", 2.95, 3.85), ("POOH", 4.55, 5.55), ("PO• + •OH", 6.35, 8.55)]
for t, a, b in CH: box(a, b, y0, y1, t, fs=12.5 if "\n" not in t else 11)
for (lab, a, b) in [("light (hν)", 1.05, 1.65), ("+ O$_2$", 2.35, 2.95), ("+ P–H", 3.85, 4.55), ("hν", 5.55, 6.35)]:
    arrow((a + 0.03, (y0 + y1) / 2), (b - 0.03, (y0 + y1) / 2))
    ax.text((a + b) / 2, y1 + 0.08, lab, ha="center", va="bottom", fontsize=10, color=GRY)
# feedback loop: radicals restart the chain
lx0, lx1, ly = 7.1, 2.0, 3.86
ax.plot([lx0, lx0, lx1], [y1, ly, ly], color=MID, lw=1.6, solid_capstyle="round")
arrow((lx1, ly), (lx1, y1 + 0.01), color=MID)
ax.text((lx0 + lx1) / 2, ly + 0.07, "•OH and PO• attack more binder: the chain restarts and speeds up", ha="center", va="bottom", fontsize=10.5, color=MID, fontweight="bold")
# products
p0, p1 = 1.75, 2.45
box(0.0, 2.75, p0, p1, "STABILISERS SLOW IT\nHALS trap P• and POO•\nUV absorbers block light", fc=GOLD, tc=DK, fs=10.5)
arrow((1.55, p1), (2.0, y0 - 0.02), color=ORG, ls=(0, (3, 2)), lw=1.4); arrow((2.35, p1), (3.3, y0 - 0.02), color=ORG, ls=(0, (3, 2)), lw=1.4)
box(4.05, 6.05, p0, p1, "Hydroperoxides\nand alcohols", fc=P, tc="white", fs=11)
box(6.35, 8.55, p0, p1, "β-scission: ketones, acids,\nlactones; chains break", fc=P, tc="white", fs=11)
arrow((5.05, y0), (5.05, p1 + 0.01)); arrow((7.45, y0), (7.45, p1 + 0.01))
# infrared read-out
ax.text(5.05, 1.42, "O–H  3100–3650 cm$^{-1}$  rises", ha="center", va="center", fontsize=11, color=ORG, fontweight="bold")
ax.text(7.45, 1.47, "C=O  1690–1800 cm$^{-1}$  rises", ha="center", va="center", fontsize=11, color=ORG, fontweight="bold")
ax.text(7.45, 1.17, "C–H  2840–3000 cm$^{-1}$  falls", ha="center", va="center", fontsize=11, color=ORG, fontweight="bold")
ax.text(0.0, 1.3, "WHAT INFRARED SEES\nIN THE FIRST 100 HOURS", ha="left", va="center", fontsize=10.5, color=ORG, fontweight="bold", linespacing=1.25)
ax.add_patch(FancyArrowPatch((2.95, 1.3), (3.9, 1.3), arrowstyle="-|>", mutation_scale=12, color=ORG, lw=1.4))
# what the eye sees later
box(4.05, 8.55, 0.05, 0.72, "Surface erodes, roughens and exposes pigment\n→ gloss loss, colour shift, chalking", fc="#E6E1EE", tc=DK, fs=10.5, bold=False)
for x in (5.05, 7.45): arrow((x, 1.0 if x == 7.45 else 1.25), (x, 0.73), color=GRY, lw=1.2)
ax.text(0.0, 0.38, "WHAT THE EYE SEES,\nWEEKS TO MONTHS LATER", ha="left", va="center", fontsize=10.5, color=GRY, fontweight="bold", linespacing=1.25)
ax.add_patch(FancyArrowPatch((2.95, 0.38), (3.9, 0.38), arrowstyle="-|>", mutation_scale=12, color=GRY, lw=1.4))
save(fig, "fig_mechanism", tight=False)

# ======================================================================  2. UV dose: test hours vs Florida sun (6.4 x 3.35 in)
FL = 0.011 * 280e3                          # kJ/m2/nm at 340 nm per South Florida year = 3,080
g154 = 0.89 * 3.6 * 8 / 12                  # kJ/m2/nm per test hour, cycle 1: 8 h light in every 12 h
xen = 0.55 * 3.6                            # kJ/m2/nm per test hour, continuous light
h = np.linspace(0, 2200, 200)
fig, ax = plt.subplots(figsize=(6.4, 3.35))
ax.plot(h, g154 * h, color=ORG, lw=2.4, label="ASTM G154 cycle 1: UVA-340 at 0.89 W/m²/nm, 8 h light in 12 h")
ax.plot(h, xen * h, color=P, lw=2.0, ls=(0, (5, 2)), label="Xenon arc at 0.55 W/m²/nm, continuous light")
ax.axhline(FL, color=GRY, lw=0.9, ls=(0, (4, 3)))
ax.plot([100], [g154 * 100], "o", color=GOLD, ms=8, mec=ORG, zorder=5)
ax.annotate("Our read-out: 100 h = 214 kJ/m²/nm ≈ 25 days of Florida sun", (100, g154 * 100), xytext=(420, 260), va="center", fontsize=10.5, color=ORG, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=ORG, lw=1, shrinkA=2, shrinkB=4))
for hh, c in [(FL / g154, ORG), (FL / xen, P)]: ax.plot([hh], [FL], "o", color=c, ms=6, zorder=5)
ax.text(60, FL + 70, "1 South Florida year = 3,080 kJ/m²/nm", fontsize=10, color=GRY, ha="left", va="bottom")
ax.text(1640, FL - 260, "1 Florida year =\n1,442 h (G154)\n1,556 h (xenon)", fontsize=10, color=DK, va="top", ha="left", linespacing=1.3)
ax.annotate("2,000 h screen ≈ 1.3–1.4 Florida years", (2000, (g154 + xen) / 2 * 2000), xytext=(60, 3720), fontsize=10.5, color=P, fontweight="bold", ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=P, lw=1, shrinkA=2, shrinkB=4))
ax.set_xlim(0, 2200); ax.set_ylim(0, 4900); ax.set_xlabel("Hours in the test chamber"); ax.set_ylabel("UV dose at 340 nm (kJ/m²/nm)")
ax.yaxis.set_major_formatter(fmt); ax.xaxis.set_major_formatter(fmt)
ax.legend(handles=ax.get_lines()[:2], labels=["ASTM G154 cycle 1 (UVA-340): 8 h light in every 12 h", "Xenon arc: continuous light"], loc="upper left", fontsize=10, handlelength=2.2, borderaxespad=0.1)
save(fig, "fig_dose")
print("dose check: G154 100 h =", round(g154 * 100, 1), "kJ ->", round(g154 * 100 / FL * 365, 1), "days; 1 yr:", round(FL / g154), "h (G154),", round(FL / xen), "h (xenon); 2000 h:",
      round(g154 * 2000 / FL, 2), round(xen * 2000 / FL, 2), "yr")

# ======================================================================  3. ATR probe depth (6.4 x 3.35 in)
nu = np.linspace(650, 4000, 600); lam_um = 1e4 / nu; th = np.radians(45); n2 = 1.5
dp = lambda n1, v=nu: (1e4 / v) / (2 * np.pi * n1 * np.sqrt(np.sin(th) ** 2 - (n2 / n1) ** 2))
fig, ax = plt.subplots(figsize=(6.4, 3.35))
for lo, hi, lab in [(3100, 3650, "O–H"), (2840, 3000, "C–H"), (1690, 1800, "C=O")]:
    ax.axvspan(lo, hi, color=TINT, lw=0); ax.text((lo + hi) / 2, 3.78, lab, ha="center", fontsize=10.5, color=P, fontweight="bold")
ax.plot(nu, dp(2.4), color=P, lw=2.4, label="Diamond crystal (n = 2.4)")
ax.plot(nu, dp(4.0), color=GOLD, lw=2.4, label="Germanium crystal (n = 4.0)")
for v in (3400, 1730):
    d = float(dp(2.4, v)); ax.plot([v], [d], "o", color=P, ms=6)
    ax.annotate(f"{d:.2f} µm", (v, d), textcoords="offset points", xytext=(-6, 10), ha="right", fontsize=10.5, color=P, fontweight="bold")
ax.text(2300, 2.1, "the failing skin\nis about 1–2 µm thick", fontsize=10, color=GRY, ha="center")
ax.set_xlim(4000, 650); ax.set_ylim(0, 4.1); ax.set_xlabel("Wavenumber (cm$^{-1}$)"); ax.set_ylabel("Depth the infrared reads (µm)")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=10)
save(fig, "fig_atr_depth")

# ======================================================================  4. Simulation diagnostics (4.05 x 2.72 in each)
yT, yh, q = 10 ** Z["yT"], 10 ** Z["yhT"], float(Z["q"])
fig, ax = plt.subplots(figsize=(4.05, 2.72))
lim = [600, 12000]; xx = np.array(lim)
ax.fill_between(xx, xx / 10 ** q, xx * 10 ** q, color=TINT, lw=0, label="90% range"); ax.plot(xx, xx, color=GRY, lw=0.8)
ax.scatter(yT, yh, s=14, color=P, alpha=0.85, lw=0, zorder=3, label="Unseen recipe")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
for a in (ax.xaxis, ax.yaxis): a.set_major_formatter(fmt); a.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([1000, 2000, 5000, 10000]); ax.set_yticks([1000, 2000, 5000, 10000])
ax.set_xlabel("Actual hours to failure"); ax.set_ylabel("Forecast from 100 h"); ax.legend(fontsize=9.5, loc="upper left", handletextpad=0.3); tag(ax)
save(fig, "fig_sim_parity")
W = R["window"]; T = [int(k) for k in W]
fig, ax = plt.subplots(figsize=(4.05, 2.72))
ax.plot(T, [W[str(k)]["gloss"][0] for k in T], "o-", color=LAV, lw=1.8, ms=5, label="Early gloss")
ax.plot(T, [W[str(k)]["ci"][0] for k in T], "o-", color=GOLD, lw=1.8, ms=5, label="Carbonyl index")
ax.plot(T, [W[str(k)]["full"][0] for k in T], "o-", color=P, lw=2.4, ms=6, label="Our model")
ax.set_xscale("log"); ax.set_xticks(T); ax.set_xticklabels([str(k) for k in T]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_ylim(0, 38)
ax.set_xlabel("Hours of exposure used"); ax.set_ylabel("Median error (%)"); ax.legend(fontsize=9.5, loc="lower left", ncol=1, handlelength=1.6); tag(ax)
save(fig, "fig_sim_window")
L = R["learning"]; n = [int(k) for k in L]; mu = np.array([L[str(k)][0] for k in n]); sd = np.array([L[str(k)][1] for k in n])
fig, ax = plt.subplots(figsize=(4.05, 2.72)); ax.fill_between(n, mu - sd, mu + sd, color=TINT, lw=0); ax.plot(n, mu, "o-", color=P, lw=2.4, ms=6)
ax.axvspan(60, 80, color=GOLD, alpha=0.25, lw=0); ax.text(70, 20.2, "plan:\n60–80", ha="center", va="top", fontsize=10, color=ORG, fontweight="bold")
ax.set_ylim(0, 22); ax.set_xlabel("Recipes used to train the model"); ax.set_ylabel("Median error (%)"); tag(ax)
save(fig, "fig_sim_learning")

# ======================================================================  5. Encode: difference spectra at 24/48/100 h (3.87 x 1.46 in, slide 3)
NUe = np.arange(4000.0, 648.0, -2.0)
lor = lambda c, w: (0.5 * w) ** 2 / ((NUe - c) ** 2 + (0.5 * w) ** 2)
gau = lambda c, w: np.exp(-4 * np.log(2) * (NUe - c) ** 2 / w ** 2)
PROD = 0.5 * (0.5 * lor(1708, 30) + 0.15 * lor(1640, 30)) + 0.5 * (0.35 * gau(3400, 300) + 0.30 * lor(1778, 28)) - 0.3 * (lor(2955, 20) * .18 + lor(2930, 25) * .14 + lor(2873, 20) * .10)
def X_of_t(k, s, tmax):
    t = np.arange(0, tmax + 1.0); X = np.zeros_like(t)
    for j in range(1, len(t)): X[j] = X[j - 1] + k * (1 + 3 * X[j - 1]) / (1 + 4 * s * np.exp(-t[j - 1] / 400))
    return t, X
_, Xe = X_of_t(2.6e-4, 0.3, 120)
fig = plt.figure(figsize=(2.76, 1.35)); ax = fig.add_axes([0.09, 0.3, 0.89, 0.55])
for T_, c, lw in [(24, LAV, 1.4), (48, MID, 1.6), (100, P, 2.2)]:
    ax.plot(NUe, Xe[T_] * PROD * 1e3, color=c, lw=lw, label=f"{T_} h")
ymin = (Xe[100] * PROD * 1e3).min()
for lo, hi in [(3100, 3650), (2840, 3000), (1690, 1800)]: ax.axvspan(lo, hi, color=GOLD, alpha=0.18, lw=0)
ax.text(3375, 1.0, "O–H rises", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9.5, color=ORG, fontweight="bold")
ax.text(1745, 1.0, "C=O rises", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9.5, color=ORG, fontweight="bold")
ax.text(2790, ymin * 0.9, "C–H falls", ha="left", va="center", fontsize=9.5, color=ORG, fontweight="bold")
ax.axhline(0, color=GRY, lw=0.6)
ax.set_xlim(4000, 650); ax.set_xticks([4000, 3000, 2000, 1000]); ax.tick_params(labelsize=9, pad=2)
ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=9.5, labelpad=1); ax.set_ylabel("change", fontsize=9)
ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(0.4, 0.58), handlelength=1.0, labelspacing=0.15, borderaxespad=0)
ax.text(1.0, 0.07, "SIMULATED", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=ORG, fontweight="bold")
save(fig, "fig_encode", tight=False)

# ======================================================================  6. Predict: forecast vs actual with the 90% range (2.76 x 1.35 in, slide 4)
fig = plt.figure(figsize=(2.76, 1.35)); ax = fig.add_axes([0.2, 0.3, 0.78, 0.62])
ax.fill_between(xx, xx / 10 ** q, xx * 10 ** q, color=TINT, lw=0, label="90% range"); ax.plot(xx, xx, color=GRY, lw=0.8)
ax.scatter(yT, yh, s=9, color=P, alpha=0.85, lw=0, zorder=3)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
for a_ in (ax.xaxis, ax.yaxis): a_.set_major_formatter(fmt); a_.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([1000, 2000, 5000]); ax.set_yticks([1000, 5000]); ax.tick_params(labelsize=9, pad=2)
ax.set_xlabel("Actual hours to failure", fontsize=9.5, labelpad=1); ax.set_ylabel("Forecast (h)", fontsize=9.5, labelpad=1)
ax.legend(fontsize=9, loc="upper left", handlelength=1.0, borderaxespad=0.1)
ax.text(1.0, 0.04, "SIMULATED", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=ORG, fontweight="bold")
save(fig, "fig_predict", tight=False)
