"""
Modelling on open data for the First 100 Hours deck.
  A. Real ATR-FTIR weathering spectra (Kyoto University, Zenodo 10.5281/zenodo.22971354, CC BY 4.0)
  B. India climate layer from NASA POWER climatology (open satellite / reanalysis data)
  C. Decision analysis on the held-out simulated recipes
usage: python3 open_model.py <sim_out_dir> <open_data_dir> <fig_dir>
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

SIM, OD, FIG = sys.argv[1], sys.argv[2], sys.argv[3]
P, GOLD, ORG, MID, LAV, DK, GRY, TINT, GRN = "#502B8B", "#F2A01E", "#C8601A", "#7A5BB5", "#B9A6E0", "#241A3D", "#6B6480", "#F1ECF9", "#2E9E5B"
fam = "TeX Gyre Adventor" if any("Adventor" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": fam, "font.size": 10.5, "axes.edgecolor": GRY, "axes.labelcolor": DK, "text.color": DK, "xtick.color": DK, "ytick.color": DK,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.labelsize": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "mathtext.fontset": "custom", "mathtext.rm": fam, "mathtext.it": fam + ":italic", "mathtext.bf": fam + ":bold", "savefig.dpi": 260, "legend.frameon": False})
def save(fig, name): fig.savefig(f"{FIG}/{name}.png", facecolor="white", bbox_inches="tight", pad_inches=0.03); plt.close(fig); print("fig", name)
OUT = {}

# ====================================================================== A. real spectra
Z = np.load(f"{OD}/kyoto_spectra_reduced.npz"); wn = Z["wn"]
def area(a, lo, hi, b1, b2, moment=False):
    m = (wn >= lo) & (wn <= hi); x = wn[m]; y = a[:, m]
    y1 = a[:, (wn >= b1[0]) & (wn <= b1[1])].mean(1); y2 = a[:, (wn >= b2[0]) & (wn <= b2[1])].mean(1); x1, x2 = np.mean(b1), np.mean(b2)
    yb = (y - (y1[:, None] + (y2 - y1)[:, None] * (x - x1) / (x2 - x1))) * np.abs(np.gradient(x))
    if not moment: return yb.sum(1)
    w = np.clip(yb, 0, None); c = (w * x).sum(1) / w.sum(1); return (w * (x - c[:, None]) ** 2).sum(1) / w.sum(1)
CB = ((1850, 1880), (1560, 1590))
def feats(a):
    oh = area(a, 3100, 3650, (3660, 3700), (3030, 3060)); ch = area(a, 2840, 3000, (3010, 3040), (2760, 2800)); co = area(a, 1650, 1850, *CB)
    acid = area(a, 1690, 1720, *CB); est = area(a, 1720, 1750, *CB); lac = area(a, 1760, 1800, *CB); cc = area(a, 1620, 1660, *CB); var = area(a, 1650, 1850, *CB, moment=True)
    F = np.column_stack([oh / co, ch / co, lac / co, acid / est, cc / co, np.log(var)])
    return F, co / ch, acid / est                      # six band features, classical carbonyl index (C=O / C-H), acid-to-ester ratio
NAMES = ["O–H / C=O", "C–H / C=O", "lactone / C=O", "acid / ester", "C=C / C=O", "carbonyl spread"]
DATA = {}
for coat, lab in (("SFC", "acrylic, semi-film-forming"), ("PC", "alkyd, penetrating")):
    for irr in (60, 120):
        a = Z[f"{coat}_{irr}W_A"]; t = Z[f"{coat}_{irr}W_t"]; F, ci, ae = feats(a)
        DATA[(coat, irr)] = dict(F=F, ci=ci, ae=ae, t=t, dose=irr * t / 1000.0)   # dose in kWh/m2 (xenon, 300-400 nm band as stated by the dataset)
R = {"source": "Teramoto, Y. (Kyoto University). Data and code for GAWNSPLS analysis of accelerated weathering spectra of wood coatings. Zenodo, doi:10.5281/zenodo.22971354 (2026), CC BY 4.0",
     "spectra": {f"{c}_{i}W": int(len(v["t"])) for (c, i), v in DATA.items()}, "replicates_per_time": 10}
# A1 repeatability and the 100 h signal
for coat in ("SFC", "PC"):
    d = DATA[(coat, 60)]; t = d["t"]; r = {}
    for key, lab in (("ae", "acid_ester"), ("ci", "carbonyl_index")):
        v = d[key]; rs = [float(np.std(v[t == tt], ddof=1) / abs(np.mean(v[t == tt])) * 100) for tt in sorted(set(t)) if tt <= 500]
        m0, m1 = v[t == 0], v[t == 100]; sp = np.sqrt((m0.var(ddof=1) + m1.var(ddof=1)) / 2)
        r[lab] = dict(rel_sd_pct_median=float(np.median(rs)), at0=float(m0.mean()), at100=float(m1.mean()), change_pct=float((m1.mean() / m0.mean() - 1) * 100), signal_to_noise=float(abs(m1.mean() - m0.mean()) / sp))
    R[f"{coat}_first100h_60W"] = r
# A2 dose collapse: does the 120 W/m2 series at time t match the 60 W/m2 series at 2t?
for coat in ("SFC", "PC"):
    a, b = DATA[(coat, 60)], DATA[(coat, 120)]; res = {}
    for key, lab in (("ae", "acid_ester"), ("ci", "carbonyl_index")):
        t60 = np.array(sorted(set(a["t"]))); m60 = np.array([a[key][a["t"] == tt].mean() for tt in t60])
        t120 = np.array([tt for tt in sorted(set(b["t"])) if tt > 0]); m120 = np.array([b[key][b["t"] == tt].mean() for tt in t120])
        pred = np.interp(2 * t120, t60, m60); res[lab] = dict(times_120W=t120.tolist(), value_120W=m120.round(4).tolist(), value_60W_at_double_time=pred.round(4).tolist(), mean_abs_diff_pct=float(np.mean(np.abs(m120 / pred - 1)) * 100))
    R[f"{coat}_dose_collapse"] = res
# A3 dose clock: train on one irradiance, predict the UV dose behind each spectrum of the other
def clock(coat, tr, te, tmax=None):
    a, b = DATA[(coat, tr)], DATA[(coat, te)]
    ma = np.ones(len(a["t"]), bool) if tmax is None else a["t"] <= tmax; mb = (b["dose"] <= a["dose"][ma].max()) & (np.ones(len(b["t"]), bool) if tmax is None else b["t"] <= tmax)
    out = {}
    for lab, Xa, Xb in (("six_bands", a["F"], b["F"]), ("carbonyl_index", a["ci"][:, None], b["ci"][:, None])):
        mdl = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
        mdl.fit(Xa[ma], np.sqrt(a["dose"][ma])); p = np.clip(mdl.predict(Xb[mb]), 0, None) ** 2; y = b["dose"][mb]
        out[lab] = dict(r2=float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()), rmse_kwh=float(np.sqrt(np.mean((y - p) ** 2))), n_test=int(mb.sum()), pred=p, y=y)
    return out
CL = {}
for coat in ("SFC", "PC"):
    for tr, te in ((60, 120), (120, 60)):
        c = clock(coat, tr, te, tmax=500 if coat == "PC" else None); CL[(coat, tr, te)] = c
        R[f"{coat}_dose_clock_train{tr}_test{te}"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("pred", "y")} for k, v in c.items()}
OUT["real_spectra"] = R

# figures: dose collapse and dose clock (acrylic)
fig, ax = plt.subplots(figsize=(4.05, 2.72))
for irr, c, mk in ((60, P, "o"), (120, GOLD, "s")):
    d = DATA[("SFC", irr)]; ts = sorted(set(d["t"])); x = [irr * tt / 1000 for tt in ts]; m = [d["ae"][d["t"] == tt].mean() for tt in ts]; s = [d["ae"][d["t"] == tt].std(ddof=1) for tt in ts]
    ax.errorbar(x, m, yerr=s, fmt=mk + "-", color=c, ms=5, lw=1.6, capsize=2, label=f"{irr} W/m²", zorder=3 if irr == 120 else 2)
ax.set_xlabel("UV dose (kWh/m²)"); ax.set_ylabel("Acid / ester band ratio"); ax.legend(fontsize=9.5, loc="lower right", title="Lamp intensity", title_fontsize=9.5)
ax.annotate("first 100 h", (6, DATA[("SFC", 60)]["ae"][DATA[("SFC", 60)]["t"] == 100].mean()), xytext=(14, 0.70), fontsize=9.5, color=ORG, fontweight="bold", arrowprops=dict(arrowstyle="-", color=ORG, lw=1))
ax.text(1.0, 1.01, "REAL DATA", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=GRN, fontweight="bold")
save(fig, "fig_real_dose")
fig, ax = plt.subplots(figsize=(4.05, 2.72)); c = CL[("SFC", 60, 120)]; rng = np.random.default_rng(1)
for lab, col, nm in (("carbonyl_index", GOLD, "Carbonyl index"), ("six_bands", P, "Our six bands")):
    ax.scatter(c[lab]["y"] + rng.normal(0, 0.6, len(c[lab]["y"])), c[lab]["pred"], s=13, color=col, alpha=0.8, lw=0, label=f"{nm} (R² {c[lab]['r2']:.2f})", zorder=3 if lab == "six_bands" else 2)
lim = [-3, max(66, float(max(c["six_bands"]["pred"].max(), c["carbonyl_index"]["pred"].max())) + 3)]; ax.plot(lim, lim, color=GRY, lw=0.8); ax.set_xlim(-3, 66); ax.set_ylim(lim)
ax.set_xlabel("Actual UV dose (kWh/m²)"); ax.set_ylabel("Dose read from spectrum"); ax.legend(fontsize=9.5, loc="upper left", handletextpad=0.2, borderaxespad=0.1)
ax.text(1.0, 1.01, "REAL DATA", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=GRN, fontweight="bold")
save(fig, "fig_real_clock")
# compact versions of both charts for main slide 4 (shown at native size)
with plt.rc_context({"font.size": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.labelsize": 9}):
    fig, ax = plt.subplots(figsize=(2.9, 1.72))
    for irr, c_, mk in ((60, P, "o"), (120, GOLD, "s")):
        d = DATA[("SFC", irr)]; ts = sorted(set(d["t"])); x = [irr * tt / 1000 for tt in ts]; m = [d["ae"][d["t"] == tt].mean() for tt in ts]; s_ = [d["ae"][d["t"] == tt].std(ddof=1) for tt in ts]
        ax.errorbar(x, m, yerr=s_, fmt=mk + "-", color=c_, ms=3.5, lw=1.3, capsize=1.5, elinewidth=0.8, label=f"{irr} W/m² lamp", zorder=3 if irr == 120 else 2)
    ax.set_xlabel("UV dose (kWh/m²)", labelpad=1); ax.set_ylabel("Acid / ester ratio", labelpad=1); ax.legend(fontsize=8.5, loc="lower right", handletextpad=0.3, borderaxespad=0.1, labelspacing=0.2)
    ax.annotate("first 100 h", (6, DATA[("SFC", 60)]["ae"][DATA[("SFC", 60)]["t"] == 100].mean()), xytext=(16, 0.69), fontsize=8.5, color=ORG, fontweight="bold", arrowprops=dict(arrowstyle="-", color=ORG, lw=0.8))
    ax.text(1.0, 1.02, "REAL DATA", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=GRN, fontweight="bold")
    save(fig, "fig_real_dose_c")
    fig, ax = plt.subplots(figsize=(2.9, 1.72)); c = CL[("SFC", 60, 120)]; rng = np.random.default_rng(1)
    for lab, col, nm in (("carbonyl_index", GOLD, "Carbonyl index"), ("six_bands", P, "Our six bands")):
        ax.scatter(c[lab]["y"] + rng.normal(0, 0.6, len(c[lab]["y"])), c[lab]["pred"], s=7, color=col, alpha=0.8, lw=0, label=f"{nm}: R² {c[lab]['r2']:.2f}", zorder=3 if lab == "six_bands" else 2)
    lim = [-3, max(66, float(max(c["six_bands"]["pred"].max(), c["carbonyl_index"]["pred"].max())) + 3)]; ax.plot(lim, lim, color=GRY, lw=0.8); ax.set_xlim(-3, 66); ax.set_ylim(lim)
    ax.set_xlabel("Actual UV dose (kWh/m²)", labelpad=1); ax.set_ylabel("Dose read from spectrum", labelpad=1); ax.legend(fontsize=8.5, loc="upper left", handletextpad=0.1, borderaxespad=0.0, labelspacing=0.2, markerscale=1.6)
    ax.text(1.0, 1.02, "REAL DATA", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=GRN, fontweight="bold")
    save(fig, "fig_real_clock_c")

# real difference spectra, acrylic at 60 W/m2
fig, ax = plt.subplots(figsize=(4.05, 2.72)); A = Z["SFC_60W_A"]; t = Z["SFC_60W_t"]; m = (wn <= 1900) & (wn >= 1500)
ref = area(A, 1720, 1750, *CB); An = A / ref[:, None]; base = An[t == 0].mean(0)
for tt, col, lw in ((100, LAV, 1.4), (500, MID, 1.7), (1500, P, 2.2)): ax.plot(wn[m], (An[t == tt].mean(0) - base)[m] * 100, color=col, lw=lw, label=f"{tt} h")
ax.axhline(0, color=GRY, lw=0.6); ax.axvspan(1690, 1720, color=GOLD, alpha=0.2, lw=0); ax.text(1705, 1.0, "acid C=O", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9.5, color=ORG, fontweight="bold")
ax.set_xlim(1900, 1500); ax.set_xlabel("Wavenumber (cm$^{-1}$)"); ax.set_ylabel("Change vs 0 h"); ax.set_yticks([]); ax.legend(fontsize=9.5, loc="upper left")
ax.text(1.0, 1.01, "REAL DATA", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=GRN, fontweight="bold")
save(fig, "fig_real_spectra")

# ====================================================================== B. India climate layer
C = json.load(open(f"{OD}/nasa_power_climatology.json")); DAYS = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]); RGAS = 8.314
def city(name):
    c = C[name]; uva = np.array(c["ALLSKY_SFC_UVA"][:12]) * DAYS; T = np.array(c["T2M"][:12]) + 273.15; rh = np.array(c["RH2M"][:12])
    return dict(uva=uva, T=T, rh=rh, annual_uva=float(uva.sum()), t_mean=float(c["T2M"][12]), rh_mean=float(c["RH2M"][12]), rain_mm=float(c["PRECTOTCORR"][12] * 365.25))
ref = city("Miami_Florida"); EA = (9e3, 21e3)          # J/mol: published range for weathering of engineering thermoplastics (Pickett et al.), used as a sensitivity band only
def twd(c, ea): return float((c["uva"] * np.exp(-ea / RGAS * (1 / c["T"] - 1 / 298.15))).sum())
G154_PER_FL_YEAR = 3080.0 / (0.89 * 3.6 * 8 / 12)     # chamber hours of ASTM G154 cycle 1 that deliver one South Florida year of 340 nm dose
IND = {}
for nm in ("Cochin", "Mahabaleshwar", "Mumbai", "Chennai", "Bengaluru", "Kolkata", "Delhi", "Jodhpur", "Miami_Florida"):
    c = city(nm); rel = c["annual_uva"] / ref["annual_uva"]; tw = [twd(c, e) / twd(ref, e) for e in EA]
    IND[nm] = dict(annual_uva_kwh_m2=round(c["annual_uva"], 1), uv_dose_vs_florida=round(rel, 3), temp_weighted_vs_florida=[round(v, 3) for v in tw], humid_dose_share=round(float(c["uva"][c["rh"] >= 75].sum() / c["uva"].sum()), 3),
                   mean_temp_c=c["t_mean"], mean_rh_pct=c["rh_mean"], rain_mm_per_year=round(c["rain_mm"]), g154_hours_per_year_uv_only=round(rel * G154_PER_FL_YEAR))
OUT["india_climate"] = dict(source=C["_source"], humid_month_rule="relative humidity of the month >= 75%", activation_energy_band_kj_mol=[9, 21], g154_hours_per_florida_year=round(G154_PER_FL_YEAR), cities=IND)
fig, ax = plt.subplots(figsize=(6.4, 3.35))
ax.axvline(1.0, color=GRY, lw=0.9, ls=(0, (4, 3))); ax.axhline(100 * 4 / 12, color=GOLD, lw=1.2, ls=(0, (4, 3)))
ax.text(0.853, 100 * 4 / 12 - 2, "ASTM G154 cycle 1:\nwet 33% of the time", ha="left", va="top", fontsize=9.5, color=ORG, linespacing=1.2)
OFF = {"Cochin": (6, 5), "Mahabaleshwar": (-6, 6), "Mumbai": (-6, -13), "Chennai": (4, -13), "Bengaluru": (6, 5), "Kolkata": (6, 4), "Delhi": (6, 4), "Jodhpur": (6, 4), "Miami_Florida": (6, -12)}
for nm, v in IND.items():
    fl = nm == "Miami_Florida"; x = v["uv_dose_vs_florida"]; y = v["humid_dose_share"] * 100
    ax.scatter([x], [y], s=70 if not fl else 90, color=GOLD if fl else P, marker="D" if fl else "o", edgecolor="white", lw=1, zorder=3)
    ax.annotate("South Florida benchmark" if fl else nm, (x, y), textcoords="offset points", xytext=OFF[nm], fontsize=10, color=ORG if fl else DK, fontweight="bold", ha="right" if OFF[nm][0] < 0 else "left")
ax.set_xlim(0.85, 1.1); ax.set_ylim(-5, 105); ax.set_xlabel("Yearly UV-A dose, relative to South Florida (= 1.0)"); ax.set_ylabel("Share of UV dose arriving\nin humid months (%)")
ax.text(1.0, 1.01, "OPEN NASA DATA", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=GRN, fontweight="bold")
save(fig, "fig_india")

# ====================================================================== C. decision analysis on held-out simulated recipes
S = np.load(f"{SIM}/sim_arrays.npz", allow_pickle=True); yT, yh, q = 10 ** S["yT"], 10 ** S["yhT"], float(S["q"]); lo, hi = yh / 10 ** q, yh * 10 ** q
DEC = {}
for tgt in (1500, 2000, 2500, 3000):
    adv, drop = lo > tgt, hi < tgt; full = ~(adv | drop)
    DEC[str(tgt)] = dict(n=int(len(yT)), advance=int(adv.sum()), drop=int(drop.sum()), full_test=int(full.sum()), verdict_at_100h_pct=float((adv | drop).mean() * 100),
                         wrong_drops=int((drop & (yT >= tgt)).sum()), wrong_advances=int((adv & (yT < tgt)).sum()),
                         chamber_hours_saved_pct=float(((adv | drop) * (2000 - 100)).sum() / (len(yT) * 2000) * 100))
OUT["decision_analysis"] = dict(note="80 held-out simulated recipes; rule: advance if the whole 90% range is above the target, drop if all of it is below, else full test", by_target_hours=DEC)
json.dump(OUT, open(f"{SIM}/open_model_results.json", "w"), indent=1)
r = OUT["real_spectra"]
for k in ("SFC_first100h_60W", "PC_first100h_60W", "SFC_dose_collapse", "PC_dose_collapse"): print(k, json.dumps({a: {b: (round(c, 2) if isinstance(c, float) else c) for b, c in v.items() if "value" not in b and "times" not in b} for a, v in r[k].items()}))
for k in r:
    if "clock" in k: print(k, {a: (round(v["r2"], 3), round(v["rmse_kwh"], 1), v["n_test"]) for a, v in r[k].items()})
for nm, v in IND.items(): print(f"{nm:15s} UVA {v['annual_uva_kwh_m2']:6.1f} rel {v['uv_dose_vs_florida']:.3f} Tw {v['temp_weighted_vs_florida']} humid {v['humid_dose_share']:.2f} T {v['mean_temp_c']} RH {v['mean_rh_pct']} G154h {v['g154_hours_per_year_uv_only']}")
for k, v in DEC.items(): print(k, v)
