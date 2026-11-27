"""
First 100 Hours - pipeline dry run on SIMULATED spectra.

Purpose: check that the feature pipeline and models recover hours-to-failure when
the assumed early-chemistry link exists, and size the experiment (how many
formulations, how much spectral noise, how long a window).
This is NOT coating data. Every kinetic setting below is a simulation setting.

usage: python3 sim.py <outdir>
"""
import sys, json
import numpy as np
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedKFold

OUT = sys.argv[1]
RNG = np.random.default_rng(20261115)
NU = np.arange(4000.0, 648.0, -2.0)           # wavenumber grid, cm-1
PULLS = [24, 48, 100, 200, 500]               # candidate pull times, h

def lor(c, w, h=1.0):  return h * (0.5 * w) ** 2 / ((NU - c) ** 2 + (0.5 * w) ** 2)
def gau(c, w, h=1.0):  return h * np.exp(-4 * np.log(2) * (NU - c) ** 2 / w ** 2)

# ---- band library (positions follow published acrylic / styrene-acrylic assignments)
BASE = [(2955, 20, .18), (2930, 25, .14), (2873, 20, .10), (1728, 22, .75), (1450, 25, .20),
        (1385, 20, .12), (1238, 35, .35), (1160, 40, .55), (1145, 30, .50), (1065, 30, .20),
        (990, 25, .12), (842, 20, .10), (750, 20, .10)]
STY = [(3026, 15, .06), (1602, 12, .08), (1493, 10, .12), (760, 14, .25), (700, 12, .45)]
B_OH = gau(3400, 300)            # hydroperoxide / alcohol / acid O-H
B_ACID = lor(1708, 30)           # carboxylic acid, ketone C=O
B_LAC = lor(1778, 28)            # gamma-lactone / perester C=O
B_CC = lor(1640, 30)             # C=C from Norrish II
B_CH = sum(lor(c, w, h) for c, w, h in BASE[:3])   # C-H stretch (lost)

def make_population(n):
    """Latent formulation properties. All distributions are simulation settings."""
    return dict(
        k=2.0e-4 * np.exp(RNG.normal(0, 0.6, n)),      # initiation rate, 1/h
        s=RNG.uniform(0, 1, n),                         # stabiliser loading (relative)
        p=RNG.uniform(0.2, 0.8, n),                     # scission-pathway fraction
        zTg=RNG.normal(0, 1, n),                        # binder Tg descriptor (standardised)
        sty=RNG.uniform(0, 0.4, n),                     # styrene fraction
        eps=RNG.normal(0, 0.12, n),                     # unexplained film-toughness scatter
        hv=np.exp(RNG.normal(0, 0.05, (n, len(BASE)))), # band-height variation
        G0=RNG.normal(80, 5, n),                        # initial 60-degree gloss, GU
        dnoise=np.exp(RNG.normal(0, 0.10, n)),          # 10% error on the recorded stabiliser loading
    )

A_AUTO, K_STAB, TAU_STAB, M_GLOSS = 3.0, 4.0, 400.0, 4.0

def kinetics(pop, tmax=40000):
    """dX/dt = k (1 + a X) / (1 + K s exp(-t/tau)); failure when w X = Xc (gloss 50%)."""
    n = len(pop["k"]); t = np.arange(0, tmax + 1.0)
    X = np.zeros((n, len(t)))
    for j in range(1, len(t)):
        x = X[:, j - 1]
        X[:, j] = x + pop["k"] * (1 + A_AUTO * x) / (1 + K_STAB * pop["s"] * np.exp(-t[j - 1] / TAU_STAB))
    w = 0.5 + pop["p"]
    Xc = 0.5 * np.exp(0.3 * pop["zTg"] + pop["eps"])
    ratio = (w[:, None] * X) / Xc[:, None]
    tf = np.array([t[np.argmax(r >= 1.0)] if (r >= 1.0).any() else np.nan for r in ratio])
    return t, X, ratio, tf

def true_spectrum(pop, i, x):
    a = sum(lor(c, w, h * pop["hv"][i, j]) for j, (c, w, h) in enumerate(BASE))
    a = a + pop["sty"][i] * sum(lor(c, w, h) for c, w, h in STY)
    p = pop["p"][i]
    prod = p * (0.50 * B_ACID + 0.15 * B_CC) + (1 - p) * (0.35 * B_OH + 0.30 * B_LAC)
    return a + x * prod - x * 0.30 * B_CH

def measure(a, sigma, nscan=3):
    """ATR measurement model: contact scaling, baseline offset and tilt, white noise; mean of nscan."""
    out = np.zeros_like(a)
    for _ in range(nscan):
        c = np.exp(RNG.normal(0, 0.03)); b0 = RNG.normal(0, 2e-3); b1 = RNG.normal(0, 2e-3)
        out += c * a + b0 + b1 * (NU - NU.mean()) / (NU.max() - NU.min()) + RNG.normal(0, sigma, a.shape)
    return out / nscan

M_FLAT1 = (NU > 3850) & (NU < 3950); M_FLAT2 = (NU > 2000) & (NU < 2400)
M_REF = (NU > 1425) & (NU < 1475)
def preprocess(a):
    """Two-point linear baseline, then normalise to the 1450 cm-1 CH2 bend area."""
    x1, x2 = NU[M_FLAT1].mean(), NU[M_FLAT2].mean(); y1, y2 = a[M_FLAT1].mean(), a[M_FLAT2].mean()
    a = a - (y1 + (y2 - y1) * (NU - x1) / (x2 - x1))
    return a / np.trapezoid(a[M_REF][::-1], NU[M_REF][::-1])

WIN = dict(OH=(3100, 3650), LAC=(1760, 1800), ACID=(1690, 1720), CC=(1620, 1660), CH=(2840, 3000))
def band(d, lo, hi):
    m = (NU >= lo) & (NU <= hi); return np.trapezoid(d[m][::-1], NU[m][::-1])

def simulate(pop, sigma=1e-3):
    t, X, ratio, tf = kinetics(pop)
    n = len(tf); spec = {}
    for T in [0] + PULLS:
        spec[T] = np.array([preprocess(measure(true_spectrum(pop, i, X[i, T]), sigma)) for i in range(n)])
    gloss = {T: pop["G0"] / (1 + ratio[:, T] ** M_GLOSS) + RNG.normal(0, 0.8, n) for T in [0] + PULLS}
    return dict(t=t, X=X, tf=tf, spec=spec, gloss=gloss)

def features(sim, pop, T):
    """Early-signature features using all pulls <= T."""
    pulls = [q for q in PULLS if q <= T]
    D = {q: sim["spec"][q] - sim["spec"][0] for q in pulls}
    cols, names = [], []
    tt = np.array(pulls, float)
    for nm, (lo, hi) in WIN.items():
        B = np.array([[band(D[q][i], lo, hi) for q in pulls] for i in range(len(pop["k"]))])
        rate = (B * tt).sum(1) / (tt ** 2).sum()            # least-squares slope through origin, per hour
        cols.append(rate); names.append("rate_" + nm)
    m = (NU >= 1600) & (NU <= 1850)
    cols.append(np.log10(D[pulls[-1]][:, m].var(1))); names.append("logvar_carbonyl")
    F = np.column_stack(cols)
    sgn = np.sign(F[:, :5]); F[:, :5] = sgn * np.log10(1 + np.abs(F[:, :5]) / 1e-4)   # signed log scale
    ci = np.array([band(D[pulls[-1]][i], 1600, 1800) / abs(band(sim["spec"][0][i], 2800, 3050)) for i in range(len(pop["k"]))])
    gl = np.column_stack([sim["gloss"][q] - sim["gloss"][0] for q in pulls])
    desc = np.column_stack([pop["s"] * pop["dnoise"], pop["zTg"], pop["sty"]])
    return dict(F=F, names=names, ci=ci.reshape(-1, 1), gloss=gl, desc=desc, Dlast=D[pulls[-1]] / pulls[-1])

def enet():  return make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.2, .5, .8], cv=5, max_iter=20000))
def lin():   return make_pipeline(StandardScaler(), LinearRegression())

def mape(y, yh):  return float(np.median(np.abs(10 ** yh - 10 ** y) / 10 ** y) * 100)
def r2(y, yh):    return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())

def cv_eval(Xm, y, model_fn, seed=1):
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed)
    pred = np.zeros((5, len(y)))
    for j, (tr, te) in enumerate(rkf.split(Xm)):
        pred[j // 5, te] = model_fn().fit(Xm[tr], y[tr]).predict(Xm[te])
    yh = pred.mean(0)
    return mape(y, yh), r2(y, yh), yh

res = {}
# ------------------------------------------------------------------ main experiment
N = 120
pop = make_population(N); sim = simulate(pop)
ok = ~np.isnan(sim["tf"]); assert ok.all()
y = np.log10(sim["tf"])
res["tf_quantiles_h"] = [float(q) for q in np.percentile(sim["tf"], [5, 25, 50, 75, 95])]
res["X_at_100h_median"] = float(np.median(sim["X"][:, 100]))
res["gloss_loss_at_100h_true_max_GU"] = float(np.max(pop["G0"] - pop["G0"] / (1 + ((0.5 + pop["p"]) * sim["X"][:, 100] / (0.5 * np.exp(0.3 * pop["zTg"] + pop["eps"]))) ** M_GLOSS)))
res["frac_of_failure_extent_at_100h"] = [float(v) for v in np.percentile([sim["X"][i, 100] / sim["X"][i, int(sim["tf"][i])] for i in range(N)], [5, 50, 95])]
f = features(sim, pop, 100)
sets = {
    "Early gloss change only": (f["gloss"], lin),
    "Formulation descriptors only": (f["desc"], lin),
    "Single carbonyl index": (np.log10(np.clip(f["ci"], 1e-5, None)), lin),
    "Band features (spectra only)": (f["F"], enet),
    "Band features + descriptors": (np.column_stack([f["F"], f["desc"]]), enet),
}
res["models_100h"] = {}; preds = {}
for k_, (Xm, fn) in sets.items():
    m, r, yh = cv_eval(Xm, y, fn); res["models_100h"][k_] = dict(mdape=m, r2=r); preds[k_] = yh
res["mean_baseline_mdape"] = mape(y, np.full_like(y, y.mean()))
_m = enet().fit(np.column_stack([f["F"], f["desc"]]), y)
res["enet_coef"] = dict(zip(f["names"] + ["stabiliser_loading", "binder_Tg", "styrene_fraction"], [float(c) for c in _m[-1].coef_]))
res["enet_alpha_l1"] = [float(_m[-1].alpha_), float(_m[-1].l1_ratio_)]

# ------------------------------------------------------------------ window study
res["window"] = {}
for T in PULLS:
    fT = features(sim, pop, T)
    a = cv_eval(np.column_stack([fT["F"], fT["desc"]]), y, enet)[:2]
    b = cv_eval(np.log10(np.clip(fT["ci"], 1e-5, None)), y, lin)[:2]
    c = cv_eval(fT["gloss"], y, lin)[:2]
    res["window"][T] = dict(full=a, ci=b, gloss=c)

# ------------------------------------------------------------------ learning curve (fixed held-out test of 80)
popT = make_population(80); simT = simulate(popT); fT = features(simT, popT, 100)
XT = np.column_stack([fT["F"], fT["desc"]]); yT = np.log10(simT["tf"])
popP = make_population(200); simP = simulate(popP); fP = features(simP, popP, 100)
XP = np.column_stack([fP["F"], fP["desc"]]); yP = np.log10(simP["tf"])
res["learning"] = {}
for n in [15, 30, 60, 100, 150, 200]:
    errs = []
    for rep in range(12):
        idx = np.random.default_rng(rep).choice(200, n, replace=False)
        errs.append(mape(yT, enet().fit(XP[idx], yP[idx]).predict(XT)))
    res["learning"][n] = [float(np.mean(errs)), float(np.std(errs))]

# ------------------------------------------------------------------ split-conformal 90% interval (train 120, calibrate 80, test 80)
mdl = enet().fit(XP[:120], yP[:120])
q = np.quantile(np.abs(yP[120:] - mdl.predict(XP[120:])), np.ceil(0.9 * 81) / 80)
cover = float(np.mean(np.abs(yT - mdl.predict(XT)) <= q))
res["conformal"] = dict(target=0.90, coverage=cover, halfwidth_log10=float(q), factor=float(10 ** q))
yhT = mdl.predict(XT)

# ------------------------------------------------------------------ noise tolerance
res["noise"] = {}
for sg in [0.25e-3, 0.5e-3, 1e-3, 2e-3, 4e-3, 8e-3]:
    s2 = simulate(pop, sigma=sg); f2 = features(s2, pop, 100)
    res["noise"][sg] = dict(full=cv_eval(np.column_stack([f2["F"], f2["desc"]]), y, enet)[:2],
                            ci=cv_eval(np.log10(np.clip(f2["ci"], 1e-5, None)), y, lin)[:2])

# ------------------------------------------------------------------ PLS coefficient map on the full difference spectrum
pls = make_pipeline(StandardScaler(with_std=False), PLSRegression(n_components=4)).fit(f["Dlast"], y)
coef = pls[-1].coef_.ravel()
res["pls_cv"] = cv_eval(f["Dlast"], y, lambda: make_pipeline(StandardScaler(with_std=False), PLSRegression(n_components=4)))[:2]

json.dump(res, open(OUT + "/sim_results.json", "w"), indent=1)
i_short, i_long = int(np.argmin(sim["tf"])), int(np.argmax(sim["tf"]))
i_q25 = int(np.argsort(sim["tf"])[N // 5]); i_q75 = int(np.argsort(sim["tf"])[4 * N // 5])
np.savez(OUT + "/sim_arrays.npz", NU=NU, y=y, tf=sim["tf"], coef=coef, yT=yT, yhT=yhT, q=q,
         **{"pred_" + str(i): v for i, v in enumerate(preds.values())}, pred_names=np.array(list(preds.keys())),
         s0_a=sim["spec"][0][i_q25], s100_a=sim["spec"][100][i_q25], s0_b=sim["spec"][0][i_q75], s100_b=sim["spec"][100][i_q75],
         tf_ab=np.array([sim["tf"][i_q25], sim["tf"][i_q75]]),
         t=sim["t"][:6001], Xa=sim["X"][i_q25, :6001], Xb=sim["X"][i_q75, :6001],
         Ga=(pop["G0"][i_q25] / (1 + (((0.5 + pop["p"][i_q25]) * sim["X"][i_q25, :6001]) / (0.5 * np.exp(0.3 * pop["zTg"][i_q25] + pop["eps"][i_q25]))) ** M_GLOSS)) / pop["G0"][i_q25],
         Gb=(pop["G0"][i_q75] / (1 + (((0.5 + pop["p"][i_q75]) * sim["X"][i_q75, :6001]) / (0.5 * np.exp(0.3 * pop["zTg"][i_q75] + pop["eps"][i_q75]))) ** M_GLOSS)) / pop["G0"][i_q75])
print(json.dumps(res, indent=1))
