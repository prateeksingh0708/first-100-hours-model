"""
Ray-traced product scene for "The First 100 Hours" (numpy; no Blender in this environment).
Three stations on a lab bench: UV weathering chamber (existing), ATR-FTIR bench (existing), verdict software (new).
usage: python3 product_render.py <figdir> [view ...]     views: hero chamber bench screen
Schematic: equipment is simplified and not to scale.
"""
import sys, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

FIG = sys.argv[1]; WANT = [a for a in sys.argv[2:] if not a.isdigit()] or ["hero", "chamber", "bench", "screen", "rack"]; QUICK = any(a.isdigit() for a in sys.argv[2:])
unit = lambda v: np.asarray(v, float) / np.linalg.norm(v)
A3 = lambda *v: np.array(v, float)
INF = np.inf

# ------------------------------------------------------------------ dashboard texture for the laptop screen
def dashboard(w=900, h=600):
    im = Image.new("RGB", (w, h), (248, 246, 252)); d = ImageDraw.Draw(im)
    fb = font_manager.findfont(font_manager.FontProperties(family="TeX Gyre Adventor", weight="bold"))
    f1, f2 = ImageFont.truetype(fb, 40), ImageFont.truetype(fb, 30)
    d.rectangle([0, 0, w, 92], fill=(80, 43, 139)); d.text((28, 22), "THE FIRST 100 HOURS", font=f1, fill=(255, 255, 255))
    d.rounded_rectangle([w - 250, 20, w - 28, 72], 12, fill=(242, 160, 30)); d.text((w - 226, 28), "VERDICT", font=f2, fill=(36, 26, 61))
    rows = [("Recipe A", 0.62, 0.90, (46, 158, 91), "ADVANCE"), ("Recipe B", 0.36, 0.64, (242, 160, 30), "FULL TEST"), ("Recipe C", 0.10, 0.36, (210, 69, 59), "DROP"), ("Recipe D", None, None, (242, 160, 30), "FULL TEST")]
    x0, x1 = 235, 630
    d.line([(x0 + 0.47 * (x1 - x0), 112), (x0 + 0.47 * (x1 - x0), h - 30)], fill=(200, 96, 26), width=5)
    for i, (n, a, b, c, lab) in enumerate(rows):
        y = 170 + i * 108
        d.text((28, y - 20), n, font=f2, fill=(36, 26, 61))
        d.line([(x0, y), (x1, y)], fill=(217, 204, 240), width=6)
        if a is not None:
            d.rounded_rectangle([x0 + a * (x1 - x0), y - 17, x0 + b * (x1 - x0), y + 17], 10, fill=(185, 166, 224), outline=(80, 43, 139), width=3)
            xm = x0 + (a + b) / 2 * (x1 - x0); d.ellipse([xm - 17, y - 17, xm + 17, y + 17], fill=(80, 43, 139))
        d.rounded_rectangle([662, y - 28, w - 28, y + 28], 12, fill=c); tw = d.textlength(lab, font=f2); d.text((662 + (w - 28 - 662 - tw) / 2, y - 19), lab, font=f2, fill=(255, 255, 255))
    return np.asarray(im, float) / 255
TEX = dashboard()

# ------------------------------------------------------------------ primitives and materials
OBJ = []
def box_planes(lo, hi):
    lo, hi = A3(*lo), A3(*hi); pl = []
    for k in range(3):
        n = np.zeros(3); n[k] = 1; pl.append((n.copy(), hi[k])); pl.append((-n, -lo[k]))
    return pl
def obox_planes(c, half, R):
    c = A3(*c); pl = []
    for k in range(3):
        n = R[:, k]; pl.append((n, n @ c + half[k])); pl.append((-n, -(n @ c) + half[k]))
    return pl
def flat(color, spec=0.12, shin=30, emis=0.0):
    color = A3(*color)
    return lambda P, N: (np.tile(color, (len(P), 1)), np.full(len(P), spec), np.full(len(P), shin), np.full(len(P), emis))
def convex(planes, mat, name=""): OBJ.append(dict(kind="convex", planes=planes, mat=mat, name=name))
def box(lo, hi, mat, name=""): convex(box_planes(lo, hi), mat, name)
def cyl(a, b, r, mat, name=""): OBJ.append(dict(kind="cyl", a=A3(*a), b=A3(*b), r=r, mat=mat, name=name))

WHITE, METAL, DARK = (0.93, 0.93, 0.95), (0.72, 0.74, 0.78), (0.20, 0.19, 0.26)
PUR, PUR2, GOLD, TERRA = (0.314, 0.169, 0.545), (0.478, 0.357, 0.71), (0.949, 0.627, 0.118), (0.74, 0.33, 0.22)
PANEL_COLS = [(0.66, 0.26, 0.16), (0.62, 0.24, 0.15), (0.70, 0.31, 0.20), (0.76, 0.44, 0.33), (0.64, 0.25, 0.16), (0.72, 0.36, 0.25), (0.67, 0.27, 0.17), (0.80, 0.52, 0.42)]

# ---- bench
def bench_mat(P, N):
    n = len(P); return np.ones((n, 3)), np.zeros(n), np.full(n, 10.0), np.zeros(n)
box((-60, -2, -60), (60, 0, 60), bench_mat, "bench")

# ---- station 1: UV weathering chamber (slanted panel rack, lamps visible through the open section)
CX0, CX1, CZ0, CZ1, CY1, CY2 = -10.4, -4.2, -2.0, 2.0, 1.7, 5.3
ZT = 0.0                                             # z of the slant's top edge
SD = unit((0, CY2 - CY1, ZT - CZ1)); SN = unit((0, CZ1 - ZT, CY2 - CY1)); SL = float(np.hypot(CY2 - CY1, CZ1 - ZT)); SP0 = A3(0, CY1, CZ1)
def lower_mat(P, N):
    n = len(P); col = np.tile(A3(*WHITE), (n, 1)); spec = np.full(n, 0.15); shin = np.full(n, 30.0); em = np.zeros(n)
    fr = N[:, 2] > 0.9; u = (P[:, 0] - CX0) / (CX1 - CX0); y = P[:, 1]
    m = fr & (y > 0.14) & (y < 0.36); col[m] = PUR
    m = fr & (u > 0.70) & (u < 0.94) & (y > 0.68) & (y < 1.4); col[m] = (0.10, 0.10, 0.20)
    m = fr & (u > 0.73) & (u < 0.91) & (y > 0.92) & (y < 1.22); col[m] = (0.55, 0.90, 1.0); em[m] = 0.9
    for k in range(3):
        m = fr & ((u - (0.08 + 0.09 * k)) ** 2 * (CX1 - CX0) ** 2 + (y - 1.05) ** 2 < 0.15 ** 2); col[m] = DARK
    return col, spec, shin, em
box((CX0, 0, CZ0), (CX1, CY1, CZ1), lower_mat, "chamber-base")
LAMP_V = [0.2, 0.4, 0.6, 0.8]
def upper_mat(P, N):
    n = len(P); col = np.tile(A3(0.86, 0.87, 0.91), (n, 1)); spec = np.full(n, 0.05); shin = np.full(n, 40.0); em = np.zeros(n)
    sl = N @ SN > 0.9; u = (P[:, 0] - CX0) / (CX1 - CX0); v = ((P - SP0) @ SD) / SL
    pz = sl & (u > 0.03) & (u < 0.61) & (v > 0.06) & (v < 0.94)
    if pz.any():
        cu = (u[pz] - 0.03) / 0.58 * 4; cv = (v[pz] - 0.06) / 0.88 * 2; iu = np.floor(cu).astype(int); iv = np.floor(cv).astype(int); fu = cu - iu; fv = cv - iv
        pan = (fu > 0.07) & (fu < 0.93) & (fv > 0.05) & (fv < 0.95)
        c = np.tile(A3(0.66, 0.46, 0.98), (pz.sum(), 1)); e = np.full(pz.sum(), 0.95)
        pc = np.array(PANEL_COLS)[(iu + 4 * iv) % 8]; c[pan] = pc[pan]; e[pan] = 0.0
        col[pz] = c; em[pz] = e
    lz = sl & (u > 0.645) & (u < 0.965) & (v > 0.06) & (v < 0.94)
    if lz.any():
        g = sum(np.exp(-((v[lz] - vk) / 0.065) ** 2) for vk in LAMP_V)
        col[lz] = A3(0.10, 0.06, 0.20) + np.clip(g, 0, 1)[:, None] * A3(0.50, 0.34, 0.75); em[lz] = 1.0
    return col, spec, shin, em
convex(box_planes((CX0, CY1, CZ0), (CX1, CY2, CZ1)) + [(SN, SN @ SP0)], upper_mat, "chamber-top")
box((CX0 - 0.1, CY2, CZ0 - 0.1), (CX1 + 0.1, CY2 + 0.12, ZT + 0.25), flat((0.62, 0.63, 0.70), 0.3), "chamber-lid")
for vk in LAMP_V:
    p = SP0 + vk * SL * SD + 0.10 * SN
    cyl((CX0 + 0.655 * (CX1 - CX0), p[1], p[2]), (CX0 + 0.955 * (CX1 - CX0), p[1], p[2]), 0.085, flat((0.90, 0.80, 1.0), 0, 10, 1.0), "lamp")

# ---- station 2: ATR-FTIR spectrometer with pressure clamp and a coated panel on the crystal
def ftir_mat(P, N):
    n = len(P); col = np.tile(A3(*WHITE), (n, 1)); spec = np.full(n, 0.18); shin = np.full(n, 40.0); em = np.zeros(n)
    fr = N[:, 2] > 0.9; y = P[:, 1]; m = fr & (y > 0.16) & (y < 0.36); col[m] = PUR
    m = fr & ((P[:, 0] - 1.75) ** 2 + (y - 1.05) ** 2 < 0.1 ** 2); col[m] = (0.3, 0.95, 0.5); em[m] = 1.0
    return col, spec, shin, em
box((-2.3, 0, -1.8), (2.3, 1.5, 1.5), ftir_mat, "ftir-body")
box((-2.3, 1.5, -1.8), (2.3, 2.15, -0.95), flat(WHITE, 0.18, 40), "ftir-optics")
box((-1.0, 1.5, -0.35), (1.0, 1.6, 1.4), flat((0.24, 0.24, 0.30), 0.5, 60), "stage")
cyl((0, 1.6, 0.6), (0, 1.64, 0.6), 0.17, flat((0.55, 0.80, 0.97), 0.9, 100), "crystal")
box((-0.85, 1.64, 0.05), (1.25, 1.69, 1.15), flat(TERRA, 0.35, 50), "panel-on-crystal")
cyl((0, 1.5, -0.62), (0, 3.05, -0.62), 0.11, flat(METAL, 0.8, 80), "post")
box((-0.17, 2.82, -0.8), (0.17, 3.05, 0.78), flat(PUR, 0.3, 50), "arm")
cyl((0, 1.69, 0.6), (0, 2.82, 0.6), 0.085, flat(METAL, 0.8, 80), "tip")
cyl((0, 3.05, 0.6), (0, 3.3, 0.6), 0.25, flat(GOLD, 0.4, 50), "knob")
for k, c in enumerate([PANEL_COLS[3], PANEL_COLS[5], PANEL_COLS[0]]):      # panels waiting on the bench
    box((-3.75 + 0.06 * k, 0.05 * k, 2.2 - 0.05 * k), (-2.25 + 0.06 * k, 0.05 * (k + 1), 3.1 - 0.05 * k), flat(c, 0.3, 40), "waiting-panel")

# ---- station 3: laptop running the verdict software
LX = 6.3
def base_mat(P, N):
    n = len(P); col = np.tile(A3(0.27, 0.27, 0.32), (n, 1)); spec = np.full(n, 0.35); shin = np.full(n, 50.0); em = np.zeros(n)
    top = N[:, 1] > 0.9; lx = P[:, 0] - LX; lz = P[:, 2]
    kb = top & (np.abs(lx) < 1.6) & (lz > -0.35) & (lz < 0.75)
    if kb.any():
        gx = (lx[kb] * 4.2) % 1; gz = (lz[kb] * 4.2) % 1; key = (gx > 0.12) & (gx < 0.88) & (gz > 0.12) & (gz < 0.88)
        c = np.tile(A3(0.10, 0.10, 0.13), (kb.sum(), 1)); c[key] = (0.36, 0.36, 0.42); col[kb] = c
    m = top & (np.abs(lx) < 0.5) & (lz > 0.95) & (lz < 1.55); col[m] = (0.34, 0.34, 0.40)
    return col, spec, shin, em
box((LX - 1.9, 0, -0.6), (LX + 1.9, 0.12, 1.75), base_mat, "laptop-base")
TH = np.radians(16); SV = A3(0, np.cos(TH), -np.sin(TH)); SW = A3(0, np.sin(TH), np.cos(TH)); SU = A3(1, 0, 0)
SHW, SHH = 1.9, 1.3; SC = A3(LX, 0.12, -0.6) + SV * SHH
def screen_mat(P, N):
    n = len(P); col = np.tile(A3(0.16, 0.16, 0.20), (n, 1)); spec = np.full(n, 0.3); shin = np.full(n, 50.0); em = np.zeros(n)
    fr = N @ SW > 0.9; u = ((P - SC) @ SU) / SHW; v = ((P - SC) @ SV) / SHH
    m = fr & (np.abs(u) < 0.94) & (np.abs(v) < 0.91)
    if m.any():
        iu = np.clip(((u[m] / 0.94 + 1) / 2 * (TEX.shape[1] - 1)).astype(int), 0, TEX.shape[1] - 1); iv = np.clip(((1 - (v[m] / 0.91 + 1) / 2) * (TEX.shape[0] - 1)).astype(int), 0, TEX.shape[0] - 1)
        col[m] = TEX[iv, iu]; em[m] = 0.92
    return col, spec, shin, em
convex(obox_planes(SC, (SHW, SHH, 0.05), np.stack([SU, SV, SW], 1)), screen_mat, "laptop-screen")


# ---- outdoor exposure rack (used only by the "rack" view; stands far to the right, outside every other frame)
RKX, RKC = 45.0, None
RKA = np.radians(45); RKY = A3(0, np.cos(RKA), -np.sin(RKA)); RKN = A3(0, np.sin(RKA), np.cos(RKA)); RKC = A3(RKX, 2.75, 0.0); RKH = (3.4, 1.75, 0.07)
RACK_COLS = [(0.66, 0.26, 0.16), (0.80, 0.52, 0.42), (0.93, 0.87, 0.74), (0.36, 0.42, 0.58), (0.70, 0.31, 0.20), (0.56, 0.62, 0.50), (0.88, 0.70, 0.36), (0.62, 0.24, 0.15), (0.78, 0.80, 0.84),
             (0.44, 0.30, 0.56), (0.76, 0.44, 0.33), (0.90, 0.84, 0.70), (0.52, 0.58, 0.47), (0.67, 0.27, 0.17), (0.40, 0.46, 0.62), (0.85, 0.66, 0.34), (0.72, 0.36, 0.25), (0.82, 0.84, 0.87)]
def rack_mat(P, N):
    n = len(P); col = np.tile(A3(0.74, 0.76, 0.80), (n, 1)); spec = np.full(n, 0.2); shin = np.full(n, 40.0); em = np.zeros(n)
    f = N @ RKN > 0.9; d = P - RKC; u = d[:, 0] / (2 * RKH[0]) + 0.5; v = (d @ RKY) / (2 * RKH[1]) + 0.5
    z = f & (u > 0.025) & (u < 0.975) & (v > 0.05) & (v < 0.95)
    if z.any():
        cu = (u[z] - 0.025) / 0.95 * 6; cv = (v[z] - 0.05) / 0.9 * 3; iu = np.floor(cu).astype(int); iv = np.floor(cv).astype(int); fu = cu - iu; fv = cv - iv
        pan = (fu > 0.06) & (fu < 0.94) & (fv > 0.06) & (fv < 0.94); c = col[z].copy(); c[pan] = np.array(RACK_COLS)[(iu + 6 * iv) % 18][pan]; col[z] = c; spec[z] = np.where(pan, 0.05, 0.2)
    return col, spec, shin, em
convex(obox_planes(RKC, RKH, np.stack([A3(1, 0, 0), RKY, RKN], 1)), rack_mat, "rack-board")
for sx_ in (-2.9, 2.9):
    box((RKX + sx_ - 0.09, 0, 1.05), (RKX + sx_ + 0.09, 1.52, 1.23), flat(METAL, 0.3), "rack-leg")
    box((RKX + sx_ - 0.09, 0, -1.35), (RKX + sx_ + 0.09, 3.9, -1.17), flat(METAL, 0.3), "rack-leg")
    box((RKX + sx_ - 0.07, 0.0, -1.3), (RKX + sx_ + 0.07, 0.12, 1.2), flat(METAL, 0.3), "rack-leg")
box((RKX - 3.1, 0.55, 1.07), (RKX + 3.1, 0.69, 1.21), flat(METAL, 0.3), "rack-leg")

# ---- flow arrows lying on the bench
def arrow(xa, xb, z, col=PUR2):
    box((xa, 0, z - 0.2), (xb - 0.6, 0.04, z + 0.2), flat(col, 0.1), "arrow")
    pl = [(A3(0, 1, 0), 0.04), (A3(0, -1, 0), 0.0), (A3(-1, 0, 0), -(xb - 0.6))]
    for sgn in (1, -1):
        n = unit((0.5, 0, sgn * 0.6)); pl.append((n, n @ A3(xb, 0, z)))
    convex(pl, flat(col, 0.1), "arrow")
arrow(-3.95, -2.55, 1.0, GOLD); arrow(2.6, 4.1, 1.0, GOLD)

# ------------------------------------------------------------------ tracing
def hit_convex(ro, rd, planes):
    n_ = len(ro); tn = np.full(n_, -INF); tf = np.full(n_, INF); nrm = np.zeros((n_, 3)); ok = np.ones(n_, bool)
    for n, d in planes:
        den = rd @ n; dist = d - ro @ n
        par = np.abs(den) < 1e-12; ok &= ~(par & (dist < 0))
        with np.errstate(divide="ignore", invalid="ignore"): t = dist / den
        ent = (den < 0) & ~par; ext = (den > 0) & ~par
        upd = ent & (t > tn); tn = np.where(upd, t, tn); nrm[upd] = n
        tf = np.where(ext & (t < tf), t, tf)
    hit = ok & (tn <= tf) & (tn > 1e-4)
    return np.where(hit, tn, INF), nrm
def hit_cyl(ro, rd, a, b, r):
    n_ = len(ro); ax = b - a; L = np.linalg.norm(ax); ax = ax / L; oc = ro - a
    rda = rd @ ax; oca = oc @ ax; rp = rd - rda[:, None] * ax; op = oc - oca[:, None] * ax
    A = (rp * rp).sum(1); B = (rp * op).sum(1); C = (op * op).sum(1) - r * r; disc = B * B - A * C
    t = np.full(n_, INF); nrm = np.zeros((n_, 3)); m = (disc > 0) & (A > 1e-12)
    if m.any():
        t1 = (-B[m] - np.sqrt(disc[m])) / A[m]; y = oca[m] + t1 * rda[m]; ok = (t1 > 1e-4) & (y >= 0) & (y <= L)
        idx = np.where(m)[0][ok]; t[idx] = t1[ok]; nrm[idx] = (op[idx] + rp[idx] * t1[ok, None]) / r
    for c0, nn in ((a, -ax), (b, ax)):
        den = rd @ nn
        with np.errstate(divide="ignore", invalid="ignore"): tc = ((c0 - ro) @ nn) / den
        pc = ro + rd * np.where(np.isfinite(tc), tc, 0)[:, None] - c0; rr = (pc * pc).sum(1) - (pc @ nn) ** 2
        u = (np.abs(den) > 1e-12) & (tc > 1e-4) & (rr <= r * r) & (tc < t); t[u] = tc[u]; nrm[u] = nn
    return t, nrm
def trace(ro, rd, skip_bench=False):
    T = np.full(len(ro), INF); N = np.zeros((len(ro), 3)); ID = np.full(len(ro), -1)
    for i, o in enumerate(OBJ):
        if skip_bench and i == 0: continue
        t, n = hit_convex(ro, rd, o["planes"]) if o["kind"] == "convex" else hit_cyl(ro, rd, o["a"], o["b"], o["r"])
        u = t < T; T[u] = t[u]; N[u] = n[u]; ID[u] = i
    return T, N, ID
LIGHT = unit((-0.35, 0.70, 0.62)); FILL = unit((0.6, 0.3, 0.75))
def shade(ro, rd):
    T, N, ID = trace(ro, rd); hit = ~np.isinf(T); P = ro + rd * np.where(hit, T, 0)[:, None]
    n = len(ro); col = np.ones((n, 3)); spec = np.zeros(n); shin = np.full(n, 30.0); em = np.zeros(n)
    for i, o in enumerate(OBJ):
        m = ID == i
        if m.any(): col[m], spec[m], shin[m], em[m] = o["mat"](P[m], N[m])
    shadow = np.ones(n)
    if hit.any():
        st, _, sid = trace(P[hit] + N[hit] * 2e-3, np.tile(LIGHT, (hit.sum(), 1)), skip_bench=True)
        lampid = np.array([o["name"] == "lamp" for o in OBJ] + [False]); occ = ~np.isinf(st) & ~lampid[sid]
        shadow[hit] = np.where(occ, 0.38, 1.0)
    diff = np.clip(N @ LIGHT, 0, 1); h = LIGHT - rd; h /= np.linalg.norm(h, axis=1, keepdims=True)
    sp = np.clip((N * h).sum(1), 0, 1) ** shin * spec; fill = 0.16 * np.clip(N @ FILL, 0, 1)
    lit = col * (0.45 + 0.50 * diff * shadow + fill)[:, None] + (sp * shadow)[:, None]
    out = lit * (1 - em)[:, None] + col * em[:, None]
    b = ID == 0
    if b.any():                                           # bench: white with soft grey shadows, fading to white far away
        fade = np.minimum(np.clip((np.linalg.norm(P[b][:, [0, 2]] - A3(-1, 0), axis=1) - 13) / 10, 0, 1), np.clip((np.linalg.norm(P[b][:, [0, 2]] - A3(45, 0), axis=1) - 7) / 5, 0, 1))
        out[b] = (1.0 - 0.24 * (1 - shadow[b]) / 0.62 * (1 - fade))[:, None] * np.ones(3)
    out[~hit] = 1.0
    return np.clip(out, 0, 1), ID, N

def render(name, cam, look, fov, W, H, SS=3, anchors=None):
    if QUICK: W, H, SS = W // 3, H // 3, 1
    cam, look = A3(*cam), A3(*look); f = unit(look - cam); r = unit(np.cross(f, [0, 1, 0])); u = np.cross(r, f)
    sx = np.tan(np.radians(fov) / 2); sy = sx * H / W; w, h = W * SS, H * SS
    ii, jj = np.meshgrid(np.arange(w), np.arange(h)); px = ((ii + 0.5) / w * 2 - 1) * sx; py = (1 - (jj + 0.5) / h * 2) * sy
    d = f[None, None] + px[..., None] * r + py[..., None] * u; rd = (d / np.linalg.norm(d, axis=-1, keepdims=True)).reshape(-1, 3)
    img = np.zeros((len(rd), 3)); ids = np.zeros(len(rd), int); nn = np.zeros((len(rd), 3)); CH = 350000
    for k in range(0, len(rd), CH): img[k:k + CH], ids[k:k + CH], nn[k:k + CH] = shade(np.tile(cam, (len(rd[k:k + CH]), 1)), rd[k:k + CH])
    img = img.reshape(h, w, 3); ids = ids.reshape(h, w); nn = nn.reshape(h, w, 3)
    ids2 = np.where(ids <= 0, 0, ids)                    # bench and background count as one region
    edge = np.zeros((h, w), bool)
    for dy, dx in ((0, 1), (1, 0)):
        a, b = ids2[:h - dy, :w - dx], ids2[dy:, dx:]; na, nb = nn[:h - dy, :w - dx], nn[dy:, dx:]
        e = (a != b) | ((a == b) & (a > 0) & ((na * nb).sum(-1) < 0.9)); edge[:h - dy, :w - dx] |= e; edge[dy:, dx:] |= e
    lamp = np.isin(ids, [i for i, o in enumerate(OBJ) if o["name"] == "lamp"]); edge &= ~lamp
    img[edge] *= 0.62
    img = img.reshape(H, SS, W, SS, 3).mean((1, 3))
    Image.fromarray((img * 255).astype(np.uint8)).save(f"{FIG}/product_{name}.png")
    out = {}
    for k, p in (anchors or {}).items():
        v = A3(*p) - cam; z = v @ f; out[k] = [float(((v @ r) / z / sx + 1) / 2), float((1 - (v @ u) / z / sy) / 2)]
    print("rendered", name, W, H, out); return out

HCX = (-7.1 + LX) / 2
VIEWS = {
    "hero": dict(cam=(HCX, 10.5, 36.0), look=(HCX, 2.5, 0.3), fov=34.5, W=1800, H=615, SS=3,
                 anchors=dict(chamber=(-7.1, 0, 3.4), bench=(0, 0, 3.4), laptop=(LX, 0, 3.4), chamber_top=(-7.1, CY2 + 0.12, -0.7), bench_top=(0, 3.3, 0.6), laptop_top=(LX, 0.12 + 2 * SHH * np.cos(TH), -0.6 - 2 * SHH * np.sin(TH)))),
    "chamber": dict(cam=(-7.1, 8.0, 19.0), look=(-7.1, 2.6, 0.3), fov=40, W=1200, H=560, SS=3),
    "bench": dict(cam=(0.0, 6.5, 13.0), look=(0.0, 1.6, 0.4), fov=36, W=1200, H=560, SS=3),
    "rack": dict(cam=(45.0, 6.6, 19.5), look=(45.0, 1.95, 0.0), fov=31, W=1200, H=560, SS=3),
    "screen": dict(cam=(LX, 4.5, 11.5), look=(LX, 1.3, 0.5), fov=36, W=1200, H=560, SS=3),
}
anc = {}
ALL_OBJ = list(OBJ)
for v in WANT:
    OBJ[:] = ALL_OBJ if v == "hero" else [o for o in ALL_OBJ if o["name"] != "arrow"]
    anc[v] = render(v, **VIEWS[v])
OBJ[:] = ALL_OBJ
if "hero" in anc: json.dump(anc["hero"], open(f"{FIG}/product_hero.json", "w"))

# cropped close-ups used on slide 4 and appendix D4
for n, (a, b) in {"chamber": (0.2, 0.748), "bench": (0.205, 0.795), "screen": (0.25, 0.75), "rack": (0.16, 0.84)}.items():
    if n in anc:
        im = Image.open(f"{FIG}/product_{n}.png"); w_, h_ = im.size; im.crop((int(a * w_), 0, int(b * w_), h_)).save(f"{FIG}/product_{n}_c.png")
