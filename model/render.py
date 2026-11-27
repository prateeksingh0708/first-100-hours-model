"""
Ray-traced cutaway renders of an exterior paint film (numpy; stands in for Blender).
Scene: steel substrate, primer, pigmented acrylic film with TiO2 cross-sections on the cut face,
oxidised surface skin, and a diamond ATR crystal pressed on the top surface.
usage: python3 render.py <outdir> <state: fresh|weathered> [width]
Schematic: layer thicknesses are exaggerated for legibility.
"""
import sys
import numpy as np

OUT, STATE = sys.argv[1], sys.argv[2]
W = int(sys.argv[3]) if len(sys.argv) > 3 else 1600
H = int(W * 9 / 16); SS = 2
rng = np.random.default_rng(7)
WEATHERED = STATE == "weathered"

# ---------------- geometry
X0, X1, Z0, Z1 = -4.0, 4.0, -3.0, 0.0
Y_SUB0, Y_SUB1, Y_PRI1, Y_TOP = -1.25, -0.55, -0.35, 1.0
TOP = Y_TOP - (0.06 if WEATHERED else 0.0)

def box(x0, x1, y0, y1, z0, z1):
    return [(np.array([1, 0, 0.]), x1), (np.array([-1, 0, 0.]), -x0), (np.array([0, 1, 0.]), y1),
            (np.array([0, -1, 0.]), -y0), (np.array([0, 0, 1.]), z1), (np.array([0, 0, -1.]), -z0)]

def prism(xb0, xb1, xt0, xt1, yb, yt, z0, z1):
    planes = [(np.array([0, 1, 0.]), yt), (np.array([0, -1, 0.]), -yb), (np.array([0, 0, 1.]), z1), (np.array([0, 0, -1.]), -z0)]
    for (xa, ya), (xb, yb_) in [((xb1, yb), (xt1, yt)), ((xt0, yt), (xb0, yb))]:
        e = np.array([xb - xa, yb_ - ya, 0.]); n = np.array([e[1], -e[0], 0.]); n /= np.linalg.norm(n)
        planes.append((n, n @ np.array([xa, ya, 0.])))
    return planes

OBJ = [
    dict(name="substrate", planes=box(X0, X1, Y_SUB0, Y_SUB1, Z0, Z1)),
    dict(name="primer", planes=box(X0, X1, Y_SUB1, Y_PRI1, Z0, Z1)),
    dict(name="film", planes=box(X0, X1, Y_PRI1, TOP, Z0, Z1)),
    dict(name="crystal", planes=prism(1.35, 2.75, 0.7, 3.4, TOP, TOP + 1.25, -2.4, -0.6)),
    dict(name="floor", planes=box(-30, 30, -2.0, Y_SUB0 - 0.25, -30, 30)),
]
# pigment discs on the cut face (z = 0) and protruding spheres on the weathered top
discs = []
while len(discs) < 150:
    x, y, r = rng.uniform(X0 + 0.1, X1 - 0.1), rng.uniform(Y_PRI1 + 0.12, TOP - 0.08), rng.uniform(0.045, 0.1)
    if all((x - a) ** 2 + (y - b) ** 2 > (r + c + 0.02) ** 2 for a, b, c in discs): discs.append((x, y, r))
discs = np.array(discs)
SPH = []
if WEATHERED:
    while len(SPH) < 140:
        x, z, r = rng.uniform(X0 + 0.15, X1 - 0.15), rng.uniform(Z0 + 0.15, Z1 - 0.12), rng.uniform(0.05, 0.1)
        if 1.2 < x < 2.9 and -2.5 < z < -0.5: continue          # under the crystal: keep contact area clear
        if all((x - a) ** 2 + (z - b) ** 2 > (r + c + 0.03) ** 2 for a, _, b, c in SPH): SPH.append((x, TOP - 0.25 * r, z, r))
SPH = np.array(SPH) if SPH else np.zeros((0, 4))

LIGHT = np.array([-0.12, 0.85, 0.5]); LIGHT /= np.linalg.norm(LIGHT)
CAM, LOOK = np.array([0.0, 6.0, 15.0]), np.array([0.0, 0.4, -1.4])

def hit_convex(ro, rd, planes):
    n_ = len(ro); tn = np.full(n_, -np.inf); tf = np.full(n_, np.inf); nrm = np.zeros((n_, 3)); ok = np.ones(n_, bool)
    for n, d in planes:
        den = rd @ n; dist = d - ro @ n
        par = np.abs(den) < 1e-12; ok &= ~(par & (dist < 0))
        with np.errstate(divide="ignore", invalid="ignore"): t = dist / den
        ent = (den < 0) & ~par; ext = (den > 0) & ~par
        upd = ent & (t > tn); tn = np.where(upd, t, tn); nrm[upd] = n
        tf = np.where(ext & (t < tf), t, tf)
    hit = ok & (tn <= tf) & (tn > 1e-4)
    return np.where(hit, tn, np.inf), nrm

def hit_spheres(ro, rd):
    best = np.full(len(ro), np.inf); nrm = np.zeros((len(ro), 3))
    for cx, cy, cz, r in SPH:
        c = np.array([cx, cy, cz]); oc = ro - c; b = (oc * rd).sum(1); cc = (oc * oc).sum(1) - r * r
        disc = b * b - cc; m = disc > 0; t = np.full(len(ro), np.inf); t[m] = -b[m] - np.sqrt(disc[m])
        t[t < 1e-4] = np.inf; u = t < best; best[u] = t[u]; nrm[u] = (ro[u] + rd[u] * t[u, None] - c) / r
    return best, nrm

def trace(ro, rd):
    T = np.full(len(ro), np.inf); N = np.zeros((len(ro), 3)); ID = np.full(len(ro), -1)
    for i, o in enumerate(OBJ):
        t, n = hit_convex(ro, rd, o["planes"]); u = t < T; T[u] = t[u]; N[u] = n[u]; ID[u] = i
    if len(SPH):
        t, n = hit_spheres(ro, rd); u = t < T; T[u] = t[u]; N[u] = n[u]; ID[u] = 99
    return T, N, ID

def shade(ro, rd):
    T, N, ID = trace(ro, rd); P = ro + rd * np.where(np.isinf(T), 0, T)[:, None]
    col = np.tile(np.array([1.0, 1.0, 1.0]), (len(ro), 1)); spec_k = np.zeros(len(ro)); shin = np.full(len(ro), 30.0)
    hitm = ~np.isinf(T)
    # ---- materials
    m = ID == 0; col[m] = np.array([0.55, 0.57, 0.6]) * (0.92 + 0.08 * np.sin(P[m, 0] * 40 + 3 * np.sin(P[m, 2] * 5)))[:, None]; spec_k[m] = 0.6; shin[m] = 60
    m = ID == 1; col[m] = np.array([0.62, 0.66, 0.6]); spec_k[m] = 0.1
    m = ID == 2
    if m.any():
        p = P[m]; c = np.tile(np.array([0.93, 0.91, 0.97]), (m.sum(), 1))
        depth = np.clip(TOP - p[:, 1], 0, None)
        ox = (0.85 if WEATHERED else 0.0) * np.exp(-depth / 0.16)
        c = c * (1 - ox[:, None]) + np.array([0.86, 0.5, 0.18]) * ox[:, None]
        front = N[m, 2] > 0.5
        if front.any():
            q = p[front]; inside = np.zeros(len(q), bool)
            for x, y, r in discs: inside |= (q[:, 0] - x) ** 2 + (q[:, 1] - y) ** 2 < r * r
            cf = c[front]; cf[inside] = np.array([1.0, 1.0, 1.0]); c[front] = cf
            c[front & False] = c[front & False]
        topf = N[m, 1] > 0.5
        if topf.any() and WEATHERED:
            q = p[topf]; nz = 0.9 + 0.1 * np.sin(q[:, 0] * 23) * np.sin(q[:, 2] * 19)
            c[topf] = c[topf] * nz[:, None]
        col[m] = c; spec_k[m] = 0.35 if not WEATHERED else 0.08; shin[m] = 80 if not WEATHERED else 12
    m = ID == 3; col[m] = np.array([0.55, 0.78, 0.95]); spec_k[m] = 1.0; shin[m] = 120
    m = ID == 4; col[m] = np.array([1.0, 1.0, 1.0]); spec_k[m] = 0.0
    m = ID == 99; col[m] = np.array([1.0, 1.0, 1.0]); spec_k[m] = 0.3; shin[m] = 40
    # ---- lighting with shadows
    diff = np.clip(N @ LIGHT, 0, 1)
    sro = P + N * 2e-3; st, _, _ = trace(sro[hitm], np.tile(LIGHT, (hitm.sum(), 1)))
    shadow = np.ones(len(ro)); shadow[hitm] = np.where(np.isinf(st), 1.0, 0.35)
    h = LIGHT - rd; h /= np.linalg.norm(h, axis=1, keepdims=True)
    spec = np.clip((N * h).sum(1), 0, 1) ** shin * spec_k
    fill = 0.25 * np.clip(N @ np.array([0.6, 0.3, 0.75]) / np.linalg.norm([0.6, 0.3, 0.75]), 0, 1)
    out = col * (0.32 + 0.68 * diff * shadow + fill)[:, None] + (spec * shadow)[:, None]
    m = ID == 3                                     # glassy crystal: fresnel rim
    if m.any():
        fr = (1 - np.abs((N[m] * rd[m]).sum(1))) ** 3
        out[m] = out[m] * 0.75 + 0.25 + fr[:, None] * 0.3
    m = ID == 4; out[m] = (1.0 - 0.22 * (1 - shadow[m]) / 0.65)[:, None]
    sky = np.array([1.0, 1.0, 1.0]); out[~hitm] = sky
    return np.clip(out, 0, 1)

def camera_rays(w, h):
    f = LOOK - CAM; f /= np.linalg.norm(f); r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    fov = np.radians(40); sx = np.tan(fov / 2); sy = sx * h / w
    ii, jj = np.meshgrid(np.arange(w), np.arange(h))
    px = ((ii + 0.5) / w * 2 - 1) * sx; py = (1 - (jj + 0.5) / h * 2) * sy
    d = f[None, None] + px[..., None] * r + py[..., None] * u
    return (d / np.linalg.norm(d, axis=-1, keepdims=True)).reshape(-1, 3), (f, r, u, sx, sy)

def project(pts, basis, w, h):
    f, r, u, sx, sy = basis; v = np.asarray(pts) - CAM; z = v @ f
    return np.stack([((v @ r) / z / sx + 1) / 2 * w, (1 - (v @ u) / z / sy) / 2 * h], 1)

w, h = W * SS, H * SS
rd, basis = camera_rays(w, h); img = np.zeros((len(rd), 3))
for k in range(0, len(rd), 400000):
    img[k:k + 400000] = shade(np.tile(CAM, (len(rd[k:k + 400000]), 1)), rd[k:k + 400000])
img = img.reshape(h, w, 3).reshape(H, SS, W, SS, 3).mean((1, 3))
np.save(f"{OUT}/render_{STATE}.npy", (img * 255).astype(np.uint8))

# ---------------- annotate
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
fam = "Carlito" if any("Carlito" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": fam})
P_, GOLD, ORG, DK = "#502B8B", "#F2A01E", "#C8601A", "#241A3D"
fig = plt.figure(figsize=(W / 200, H / 200), dpi=200); ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis("off")
pj = lambda p: project([p], basis, W, H)[0]
# IR beam inside the crystal
zc = -1.5; a, b, c = pj([0.95, TOP + 1.05, zc]), pj([2.05, TOP, zc]), pj([3.15, TOP + 1.05, zc])
ax.plot([a[0], b[0], c[0]], [a[1], b[1], c[1]], color="#E0301E", lw=3.0, solid_capstyle="round")
ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color="#E0301E", lw=0, mutation_scale=18))
e0 = pj([1.85, TOP - 0.06, zc]); e1 = pj([2.25, TOP - 0.06, zc]); ax.plot([e0[0], e1[0]], [e0[1], e1[1]], color="#E0301E", lw=2, alpha=0.6)
def num(p3, k, color):
    p = pj(p3); r = 15 * W / 1600
    ax.add_patch(plt.Circle(p, r, color=color, zorder=5)); ax.add_patch(plt.Circle(p, r, fill=False, ec="white", lw=1.5, zorder=6))
    ax.text(p[0], p[1], str(k), color="white", fontsize=10.5 * W / 1600, fontweight="bold", ha="center", va="center", zorder=7)
num([3.05, TOP + 0.9, -0.6], 1, P_)
num([2.05, TOP + 0.02, -0.6], 2, "#E0301E")
num([-3.3, TOP - 0.06, 0], 3, ORG if WEATHERED else P_)
if WEATHERED: num([-1.0, TOP + 0.08, -2.0], 4, ORG)
num([-1.5, (Y_PRI1 + TOP) / 2 - 0.15, 0], 5, DK)
num([0.5, (Y_SUB1 + Y_PRI1) / 2, 0], 6, DK)
num([2.0, (Y_SUB0 + Y_SUB1) / 2, 0], 7, DK)
fig.savefig(f"{OUT}/render_{STATE}.png", dpi=200); print("saved", STATE)
