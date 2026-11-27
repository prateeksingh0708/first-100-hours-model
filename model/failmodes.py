"""Procedural shaded renders of six exterior-paint failure modes (left half fresh, right half failed).
usage: python3 failmodes.py <figdir> [W H]"""
import sys
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

FIG = sys.argv[1]
W, H = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (720, 470)
yy, xx = np.mgrid[0:H, 0:W].astype(float)
SPLIT = W // 2

def fbm(seed, base=6, octaves=5, persist=0.5, aspect=(1.0, 1.0)):
    rng = np.random.default_rng(seed); out = np.zeros((H, W)); amp, tot = 1.0, 0.0
    for o in range(octaves):
        gh = max(2, int(base * 2 ** o * aspect[0])); gw = max(2, int(base * 2 ** o * W / H * aspect[1]))
        g = rng.random((gh + 3, gw + 3))
        up = zoom(g, ((H + 40) / (gh + 3), (W + 40) / (gw + 3)), order=3)[20:20 + H, 20:20 + W]
        out += amp * up; tot += amp; amp *= persist
    out /= tot
    return (out - out.min()) / (out.max() - out.min() + 1e-9)

def shade(height, color, ks, shin, k_norm=1.0, light=(-0.45, -0.55, 0.70)):
    gy, gx = np.gradient(height)
    n = np.dstack([-gx * k_norm, -gy * k_norm, np.ones_like(height)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
    L = np.array(light); L /= np.linalg.norm(L); V = np.array([0, 0, 1.0]); Hh = (L + V) / np.linalg.norm(L + V)
    diff = np.clip(n @ L, 0, 1); spec = np.clip(n @ Hh, 0, 1) ** shin
    sun = 0.92 + 0.08 * (1 - yy / H) * (1 - 0.5 * xx / W)          # soft sunlight falloff
    img = color * (0.30 + 0.78 * diff[..., None]) * sun[..., None] + (ks * spec)[..., None]
    return np.clip(img, 0, 1)

def rgb(h): return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
BASE = rgb("#B4553D")                                                   # terracotta exterior emulsion
bulge = 60 * np.sin(np.pi * xx / W) * np.sin(np.pi * yy / H)             # gentle curvature -> highlight band
FRESH_H = bulge + 0.6 * fbm(1, base=40, octaves=3)

def fresh():
    col = np.ones((H, W, 3)) * BASE * (0.97 + 0.06 * fbm(2, base=3)[..., None])
    return shade(FRESH_H, col, ks=0.55, shin=55, k_norm=1.0)

def compose(failed, name, label_r="failed"):
    left = fresh(); img = left.copy(); img[:, SPLIT:] = failed[:, SPLIT:]
    im = Image.fromarray((img * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for y0 in range(0, H, 16): d.line([(SPLIT, y0), (SPLIT, y0 + 9)], fill=(255, 255, 255), width=3)
    fp = font_manager.findfont(font_manager.FontProperties(family="TeX Gyre Adventor", weight="bold"))
    f = ImageFont.truetype(fp, int(W * 0.082))
    for x, t in [(18, "NEW"), (SPLIT + 18, label_r.upper())]:
        tw = d.textlength(t, font=f); fs_ = int(W * 0.082); d.rounded_rectangle([x - 8, H - fs_ - 30, x + tw + 10, H - 12], 10, fill=(36, 26, 61))
        d.text((x + 1, H - fs_ - 28), t, font=f, fill=(255, 255, 255))
    im.save(f"{FIG}/fm_{name}.png"); print(name)

# 1 gloss loss: rough micro-texture, no specular
h = bulge + 5.5 * fbm(3, base=60, octaves=4) + 2.0 * fbm(4, base=140, octaves=2)
col = np.ones((H, W, 3)) * BASE * 0.95
compose(shade(h, col, ks=0.05, shin=6, k_norm=1.0), "gloss")

# 2 colour fade: lighter, desaturated, slight haze
faded = BASE * 0.45 + rgb("#E9D3C6") * 0.55
col = np.ones((H, W, 3)) * faded * (0.96 + 0.08 * fbm(5, base=5)[..., None])
h = bulge + 3.0 * fbm(6, base=60, octaves=3)
compose(shade(h, col, ks=0.06, shin=8), "fade")

# 3 chalking: white powder with a finger swipe revealing colour
pw = fbm(7, base=50, octaves=4); speck = (np.random.default_rng(8).random((H, W)) > 0.55).astype(float)
chalk = np.clip(0.55 + 0.6 * (pw - 0.5) + 0.25 * gaussian_filter(speck, 0.8), 0, 1)
cx = SPLIT + 0.55 * (W - SPLIT) + 22 * np.sin(yy / 55.0); taper = np.clip((yy - 40) / 70, 0, 1) * np.clip((H - 70 - yy) / 70, 0, 1)
swipe = np.exp(-((xx - cx) / 34) ** 2) * (taper ** 1.5)
chalk *= (1 - 0.92 * swipe)
col = BASE * (1 - 0.82 * chalk[..., None]) + np.array([0.96, 0.95, 0.93]) * 0.82 * chalk[..., None]
h = bulge + 4.0 * fbm(9, base=70, octaves=3) + 3.0 * chalk
compose(shade(h, col, ks=0.04, shin=6), "chalk")

# 4 cracking: Voronoi network with dark gaps and lifted edges
rng = np.random.default_rng(10); seeds = np.c_[rng.uniform(SPLIT - 40, W + 40, 55), rng.uniform(-40, H + 40, 55)]
d = np.sqrt((xx[..., None] - seeds[:, 0]) ** 2 + (yy[..., None] - seeds[:, 1]) ** 2); d.sort(axis=2)
edge = d[..., 1] - d[..., 0]; wob = 2.0 + 2.5 * fbm(11, base=30, octaves=3)
crack = np.clip(1 - edge / wob, 0, 1); rim = np.clip(1 - edge / (wob * 3.2), 0, 1) - crack
col = np.ones((H, W, 3)) * BASE * 0.9 * (1 - 0.85 * crack[..., None]) + np.array([0.2, 0.18, 0.17]) * 0.6 * crack[..., None]
h = bulge + 3.0 * fbm(12, base=60, octaves=3) - 12 * crack + 4 * rim
compose(shade(h, col, ks=0.05, shin=8), "crack")

# 5 algae and dirt pick-up: blotches denser towards the bottom plus rain-run streaks
blot = fbm(13, base=7, octaves=5); grad = np.clip((yy / H) ** 1.4 * 1.5, 0, 1)
streak = fbm(14, base=4, octaves=4, aspect=(0.08, 3.0))
mask = np.clip((blot * 0.8 + 0.5 * grad + 0.35 * streak - 0.75) * 3.0, 0, 1)
algae = rgb("#3B4626") * 0.7 + rgb("#1D1F17") * 0.3
dirt = BASE * 0.75 * (1 - 0.25 * streak[..., None])
col = dirt * (1 - mask[..., None]) + algae * mask[..., None]
h = bulge + 3.5 * fbm(15, base=70, octaves=3) + 3 * mask
compose(shade(h, col, ks=0.03, shin=6), "algae")

# 6 blistering and peeling: domed blisters + peeled flakes exposing grey primer
rng = np.random.default_rng(16); hb = np.zeros((H, W))
for _ in range(28):
    cx, cy, r = rng.uniform(SPLIT + 10, W - 20), rng.uniform(20, H - 60), rng.uniform(10, 26)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2); hb += np.where(rr < r, np.sqrt(np.clip(r * r - rr * rr, 0, None)), 0)
peel = (fbm(17, base=5, octaves=5) > 0.64).astype(float); peel = gaussian_filter(peel, 1.2)
primer = rgb("#B9B4AA")
col = np.ones((H, W, 3)) * BASE * 0.93 * (1 - peel[..., None]) + primer * peel[..., None]
h = bulge + 2.5 * fbm(18, base=60, octaves=3) + 0.9 * hb - 6 * peel
compose(shade(h, col, ks=0.18, shin=20), "blister", "failed")
