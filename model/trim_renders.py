"""Trim the white margins of the film cross-section renders: render_<state>.png -> render_<state>_c.png.
usage: python3 trim_renders.py <figdir>"""
import sys
import numpy as np
from PIL import Image
FIG = sys.argv[1]
for state in ("fresh", "weathered"):
    im = Image.open(f"{FIG}/render_{state}.png").convert("RGB"); a = np.asarray(im)
    ys, xs = np.where((a < 250).any(-1)); pad = 12
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, im.width), min(ys.max() + pad, im.height))
    im.crop(box).save(f"{FIG}/render_{state}_c.png"); print(state, box)
