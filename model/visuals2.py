"""Extra visuals for the Destiny-style deck: equations, risk heat-map background.
usage: python3 visuals2.py <figdir>"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
FIG = sys.argv[1]
DK = "#241A3D"
plt.rcParams.update({"mathtext.fontset": "cm"})
EQ = {
 "eq_rateH": r"$r_{b}=\dfrac{\sum_{k} H_k\,\Delta\bar{A}_{b}(H_k)}{\sum_{k} H_k^{2}},\qquad H_k\in\{H_{24},H_{48},H_{100}\}$",
 "eq_model": r"$\log_{10}\hat{t}_f=\beta_0+\sum_{b}\beta_b\,r_b+\beta_v\,v+\sum_{j=1}^{3}\gamma_j\,d_j$",
 "eq_mdape": r"$\mathrm{MdAPE}=\mathrm{median}_i\,\left|\hat{t}_{f,i}-t_{f,i}\right|\,/\,t_{f,i}$",
 "eq_cov": r"$\widehat{\mathrm{cov}}=\dfrac{1}{m}\sum_{i=1}^{m}\mathbf{1}\left[t_{f,i}\in\mathrm{PI}_{0.9}(x_i)\right]$",
 "eq_atr_i": r"$d_p=\lambda\,/\,[\,2\pi n_1\,(\sin^2\theta-(n_2/n_1)^2)^{1/2}\,]$",
 "eq_rateH_i": r"$r_b=\Sigma_k\,H_k\,\Delta\bar{A}_b(H_k)\ /\ \Sigma_k\,H_k^2,\quad k\in\{24,48,100\,\mathrm{h}\}$",
 "eq_model_i": r"$\log_{10}\hat{t}_f=\beta_0+\Sigma_b\,\beta_b r_b+\beta_v v+\Sigma_j\,\gamma_j d_j$",
}
for k, tex in EQ.items():
    fig = plt.figure(figsize=(0.1, 0.1)); fig.text(0, 0, tex, fontsize=15, color=DK)
    fig.savefig(f"{FIG}/{k}.png", dpi=300, bbox_inches="tight", pad_inches=0.03, transparent=True); plt.close(fig)
# risk heat-map background: likelihood (x) by impact (y), 1-5 each
cm = LinearSegmentedColormap.from_list("rag", ["#2E9E5B", "#9CCB4F", "#F2D21E", "#F2A01E", "#D2453B"])
x = np.linspace(0.5, 5.5, 600); y = np.linspace(0.5, 5.5, 450); X, Y = np.meshgrid(x, y)
Z = (X * Y) / 25.0
fig = plt.figure(figsize=(6, 4.5), dpi=200); ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(Z, origin="lower", extent=[0.5, 5.5, 0.5, 5.5], cmap=cm, vmin=0.04, vmax=0.8, aspect="auto", interpolation="bicubic")
xx = np.linspace(2.4, 5.5, 200); ax.plot(xx, 12 / xx, color="white", lw=2.2, ls=(0, (5, 3)))
for v in np.arange(1.5, 5.5, 1.0):
    ax.axvline(v, color="white", lw=0.6, alpha=0.45); ax.axhline(v, color="white", lw=0.6, alpha=0.45)
ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5); ax.axis("off")
fig.savefig(f"{FIG}/risk_bg.png", dpi=200); plt.close(fig)
print("ok")
