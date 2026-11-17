"""그림 공용 (dataviz 기본 팔레트, 밝은 배경, 단일 축)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.rq2.ensemble_final.loader import RES

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("blue", ["#f3f8fe", "#cde2fb", "#8dbdf3", "#4a95e6", "#2a78d6", "#1b5cad", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#0d366b", "#4a95e6", "#f0efec", "#f19a76", "#a2300c"])

for f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    try:
        matplotlib.font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.family"] = f
        break
    except Exception:
        pass
plt.rcParams.update({"axes.unicode_minus": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "savefig.dpi": 150,
                     "legend.frameon": False})


def save(fig, name):
    p = RES / "figures" / f"{name}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    return p


def heatmap(ax, M, labels, title, vmin=None, vmax=None, cmap=SEQ, fmt="{:.2f}"):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = M[i, j]
            if np.isfinite(v):
                lo, hi = im.norm.vmin, im.norm.vmax
                dark = (v - lo) / (hi - lo + 1e-12) > 0.6
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=5.5, color="white" if dark else INK)
    ax.set_title(title, fontsize=9, loc="left")
    return im


def dot_ci(ax, labels, est, lo, hi, color=CAT[0], ref=None, highlight=()):
    y = np.arange(len(labels))
    for k in range(len(labels)):
        c = CAT[1] if labels[k] in highlight else color
        ax.plot([lo[k], hi[k]], [y[k], y[k]], color=c, lw=2, solid_capstyle="round")
        ax.plot(est[k], y[k], "o", ms=6, color=c, mec=SURF, mew=1.5)
    ax.set_yticks(y, labels, fontsize=7)
    ax.invert_yaxis()
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=1, ls="--")
