"""그림 공용 (ensemble 과 같은 팔레트·글꼴, 저장 위치만 이 실험 폴더)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.rq2.phase1.sensitivity.common import FIG  # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DIV = LinearSegmentedColormap.from_list("div", ["#a2300c", "#f19a76", "#f0efec", "#4a95e6", "#0d366b"])
COL = {"BSTS_005930": CAT[0], "BSTS_000660": CAT[2], "DIFF_005930": CAT[6], "SIMPLEX_005930": CAT[1],
       "SIMPLEX_000660": CAT[3], "005930": CAT[0], "000660": CAT[2], "005935": CAT[4]}

for f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    try:
        matplotlib.font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.family"] = [f, "DejaVu Sans"]
        break
    except Exception:  # noqa: BLE001
        pass
plt.rcParams.update({"axes.unicode_minus": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "savefig.dpi": 150,
                     "legend.frameon": False})


def _ascii_glyphs(fig):
    """한글 글꼴에 없는 글리프(U+2212, 결합 circumflex)를 저장 직전에 바꾼다."""
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        s = t.get_text()
        if s and ("−" in s or "̂" in s):
            t.set_text(s.replace("−", "-").replace("̂", ""))


def save(fig, name):
    _ascii_glyphs(fig)
    p = FIG / f"{name}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    return p
