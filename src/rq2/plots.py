"""RQ2 그림. 결과 표(csv)만 읽어 그린다 (재실행 없이 다시 그릴 수 있게)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "violet": "#6250d6",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "band": "#86b6ef"}
NAMES = {"005930": "삼성전자 (005930)", "000660": "SK하이닉스 (000660)", "005935": "삼성전자우 (005935, 음성통제)"}

plt.rcParams.update({"font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["ink2"],
                     "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
                     "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "figure.dpi": 130,
                     "savefig.bbox": "tight", "lines.linewidth": 1.6})


def _events(ax, tl, labels=True):
    for k, ls in [("listing", ":"), ("announce", "--"), ("effective", "-")]:
        ax.axvline(tl[k], color=C["ink2"], ls=ls, lw=0.9)
        if labels:
            ax.text(tl[k], ax.get_ylim()[1], {"listing": " 상장", "announce": " 발표", "effective": " 시행"}[k],
                    va="top", ha="left", fontsize=8, color=C["ink2"])


def counterfactual(path_csv, tl, out_png):
    d = pd.read_csv(path_csv, parse_dates=["date"], dtype={"code": str})
    codes = list(d.code.unique())
    fig, axes = plt.subplots(len(codes), 1, figsize=(9, 2.8 * len(codes)), sharex=True)
    for ax, code in zip(np.atleast_1d(axes), codes):
        g = d[d.code == code]
        g = g[g.date >= "2026-03-01"]
        ax.plot(g.date, g.y, color=C["ink"], lw=1.1, label="실제 log RV")
        p = g.dropna(subset=["cf"])
        ax.fill_between(p.date, p.lo, p.hi, color=C["band"], alpha=0.45, lw=0, label="반사실 95% 구간")
        ax.plot(p.date, p.cf, color=C["blue"], lw=1.6, label="반사실 (BSTS)")
        ax.plot(g.date, g.fit, color=C["blue"], lw=0.9, alpha=0.6)
        ax.set_title(NAMES.get(code, code), loc="left", fontsize=10)
        _events(ax, tl)
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=8, frameon=False, ncol=3)
    fig.supylabel("log 실현변동성 (5분 RV)")
    fig.savefig(out_png)
    plt.close(fig)


def forest(df, label_col, out_png, title, windows=("transition", "post_effective"), ref_label=None):
    """df: code, window, label, effect_log, ci_lo, ci_hi."""
    codes = [c for c in NAMES if c in df.code.unique()]
    labels = list(dict.fromkeys(df[label_col]))
    fig, axes = plt.subplots(1, len(codes), figsize=(4.2 * len(codes), 0.32 * len(labels) + 1.5), sharey=True)
    cols = [C["blue"], C["orange"]]
    for ax, code in zip(np.atleast_1d(axes), codes):
        for j, w in enumerate(windows):
            g = df[(df.code == code) & (df.window == w)].set_index(label_col).reindex(labels)
            yy = np.arange(len(labels)) + (j - 0.5) * 0.3
            ax.errorbar(g.effect_log, yy, xerr=[g.effect_log - g.ci_lo, g.ci_hi - g.effect_log], fmt="o",
                        ms=4, color=cols[j], elinewidth=1.2, capsize=0, label=w)
        ax.axvline(0, color=C["ink2"], lw=0.8)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(NAMES[code], fontsize=9, loc="left")
        ax.set_xlabel("효과 (log RV, 실제 − 반사실)")
        if ref_label in labels:
            ax.axhspan(labels.index(ref_label) - 0.5, labels.index(ref_label) + 0.5, color=C["grid"], zorder=0)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc="upper right", ncol=len(windows))
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.savefig(out_png)
    plt.close(fig)


def placebo_hist(plac: pd.DataFrame, actual: pd.DataFrame, value_cols, out_png, title, group_col="code",
                 xlabels=None):
    """value_cols 하나당 한 줄. 주황 선 = 실제 값."""
    value_cols = [value_cols] if isinstance(value_cols, str) else list(value_cols)
    xlabels = xlabels or value_cols
    codes = [c for c in NAMES if c in plac[group_col].unique()]
    fig, axes = plt.subplots(len(value_cols), len(codes), figsize=(4 * len(codes), 2.6 * len(value_cols) + 0.6),
                             squeeze=False)
    for r, (vc, xl) in enumerate(zip(value_cols, xlabels)):
        for ax, code in zip(axes[r], codes):
            v = plac.loc[plac[group_col] == code, vc].dropna()
            ax.hist(v, bins=30, color=C["band"], edgecolor="white", linewidth=0.5)
            for x in actual.loc[actual[group_col] == code, vc]:
                ax.axvline(x, color=C["orange"], lw=2)
            ax.axvline(0, color=C["ink2"], lw=0.8)
            if r == 0:
                ax.set_title(NAMES[code], fontsize=9, loc="left")
            ax.set_xlabel(xl, fontsize=8)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png)
    plt.close(fig)


def gap_history(L: pd.DataFrame, tl, out_png):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for code, col in [("005930", C["blue"]), ("000660", C["orange"]), ("005935", C["aqua"])]:
        gap = (L[code] - L["don_mean"]).resample("ME").mean()
        ax.plot(gap.index, gap.values, color=col, label=NAMES[code])
    ax.axhline(0, color=C["ink2"], lw=0.8)
    _events(ax, tl, labels=False)
    ax.set_ylabel("log GK − donor 평균 (월평균)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("처치 종목의 상대 변동성 추이 (F1 일별 OHLC, Garman-Klass)", fontsize=10, loc="left")
    fig.savefig(out_png)
    plt.close(fig)
