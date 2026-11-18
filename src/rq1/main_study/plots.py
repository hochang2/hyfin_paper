"""RQ1 그림. 스타일은 RQ2(src/rq2/main_study/plots.py)와 같게 맞춘다."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "violet": "#6250d6",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "band": "#86b6ef"}
NAMES = {"005930": "삼성전자", "000660": "SK하이닉스", "005935": "삼성전자우"}

plt.rcParams.update({"font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["ink2"],
                     "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
                     "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "figure.dpi": 130,
                     "savefig.bbox": "tight", "lines.linewidth": 1.4})


def _listing(ax, tl, end, label=True):
    ax.axvspan(tl["listing"], end, color=C["band"], alpha=0.12, lw=0)
    ax.axvline(tl["listing"], color=C["ink2"], ls=":", lw=0.9)
    ax.axvline(pd.Timestamp("2026-05-22"), color=C["ink2"], ls="-", lw=0.5, alpha=0.5)
    if label:
        ax.text(tl["listing"], ax.get_ylim()[1], " 상장 05-27", va="top", ha="left", fontsize=8, color=C["ink2"])


def data_overview(df: pd.DataFrame, cfg: dict, out_png, smooth: int = 5):
    """위: 처리·음성통제 종목과 donor 평균의 log RV (5일 이동평균).
    아래: 처리 종목 − donor 평균 (DiD 의 원재료. 추정 결과가 아니라 원자료 차이)."""
    tl = cfg["timeline"]
    w = df.pivot(index="date", columns="code", values="y")
    don = df[df.role == "donor"].groupby("date").y.mean()
    semis = df[df.role == "neg_semis"].groupby("date").y.mean()
    sm = lambda s: s.rolling(smooth, min_periods=1).mean()  # noqa: E731

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=[3, 2])
    ax = axes[0]
    ax.plot(sm(don), color=C["ink2"], lw=2, label="donor 175종목 평균")
    ax.plot(sm(semis), color=C["ink2"], lw=1, ls="--", label="반도체 음성통제 7종목 평균")
    ax.plot(sm(w["005930"]), color=C["blue"], label="삼성전자")
    ax.plot(sm(w["000660"]), color=C["orange"], label="SK하이닉스")
    ax.plot(sm(w["005935"]), color=C["aqua"], lw=1, label="삼성전자우 (음성통제)")
    ax.set_ylabel("log RV (5분, 정규장)")
    ax.set_title(f"RQ1 분석 표본: log 실현변동성 ({smooth}일 이동평균)", loc="left", fontsize=11)
    ax.legend(fontsize=8, ncol=2, frameon=False, loc="upper left")
    end = w.index.max()
    ax.set_xlim(w.index.min(), end)
    _listing(ax, tl, end)

    ax = axes[1]
    ax.axhline(0, color=C["ink2"], lw=0.6)
    ax.plot(sm(w["005930"] - don), color=C["blue"], label="삼성전자 − donor 평균")
    ax.plot(sm(w["000660"] - don), color=C["orange"], label="SK하이닉스 − donor 평균")
    ax.plot(sm(w["005930"] - w["005935"]), color=C["violet"], lw=1, ls="--", label="삼성전자 − 삼성전자우")
    ax.set_ylabel("log RV 차이")
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper left")
    _listing(ax, tl, end, label=False)
    ax.text(0, -0.22, "원자료 차이이며 추정 결과가 아니다. 얇은 실선 = 05-22 (RQ2 sup-F 추정 시점).",
            transform=ax.transAxes, fontsize=8, color=C["ink2"])
    fig.savefig(out_png)
    plt.close(fig)


def placebo_hist(null: pd.Series, res: pd.DataFrame, out_png):
    """donor 가짜 처리 β 분포와 실제 처리 종목 β."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(null, bins=30, color=C["grid"], edgecolor=C["ink2"], lw=0.5, label=f"donor placebo β (n={len(null)})")
    for r, col in zip(res.itertuples(), [C["blue"], C["orange"]]):
        ax.axvline(r.beta, color=col, lw=2,
                   label=f"{NAMES.get(r.code, r.code)} β={r.beta:+.2f}, 순열 p={r.perm_p:.3f}, Holm {r.holm_p:.3f}")
    ax.axvline(0, color=C["ink2"], lw=0.6)
    ax.set_xlabel("DiD β (log RV, 상장 후 − 상장 전, donor 대비)")
    ax.set_ylabel("종목 수")
    ax.set_title("RQ1 주 검정: donor placebo-in-space 분포", loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.savefig(out_png)
    plt.close(fig)


def event_study(es: pd.DataFrame, out_png):
    """5거래일 구간별 계수. 회색 띠 = donor placebo 2.5~97.5%."""
    codes = list(es.code.unique())
    fig, axes = plt.subplots(1, len(codes), figsize=(11, 4), sharey=True)
    for ax, code, col in zip(np.atleast_1d(axes), codes, [C["blue"], C["orange"]]):
        e = es[es.code == code].sort_values("bin")
        ax.fill_between(e.bin, e.placebo_lo, e.placebo_hi, color=C["grid"], alpha=0.9, lw=0, label="placebo 95%")
        ax.axhline(0, color=C["ink2"], lw=0.6)
        ax.axvline(-0.5, color=C["ink2"], ls=":", lw=0.9)
        ax.plot(e.bin, e.coef, "o-", color=col, ms=4, label="계수")
        ax.set_title(NAMES.get(code, code), loc="left", fontsize=10)
        ax.set_xlabel("상장 기준 5거래일 구간 (0 = 05-27~)")
    np.atleast_1d(axes)[0].set_ylabel("log RV, 기준 = −9 이전 구간")
    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("RQ1 이벤트 스터디 (donor 대비)", x=0.02, ha="left", fontsize=11)
    fig.savefig(out_png)
    plt.close(fig)


def robustness_forest(df: pd.DataFrame, out_png):
    """사양별 β. 채운 점 = p < 0.05, 빈 점 = p ≥ 0.05. 모형 p 사양은 회색 글씨."""
    d = df[df.code.isin(["005930", "000660"])].copy()
    specs = list(dict.fromkeys(d.spec))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(specs) + 1.2))
    for k, (code, col, off) in enumerate([("005930", C["blue"], -0.15), ("000660", C["orange"], 0.15)]):
        e = d[d.code == code].set_index("spec").reindex(specs)
        y = np.arange(len(specs)) + off
        sig = e.p.astype(float) < 0.05
        ax.scatter(e.beta[sig], y[sig.values], color=col, s=30, zorder=3, label=f"{NAMES[code]} (p<0.05)")
        ax.scatter(e.beta[~sig], y[~sig.values], facecolor="white", edgecolor=col, s=30, zorder=3,
                   label=f"{NAMES[code]} (p≥0.05)")
    ax.axvline(0, color=C["ink2"], lw=0.6)
    ax.set_yticks(range(len(specs)))
    labels = [s + ("  ※모형 p" if "모형" in str(d[d.spec == s].p_type.iloc[0]) else "") for s in specs]
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("효과 (log RV)")
    ax.set_title("RQ1 강건성: 사양별 상장 효과 추정치", loc="left", fontsize=11)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower right")
    fig.savefig(out_png)
    plt.close(fig)


def supf_curves(F: pd.DataFrame, cfg: dict, out_png):
    """후보 변화 시점별 F 통계량."""
    cols = [C["blue"], C["orange"], C["aqua"], C["ink2"], C["violet"]]
    fig, ax = plt.subplots(figsize=(10, 4))
    for (name, s), col in zip(F.items(), cols):
        ax.plot(s.index, s.values, color=col, lw=1.3, ls="--" if "음성" in name else "-", label=name)
    ax.axhline(8.85, color=C["ink2"], lw=0.7, ls=":")
    ax.text(F.index[0], 8.85, " Andrews 5% (8.85)", va="bottom", fontsize=7, color=C["ink2"])
    ax.axvline(cfg["timeline"]["listing"], color=C["ink2"], ls=":", lw=0.9)
    ax.set_ylabel("F (평균 이동)")
    ax.set_title("RQ1 탐색: 변화 시점별 F 통계량 (log RV 차이)", loc="left", fontsize=11)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    fig.savefig(out_png)
    plt.close(fig)
