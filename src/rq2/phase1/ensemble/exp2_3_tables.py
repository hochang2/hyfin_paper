"""Exp2 선택적 앙상블 · Exp3 종목/국면 적응 — 예열 이후 전체 평가 구간의 온라인 성능 표 (서술용).

주의: 여기 표는 수십 개 전략을 같은 표본에서 비교하므로 '최고' 전략에는 선택 편향이 있다.
선택 편향을 반영한 판정은 Exp4 (선정 구간과 최종 구간 분리 + reality check) 에서만 한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.rq2.phase1.ensemble import evaluate as EV
from src.rq2.phase1.ensemble import figures as FG
from src.rq2.phase1.ensemble import loader as L

DSS = ("gk", "rv5", "rv5d", "gkd")


def tables():
    Ts, Ws = [], []
    for ds in DSS:
        D = EV.load_eval(ds)
        Ts.append(EV.metrics(D, ds))
        Ws.append(EV.weight_stability(ds))
    T = pd.concat(Ts, ignore_index=True)
    Wst = pd.concat(Ws, ignore_index=True)
    e2 = T[~T.family.isin(["E3"])]
    e3 = T[T.family.eq("E3") | T.strategy.isin(["simplex_l2", "invmse_top3", "single_bsts", "old_stack", "old_stack_replica"])]
    e3 = e3.assign(base=e3.strategy.str.split("|").str[0], scope=e3.strategy.str.split("|").str[1].fillna("global"))
    L.save_table(e2, "exp2_selective_online")
    L.save_table(e3, "exp3_adaptive_online")
    L.save_table(Wst, "exp2_weight_stability")
    # 요약: 자료별로 창 평균 상대 RMSE
    S = T.groupby(["dataset", "strategy", "family"]).agg(rel_bsts=("rel_bsts", "mean"), rel_old=("rel_old", "mean"),
                                                         win_vs_bsts=("win_vs_bsts", "mean"), bias=("bias", "mean")).reset_index()
    S["rank"] = S.groupby("dataset").rel_bsts.rank()
    L.save_table(S, "exp2_3_summary_by_dataset")
    return T, S, Wst


def regime_breakdown():
    """Exp3: 국면별(사전 관측 변수) 성능 — 고정 기준 전략 vs 국면 적응 변형."""
    rows = []
    for ds in ("gk", "rv5", "rv5d"):
        D = EV.load_eval(ds)
        F = L.regime_frame(ds)[["i0", "vol_hi", "trend"]]
        D = D.merge(F, on="i0")
        keep = ["single_bsts", EV.ref_old(ds), "simplex_l2", "simplex_l2|regime_vol", "simplex_l2|regime_trend",
                "invmse_top3", "invmse_top3|regime_vol", "invmse_top3|regime_trend"]
        D = D[D.strategy.isin(keep)]
        for col in ("vol_hi", "trend"):
            for (w, lab, s), g in D.groupby(["window", col, "strategy"]):
                rows.append({"dataset": ds, "window": w, "split": col, "regime": str(lab), "strategy": s,
                             "n": len(g), "rmse": np.sqrt(np.mean(g.err ** 2)), "bias": g.err.mean()})
    T = pd.DataFrame(rows)
    ref = T[T.strategy == "single_bsts"].set_index(["dataset", "window", "split", "regime"]).rmse
    T["rel_bsts"] = T.rmse / T.set_index(["dataset", "window", "split", "regime"]).index.map(ref)
    L.save_table(T, "exp3_regime_breakdown")
    return T


SHOW = ["old_stack", "old_stack_replica", "eq_all", "ftl_single", "top3", "invmse_top3", "bc_invmse_top3", "bc_bsts",
        "mv_lw", "greedy", "pair_lowcorr", "simplex", "simplex_cap", "simplex_l2", "bsts_shrink", "h_bsts_ols_tfm",
        "h_bsts_ols_chr", "h_bsts_chr_uc", "h_bsts_ols_bestml", "stat_simplex", "simplex_l2|code", "simplex_l2|hier",
        "simplex_l2|glocal", "simplex_l2|regime_vol", "simplex_l2|regime_trend", "meta_ftl", "single_ols"]


def fig(S):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(17, 7), sharey=True)
    for ax, ds in zip(axs, DSS):
        s = S[S.dataset == ds].set_index("strategy").reindex(SHOW)
        y = np.arange(len(SHOW))
        col = [FG.CAT[1] if k.startswith("old") else (FG.CAT[6] if "|" in k else FG.CAT[0]) for k in SHOW]
        ax.scatter(s.rel_bsts, y, c=col, s=28, zorder=3, edgecolors=FG.SURF, linewidths=1)
        ax.axvline(1, color=FG.INK2, lw=1, ls="--")
        ax.set_yticks(y, SHOW, fontsize=7)
        ax.set_title(f"{ds}: 창 평균 RMSE (BSTS=1)", fontsize=9, loc="left")
        ax.set_xlim(0.85, 1.25)
    axs[0].invert_yaxis()
    fig.text(0.01, -0.01, "주황=기존 stacking · 보라=종목/국면 변형 · 파랑=Exp2 전략. 예열 이후 전체 구간(선택 편향 미보정, Exp4 참조).",
             fontsize=8, color=FG.INK2)
    FG.save(fig, "exp2_3_rel_rmse_by_dataset")


def run():
    T, S, W = tables()
    regime_breakdown()
    fig(S)
    return T, S


if __name__ == "__main__":
    run()
