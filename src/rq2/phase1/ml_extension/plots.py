"""ML 확장 그림. 표(results/tables/*.csv)만 읽어 그린다 (논문용 PNG, 밝은 배경).

색 = 모형 계열 (고정 순서, 검증된 범주 팔레트), 모양 = 계열 (색 외 2차 부호화). 축 이름 옆에 모형명이 붙어 색만으로 식별하지 않는다.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.rq2.phase1.ml_extension import data_pipeline as DP

TAB = DP.RES / "tables"
FIG = DP.RES / "figures"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FAM_C = {"statistical": "#2a78d6", "ML": "#eb6834", "ML-tuned": "#1baf7a", "foundation": "#eda100",
         "ensemble": "#e87ba4", "naive": "#8a8984", "other": "#8a8984"}
FAM_M = {"statistical": "o", "ML": "s", "ML-tuned": "D", "foundation": "^", "ensemble": "P", "naive": "X", "other": "o"}
FAM_ORDER = ["naive", "statistical", "ML", "ML-tuned", "foundation", "ensemble"]
DS_LABEL = {"rv5": "5분봉 log RV (2025-10~2026-07, 3종목)", "gk": "일별 log GK (2016~2026-02, 3종목)"}

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False})


def _t(name):
    return pd.read_csv(TAB / f"{name}.csv", dtype={"code": str})


def _legend(fig, fams):
    """그림 아래 한 줄 범례 (자료 점을 가리지 않게)."""
    h = [plt.Line2D([], [], ls="", marker=FAM_M[f], color=FAM_C[f], ms=7, label=f) for f in FAM_ORDER if f in fams]
    fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))


def _end_labels(ax, ends, min_gap):
    """선 끝 이름표가 겹치지 않게 세로로 밀어낸다. ends = [(x, y, text)]."""
    ends = sorted(ends, key=lambda e: e[1])
    ys = []
    for _, y, _ in ends:
        ys.append(max(y, ys[-1] + min_gap) if ys else y)
    for (x, y, t), yl in zip(ends, ys):
        ax.annotate(t, (x, y), xytext=(x + 1.2, yl), textcoords="data", fontsize=8, color=INK2, va="center",
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8) if abs(yl - y) > 1e-9 else None)


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_effect_rmse():
    S = _t("exp1_forecast_summary")
    S = S[(S.subset == "all") & (S.horizon == "h30")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.5))
    fams = set()
    for ax, ds in zip(axes, ["rv5", "gk"]):
        s = S[S.dataset == ds].sort_values("eff_rmse", ascending=False)
        y = np.arange(len(s))
        for f in s.family.unique():
            k = s.family == f
            ax.scatter(s.eff_rmse[k], y[k.values], c=FAM_C[f], marker=FAM_M[f], s=42, zorder=3,
                       edgecolors=SURF, linewidths=1.5)
        ax.set_yticks(y, s.model)
        ref = s[s.model == "ols"].eff_rmse
        if len(ref):
            ax.axvline(ref.iloc[0], color=INK2, lw=1, ls="--", zorder=1)
            ax.annotate("OLS", (ref.iloc[0], len(s) - 0.5), color=INK2, fontsize=8, ha="left", xytext=(3, 0),
                        textcoords="offset points")
        ax.set_xlabel("30일 창평균 효과 추정오차 RMSE (log)  — 작을수록 좋음")
        ax.set_title(DS_LABEL[ds], fontsize=9, loc="left")
        ax.grid(axis="y", visible=False)
        fams |= set(s.family)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    _legend(fig, fams)
    fig.suptitle("실험 1. 가짜 개입일에서의 효과 추정오차 (참효과 0)", x=0.01, ha="left", fontsize=11)
    _save(fig, "exp1_effect_rmse_h30")


def fig_horizon():
    S = _t("exp1_forecast_summary")
    S = S[S.subset == "all"]
    pick = ["naive_mean", "ols", "bsts", "lgbm", "chronos2_cov", "ens_equal"]
    hz = ["h5", "h10", "h20", "h30", "h61"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, ds in zip(axes, ["rv5", "gk"]):
        s = S[(S.dataset == ds) & S.model.isin(pick)]
        ends = []
        for m in pick:
            g = s[s.model == m].set_index("horizon").reindex(hz)
            if g.eff_rmse.isna().all():
                continue
            f = g.family.dropna().iloc[0]
            x = [int(h[1:]) for h in hz]
            ax.plot(x, g.eff_rmse, color=FAM_C[f], lw=2, marker=FAM_M[f], ms=6, mec=SURF, mew=1.5)
            last = g.eff_rmse.dropna()
            ends.append((x[hz.index(last.index[-1])], last.iloc[-1], m))
        lo, hi = ax.get_ylim()
        _end_labels(ax, ends, 0.045 * (hi - lo))
        ax.set_xticks([5, 10, 20, 30, 61])
        ax.set_xlabel("평가창 길이 (거래일)")
        ax.set_ylabel("창평균 효과 추정오차 RMSE (log)")
        ax.set_title(DS_LABEL[ds], fontsize=9, loc="left")
        ax.set_xlim(3, 75)
    fig.suptitle("실험 1. 지평별 효과 추정오차", x=0.01, ha="left", fontsize=11)
    _save(fig, "exp1_effect_rmse_by_horizon")


# time-보정 검정이 모든 과제에서 가능한 창 (rv5 h30 은 겹침 보정 후 검정 불가 → 선정 규칙과 같은 h10)
TEST_WIN = {"rv5": ("h10", "10일 창"), "gk": ("h30", "30일 창")}


def fig_power(selected=None):
    P = _t("exp2_power_curve")
    pick = ["ols", "bsts", "lgbm", "xgb", "ens_equal"] + ([selected] if selected and selected not in
                                                          ["ols", "bsts", "lgbm", "xgb", "ens_equal"] else [])
    from src.rq2.phase1.ml_extension.models import family
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, ds in zip(axes, ["rv5", "gk"]):
        p = P[(P.dataset == ds) & (P.window == TEST_WIN[ds][0]) & P.model.isin(pick)]
        p = p.groupby(["model", "effect_pct"]).time_power.mean().reset_index()
        ends = []
        for m in pick:
            g = p[p.model == m].sort_values("effect_pct")
            if g.empty:
                continue
            f = family(m)
            ax.plot(-g.effect_pct, g.time_power, color=FAM_C[f], lw=2, ls="-" if m != selected else "--")
            ends.append((-g.effect_pct.iloc[0], g.time_power.iloc[0], m + (" (선정)" if m == selected else "")))
        _end_labels(ax, ends, 0.045)
        ax.axhline(0.8, color=INK2, lw=1, ls=":")
        ax.axhline(0.05, color=INK2, lw=1, ls=":")
        ax.set_xlabel("주입한 RV 감소 (%)")
        ax.set_ylabel("기각률 (time 보정, α=0.05)")
        ax.set_title(DS_LABEL[ds] + " · " + TEST_WIN[ds][1], fontsize=9, loc="left")
        ax.set_xlim(0, 70)
    fig.suptitle("실험 2. 검정력 곡선 (시나리오 A 즉시·상수 효과)", x=0.01, ha="left", fontsize=11)
    _save(fig, "exp2_power_curve")


def fig_fpr():
    S = _t("exp2_synthetic_summary")
    S = S[(S.effect_pct == 0) & (S.scenario == "A_immediate")]
    sp = _t("exp2_space_permutation")
    sp = sp[(sp.effect_pct == 0) & (sp.window == "post_effective")].set_index("model").space_reject
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.5), sharex=True, gridspec_kw={"wspace": 0.5})
    for ax, ds in zip(axes, ["rv5", "gk"]):
        s = S[(S.dataset == ds) & (S.window == TEST_WIN[ds][0])].sort_values("time_reject", ascending=False)
        y = np.arange(len(s))
        for i, r in enumerate(s.itertuples()):
            c = FAM_C[r.family]
            if np.isfinite(r.native_reject):
                ax.plot([r.native_reject, r.time_reject], [i, i], color=GRID, lw=2, zorder=1)
                ax.scatter(r.native_reject, i, facecolors=SURF, edgecolors=c, s=36, marker=FAM_M[r.family], zorder=3)
            ax.scatter(r.time_reject, i, c=c, s=40, marker=FAM_M[r.family], zorder=4, edgecolors=SURF, linewidths=1.2)
            if ds == "rv5" and r.model in sp.index:
                ax.scatter(sp[r.model], i, marker="|", color=INK, s=60, zorder=5)
        ax.set_yticks(y, s.model)
        ax.axvline(0.05, color=INK2, lw=1, ls="--")
        ax.set_xlabel("효과 0 일 때 기각률 (명목 0.05)")
        ax.set_title(DS_LABEL[ds] + " · " + TEST_WIN[ds][1], fontsize=9, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(0, max(0.5, float(np.nanmax(S.native_reject)) + 0.03))
    fig.text(0.5, 0.0, "빈 표시 = 모형 자체(native) 구간, 채운 표시 = time 보정, | = space 순열 (실제일 post 61일, rv5 만)",
             ha="center", va="top", fontsize=8, color=INK2)
    fig.suptitle("진단 7. 위양성률 — 모형 자체 구간 vs 보정", x=0.01, ha="left", fontsize=11)
    _save(fig, "diag07_fpr_native_vs_calibrated")


def fig_tuning():
    T = _t("exp3_tuning_before_after")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    vc = {"default": INK2, "tuned_A_forecast": "#2a78d6", "tuned_B_effect": "#eb6834"}
    vm = {"default": "o", "tuned_A_forecast": "s", "tuned_B_effect": "D"}
    for ax, ds in zip(axes, ["rv5", "gk"]):
        t = T[T.dataset == ds]
        kinds = list(t.kind.unique())
        for i, k in enumerate(kinds):
            g = t[t.kind == k].set_index("variant")
            ax.plot(g.effect_rmse_h30.values, [i] * len(g), color=GRID, lw=2, zorder=1)
            for v, r in g.iterrows():
                ax.scatter(r.effect_rmse_h30, i, color=vc[v], marker=vm[v], s=44, zorder=3, edgecolors=SURF,
                           label=v if i == 0 else None)
        ax.set_yticks(range(len(kinds)), kinds)
        ax.set_xlabel("30일 효과 추정오차 RMSE (log)")
        ax.set_title(DS_LABEL[ds], fontsize=9, loc="left")
        ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("실험 3. Optuna 튜닝 전후 (같은 과제끼리 비교)", x=0.01, ha="left", fontsize=11)
    _save(fig, "exp3_tuning_before_after")


def fig_corr():
    C = _t("exp4_error_correlation_h30")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"wspace": 0.45})
    for ax, ds in zip(axes, ["rv5", "gk"]):
        c = C[C.dataset == ds].drop(columns="dataset").set_index("model")
        c = c.loc[:, c.index]
        im = ax.imshow(c.values, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(c)), c.index, rotation=90, fontsize=7)
        ax.set_yticks(range(len(c)), c.index, fontsize=7)
        ax.grid(False)
        ax.set_title(DS_LABEL[ds], fontsize=9, loc="left")
    fig.colorbar(im, ax=axes, shrink=0.7, label="창평균(30일) 오차 상관")
    fig.suptitle("실험 4. 모형 간 효과 추정오차 상관 (종목별 평균 제거)", x=0.01, ha="left", fontsize=11)
    _save(fig, "exp4_error_correlation")


def fig_real():
    R = _t("exp5_real_effect_rv5")
    codes = ["005930", "000660", "005935"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 9), sharex=True, sharey=True)
    order = R[(R.code == "005930") & (R.window == "post_effective")].sort_values("effect").model.tolist()
    for i, w in enumerate(["transition", "post_effective"]):
        for j, c in enumerate(codes):
            ax = axes[i, j]
            r = R[(R.code == c) & (R.window == w)].set_index("model").reindex(order).dropna(subset=["effect"])
            y = np.arange(len(r))
            for k, (m, x) in enumerate(r.iterrows()):
                col = FAM_C[x.family]
                ax.plot([x.time_ci_lo + x.m0, x.time_ci_hi + x.m0], [k, k], color=col, lw=1.5, alpha=.7)
                ax.scatter(x.effect, k, color=col, marker=FAM_M[x.family], s=50 if x.primary else 30,
                           edgecolors=INK if x.primary else SURF, linewidths=1.2, zorder=3)
            ax.axvline(0, color=INK2, lw=1)
            ax.set_yticks(y, r.index, fontsize=7)
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{c} · {w}", fontsize=9, loc="left")
            if i == 1:
                ax.set_xlabel("효과 (log RV)")
    fig.suptitle("실험 5. 실제 개입일 효과 — 모형별 (선: time 보정 95% 구간, post 61일은 비겹침 창 1개라 구간 없음;"
                 " 검은 테두리: 사전 지정 주 모형)",
                 x=0.01, ha="left", fontsize=11)
    _save(fig, "exp5_real_effect_by_model")


def make_all():
    import json
    sel = None
    f = TAB / "exp5_model_selection.json"
    if f.exists():
        sel = json.loads(f.read_text(encoding="utf-8")).get("selected")
    for fn in [fig_effect_rmse, fig_horizon, lambda: fig_power(sel), fig_fpr, fig_tuning, fig_corr, fig_real]:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print("figure failed:", getattr(fn, "__name__", "fig"), repr(e))


if __name__ == "__main__":
    make_all()
