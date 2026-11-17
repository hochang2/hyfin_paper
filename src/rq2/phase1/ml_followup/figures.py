"""후속 실험 그림: py -3.13 -m src.rq2.phase1.ml_followup.figures

results/tables/*.csv 만 읽어 results/figures/*.png 를 만든다 (적합·캐시 불필요).
색: 모형마다 고정 색 (그림 사이에서 같은 모형 = 같은 색). 기준선은 회색 점선.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent / "results"
TAB, FIG = ROOT / "tables", ROOT / "figures"

# 참조 팔레트 (dataviz reference palette, light) — 고정 배정, 순위로 재배정하지 않음
C = {"ols": "#2a78d6", "ridge": "#eb6834", "bsts": "#1baf7a", "lgbm": "#eda100",
     "rf": "#e87ba4", "xgb": "#008300", "enet": "#6250d6"}
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#9b9a95"
MARK = {"ols": "o", "ridge": "s", "bsts": "D", "lgbm": "^", "rf": "v", "xgb": "P", "enet": "X"}

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
    "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 5,
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def _t(name: str) -> pd.DataFrame:
    return pd.read_csv(TAB / f"{name}.csv")


def _save(fig, name: str, title: str, note: str | None = None, top: float = 1.03):
    fig.suptitle(title, x=0.01, y=top, ha="left", va="bottom", fontsize=11, color=INK, fontweight="bold")
    if note:
        fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=8, color=INK2)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("saved", name)


def _ref(ax, y, label=None, horizontal=True, left=False):
    (ax.axhline if horizontal else ax.axvline)(y, color=MUTED, lw=1, ls="--", zorder=1)
    if label and horizontal:
        ax.annotate(label, (0 if left else 1, y), xycoords=("axes fraction", "data"), xytext=(4 if left else -2, 3),
                    textcoords="offset points", ha="left" if left else "right", va="bottom", fontsize=8, color=INK2)


# ---------------------------------------------------------------- Exp5
def fig_exp5_fpr():
    d = _t("exp5_time_null_audit_gk")
    wins = ["h5", "h10", "h20", "h30"]
    p = d[d.window.isin(wins)].pivot_table(index="window", columns="method", values="fpr", aggfunc="median").loc[wins]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(wins))
    bad = {"naive": "o", "hac": "s", "mbb": "^", "sb": "v"}
    for m, mk in bad.items():
        ax.plot(x, p[m], color=MUTED, marker=mk, lw=1.4, zorder=2)
        ax.annotate(m.upper() if m != "naive" else "naive", (x[-1], p[m].iloc[-1]), xytext=(6, {"hac": 6, "naive": -6}.get(m, 0)),
                    textcoords="offset points", va="center", fontsize=8, color=INK2)
    for m, col in (("neff", C["ols"]), ("strict", C["ridge"])):
        ax.plot(x, p[m], color=col, marker="o", zorder=3)
        dy = 6 if m == "strict" else -8
        ax.annotate(m, (x[-1], p[m].iloc[-1]), xytext=(6, dy), textcoords="offset points",
                    va="center", fontsize=8, color=INK, fontweight="bold")
    _ref(ax, 0.05, "명목 5%", left=True)
    ax.set_xticks(x, wins)
    ax.set_xlim(-0.2, len(wins) - 0.4)
    ax.set_ylabel("시간 null 오기각률 (FPR)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    _save(fig, "fig1_exp5_fpr_by_method",
          "Exp5 · 겹친 창을 무시한 추론은 창이 길수록 과대 기각한다",
          "GK 조밀 가짜 개입일 감사, 모형별 FPR 의 중앙값. 회색 = 자기상관 무시/부분 보정, 색 = 유효 표본 보정(neff·strict).")


# ---------------------------------------------------------------- Exp1 + Exp2 forest
def fig_dmse_forest():
    e1 = _t("exp1_global_dmse_vs_local_ols")
    e1 = e1[e1.group == "donor"]
    pick1 = [("Local|-|bsts", "h10"), ("Local|-|bsts", "h30"), ("G2_feat|POST|lgbm", "h30"),
             ("G2_feat|LOSO|lgbm", "h30"), ("G3_recal|LOSO|enet", "post_effective"),
             ("G2_feat|LOSO|lgbm", "h10")]
    e2 = _t("exp2_transfer_dmse_cluster_ci")
    pick2 = [("multi_1000|lin_resid", "h10"), ("multi_1000|lin_resid", "h30"),
             ("multi_1000|lin_resid", "post_effective"), ("multi_same|lin_resid", "h30"),
             ("multi_1000|lgbm_recal", "h30"), ("self_1000|lin_resid", "h30"),
             ("self_1000|lgbm_recal", "h10"), ("multi_1000|lgbm_zs", "h30")]

    def rows(df, picks, col):
        out = []
        for m, w in picks:
            r = df[(df.model == m) & (df.window == w)].iloc[0]
            out.append((f"{m}  [{w}]", r[col], r.ci_lo, r.ci_hi))
        return out

    panels = [("Exp1 Global/Local · 기준 Local OLS (donor 40)", rows(e1, pick1, "d_mse")),
              ("Exp2 GK→rv5 전이 · 기준 rv5 OLS (donor 20)", rows(e2, pick2, "d_mse_vs_rv5_ols"))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"wspace": 0.95})
    for ax, (ttl, rs) in zip(axes, panels):
        y = np.arange(len(rs))[::-1]
        for yi, (lab, v, lo, hi) in zip(y, rs):
            sig = hi < 0 or lo > 0
            col = (C["ols"] if v < 0 else C["ridge"]) if sig else MUTED
            ax.plot([lo, hi], [yi, yi], color=col, lw=2, solid_capstyle="round")
            ax.plot(v, yi, "o", color=col, ms=6, mec="white", mew=1.5, zorder=3)
        ax.set_yticks(y, [r[0] for r in rs], fontsize=8)
        _ref(ax, 0, horizontal=False)
        ax.set_title(ttl, loc="left")
        ax.set_xlabel("dMSE (음수 = 기준보다 정확)")
        ax.grid(axis="y", visible=False)
        lim = max(abs(min(r[2] for r in rs)), 0.06)
        ax.set_xlim(-lim * 1.1, None)
    _save(fig, "fig2_dmse_forest_exp1_exp2",
          "Exp1·Exp2 · 기준 대비 개선은 작고, 유의한 것은 Local BSTS 와 multi GK 선형 잔차 전이뿐",
          "점 = dMSE, 선 = 종목 cluster bootstrap 95% CI. 파랑 = 유의하게 개선, 주황 = 유의하게 악화, 회색 = 0 포함. "
          "두 패널은 기준 모형·donor 표본이 달라 서로 직접 비교하지 않는다. G0_raw(원 수준, dMSE +0.30~0.34)는 축 범위 때문에 생략.")


# ---------------------------------------------------------------- Exp4
def fig_exp4_gk_length():
    d = _t("exp4_gk_length_curve")
    models = ["ols", "ridge", "lgbm", "rf", "bsts"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"wspace": 0.25})
    for ax, (g, w, ttl) in zip(axes, [("donor", "h10", "donor 30종목 · h10"), ("treated", "h61", "처치 3종목 · h61")]):
        x = d[(d.group == g) & (d.window == w)]
        for m in models:
            s = x[x.model == m].sort_values("L")
            if s.empty:
                continue
            ax.plot(s.L, s.rmse, color=C[m], marker=MARK[m], label=m.upper() if m in ("ols", "rf", "bsts") else m)
        ax.set_xscale("log")
        ax.set_xticks([60, 130, 250, 500, 1000, 2000], ["60", "130", "250", "500", "1000", "2000"])
        ax.minorticks_off()
        ax.set_xlabel("GK 학습 길이 L (일, 로그 척도)")
        ax.set_ylabel("반사실 RMSE")
        ax.set_title(ttl, loc="left")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="lower left", bbox_to_anchor=(0.06, 0.95), fontsize=8, handlelength=1.6)
    _save(fig, "fig3_exp4_gk_length_curve",
          "Exp4 · 학습 기간을 늘리면 모든 모형이 나빠진다 (Q1: 표본 부족이 주원인 아님)",
          "평가 2023-06 이후 10일 간격 가짜 개입일. BSTS 는 비용 때문에 L ≤ 500 만 적합. 처치 패널은 3종목이라 기술통계.")


def fig_exp4_nt():
    d = _t("exp4_nt_budget")
    d = d[d.window == "h10"]
    Bs = sorted(d.B.unique())
    fig, axes = plt.subplots(1, len(Bs), figsize=(10, 3.4), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, B in zip(axes, Bs):
        x = d[d.B == B]
        for a in ("ridge", "lgbm"):
            s = x[x.algo == a].sort_values("N")
            ax.plot(s.N, s.rmse, color=C[a], marker=MARK[a], label=a)
            ax.annotate(a, (s.N.iloc[-1], s.rmse.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
        ax.set_xscale("log", base=2)
        ns = sorted(x.N.unique())
        ax.set_xticks(ns, [f"N{n}\nT{int(B // n)}" for n in ns], fontsize=8)
        ax.minorticks_off()
        ax.set_title(f"행 예산 B = {B}", loc="left")
        ax.set_xlim(ns[0] / 1.4, ns[-1] * 2.2)
    axes[0].set_ylabel("반사실 RMSE (h10)")
    _save(fig, "fig4_exp4_nt_budget",
          "Exp4 · 같은 자료량이면 기간보다 종목을 늘리는 편이 낫다 (Q2)",
          "N = 학습 종목 수, T = 종목당 일수 (N×T = B). 종목당 기간이 너무 짧아지면(N64) 비선형 LGBM 이 먼저 무너진다.")


# ---------------------------------------------------------------- Exp6
def fig_exp6_ratio():
    d = _t("exp6_rmse_ratio_vs_ols")
    d = d[d.window == "h20"]
    models = ["ridge", "bsts", "lgbm", "xgb", "rf"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, fam in zip(axes, ["market", "theory"]):
        x = d[d.family == fam].groupby(["model", "n"]).rmse_ratio_vs_ols.median().reset_index()
        for m in models:
            s = x[x.model == m].sort_values("n")
            ax.plot(s.n, s.rmse_ratio_vs_ols, color=C[m], marker=MARK[m], label=m.upper() if m in ("rf", "bsts") else m)
        _ref(ax, 1.0, "OLS = 1", left=True)
        ax.set_xscale("log")
        ax.set_xticks([90, 190, 500, 1000], ["90", "190", "500", "1000"])
        ax.minorticks_off()
        ax.set_xlabel("학습 길이 n")
        ax.set_title({"market": "market 계열 (실제 donor 공변량 위 DGP)", "theory": "theory 계열 (이론 DGP)"}[fam], loc="left")
    axes[0].set_ylabel("OLS 대비 RMSE 비율 (h20)")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=5, loc="lower left", bbox_to_anchor=(0.06, 0.95), fontsize=8, handlelength=1.6)
    _save(fig, "fig5_exp6_rmse_ratio",
          "Exp6 · rv5 규모(n ≤ 190)에서는 OLS/ridge 를 이기기 어렵다",
          "7개 시나리오의 중앙값, 같은 반복끼리 짝지은 비율. BSTS 는 n ≤ 500 · 40 반복. 트리 모형은 n = 1000 비선형·상호작용에서만 OLS 를 앞선다.")


# ---------------------------------------------------------------- Exp7
def fig_exp7_pool():
    d = _t("exp7_pool_real_effect")
    d = d[(d.model == "ols") & (d.window == "post_effective")]
    acc = _t("exp7_pool_accuracy")
    acc = acc[(acc.model == "ols") & (acc.window == "h30")].set_index("pool").rmse
    pools = acc.sort_values().index.tolist()
    codes = [(5930, "005930 (처치)", C["ols"]), (660, "000660 (처치)", C["ridge"]), (5935, "005935 (음성통제)", C["bsts"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, gridspec_kw={"wspace": 0.08, "width_ratios": [1, 1.4]})
    y = np.arange(len(pools))[::-1]
    axes[0].barh(y, acc.loc[pools], color=C["ols"], height=0.6)
    for yi, v in zip(y, acc.loc[pools]):
        axes[0].annotate(f"{v:.3f}", (v, yi), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    axes[0].set_title("처치 종목 가짜 개입일 RMSE (OLS, h30)", loc="left")
    axes[0].set_yticks(y, [f"{p}" for p in pools])
    axes[0].grid(axis="y", visible=False)
    axes[0].set_xlim(0, acc.max() * 1.2)
    for code, lab, col in codes:
        s = d[d.code == code].set_index("pool").effect_pct.reindex(pools)
        axes[1].plot(s.values, y, "o", color=col, ms=7, mec="white", mew=1.5, label=lab, zorder=3)
    _ref(axes[1], 0, horizontal=False)
    axes[1].set_title("실제 발표일 효과 % (OLS, post_effective)", loc="left")
    axes[1].set_xlabel("효과 (%)")
    axes[1].grid(axis="y", visible=False)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.5, 0.95), ncol=3, fontsize=8)
    _save(fig, "fig6_exp7_donor_pool",
          "Exp7 · donor pool 에 따라 효과 크기가 크게 흔들리지만 모두 비유의",
          "pool 은 h30 RMSE 순으로 정렬. 모든 p_neff ≥ 0.18. all 의 정확도 우위는 사전구간 기준이며 주 분석 pool(main)을 바꾸는 근거로 쓰지 않는다.")


# ---------------------------------------------------------------- Exp3
def fig_exp3_fm():
    d = _t("exp3_fm_accuracy")
    d = d[d.window == "h30"]
    order = ["chronos2_uni", "timesfm_uni", "chronos2_cov", "timesfm_xreg", "ft_rv5_cov", "ft_gk_cov"]
    lab = {"chronos2_uni": "Chronos-2 단변량", "timesfm_uni": "TimesFM 단변량", "chronos2_cov": "Chronos-2 + 공변량",
           "timesfm_xreg": "TimesFM + 공변량", "ft_rv5_cov": "Chronos-2 FT(rv5) + 공변량",
           "ft_gk_cov": "Chronos-2 FT(GK) + 공변량"}
    e2 = _t("exp2_transfer_accuracy")
    ref = e2[(e2.model == "rv5_only|ols") & (e2.window == "h30")].set_index("group").rmse
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(order))[::-1]
    for ax, g, ttl in zip(axes, ["donor", "treated"], ["donor 40종목 · h30", "처치 3종목 · h30"]):
        v = d[d.group == g].set_index("model").rmse.reindex(order)
        cols = [MUTED if "uni" in m else (C["bsts"] if m.startswith("ft") else C["ols"]) for m in order]
        ax.barh(y, v.values, color=cols, height=0.6)
        for yi, vv in zip(y, v.values):
            ax.annotate(f"{vv:.3f}", (vv, yi), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8, color=INK2,
                        bbox=dict(fc="white", ec="none", pad=0.6), zorder=4)
        ax.axvline(ref[g], color=INK, lw=1, ls="--", zorder=0.5)
        ax.annotate(f"rv5 OLS {ref[g]:.3f}", (ref[g], y[0] + 0.55), xytext=(3, 0), textcoords="offset points",
                    fontsize=8, color=INK)
        ax.set_title(ttl, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 0.8)
        ax.set_xlabel("반사실 RMSE")
    axes[0].set_yticks(y, [lab[m] for m in order])
    _save(fig, "fig7_exp3_fm_accuracy",
          "Exp3 · FM 은 donor 공변량이 있어야 쓸 만하고, fine-tune 은 처치 종목에서 오히려 나빠진다",
          "회색 = 단변량 zero-shot, 파랑 = 공변량 zero-shot, 초록 = fine-tune(2 시점×100 step, CPU). 점선 = Exp2 rv5 OLS(특성 집합이 달라 참고용, CI 없음).")


ALL = [fig_exp5_fpr, fig_dmse_forest, fig_exp4_gk_length, fig_exp4_nt, fig_exp6_ratio, fig_exp7_pool, fig_exp3_fm]

if __name__ == "__main__":
    for f in ALL:
        f()
