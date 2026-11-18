"""2단계: 기술통계·사전추세.

· 집단별 일 거래대금 합(기준구간=100 지수), 기간별 평균
· 계열별 격차 g_f,t (2X−1X 평균 로그 거래대금) 2025-01~2026-10
· 사전추세: long_pre·base 구간 격차의 선형 추세 기울기 (HAC), 주별 격차
· 개인 순매수(F9) 집단·기간별
· 시장 여건(VKOSPI, 지수 변동성) 기간별
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.rq4.core.common import CFG, FIG, PAL, ExpLog, plt_setup, save_table, vlines, TL
from src.rq4.core.data_pipeline import load_etf, load_market
from src.rq4.core.panel import P, add_outcomes, code_group, period_of

LOG = ExpLog("02_descriptive")
G = CFG["groups"]


def group_daily(start="2025-01-02"):
    e = load_etf()
    cg = code_group()
    d = e[e["code"].isin(cg) & (e["date"] >= start)].copy()
    d["group"] = d["code"].map(cg)
    s = d.groupby(["date", "group"])["value"].sum().unstack()
    return s


def main():
    plt = plt_setup()
    s = group_daily()
    per = period_of(pd.Series(s.index, index=s.index))
    tab = s.groupby(per.values).mean().T / 1e8
    tab = tab[["long_pre", "base", "trans", "post"]]
    tab["post_vs_base_pct"] = 100 * (tab["post"] / tab["base"] - 1)
    tab["base_vs_longpre_pct"] = 100 * (tab["base"] / tab["long_pre"] - 1)
    save_table(tab.round(2).reset_index().rename(columns={"index": "group"}), "t02_group_value_by_period")

    # 그림 1: 집단 거래대금 지수 (기준구간=100), 10일 이동평균
    fig, ax = plt.subplots(figsize=(9, 4.2))
    show = [("A_single_lev_etf", "A 단일종목 2X ETF(규제)"), ("B_index_2x", "B 지수 2X"), ("C_semi_2x", "C 반도체 2X"),
            ("E_index_1x", "E 지수 1X(대조)"), ("E_semi_1x", "E 반도체 1X(대조)"),
            ("SP_concentrated_1x", "SP 삼성·SK 집중 1X"), ("D_overseas_2x", "D 해외기초 2X")]
    base_mask = (s.index >= P["listing_base"][0]) & (s.index <= P["listing_base"][1])
    for i, (g, lab) in enumerate(show):
        x = s[g].loc["2025-10-01":]
        idx = 100 * x / s[g][base_mask].mean()
        ax.plot(idx.rolling(10, min_periods=3).mean(), color=PAL[i], label=lab, lw=1.6)
    ax.set_yscale("log")
    ax.set_ylabel("거래대금 합 (기준구간 평균=100, 로그축, 10일 MA)")
    vlines(ax)
    ax.legend(ncol=2, fontsize=7, loc="lower left")
    ax.set_title("그림 1. 집단별 거래대금 추이")
    fig.savefig(FIG / "f01_group_value_index.png")
    plt.close(fig)

    # 계열별 격차
    from src.rq4.core.collapsed import wide
    W, V = wide()
    rows = {}
    for f, (tr, co) in {"idx": (G["B_index_2x"], G["E_index_1x"]), "semi": (G["C_semi_2x"], G["E_semi_1x"])}.items():
        rows[f] = W[tr].mean(axis=1) - W[co].mean(axis=1)
    gap = pd.DataFrame(rows).loc["2025-01-02":]
    gap.to_csv(FIG.parent / "tables" / "t02b_gap_daily.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, (f, lab) in enumerate([("idx", "지수: B 2X - E 1X"), ("semi", "반도체: C 2X - E 1X")]):
        ax.plot(gap[f], color=PAL[i], lw=0.6, alpha=0.35)
        ax.plot(gap[f].rolling(10, min_periods=3).mean(), color=PAL[i], label=lab, lw=1.8)
        for pn, (a, b) in [("base", P["listing_base"]), ("post", P["post"])]:
            m = gap[f].loc[a:b].mean()
            ax.hlines(m, a, b, color=PAL[i], lw=1, ls="--")
    vlines(ax)
    ax.set_ylabel("평균 로그 거래대금 격차")
    ax.legend(fontsize=8)
    ax.set_title("그림 2. 계열별 2X-1X 격차 (점선: 기준·사후 평균; 미표본 일부 상품은 해당 기간에 결측)")
    fig.savefig(FIG / "f02_gap_series.png")
    plt.close(fig)

    # 사전추세: 고정 코호트(주 분석) 격차로 기울기
    from src.rq4.core.panel import main_panel
    d = main_panel()
    pt = []
    for f in ["idx", "semi"]:
        x = d[d.family == f]
        g = x[x.lev2 == 1].groupby("date").log_value.mean() - x[x.lev2 == 0].groupby("date").log_value.mean()
        for nm, (a, b) in [("long_pre", P["long_pre"]), ("listing_base", P["listing_base"]),
                           ("long_pre+base", (P["long_pre"][0], P["listing_base"][1]))]:
            y = g.loc[a:b]
            t = np.arange(len(y)) / 21.0     # 월(21거래일) 단위 기울기
            m = sm.OLS(y.values, sm.add_constant(t)).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
            pt.append(dict(family=f, window=nm, n_days=len(y), slope_per_month=m.params[1], se_hac=m.bse[1],
                           p=m.pvalues[1], level_mean=y.mean()))
        # 상장 전후 격차 수준 변화 (상장 자체가 기준선을 바꿨는지)
        lv = g.loc[P["long_pre"][0]:P["long_pre"][1]].tail(35).mean()
        lb = g.loc[P["listing_base"][0]:P["listing_base"][1]].mean()
        pt.append(dict(family=f, window="base − 직전35일(상장 전)", n_days=35, slope_per_month=np.nan, se_hac=np.nan,
                       p=np.nan, level_mean=lb - lv))
    pt = pd.DataFrame(pt)
    save_table(pt.round(4), "t02c_pretrend_slopes")

    # F9 개인 순매수 집단·기간 (원 → 억원/일), 강도 = 순매수/거래대금
    e = add_outcomes(load_etf())
    cg = code_group()
    f9 = e[e.indiv_amt.notna() & (e.date >= "2025-10-01")].copy()
    f9["group"] = f9.code.map(cg).fillna("P2_other")
    f9["period"] = period_of(f9.date)
    nb = f9.groupby(["group", "period"]).agg(net_eok_per_day=("indiv_amt", lambda x: x.sum() / 1e8),
                                             n_days=("date", "nunique"), n_prod=("code", "nunique"),
                                             intensity_mean=("indiv_net_intensity", "mean")).reset_index()
    nb["net_eok_per_day"] = nb["net_eok_per_day"] / nb["n_days"]
    save_table(nb.round(4), "t02d_indiv_net_by_group_period")

    # 시장 여건
    mk = load_market()
    mk = mk[mk.date >= "2025-10-01"].copy()
    mk["period"] = period_of(mk.date)
    for c in ["r_k200", "r_krx_semi", "r_kq150"]:
        mk["abs_" + c] = mk[c].abs() * 100
    cond = mk.groupby("period")[["vkospi", "abs_r_k200", "abs_r_krx_semi", "abs_r_kq150"]].mean()
    save_table(cond.round(3).reset_index(), "t02e_market_conditions")

    LOG.add(id="D01", phase="2_descriptive", kind="descriptive", purpose="집단별 거래대금 기간 변화",
            outcome="value 합(억원/일)", period="long_pre/base/trans/post",
            estimate="", interpretation="; ".join(f"{g}: post/base {tab.loc[g, 'post_vs_base_pct']:+.1f}%"
                                                   for g in tab.index))
    for _, r in pt.iterrows():
        LOG.add(id=f"D02_{r['family']}_{r['window'][:12]}", phase="2_descriptive", kind="pretrend",
                hypothesis="평행추세", purpose="사전 격차 선형추세 (HAC 10)", treated=r["family"],
                period=r["window"], outcome="2X−1X 평균 log 거래대금 격차", model="OLS g_t ~ t",
                estimand="월당 기울기" if np.isfinite(r["slope_per_month"]) else "상장 직후 수준 변화",
                estimate=r["slope_per_month"] if np.isfinite(r["slope_per_month"]) else r["level_mean"],
                se=r["se_hac"], p=r["p"], inference="HAC(10)")
    LOG.save()
    print(tab.round(1).to_string())
    print(pt.round(3).to_string())
    print(nb.round(3).to_string())
    print(cond.round(2).to_string())


if __name__ == "__main__":
    main()
