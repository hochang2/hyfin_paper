"""7단계: 메커니즘 (기술적·보조). 개별 투자자 식별 자료가 없으므로 '같은 투자자가 옮겨갔다' 고 말하지 않는다.
여기서 보는 것은 '총량·시점·노출의 정합성' 뿐이다.

K1 규제 상품 위축: A 집단 일 거래대금, 삼성전자·SK하이닉스 현물 거래대금 대비 비율, 상장 신기효과(지수 감쇠) 외삽 대비 사후
K2 노출 환산: 상품별 NAV 수익률 ~ 005930, 000660 수익률 (상장 전 또는 기준구간) 적재값 × 거래대금 = 종목 노출 환산 거래.
   집단별 기준→사후 변화: 잃은 노출(A) 대비 얻은 노출(B, C, SP, E). 계열 1X 변화로 보정한 값 병기.
K3 시점 연결: 일별 Δlog 거래대금의 교차상관 (A vs 대체 후보), 현물 거래대금 변화를 뺀 잔차 기준
K4 개인 순매수 ITS: F9 자료가 있는 P2 상품(지수 2X 4, 반도체 2X 1, 코스닥 2X 2, 해외 2X 3, 인버스)
   강도 = 개인 순매수/거래대금, 사후−기준 변화를 같은 길이 과거 비중첩 창 분포와 비교
K5 설정·환매(flow_won) 와 개인 순매수 상관
K6 해외 상장 상품: 자료 없음 → 식별 불가 (기록만)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.rq4.collapsed import placebo_p
from src.rq4.common import CFG, FIG, PAL, INK2, ExpLog, plt_setup, save_table, vlines
from src.rq4.data_pipeline import load_etf, load_stock_ss
from src.rq4.panel import P, add_outcomes, code_group, period_of

LOG = ExpLog("10_mechanism")
G = CFG["groups"]


def main():
    t0 = time.time()
    plt = plt_setup()
    e = add_outcomes(load_etf())
    cg = code_group()
    e["group"] = e["code"].map(cg)
    ss = load_stock_ss()
    stk = ss.pivot(index="date", columns="code", values="value")
    sret = ss.pivot(index="date", columns="code", values="ret_log")

    # ---------- K1
    A = e[e.group.isin(["A_single_lev_etf", "A_single_lev_etn", "A_single_inv"])]
    a_day = A.groupby("date")["value"].sum()
    a_day = a_day[a_day.index >= P["listing_base"][0]]
    ratio = a_day / stk[["005930", "000660"]].sum(axis=1).reindex(a_day.index)
    per = period_of(pd.Series(a_day.index, index=a_day.index))
    k1 = pd.DataFrame({"A_value_eok": a_day / 1e8, "ratio_to_spot": ratio, "period": per.values})
    k1s = k1.groupby("period")[["A_value_eok", "ratio_to_spot"]].mean()
    # 신기효과 외삽: 기준구간 log 거래대금 ~ 상장 후 경과일 (지수 감쇠)
    b = k1[k1.period == "base"]
    tt = np.arange(len(k1))
    mb = sm.OLS(np.log(b.A_value_eok.values), sm.add_constant(tt[:len(b)])).fit()
    k1["decay_fit"] = np.exp(mb.params[0] + mb.params[1] * tt)
    post = k1.period == "post"
    gap_decay = np.log(k1.loc[post, "A_value_eok"]).mean() - np.log(k1.loc[post, "decay_fit"]).mean()
    # 비율 기준 (현물 관심 대비)
    br = np.log(k1.loc[k1.period == "base", "ratio_to_spot"]).mean()
    pr = np.log(k1.loc[post, "ratio_to_spot"]).mean()
    f9a = A[A.indiv_amt.notna()].assign(period=lambda x: period_of(x.date)).groupby("period")["indiv_amt"].sum() / 1e8
    nd = k1.groupby("period").size()
    k1tab = k1s.assign(indiv_net_eok_per_day=(f9a / nd).reindex(k1s.index))
    save_table(k1tab.reset_index(), "t10_regulated_collapse")
    LOG.add(id="K1", phase="7_mechanism", kind="mechanism", purpose="규제 상품 위축의 크기",
            treated="A (단일종목 2X ETF·ETN·인버스)", outcome="거래대금, 현물 대비 비율, 개인 순매수",
            estimate=f"post/base {100 * (k1s.loc['post', 'A_value_eok'] / k1s.loc['base', 'A_value_eok'] - 1):+.1f}%",
            diagnostics=f"기준구간 감쇠율 {mb.params[1]:+.4f}/일 (신기효과); 감쇠 외삽 대비 사후 log 차 {gap_decay:+.2f}; "
                        f"현물 대비 비율 log 변화 {pr - br:+.2f}",
            interpretation=("기준구간에 A 거래는 줄지 않고 오히려 늘고 있었다 (감쇠 없음). " if mb.params[1] > 0 else
                            "기준구간에 A 거래는 이미 줄고 있었다. ") +
                           ("사후 수준은 기준구간 추세 외삽보다 훨씬 낮다 → 규제 시점의 급락은 신기효과 소멸로 설명되지 않음"
                            if gap_decay < -0.5 else "사후 수준이 추세 외삽과 비슷 → 규제 효과와 신기효과 구분 어려움"))

    # ---------- K2 노출 환산
    est = e[(e.date >= "2025-10-01") & (e.date <= P["listing_base"][1])][["date", "code", "nav_ret_log"]]
    est = est.merge(sret[["005930", "000660"]].reset_index(), on="date", how="left").dropna()
    load = {}
    for c, g in est.groupby("code"):
        if len(g) < 30:
            continue
        X = sm.add_constant(g[["005930", "000660"]])
        r = sm.OLS(g["nav_ret_log"], X).fit()
        load[c] = dict(b_sec=r.params["005930"], b_hyn=r.params["000660"], r2=r.rsquared, n=len(g))
    load = pd.DataFrame(load).T
    groups = ["A_single_lev_etf", "A_single_lev_etn", "A_single_inv", "B_index_2x", "C_semi_2x", "E_index_1x",
              "E_semi_1x", "SP_concentrated_1x", "D_overseas_2x", "D_kosdaq_2x", "D_sector_2x"]
    x = e[e.group.isin(groups) & e.code.isin(load.index) & (e.date >= P["listing_base"][0])].copy()
    x["period"] = period_of(x.date)
    x = x[x.period.isin(["base", "post"])]
    x["exp_sec"] = x.value * x.code.map(load.b_sec).abs()
    x["exp_hyn"] = x.value * x.code.map(load.b_hyn).abs()
    nd2 = x.groupby("period").date.nunique()
    k2 = x.groupby(["group", "period"])[["value", "exp_sec", "exp_hyn"]].sum().div(nd2, level="period", axis=0) / 1e8
    k2 = k2.unstack("period")
    out = pd.DataFrame({
        "value_base": k2[("value", "base")], "value_post": k2[("value", "post")],
        "sec_base": k2[("exp_sec", "base")], "sec_post": k2[("exp_sec", "post")],
        "hyn_base": k2[("exp_hyn", "base")], "hyn_post": k2[("exp_hyn", "post")]})
    out["d_sec"], out["d_hyn"] = out.sec_post - out.sec_base, out.hyn_post - out.hyn_base
    # 계열 1X 변화율로 보정한 '반사실 대비' 변화 (B·C 만: 2X 기준 × 1X 변화율)
    for g2, ctrl in [("B_index_2x", "E_index_1x"), ("C_semi_2x", "E_semi_1x")]:
        gr = out.loc[ctrl, "value_post"] / out.loc[ctrl, "value_base"]
        out.loc[g2, "d_sec_adj"] = out.loc[g2, "sec_post"] - out.loc[g2, "sec_base"] * gr
        out.loc[g2, "d_hyn_adj"] = out.loc[g2, "hyn_post"] - out.loc[g2, "hyn_base"] * gr
    lostA = out.loc[[g for g in out.index if g.startswith("A_")], ["d_sec", "d_hyn"]].sum()
    gainBC_adj = out.loc[["B_index_2x", "C_semi_2x"], ["d_sec_adj", "d_hyn_adj"]].sum()
    out["loading_mean_sec"] = x.groupby("group").apply(lambda g: g.code.map(load.b_sec).mean())
    out["loading_mean_hyn"] = x.groupby("group").apply(lambda g: g.code.map(load.b_hyn).mean())
    save_table(out.reset_index(), "t10b_exposure_equivalent")
    save_table(load.reset_index().rename(columns={"index": "code"}), "t10c_exposure_loadings")
    LOG.add(id="K2", phase="7_mechanism", kind="mechanism", purpose="삼성전자·SK하이닉스 노출 환산 거래의 이동",
            outcome="|적재값|×거래대금 (억원/일)", model="NAV 수익률 ~ 005930 + 000660 (2025-10~07-15)",
            estimate=f"A 잃은 노출: 삼성 {lostA.d_sec:,.0f}, 하이닉스 {lostA.d_hyn:,.0f} 억원/일; "
                     f"B+C 반사실 대비: 삼성 {gainBC_adj.d_sec_adj:+,.0f}, 하이닉스 {gainBC_adj.d_hyn_adj:+,.0f}",
            interpretation=f"보존 비율(반사실 보정, 하이닉스) ≈ {gainBC_adj.d_hyn_adj / -lostA.d_hyn:.1%}; "
                           "두 종목 수익률 상관이 높아 적재값 분리는 불안정 (합계 위주로 해석)")

    # ---------- K3 교차상관
    cand = {"A": ["A_single_lev_etf", "A_single_lev_etn"], "B": ["B_index_2x"], "C": ["C_semi_2x"],
            "SP": ["SP_concentrated_1x"], "E_semi": ["E_semi_1x"], "D_ovs": ["D_overseas_2x"]}
    S = pd.DataFrame({k: e[e.group.isin(v)].groupby("date").value.sum() for k, v in cand.items()})
    S = S[S.index >= P["listing_base"][0]]
    S["spot"] = stk[["005930", "000660"]].sum(axis=1).reindex(S.index)
    dS = np.log(S).diff().dropna()
    res = {}
    for k in cand:
        r = sm.OLS(dS[k], sm.add_constant(dS["spot"])).fit()
        res[k] = r.resid
    R = pd.DataFrame(res)
    k3 = []
    for k in ["B", "C", "SP", "E_semi", "D_ovs"]:
        for lag in range(-3, 4):
            for nm, m in [("base", R.index <= P["listing_base"][1]), ("trans+post", R.index >= P["transition"][0])]:
                a, bb = R.loc[m, "A"], R.loc[m, k].shift(-lag)
                k3.append(dict(series=k, lag=lag, window=nm, corr=a.corr(bb), n=int((a.notna() & bb.notna()).sum())))
    k3 = pd.DataFrame(k3)
    save_table(k3, "t10d_crosscorr")
    c0 = k3[(k3.lag == 0)].pivot(index="series", columns="window", values="corr")
    LOG.add(id="K3", phase="7_mechanism", kind="mechanism", purpose="규제 상품 감소와 대체 상품 변화의 시점 연결",
            model="Δlog 거래대금, 현물(005930+000660) Δlog 회귀 잔차의 교차상관",
            estimate="; ".join(f"{i}: base {r['base']:+.2f}, 이후 {r['trans+post']:+.2f}" for i, r in c0.iterrows()),
            interpretation="대체라면 A 감소일에 대체 상품 증가 → 음의 상관이 기대. 양의 상관 = 공통 관심 충격 (동반 증감)")

    # ---------- K4 개인 순매수 ITS
    f9codes = sorted(e.loc[e.indiv_amt.notna() & e.group.isin(["B_index_2x", "C_semi_2x", "D_kosdaq_2x",
                                                                 "D_overseas_2x", "D_inverse_2x"]), "code"].unique())
    F = e[e.code.isin(f9codes)].pivot(index="date", columns="code", values="indiv_net_intensity").sort_index()
    gmap = {c: cg[c] for c in f9codes}
    days = F.index
    ib = np.where((days >= P["listing_base"][0]) & (days <= P["listing_base"][1]))[0]
    ip = np.where((days >= P["post"][0]) & (days <= P["post"][1]))[0]
    nb, npost, ngap = len(ib), len(ip), ip[0] - ib[-1] - 1
    L = nb + ngap + npost

    def eff(b_, p_, grp):
        cs = [c for c in f9codes if gmap[c] == grp and F.iloc[np.r_[b_, p_]][c].notna().all()]
        if not cs:
            return np.nan
        return F.iloc[p_][cs].mean().mean() - F.iloc[b_][cs].mean().mean()
    k4 = []
    grps = ["B_index_2x", "C_semi_2x", "D_kosdaq_2x", "D_overseas_2x", "D_inverse_2x"]
    act = {g: eff(ib, ip, g) for g in grps}
    act["B−D_ovs"] = act["B_index_2x"] - act["D_overseas_2x"]
    s0 = np.where(days >= "2016-01-04")[0][0]
    s1 = np.where(days <= P["long_pre"][1])[0][-1]
    dist = []
    i = s0
    while i + L - 1 <= s1:
        b_, p_ = np.arange(i, i + nb), np.arange(i + nb + ngap, i + L)
        r = {g: eff(b_, p_, g) for g in grps}
        r["B−D_ovs"] = r["B_index_2x"] - r["D_overseas_2x"]
        r["start"] = days[i]
        dist.append(r)
        i += L
    dist = pd.DataFrame(dist)
    for g in grps + ["B−D_ovs"]:
        pr_ = placebo_p(act[g], dist[g])
        lv = e[e.code.isin([c for c in f9codes if gmap.get(c) == g])]
        k4.append(dict(group=g, delta_intensity=act[g], p_placebo=pr_["p"], n_placebo=pr_["n"],
                       null_sd=pr_["sd"]))
    k4 = pd.DataFrame(k4)
    lvl = e[e.code.isin(f9codes)].assign(period=lambda z: period_of(z.date))
    lvl = lvl[lvl.period.isin(["base", "trans", "post"])].groupby([lvl.code.map(gmap), "period"]).agg(
        net_eok_day=("indiv_amt", lambda s: s.sum() / 1e8), nd=("date", "nunique"),
        intensity=("indiv_net_intensity", "mean")).reset_index()
    lvl["net_eok_day"] = lvl.net_eok_day / lvl.nd
    save_table(k4, "t10e_indiv_its")
    save_table(lvl, "t10f_indiv_levels")
    for _, r in k4.iterrows():
        LOG.add(id=f"K4_{r.group}", phase="7_mechanism", kind="mechanism", hypothesis="개인 수요 (상품 수준)",
                purpose="개인 순매수 강도 변화 ITS", treated=r.group, outcome="개인 순매수/거래대금",
                model="사후−기준 평균 차", estimate=r.delta_intensity, p=r.p_placebo,
                inference=f"역사적 비중첩 창 n={r.n_placebo}, sd={r.null_sd:.3f}",
                interpretation="같은 투자자의 이동이 아니라 상품 수준 개인 순매수의 변화")

    # ---------- K5 flow vs 개인 순매수
    z = e[e.code.isin(f9codes) & (e.date >= "2025-10-01")].dropna(subset=["flow_won", "indiv_amt"])
    k5 = z.groupby(z.code.map(gmap)).apply(lambda g: g.flow_won.corr(g.indiv_amt)).rename("corr_flow_indiv")
    save_table(k5.reset_index(), "t10g_flow_indiv_corr")
    LOG.add(id="K5", phase="7_mechanism", kind="mechanism", purpose="설정·환매와 개인 순매수의 관계",
            estimate="; ".join(f"{k}: {v:+.2f}" for k, v in k5.items()),
            interpretation="양의 상관이면 개인 순매수가 LP 설정으로 이어짐 (순자산 증가 경로)")
    LOG.add(id="K6", phase="7_mechanism", kind="mechanism", status="not_identified",
            purpose="해외 상장 레버리지 상품(예: 미국 상장 삼성·하이닉스 2X, 반도체 3X)으로의 이동",
            failure_revision="DB 에 해외 상장 상품 거래·국내 투자자 해외주식 결제 자료가 없음 → 추정 불가. "
                             "국내 상장 해외기초 2X(D_overseas) 는 대조 참고만 가능하며, 이는 해외 상장 상품과 규제·상품 구조가 다르다.")

    # 그림: A 거래대금 + 감쇠 외삽, B/C/SP
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    ax.plot(k1.index, k1.A_value_eok, color=PAL[1], lw=1.4, label="A 규제 상품 거래대금")
    ax.plot(k1.index, k1.decay_fit, color=INK2, lw=1, ls="--", label="기준구간 로그선형 추세 외삽")
    ax.set_yscale("log")
    ax.set_ylabel("억원/일 (로그)")
    vlines(ax)
    ax.legend(fontsize=7)
    ax.set_title("규제 상품 거래대금과 기준구간 추세 외삽", fontsize=9, pad=14)
    ax = axes[1]
    show = out.loc[[g for g in ["A_single_lev_etf", "A_single_lev_etn", "B_index_2x", "C_semi_2x", "SP_concentrated_1x",
                                "E_semi_1x", "E_index_1x"] if g in out.index]]
    yy = np.arange(len(show))[::-1]
    ax.barh(yy + 0.2, show.d_sec, height=0.38, color=PAL[0], label="삼성전자 노출 환산 Δ")
    ax.barh(yy - 0.2, show.d_hyn, height=0.38, color=PAL[2], label="SK하이닉스 노출 환산 Δ")
    ax.set_yticks(yy)
    ax.set_yticklabels(show.index, fontsize=7)
    ax.set_xscale("symlog", linthresh=100)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_xlabel("사후 - 기준 (억원/일, symlog)")
    ax.legend(fontsize=7)
    ax.set_title("종목 노출 환산 거래 변화", fontsize=9)
    fig.savefig(FIG / "f09_mechanism.png")
    plt.close(fig)
    LOG.save()
    print(k1tab.round(3).to_string())
    print(f"decay slope {mb.params[1]:.4f}, gap vs decay {gap_decay:.2f}, ratio change {pr - br:.2f}")
    print(out.round(1).to_string())
    print("lostA", lostA.round(0).to_dict(), "gainBC_adj", gainBC_adj.round(0).to_dict())
    print(c0.round(3))
    print(k4.round(4).to_string())
    print(lvl.round(3).to_string())
    print(k5.round(3))
    print(load.loc[[c for c in ["122630", "494310", "488080", "243880", "069500", "091160", "0192L0"] if c in load.index]].round(3))
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
