"""1단계: 상품 유니버스 검증. 집단별 상품·상장일·자료 범위·F9 가용성·코호트 포함 여부와 사유.

출력: tables/t01_universe.csv, t01b_group_summary.csv, t01c_data_availability.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.rq4.core.common import CFG, ExpLog, save_table
from src.rq4.core.data_pipeline import load_etf
from src.rq4.core.panel import P, code_group

LOG = ExpLog("01_universe")


def main():
    e = load_etf()
    cg = code_group()
    e["grp_raw"] = e["grp"].str[:2]
    codes = list(cg)
    d = e[e["code"].isin(codes)]
    win = d[(d["date"] >= "2025-10-01") & (d["date"] <= "2026-10-30")]
    ndays = win["date"].nunique()
    rows = []
    for c in codes:
        x = d[d["code"] == c]
        w = win[win["code"] == c]
        if x.empty:
            rows.append(dict(code=c, group=cg[c], status="자료 없음"))
            continue
        per = {}
        for nm, (a, b) in [("long_pre", P["long_pre"]), ("base", P["listing_base"]), ("trans", P["transition"]),
                           ("post", P["post"])]:
            s = x[(x["date"] >= a) & (x["date"] <= b)]["value"]
            per[f"med_value_eok_{nm}"] = s.median() / 1e8 if len(s) else np.nan
        npos = int((w["value"] > 0).sum())
        balanced = npos == ndays
        floor = per["med_value_eok_long_pre"] * 1e8 >= CFG["cohort"]["min_pre_median_value_won"] \
            if not np.isnan(per["med_value_eok_long_pre"]) else False
        rows.append(dict(code=c, name=x["name"].iloc[-1], group=cg[c], grp_f8=x["grp_raw"].iloc[0],
                         kind=x["kind"].iloc[0], leverage=int(x["leverage"].iloc[0]),
                         index_name=x["index_name"].iloc[0], list_date=x["list_date"].iloc[0],
                         first_date=x["date"].min(), last_date=x["date"].max(), n_days_win=len(w),
                         n_value_pos_win=npos, zero_value_days_win=int((w["value"] == 0).sum()),
                         f9_days_win=int(w["indiv_amt"].notna().sum()), balanced=balanced,
                         floor_ok=floor, **per))
    u = pd.DataFrame(rows)
    main_groups = ["B_index_2x", "C_semi_2x", "E_index_1x", "E_semi_1x"]
    u["in_main"] = u["group"].isin(main_groups) & u["balanced"] & u["floor_ok"]
    u["exclusion_reason"] = np.select(
        [~u["group"].isin(main_groups), ~u["balanced"].astype(bool), ~u["floor_ok"].astype(bool)],
        ["주 분석 집단 아님(메커니즘·탐색·민감도)", "2025-10~2026-10 불균형(상장 늦음/결측)", "사전 거래대금 중앙값<1억"], "")
    save_table(u, "t01_universe")

    # P1 = A 집단 일치 확인
    p1 = sorted(e[e["grp_raw"] == "P1"]["code"].unique())
    a = sorted(CFG["groups"]["A_single_lev_etf"] + CFG["groups"]["A_single_lev_etn"] + CFG["groups"]["A_single_inv"])
    p1_ok = p1 == a
    # F9 가용 상품
    f9 = sorted(e[e["indiv_amt"].notna()]["code"].unique())

    gs = u.groupby("group").agg(n=("code", "size"), n_main=("in_main", "sum"),
                                med_value_eok_base=("med_value_eok_base", "sum"),
                                med_value_eok_post=("med_value_eok_post", "sum"),
                                f9_products=("f9_days_win", lambda s: int((s > 0).sum()))).reset_index()
    save_table(gs, "t01b_group_summary")

    avail = pd.DataFrame([
        dict(item="개인 거래대금(매수+매도)", available="없음", note="F9 는 순매수(indiv_amt)만. 33종(P1·P2)"),
        dict(item="개인 매수·매도 분리", available="없음", note="요청 결과변수 중 개인 매수/매도는 추정 불가"),
        dict(item="개인 순매수", available=f"{len(f9)}종", note=", ".join(f9)),
        dict(item="1X 대조군 개인 자료", available="없음", note="개인 기준 DID 불가 → 전체 거래대금 DID + 개인 순매수 ITS"),
        dict(item="해외상장 상품 거래", available="없음", note="F15/SEIBro 없음 → 국내상장 해외기초 2X 로만 대리"),
        dict(item="거래대금·거래량·좌수·NAV", available="392종 전체", note="proc.etf_daily (F7)"),
        dict(item="순자산 aum_won", available="2025-01~", note="F10"),
        dict(item="P1 그룹 = config A 집단", available=str(p1_ok), note=f"P1 {len(p1)}종"),
    ])
    save_table(avail, "t01c_data_availability")

    LOG.add(id="U01", phase="1_data", kind="data", purpose="상품 유니버스·코호트 확정 (추정 전)",
            treated="B(KOSPI200 2X ETF 8) ∪ C(반도체 2X ETF 3)", control="E(같은 계열 1X ETF)",
            period="2025-10-01~2026-10-30 균형", outcome="value",
            model="-", estimand="-",
            diagnostics=f"주 분석 {int(u['in_main'].sum())}종; 불균형·하한 제외 "
                        f"{int((u['group'].isin(main_groups) & ~u['in_main']).sum())}종; P1=A 일치 {p1_ok}; "
                        f"F9 {len(f9)}종; 창 내 거래대금 0인 날(주 집단) "
                        f"{int(u[u['group'].isin(main_groups)]['zero_value_days_win'].sum())}",
            interpretation="개인 거래대금·매수·매도 자료는 없다. H1 의 '개인 거래활동'은 (i) 전체 거래대금 DID 와 "
                           "(ii) F9 순매수가 있는 P2 상품의 개인 순매수 ITS 로 나눠 본다.")
    LOG.save()
    print(gs.to_string())
    print(u[u["group"].isin(main_groups) & ~u["in_main"]][["code", "name", "group", "exclusion_reason"]])
    print("P1==A:", p1_ok, "F9:", len(f9))


if __name__ == "__main__":
    main()
