"""3단계-3: 삼중차분 (H2). (2X vs 1X) × (반도체 vs 지수 계열) × Post.

식별 문제: H2 는 '반도체와 지수 중 어디서 더 늘었나'. 계열별 DID 의 차이 = DDD. 계열마다 다른 공통충격은
계열×일 FE 가 흡수하고, 남는 위협은 '반도체 계열 안에서만 2X−1X 격차가 다르게 움직이는 다른 요인'.
변형
 D1 기준 DDD (기준선 = listing_base)        = did.py H2 와 같은 값 (재현 확인)
 D2 장기 기준선 (long_pre 2025-10~05-26)     기준구간의 일시적 이상(반도체 1X 급증)을 피함. 추정대상이 다름
 D3 상품별 선형추세 (long_pre+base 로 적합 후 외삽 제거)  사전추세 보정
 D4 개인 투자자 유형 DDD (개인 vs 금융투자 순매수)  → 순매수 합이 기계적으로 0 근처라 정의 불가: 실패로 기록
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.rq4.common import ExpLog, save_table
from src.rq4.did import all_se, design
from src.rq4.estim import FE
from src.rq4.panel import P, main_panel, window

LOG = ExpLog("05_ddd")


def ddd_fit(d, y="log_value", trend=False):
    d = d.copy()
    d["semi"] = (d["family"] == "semi").astype(int)
    d["p2"] = d["lev2"] * d["post"]
    d["t2"] = d["lev2"] * d["trans"]
    d["p2s"] = d["p2"] * d["semi"]
    d["t2s"] = d["t2"] * d["semi"]
    X = ["p2", "p2s", "t2", "t2s"]
    if trend:
        # 상품별 선형추세: 사전 구간 잔차로 기울기 추정 → y 에서 제거 (2단계, Goodman-Bacon/Borusyak 식 사전추세 보정)
        pre = d[d["period"].isin(["long_pre", "base"])]
        t0 = d["date"].min()
        d["tt"] = (d["date"] - t0).dt.days / 30.4
        pre = pre.assign(tt=(pre["date"] - t0).dt.days / 30.4)
        m0 = FE(pre, y, ["tt"] if False else [], ["code", "fd"]) if False else None
        # 계열×일 FE 를 뺀 뒤 상품별 기울기
        from src.rq4.estim import demean, codes
        r = demean(pre[[y]].values, [codes(pre["fd"])])[:, 0]
        pre = pre.assign(r=r)
        sl = pre.groupby("code").apply(lambda g: np.polyfit(g["tt"], g["r"], 1)[0])
        d[y] = d[y] - d["code"].map(sl) * d["tt"]
    return FE(d, y, X, ["code", "fd"])


def main():
    full = main_panel()
    rows = []
    specs = [("D1 기준(listing_base)", window(full), False),
             ("D2 장기 기준선(long_pre)", full[full["period"].isin(["long_pre", "trans", "post"])], False),
             ("D3 상품별 선형추세 제거", full[full["period"].isin(["long_pre", "base", "trans", "post"])], True)]
    for lab, d, tr in specs:
        if lab.startswith("D3"):
            m = ddd_fit(d, trend=True)
            # 추세 제거 후 기준은 base (long_pre 행은 포함되나 post/trans 더미 외 모두 기준)
        else:
            m = ddd_fit(d)
        for kind, cl in [("cr1", "code"), ("dk", "date")]:
            s = m.summary(kind, cluster=cl)
            for t in ["p2", "p2s"]:
                rows.append(dict(spec=lab, term={"p2": "β_B(지수)", "p2s": "DDD β_C−β_B"}[t], method=kind,
                                 **s.loc[t].to_dict()))
        wb = m.wild(j=1, reps=4999)
        rows.append(dict(spec=lab, term="DDD β_C−β_B", method="wild", coef=m.b[1], p=wb["p"],
                         ci_low=wb["ci_low"], ci_high=wb["ci_high"]))
        sc = m.summary("cr1", cluster="code").loc["p2s"]
        LOG.add(id=f"T{lab[1]}", phase="3_main", kind="ddd", hypothesis="H2", purpose=lab,
                treated="C 반도체 2X vs B 지수 2X", control="계열별 1X", outcome="log 거래대금",
                model="상품 FE + 계열×일 FE, 2X×Post, 2X×Post×반도체", estimand="β_C − β_B",
                estimate=m.b[1], se=sc["se"], ci_low=sc["ci_low"], ci_high=sc["ci_high"], p=sc["p"],
                inference=f"CR1 상품; wild p={wb['p']:.3f}; DK p={m.summary('dk', cluster='date').loc['p2s', 'p']:.3g}",
                interpretation="D2 는 기준선이 다르므로 추정대상이 다름 (상장 전 대비)" if lab.startswith("D2") else "")
    out = pd.DataFrame(rows)
    save_table(out, "t05_ddd")
    LOG.add(id="T4", phase="3_main", kind="ddd", status="failed", hypothesis="H2(개인)",
            purpose="투자자 유형 DDD (개인 vs 금융투자·기관) × 2X × Post",
            failure_revision="F9 순매수는 투자자 유형 합이 0 에 가깝다 (개인 매수 ≈ LP·금융투자 매도). "
                             "유형 간 차이가 기계적으로 결정되어 DDD 가 식별 정보를 주지 않는다. 또 1X 대조군에는 F9 가 없다. "
                             "→ 개인 순매수는 mechanism.py 의 단절 시계열(ITS)+역사적 위약으로 대체.")
    LOG.save()
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
