"""8단계: 종합. 핵심 수치를 결과표에서 다시 읽어 요약표(t11_summary)를 만들고 실험 로그를 병합한다.
RQ2 와의 연결은 이 단계에서만 기록한다 (RQ4 추정에는 RQ2 결과를 쓰지 않음).
"""
from __future__ import annotations

import pandas as pd

from src.rq4.common import TAB, ExpLog, merge_logs, save_table

LOG = ExpLog("11_synthesis")


def main():
    t3 = pd.read_csv(TAB / "t03_primary_did_all_se.csv")
    rob = pd.read_csv(TAB / "t08_robustness.csv")
    hon = pd.read_csv(TAB / "t09_honestdid.csv")
    pick = lambda term, meth: t3[(t3.term == term) & (t3.method == meth)].iloc[0]
    rows = []
    for lab, term, cterm in [("H1 풀링 β", "pooled", "post_2x"), ("β_B 지수 2X", "idx", "post_idx"),
                             ("β_C 반도체 2X", "semi", "post_semi"), ("H2 β_C−β_B", "diff", "H2: post_semi − post_idx")]:
        pl = pick(term, "역사적 위약(비중첩)")
        cr = pick(cterm, "CR1 상품")
        wb = pick(cterm, "Wild cluster(Webb) 상품")
        rb = rob[(rob.group != "주") & ~rob.spec.str.contains("순유입|점유율")][term].dropna()   # log 척도 사양만
        rows.append(dict(estimand=lab, estimate=pl.coef, pct=100 * (2.718281828 ** pl.coef - 1) if term != "diff" else None,
                         p_placebo=pl.p, p_cr1=cr.p, p_wild=wb.p,
                         robust_min=rb.min(), robust_max=rb.max(), robust_share_same_sign=(rb * pl.coef > 0).mean()))
    s = pd.DataFrame(rows)
    for fam, term in [("idx", "β_B 지수 2X"), ("semi", "β_C 반도체 2X")]:
        h = hon[(hon.family == fam) & (hon.restriction == "RM")]
        ok = h[h.excludes0]
        s.loc[s.estimand == term, "honest_RM_breakdown"] = ok.M.max() if len(ok) else None   # None = M̄=0 에서도 0 배제 못함
    save_table(s, "t11_summary")
    LOG.add(id="Z1", phase="8_synthesis", kind="synthesis", purpose="핵심 추정 요약", estimate=s.round(3).to_string())
    LOG.add(id="Z2", phase="8_synthesis", kind="rq2_link", purpose="RQ2 연결 (마지막 단계에서만)",
            interpretation="RQ2: 규제 후 기초자산(삼성전자·SK하이닉스) 변동성의 유의한 감소 없음(null). "
                           "RQ4: 규제 상품 거래는 −94% 로 붕괴했지만, 국내 비규제 2X 로의 상대적 이동(풍선효과)은 확인되지 않음. "
                           "따라서 RQ2 null 을 '수요가 국내 대체 상품으로 옮겨가 기초자산 압력이 유지됐기 때문' 이라고 설명할 근거는 약하다. "
                           "남는 설명: (1) 규제 상품의 기초자산 가격영향이 원래 작았음, (2) 해외 상장 상품·현물·신용 등 관측 못한 경로, "
                           "(3) RQ2 의 검정력 한계(MDE −38~−58%).")
    LOG.save()
    merge_logs()
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
