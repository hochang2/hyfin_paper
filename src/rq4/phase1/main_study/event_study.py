"""3단계-2: 주별 이벤트 스터디. 사건 주 = 발표일(07-16) 기준 7일 단위 (0주 = 07-16~07-22, 시행 07-31 = 2주).

ES-A (주): 2025-10-01~2026-10-30, 참조 = listing_base 전체(주 추정대상과 같은 기준). 선행 = 상장 전 주들.
ES-B (HonestDiD 용): 05-27~10-30, 참조 = 발표 직전 주(−1), 선행 = 기준구간 안 −7~−2 주.
식별 문제: 평행추세 신뢰도 (질문 4). 선행계수 결합 검정(DK Wald)과 선형추세 검정.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.rq4.core.common import FIG, PAL, INK2, TL, ExpLog, plt_setup, save_table
from src.rq4.core.estim import FE
from src.rq4.core.panel import main_panel, window

LOG = ExpLog("04_event_study")


def add_week(d):
    d = d.copy()
    d["ew"] = ((d["date"] - TL["announce"]).dt.days // 7).astype(int)
    return d


def es_fit(d, ref_mask, by_family=True, y="log_value"):
    d = add_week(d)
    keys = sorted(d.loc[~ref_mask, "ew"].unique())
    cols = []
    fams = ["idx", "semi"] if by_family else ["all"]
    for f in fams:
        tf = d["lev2"] * ((d["family"] == f) if f != "all" else 1)
        for k in keys:
            c = f"{f}_w{k}"
            d[c] = (tf * ((d["ew"] == k) & ~ref_mask)).astype(float)
            if d[c].abs().sum() > 0:
                cols.append(c)
    m = FE(d, y, cols, ["code", "fd"])
    out = []
    for kind, cl in [("cr1", "code"), ("dk", "date")]:
        s = m.summary(kind, cluster=cl)
        s["method"] = kind
        s["term"] = s.index
        out.append(s)
    s = pd.concat(out).reset_index(drop=True)
    s["family"] = s["term"].str.split("_w").str[0]
    s["week"] = s["term"].str.split("_w").str[1].astype(int)
    return m, s


def wald(m, terms, kind="dk", cluster="date"):
    idx = [m.names.index(t) for t in terms]
    V = m.vcov(kind, cluster)[np.ix_(idx, idx)]
    b = m.b[idx]
    W = float(b @ np.linalg.pinv(V) @ b)
    q = np.linalg.matrix_rank(V)
    return W, q, stats.chi2.sf(W, q)


def main():
    plt = plt_setup()
    d = main_panel()
    # ES-A
    ref = (d["period"] == "base")
    m, s = es_fit(d, ref.values)
    save_table(s, "t04_event_study_A")
    rows = []
    for f in ["idx", "semi"]:
        lead = [t for t in m.names if t.startswith(f) and int(t.split("_w")[1]) < -8]
        W, q, p = wald(m, lead)
        # 선행계수 선형추세
        ss = s[(s.family == f) & (s.method == "dk") & (s.week < -8)]
        sl = np.polyfit(ss.week, ss.coef, 1)[0]
        rows.append(dict(spec="ES-A 상장 전 주들 (ref=base)", family=f, n_leads=len(lead), wald_dk=W, df=q, p=p,
                         lead_mean=ss.coef.mean(), lead_slope_per_week=sl))
        LOG.add(id=f"E01_{f}", phase="3_main", kind="event_study", hypothesis="평행추세",
                purpose="상장 전 주별 격차가 기준구간과 같은가", treated=f, period="2025-10~2026-10 주별",
                outcome="log 거래대금", model="상품 FE+계열×일 FE, 2X×주 더미, ref=listing_base",
                estimand="선행계수 결합=0", estimate=ss.coef.mean(), p=p, inference=f"DK Wald χ²({q})={W:.1f}",
                diagnostics=f"선행 평균 {ss.coef.mean():.3f}, 기울기 {sl:.4f}/주",
                interpretation="기준구간이 상장 전과 수준이 다르다 = 상장 자체가 기준선을 바꿈" if p < .05 else "")
    # 그림
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, f, col, lab in [(axes[0], "idx", PAL[0], "지수 2X - 1X"), (axes[1], "semi", PAL[1], "반도체 2X - 1X")]:
        ss = s[(s.family == f) & (s.method == "cr1")].sort_values("week")
        sd = s[(s.family == f) & (s.method == "dk")].sort_values("week")
        ax.fill_between(sd.week, sd.ci_low, sd.ci_high, color=col, alpha=0.15, lw=0, label="95% DK")
        ax.plot(ss.week, ss.ci_low, color=col, lw=0.6, ls=":")
        ax.plot(ss.week, ss.ci_high, color=col, lw=0.6, ls=":", label="95% CR1 상품")
        ax.plot(ss.week, ss.coef, color=col, lw=1.6, marker="o", ms=3)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvspan(-7.5, -0.5, color="#e4e3df", alpha=0.6, lw=0)
        for x, t in [(-7, "상장"), (0, "발표"), (2, "시행")]:
            ax.axvline(x, color=INK2, lw=0.7, ls=":")
            ax.text(x, 1.0, t, transform=ax.get_xaxis_transform(), fontsize=7, ha="center", va="bottom", color=INK2)
        ax.set_title(f"{lab} (회색=기준구간, 참조)")
        ax.set_xlabel("발표일 기준 주")
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("계수 (로그)")
    fig.savefig(FIG / "f03_event_study.png")
    plt.close(fig)

    # ES-B: 기준구간 안 선행 (HonestDiD 입력)
    w = window(d)
    w = add_week(w)
    refB = (w["ew"] == -1).values
    mB, sB = es_fit(w, refB)
    save_table(sB, "t04b_event_study_B")
    for f in ["idx", "semi"]:
        lead = [t for t in mB.names if t.startswith(f) and int(t.split("_w")[1]) < -1]
        W, q, p = wald(mB, lead)
        sc = wald(mB, lead, "cr1", "code")
        ss = sB[(sB.family == f) & (sB.method == "dk") & (sB.week < -1)]
        rows.append(dict(spec="ES-B 기준구간 안 (ref=−1주)", family=f, n_leads=len(lead), wald_dk=W, df=q, p=p,
                         wald_cr1=sc[0], p_cr1=sc[2], lead_mean=ss.coef.mean(),
                         lead_slope_per_week=np.polyfit(ss.week, ss.coef, 1)[0]))
        LOG.add(id=f"E02_{f}", phase="3_main", kind="event_study", hypothesis="평행추세(기준구간 안)",
                purpose="기준구간 안에서 격차가 평평했는가", treated=f, period="05-27~10-30",
                outcome="log 거래대금", model="ref=발표 직전 주", estimand="선행 −7~−2주 결합=0",
                estimate=ss.coef.mean(), p=p, inference=f"DK Wald χ²({q})={W:.1f}; CR1 p={sc[2]:.3g}",
                diagnostics=f"선행 기울기 {np.polyfit(ss.week, ss.coef, 1)[0]:.3f}/주")
    pt = pd.DataFrame(rows)
    save_table(pt, "t04c_pretests")
    LOG.save()
    print(pt.round(4).to_string())
    print(sB[(sB.method == "dk")][["family", "week", "coef", "se", "p"]].round(3).to_string())


if __name__ == "__main__":
    main()
