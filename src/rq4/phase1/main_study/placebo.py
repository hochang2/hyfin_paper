"""5단계-위약: 가짜 날짜, 가짜 처치집단, 가짜 결과변수, 과거 동일 달력 구간.

P1 가짜 날짜(표본 안): 상장 전 long_pre 안에서 같은 길이(기준 35·공백 10·사후 61) 창을 5일 간격으로 이동 (중첩·참고)
P2 과거 동일 달력 구간: 2016~2025 각 해 05-27~07-15 기준, 07-31~10-30 사후 (계절성 위약, 해마다 1개 → 비중첩)
P3 가짜 처치집단: 각 계열 1X 대조를 기준구간 거래대금 상위 절반(가짜 처치) vs 하위 절반 — 실제 기준·사후 창
P4 가짜 결과변수: log|NAV 수익률| 격차 — 레버리지 배수가 유지되면 규제와 무관하게 log 2 근처에서 일정해야 함
P5 가짜 처치집단(해외기초 2X·코스닥 2X): 계열 밖 2X 를 지수 1X 와 비교 — 같은 시장 2X 의 일반적 변화 참고 (식별 아님)
주 추론의 역사적 위약 분포(비중첩)는 did.py t03b 에 있음.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.rq4.core import collapsed as C
from src.rq4.core.collapsed import actual_positions, placebo_dist, placebo_p, wide, window_effect
from src.rq4.core.common import CFG, FIG, PAL, INK2, ExpLog, plt_setup, save_table
from src.rq4.core.data_pipeline import load_etf
from src.rq4.core.panel import P

LOG = ExpLog("07_placebo")
G = CFG["groups"]


def absret_wide():
    if "log_absret" in C._WIDE:
        return
    e = load_etf()
    allc = sorted({c for v in G.values() for c in v})
    d = e[e["code"].isin(allc)].copy()
    r = d["nav_ret_log"].abs()
    d["log_absret"] = np.log(r.where(r > 1e-5))
    W = d.pivot(index="date", columns="code", values="log_absret").sort_index()
    _, V = wide()
    C._WIDE["log_absret"] = (W.reindex(V.index), V)


def main():
    t0 = time.time()
    plt = plt_setup()
    W, V = wide()
    days = W.index
    ib, ip = actual_positions()
    act = window_effect(ib, ip)
    nb, nt, npost = C.window_lengths()
    rows = []

    # P1 표본 안 가짜 날짜
    p1 = placebo_dist(step=5, start=P["long_pre"][0], end=P["long_pre"][1])
    p1["kind"] = "P1 표본 안(long_pre) 5일 간격"
    # P2 과거 동일 달력 구간
    p2 = []
    for y in range(2016, 2026):
        b = np.where((days >= f"{y}-05-27") & (days <= f"{y}-07-15"))[0]
        q = np.where((days >= f"{y}-07-31") & (days <= f"{y}-10-30"))[0]
        r = window_effect(b, q)
        r["start"] = days[b[0]]
        p2.append(r)
    p2 = pd.DataFrame(p2)
    p2["kind"] = "P2 과거 같은 달력(5/27~7/15 vs 7/31~10/30)"
    for nm, dist in [("P1", p1), ("P2", p2)]:
        for k in ["pooled", "idx", "semi", "diff"]:
            x = dist[k].dropna()
            pr = placebo_p(act[k], x)
            rows.append(dict(test=nm, term=k, actual=act[k], n=len(x), null_mean=x.mean(), null_sd=x.std(),
                             null_min=x.min(), null_max=x.max(), p=pr["p"]))
    save_table(pd.concat([p1, p2]), "t07b_placebo_fake_dates")

    # P3 가짜 처치: 1X 상위 절반 vs 하위 절반 (기준구간 거래대금)
    spec3 = {}
    for f, co in [("idx", G["E_index_1x"]), ("semi", G["E_semi_1x"])]:
        cs = [c for c in co if c in V.columns]
        med = V.iloc[ib][cs].median().dropna().sort_values(ascending=False)
        h = len(med) // 2
        spec3[f] = (list(med.index[:h]), list(med.index[h:]))
    a3 = window_effect(ib, ip, spec3)
    d3 = placebo_dist(spec3)
    for k in ["pooled", "idx", "semi", "diff"]:
        pr = placebo_p(a3[k], d3[k])
        rows.append(dict(test="P3 가짜 처치(1X 상위 vs 하위)", term=k, actual=a3[k], n=pr["n"], null_mean=pr.get("mean_null"),
                         null_sd=pr["sd"], p=pr["p"]))

    # P4 가짜 결과: log|NAV 수익률|
    absret_wide()
    a4 = window_effect(ib, ip, outcome="log_absret")
    d4 = placebo_dist(outcome="log_absret")
    for k in ["pooled", "idx", "semi", "diff"]:
        pr = placebo_p(a4[k], d4[k])
        rows.append(dict(test="P4 가짜 결과(log|NAV수익률|)", term=k, actual=a4[k], n=pr["n"], null_mean=pr.get("mean_null"),
                         null_sd=pr["sd"], p=pr["p"]))

    # P5 계열 밖 2X (참고)
    for g in ["D_overseas_2x", "D_kosdaq_2x", "D_inverse_2x", "D_sector_2x"]:
        spec5 = {"idx": (G[g], G["E_index_1x"])}
        a5 = window_effect(ib, ip, spec5)
        d5 = placebo_dist(spec5)
        pr = placebo_p(a5["idx"], d5["idx"])
        rows.append(dict(test=f"P5 {g} vs 지수 1X (참고)", term="idx", actual=a5["idx"], n=pr["n"],
                         null_mean=pr.get("mean_null"), null_sd=pr["sd"], p=pr["p"],
                         note=f"nT={a5['nT_idx']}, nC={a5['nC_idx']}"))
    out = pd.DataFrame(rows)
    save_table(out, "t07_placebo")

    # 그림: 실제 vs 위약 분포 (주 비중첩 + P2)
    dist_main = pd.read_csv(FIG.parent / "tables" / "t03b_placebo_windows_nonoverlap.csv")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, k, lab in zip(axes, ["pooled", "idx", "semi", "diff"], ["풀링 β", "β_B 지수", "β_C 반도체", "β_C-β_B"]):
        x = dist_main[k].dropna()
        ax.scatter(x, np.zeros(len(x)) + 0.0, color=PAL[0], s=18, alpha=0.6, label="역사적 비중첩 창")
        y = p2[k].dropna()
        ax.scatter(y, np.zeros(len(y)) + 0.3, color=PAL[2], s=18, alpha=0.8, marker="s", label="과거 같은 달력")
        z = p1[k].dropna()
        ax.scatter(z, np.zeros(len(z)) + 0.6, color=PAL[3], s=14, alpha=0.6, marker="^", label="표본 안(중첩)")
        ax.axvline(act[k], color=PAL[1], lw=2, label="실제")
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_yticks([])
        ax.set_ylim(-0.3, 0.9)
        ax.set_title(lab)
    axes[0].legend(fontsize=6, loc="upper left")
    fig.suptitle("그림 5. 실제 효과와 위약 분포", fontsize=10)
    fig.savefig(FIG / "f05_placebo.png")
    plt.close(fig)

    rt = time.time() - t0
    for i, r in out.iterrows():
        LOG.add(id=f"PL{i + 1:02d}", phase="5_robustness", kind="placebo", hypothesis="H1/H2 위약",
                purpose=r["test"], treated=r["term"], outcome="log|NAV수익률|" if "P4" in r["test"] else "log 거래대금",
                model="접힌 격차 Δ", estimand="위약 Δ", estimate=r["actual"], p=r["p"],
                inference=f"분포 n={r['n']}, 평균 {r['null_mean']:.3f}, sd {r['null_sd']:.3f}",
                interpretation=r.get("note", "") if isinstance(r.get("note", ""), str) else "", runtime_s=rt / len(out))
    LOG.save()
    print(out.round(4).to_string())
    print(p2[["start", "pooled", "idx", "semi", "diff"]].round(3).to_string())


if __name__ == "__main__":
    main()
