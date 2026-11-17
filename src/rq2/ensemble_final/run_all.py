"""ensemble_final 전체 실행: 온라인 앙상블 예측 → Exp1~6 → 실험 로그.
py -3.13 -m src.rq2.ensemble_final.run_all [--force]   (모형 재학습 없음. 전체 약 5분, 대부분 Exp5 감사)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from src.rq2.core.explog import ExpLog
from src.rq2.ensemble_final import loader as L
from src.rq2.ensemble_final import online_run as R


def _t(name):
    return pd.read_csv(L.RES / "tables" / f"{name}.csv", dtype={"code": str})


def main(force=False):
    t0 = time.time()
    if force:
        for f in L.CACHE.glob("*.parquet"):
            f.unlink()
    for ds in ("rv5", "rv5d", "gk", "gkd"):
        R.run(ds)
    from src.rq2.ensemble_final import exp1_diversity, exp2_3_tables, exp4_oos, exp5_inference, exp6_real
    exp1_diversity.run()
    exp2_3_tables.run()
    fz, C, RC, rho = exp4_oos.run()
    exp5_inference.run()
    exp6_real.run()
    log(fz, rho)
    print(f"완료 {time.time() - t0:.0f}s")


def log(fz, rho):
    lg = ExpLog(L.RES / "experiment_log.csv")
    seed = L.SEED
    T1 = _t("exp1_model_summary")
    best = T1[T1.window.isin(["h10", "h30"]) & ~T1.model.str.startswith("old")].sort_values("rmse").groupby(["dataset", "window"]).head(1)
    lg.add(id="EF-E1-PERF", rq="RQ2", kind="diagnostic", purpose="개별 모형 효과오차 정확도·편향·창길이·종목·국면",
           data="ml_ext gk(3x238)·rv5(3x32), ml_followup rv5 donor 20·gk_dense", model="13개 기본 모형 + 기존 앙상블 (재학습 없음)",
           settings={"windows": ["h5", "h10", "h20", "h30", "h61", "post_effective"]},
           result="; ".join(f"{r.dataset}/{r.window}: {r.model} {r.rmse:.3f} (BSTS 대비 {r.rel_bsts:.3f})" for r in best.itertuples()),
           interpretation="단일 최고 모형은 자료·창마다 바뀐다 (BSTS·OLS·TimesFM cov 가 번갈아). 큰 격차 없음.", seed=seed)
    PG = _t("exp1_pair_gain_bsts")
    pg = PG.groupby("partner").agg(rho=("corr_bsts", "mean"), g=("rmse_5050_rel", "mean")).sort_values("g")
    lg.add(id="EF-E1-DIV", rq="RQ2", kind="diagnostic", purpose="오차 상관·공분산, BSTS 보완 후보",
           data="창평균·일별 오차", model="-", settings={"pair": "50/50 및 사후 최적 볼록결합(설명용)"},
           result="BSTS 와 평균 오차상관/50:50 상대RMSE: " + ", ".join(f"{k} {v.rho:.2f}/{v.g:.3f}" for k, v in pg.head(5).iterrows()),
           interpretation="UC(0.66)를 빼면 BSTS 와의 오차상관이 0.8~0.98 로 높아 분산 감소 여지가 작다. 공통 양(+) 편향은 결합으로 상쇄되지 않는다.", seed=seed)
    S = _t("exp2_3_summary_by_dataset")
    piv = S.pivot(index="strategy", columns="dataset", values="rel_bsts")
    top = piv.mean(1).sort_values().head(6)
    lg.add(id="EF-E2-SEL", rq="RQ2", kind="extension", purpose="선택적 앙상블 (성능·다양성·제약 stacking·혼합) 온라인 성능",
           data="gk(2018~), rv5(i0>=120), rv5d(i0>=120), gkd(2018~)", model="25개 전략 + meta FTL",
           settings=L.CFG["online"], result="4개 자료 평균 상대RMSE 상위: " + ", ".join(f"{k} {v:.3f}" for k, v in top.items()),
           interpretation="같은 표본에서 수십 개 전략 중 최고 → 선택 편향 있음. 판정은 Exp4.", seed=seed)
    lg.add(id="EF-E3-ADAPT", rq="RQ2", kind="extension", purpose="종목 적응(code/hier/glocal)·국면 적응(vol/trend) 가중치",
           data="같은 온라인 자료", model="simplex_l2, invmse_top3 기준", settings=L.CFG["regime"],
           result="rv5d(donor 20) 에서 hier/glocal 상대RMSE " + ", ".join(f"{k} {piv.loc[k, 'rv5d']:.3f}" for k in ["simplex_l2|hier", "simplex_l2|glocal", "simplex_l2|code"] if k in piv.index)
                  + "; 처치 rv5 " + ", ".join(f"{k} {piv.loc[k, 'rv5']:.3f}" for k in ["simplex_l2|hier", "simplex_l2|regime_trend"] if k in piv.index),
           interpretation="종목 적응의 이득은 종목이 많은 donor 단면에서만 (종목 고유 편향 보정). 처치 3종목·국면 적응은 이득 없음.", seed=seed)
    C = _t("exp4_oos_comparison")
    c = C[(C.role == "overall") & C.available]
    RCt = _t("exp4_reality_check")
    lg.add(id="EF-E4-OOS", rq="RQ2", kind="robustness", purpose="선정 구간(GK 2018-21)과 최종 구간(GK 2022-26) 분리 + rv5·donor 전이",
           data="gk, rv5, rv5d", model=f"사전 고정 선택 {fz['picks']}",
           settings={**L.CFG["oos"], "frozen_at": fz["frozen_at"]},
           result="; ".join(f"{r.dataset}/{r.window}: 상대RMSE {r.rel_bsts:.3f}, ΔMSE {r.dmse_bsts_pct:+.1f}% [{r.lo_bsts_pct:+.1f},{r.hi_bsts_pct:+.1f}]" for r in c.itertuples())
                  + f"; 선정-최종 순위상관 {rho:.2f}; RC p: " + ", ".join(f"{r.dataset}/{r.window} {r.rc_p:.3f}" for r in RCt.itertuples()),
           diagnostics="GK 는 ml_extension 에서 이미 본 자료 — 완전한 외부검증 아님. rv5 최종 구간 i0 14~21개.",
           interpretation="최종 구간에서 BSTS 대비 개선은 h10 에서만 작고 불안정, post_effective 는 악화.", seed=seed)
    A = pd.concat([_t("exp5_audit_gkd"), _t("exp5_audit_gk10_fm")])
    a = A[(A.method == "neff") & (A.null == "cf")]
    keys = ["single_bsts", "stat_simplex", "sel", "old_stack_replica", "simplex"]
    lg.add(id="EF-E5-INF", rq="RQ2", kind="placebo", purpose="앙상블 선택·가중치 학습까지 포함한 시간 placebo FPR·검정력·MDE",
           data="gk_dense 블록(32+1, 3일 격자), GK 10일 격자 블록(10+1, FM 포함)", model=", ".join(keys),
           settings=L.CFG["audit"],
           result="; ".join(f"{r.dataset}/{r.window}/{r.strategy}: FPR {r.fpr:.3f}, power-50 {r['power-50']:.2f}" for _, r in a[a.strategy.isin(keys)].iterrows()),
           interpretation="cross-fitted 귀무는 FPR 를 명목 이하로 유지하나 검정력은 BSTS 와 같거나 낮다 — 추론 개선 없음.", seed=seed)
    E6 = _t("exp6_real_effect")
    e = E6[(E6.window == "transition") & E6.code.isin(["005930", "000660"])]
    lg.add(id="EF-E6-REAL", rq="RQ2", kind="robustness", purpose="실제 규제일 효과 (탐색적 강건성; 주 분석 유지)",
           data="ml_ext space i0=190 (처치 2·음성통제 1·donor 175)", model="BSTS·최고 단일·FM·기존 stacking·사전 고정 앙상블",
           settings=RC_settings(),
           result="; ".join(f"{r.code} {r.strategy}: {r.effect_log:+.3f} (p_time {r.p_time_neff:.2f}, p_space {r.p_space:.2f})" for r in e.itertuples()),
           interpretation="모든 행에서 유의한 효과 없음. 결론(유의한 규제 효과 미검출) 유지.", seed=seed)
    lg.save()


def RC_settings():
    return {**L.CFG["real"], "picks": json.loads((L.RES / "selection_frozen.json").read_text(encoding="utf-8"))["picks"]}


if __name__ == "__main__":
    main(force="--force" in sys.argv)
