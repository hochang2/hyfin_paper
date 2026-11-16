"""RQ2 종합 표: 결과 csv 들을 모아 보고서용 표를 만든다 (재추정 없음).

py -m src.rq2.main_study.synthesis  →  results/tables/rq2_synthesis_*.csv + 콘솔 요약

- rq2_synthesis_main.csv : 설계별 핵심 효과, 모형 구간, 보정된(placebo) p, 최소검출효과(MDE)
  MDE(80% 검정력, 양측 5%) ≈ 2.8 × (귀무 분포의 표준편차). 귀무 분포 = placebo 효과 분포.
- rq2_synthesis_models.csv : 모형별 효과 + placebo_time 위양성률·RMSE (예측 성능과 인과 추정 분리)
- rq2_synthesis_robust.csv : 강건성 사양별 부호·유의성 집계
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

T = Path(__file__).resolve().parent / "results" / "tables"
W = ["transition", "post_effective"]


def r(name):
    return pd.read_csv(T / f"{name}.csv", dtype={"code": str, "series": str})


def main():
    rows = []
    prim = r("rq2_primary_bsts")
    psb = r("rq2_placebo_space_bsts")
    psb_p = r("rq2_placebo_space_bsts_pvalues")
    don = psb[~psb.treated]
    for x in prim[prim.window.isin(W)].itertuples():
        pp = psb_p[(psb_p.code == x.code) & (psb_p.stat == x.window)].perm_p.iloc[0]
        sd = don[x.window].std()
        rows.append({"design": "BSTS (주 분석, donor 반사실)", "code": x.code, "window": x.window, "effect_log": x.effect_log,
                     "effect_pct": np.expm1(x.effect_log) * 100, "model_ci": f"[{x.ci_lo:+.2f}, {x.ci_hi:+.2f}]",
                     "model_p": x.p_two_sided, "p_holm": getattr(x, "p_holm", np.nan),
                     "calibrated_p": pp, "calib_source": "BSTS placebo-in-space (donor 175)",
                     "null_sd": sd, "MDE_log": 2.8 * sd, "MDE_pct_drop": (1 - np.exp(-2.8 * sd)) * 100})
    wf = r("rq2_withinfirm")
    lh = r("rq2_longhist_pvalues")
    for x in wf[(wf.model == "within_DID_HAC_vs_listing") & wf.window.isin(W)].itertuples():
        l = lh[(lh.series == "005930-005935") & (lh.window == x.window)].iloc[0]
        se = (x.ci_hi - x.ci_lo) / (2 * 1.96)
        rows.append({"design": "같은 기업 차분 005930−005935 (상장기간 대비, HAC)", "code": "005930", "window": x.window,
                     "effect_log": x.effect_log, "effect_pct": np.expm1(x.effect_log) * 100,
                     "model_ci": f"[{x.ci_lo:+.2f}, {x.ci_hi:+.2f}]", "model_p": x.p_two_sided, "p_holm": np.nan,
                     "calibrated_p": l.emp_p_two_sided, "calib_source": "GK 차분 긴 표본 placebo (2016~2026.5)",
                     "null_sd": max(se, l.placebo_sd), "MDE_log": 2.8 * max(se, l.placebo_sd),
                     "MDE_pct_drop": (1 - np.exp(-2.8 * max(se, l.placebo_sd))) * 100})
    for x in lh[lh.series.isin(["005930", "000660", "005935"]) & lh.window.isin(W)].itertuples():
        rows.append({"design": "긴 표본 GK OLS (일별 OHLC)", "code": x.series, "window": x.window, "effect_log": x.actual,
                     "effect_pct": np.expm1(x.actual) * 100, "model_ci": "", "model_p": np.nan, "p_holm": np.nan,
                     "calibrated_p": x.emp_p_two_sided, "calib_source": f"긴 표본 placebo {x.n_placebo}개",
                     "null_sd": x.placebo_sd, "MDE_log": 2.8 * x.placebo_sd,
                     "MDE_pct_drop": (1 - np.exp(-2.8 * x.placebo_sd)) * 100})
    if (T / "rq2_longhist_usonly_pvalues.csv").exists():
        us = r("rq2_longhist_usonly_pvalues")
        for x in us[us.window.isin(W)].itertuples():
            rows.append({"design": "긴 표본 GK OLS, 미국 변수만 통제 (국내 통제 오염 대비)", "code": x.series, "window": x.window,
                         "effect_log": x.actual, "effect_pct": np.expm1(x.actual) * 100, "model_ci": "", "model_p": np.nan,
                         "p_holm": np.nan, "calibrated_p": x.emp_p_two_sided, "calib_source": f"긴 표본 placebo {x.n_placebo}개",
                         "null_sd": x.placebo_sd, "MDE_log": 2.8 * x.placebo_sd,
                         "MDE_pct_drop": (1 - np.exp(-2.8 * x.placebo_sd)) * 100})
    if (T / "rq2_withinfirm_calib.csv").exists():
        wc = r("rq2_withinfirm_calib")
        for x in wc[wc.model == "within_bsts_ctrl005935"].itertuples():
            rows.append({"design": "같은 기업 BSTS (005935 를 회귀변수로) — 탐색", "code": "005930", "window": x.window,
                         "effect_log": x.effect_main, "effect_pct": np.expm1(x.effect_main) * 100, "model_ci": "",
                         "model_p": x.model_p, "p_holm": x.model_p_holm8, "calibrated_p": x.calib_p_gk,
                         "calib_source": f"GK 긴 표본 같은 설계 placebo {x.placebo_gk_n}개 (GK 효과 {x.gk_effect:+.2f})",
                         "null_sd": x.placebo_gk_sd, "MDE_log": 2.8 * x.placebo_gk_sd,
                         "MDE_pct_drop": (1 - np.exp(-2.8 * x.placebo_gk_sd)) * 100})
    main_t = pd.DataFrame(rows)
    main_t.to_csv(T / "rq2_synthesis_main.csv", index=False, encoding="utf-8-sig")

    m = r("rq2_models")
    m = m[m.window.isin(W)][["model", "code", "window", "effect_log", "ci_lo", "ci_hi", "p_two_sided", "pre_rmse"]]
    fm = r("rq2_foundation")
    fm = fm[fm.window.isin(W)].rename(columns={"emp_p": "p_two_sided"})[["model", "code", "window", "effect_log", "ci_lo", "ci_hi", "p_two_sided"]]
    allm = pd.concat([m, fm])
    pts = r("rq2_placebo_time_summary")
    pts = pts[pts.h == "h30"][["model", "fpr", "rmse", "mean_effect", "sd_effect"]].rename(
        columns={"fpr": "placebo_fpr_h30", "rmse": "placebo_rmse_h30", "mean_effect": "placebo_bias_h30", "sd_effect": "placebo_sd_h30"})
    fmp = r("rq2_foundation_placebo").groupby("model").agg(placebo_rmse_h30=("rmse30", "mean"),
                                                          placebo_bias_h30=("h30", "mean"), placebo_sd_h30=("h30", "std")).reset_index()
    allm = allm.merge(pd.concat([pts, fmp]), on="model", how="left")
    allm.to_csv(T / "rq2_synthesis_models.csv", index=False, encoding="utf-8-sig")

    rb = r("rq2_robustness")
    rb = rb[rb.window.isin(W) & rb.effect_log.notna()]
    agg = rb.groupby(["model", "code", "window"]).agg(
        n_specs=("spec", "size"), eff_min=("effect_log", "min"), eff_median=("effect_log", "median"),
        eff_max=("effect_log", "max"), n_negative=("effect_log", lambda v: int((v < 0).sum())),
        n_sig05=("sig05", "sum"), n_sig05_negative=("effect_log", lambda v: 0)).reset_index()
    neg_sig = rb[(rb.effect_log < 0) & rb.sig05].groupby(["model", "code", "window"]).size()
    agg["n_sig05_negative"] = [int(neg_sig.get((a, b, c), 0)) for a, b, c in zip(agg.model, agg.code, agg.window)]
    agg.to_csv(T / "rq2_synthesis_robust.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(main_t.round(3).to_string())
    print(allm.pivot_table(index="model", columns=["code", "window"], values="effect_log").round(3).to_string())
    print(allm.drop_duplicates("model")[["model", "placebo_fpr_h30", "placebo_rmse_h30", "placebo_bias_h30", "placebo_sd_h30"]].round(3).to_string())
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
