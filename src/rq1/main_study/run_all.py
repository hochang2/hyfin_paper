"""RQ1 1차 연구 실행기. RQ2(main_study/run_all.py)와 같은 방식이다.

    py -m src.rq1.main_study.run_all                  # 전 단계
    py -m src.rq1.main_study.run_all --stages data    # 일부 단계만

- 설정: src/rq1/config.yaml (사전 지정 포함)
- 결과: src/rq1/main_study/results/{tables,figures}, experiment_log.{csv,md}, environment.json
- 실험 로그는 단계별로 누적된다. 같은 id 는 새 결과로 바꾼다.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.rq1.main_study import data as D
from src.rq1.main_study import did as M
from src.rq1.main_study import plots as P
from src.rq2.core.explog import ExpLog

OUT = Path(__file__).resolve().parent / "results"
TAB, FIG = OUT / "tables", OUT / "figures"
LOG = ExpLog(OUT / "experiment_log.csv")
SEED = D.CFG["seed"]


def _period(df: pd.DataFrame) -> str:
    return f"{df.date.min():%Y-%m-%d}~{df.date.max():%Y-%m-%d}"


# ====================================================================== stages
def stage_data():
    """1단계: 패널 구성·점검표·기초 그림."""
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = D.panel_long("rv5")
    miss = D.missing_cells("rv5")
    cal = D.calendar_flags()
    post_start = D.CFG["sample"]["treat_start"]

    # 역할별 점검표
    g = df.groupby("role")
    check = pd.DataFrame({
        "n_codes": g.code.nunique(),
        "n_obs": g.size(),
        "n_pre_days": df[df.post == 0].groupby("role").date.nunique(),
        "n_post_days": df[df.post == 1].groupby("role").date.nunique(),
        "n_missing_cells": miss.groupby("role").size(),
        "mean_lrv_pre": df[df.post == 0].groupby("role").y.mean(),
        "mean_lrv_post": df[df.post == 1].groupby("role").y.mean(),
    }).fillna(0)
    check["raw_change_post_minus_pre"] = check.mean_lrv_post - check.mean_lrv_pre
    check.round(4).to_csv(TAB / "rq1_data_check.csv", encoding="utf-8-sig")

    # 종목별 (처리·음성통제) 요약
    focus = df[df.role != "donor"]
    per = focus.groupby(["code", "role", "post"]).y.agg(["size", "mean", "std"]).unstack("post")
    per.columns = [f"{a}_{'post' if b else 'pre'}" for a, b in per.columns]
    per.round(4).to_csv(TAB / "rq1_data_focus_codes.csv", encoding="utf-8-sig")

    miss.to_csv(TAB / "rq1_data_missing.csv", index=False, encoding="utf-8-sig")
    flags = cal[cal.any(axis=1)].assign(post=lambda x: (x.index >= post_start).astype(int))
    flags.to_csv(TAB / "rq1_data_calendar_flags.csv", encoding="utf-8-sig")

    P.data_overview(df, D.CFG, FIG / "rq1_data_overview.png")

    LOG.add(id="RQ1-D00-DATA", rq="RQ1", kind="diagnostic",
            purpose="1단계: 분석 패널 구성과 데이터 점검 (결과 추정 아님)",
            data=f"proc.rv_daily rv5, {_period(df)}",
            model="-",
            settings={"pool": "main", "bin_days": D.BIN_DAYS, "ref_bin_max": D.REF_BIN_MAX,
                      "roles": {k: len(v) for k, v in D.groups().items()}},
            diagnostics=(f"obs={len(df)}, codes={df.code.nunique()}, days={df.date.nunique()} "
                         f"(pre {df[df.post == 0].date.nunique()}, post {df[df.post == 1].date.nunique()}); "
                         f"missing={len(miss)} ({', '.join(f'{r.code} {r.date:%Y-%m-%d}' for r in miss.itertuples())}); "
                         f"gap_day pre {int(flags[(flags.post == 0)].gap_day.sum())} / post {int(flags[(flags.post == 1)].gap_day.sum())}, "
                         f"late_open {int(flags.late_open.sum())}"),
            result="tables/rq1_data_check.csv, rq1_data_focus_codes.csv, rq1_data_missing.csv, "
                   "rq1_data_calendar_flags.csv; figures/rq1_data_overview.png",
            interpretation="결측 칸은 0 으로 채우지 않고 행을 뺀다. 시장 공백일은 모든 종목에 공통이라 DiD 날짜 고정효과가 흡수한다.",
            seed=SEED)


def stage_primary():
    """2단계: 사전 지정 주 검정 P1(005930), P2(000660). 종목별 TWFE DiD + donor 순열 p + Holm + MDE.
    이벤트 스터디 계수와 그 placebo 분포도 함께 만든다 (3단계 사전추세 검정에 쓴다)."""
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    panel = D.panel_long("rv5")
    donors = D.groups()["donor"]
    fam = {p["id"]: p["code"] for p in D.CFG["pre_registration"]["primary_family"]}

    null = M.placebo_space(panel, donors)            # donor 175개 가짜 처리 β
    null.to_csv(TAB / "rq1_placebo_space.csv", encoding="utf-8-sig")
    null_es = M.placebo_space(panel, donors, event_study=True)
    null_es.to_csv(TAB / "rq1_placebo_space_event.csv", encoding="utf-8-sig")
    mde_log = M.mde(null["D"])

    rows, es_rows = [], []
    for pid, code in fam.items():
        b = M.did_beta(panel, code, donors)
        hac = M.did_hac(panel, code, donors)
        p = M.perm_p(b, null["D"])
        rows.append({"id": pid, "code": code, "beta": b, "pct": np.expm1(b), **hac,
                     "perm_p": p, "n_placebo": len(null), "placebo_sd": float(null["D"].std(ddof=1)),
                     "mde_log": mde_log, "mde_pct_up": np.expm1(mde_log), "rank_abs": int((null["D"].abs() >= abs(b)).sum()) + 1})
        d, cols = M.design(panel, code, donors, event_study=True)
        es = M.twfe(d, cols)
        for c in cols:
            lo, hi = null_es[c].quantile([0.025, 0.975])
            es_rows.append({"code": code, "bin": int(c[1:]), "coef": es[c], "placebo_lo": lo, "placebo_hi": hi,
                            "perm_p": M.perm_p(es[c], null_es[c])})
    res = pd.DataFrame(rows)
    hp = M.holm(dict(zip(res.id, res.perm_p)))
    res["holm_p"] = res.id.map(hp)
    res.to_csv(TAB / "rq1_primary_did.csv", index=False, encoding="utf-8-sig")
    es_df = pd.DataFrame(es_rows)
    es_df.to_csv(TAB / "rq1_event_study.csv", index=False, encoding="utf-8-sig")

    P.placebo_hist(null["D"], res, FIG / "rq1_placebo_space.png")
    P.event_study(es_df, FIG / "rq1_event_study.png")

    period = _period(panel)
    for r in res.itertuples():
        LOG.add(id=f"RQ1-{r.id}-{r.code}", rq="RQ1", kind="baseline",
                purpose="사전 지정 주 검정: 상장 후 처리 종목 log RV 의 donor 대비 상대 변화 (DiD)",
                data=f"proc.rv_daily rv5, {period}, 처리 {r.code} + donor main {len(donors)}",
                model="TWFE DiD (종목 FE + 날짜 FE + Treated×Post)",
                settings={"post_start": "2026-05-27", "inference": "donor placebo-in-space |β| 순위", "seed": SEED},
                diagnostics=f"HAC(5) 보고용: β={r.beta_hac:+.3f}, se={r.se_hac:.3f}, p={r.p_hac:.3f}",
                result=(f"β={r.beta:+.3f} ({r.pct:+.1%}), 순열 p={r.perm_p:.3f} (순위 {r.rank_abs}/{r.n_placebo + 1}), "
                        f"Holm p={r.holm_p:.3f}, MDE={r.mde_log:.3f} log (+{r.mde_pct_up:.0%})"),
                interpretation="판정은 3단계에서 조건 (a)(b)(c) 를 모두 확인한 뒤 내린다.", seed=SEED)
    LOG.add(id="RQ1-P-FAMILY", rq="RQ1", kind="baseline", purpose="주 검정 family (P1, P2) 다중검정 보정",
            data="RQ1-P1, RQ1-P2", model="Holm-Bonferroni", settings={"family": list(fam.values())},
            result="; ".join(f"{r.code}: p={r.perm_p:.3f}→Holm {r.holm_p:.3f}" for r in res.itertuples()), seed=SEED)


def _anticipation(panel, treated, controls, start="2026-05-22"):
    """05-22~05-26 구간(D_ant)과 05-27 이후(D_post)를 함께 넣은 DiD."""
    d = panel[panel.code.isin([treated] + list(controls))].copy()
    tr = (d.code == treated).astype(float)
    d["D_ant"] = tr * ((d.date >= start) & (d.post == 0))
    d["D_post"] = tr * d.post
    return M.twfe(d, ["D_ant", "D_post"])


def stage_judgment():
    """3단계: 판정 조건 (b) 음성통제, (c) 상장 전 평행추세, 05-22 사전 반응. 2단계 결과 표를 읽는다."""
    panel = D.panel_long("rv5")
    g = D.groups()
    donors = g["donor"]
    prim = pd.read_csv(TAB / "rq1_primary_did.csv", dtype={"code": str}).set_index("code")
    null = pd.read_csv(TAB / "rq1_placebo_space.csv", dtype={"code": str}).set_index("code")["D"]
    null_es = pd.read_csv(TAB / "rq1_placebo_space_event.csv", dtype={"code": str}).set_index("code")
    es = pd.read_csv(TAB / "rq1_event_study.csv", dtype={"code": str})
    pre_bins = [f"b{b:+d}" for b in range(-8, 0)]

    # ---- (b) 음성통제
    neg_rows = []
    b_same = M.did_beta(panel, g["neg_same"][0], donors)
    neg_rows.append({"for": "005930", "control": "005935 삼성전자우 (같은 기업)", "beta_neg": b_same,
                     "perm_p_neg": M.perm_p(b_same, null), "null": "donor 개별 175",
                     "beta_treated": prim.loc["005930", "beta"]})
    k = len(g["neg_semis"])
    pp = M.portfolio_panel(panel, g["neg_semis"], "SEMIS7")
    b_semis = M.did_beta(pp, "SEMIS7", donors)
    null_pf = M.placebo_portfolios(panel, donors, k=k, n_draw=1000, seed=SEED)
    pd.Series(null_pf, name="beta").to_csv(TAB / "rq1_placebo_portfolio7.csv", index=False, encoding="utf-8-sig")
    neg_rows.append({"for": "000660", "control": f"반도체 {k}종목 평균 (업종)", "beta_neg": b_semis,
                     "perm_p_neg": M.perm_p(b_semis, null_pf), "null": f"donor 무작위 {k}종목 포트폴리오 1000",
                     "beta_treated": prim.loc["000660", "beta"]})
    for c in g["neg_semis"]:
        b = M.did_beta(panel, c, donors)
        neg_rows.append({"for": "000660 (개별 참고)", "control": c, "beta_neg": b,
                         "perm_p_neg": M.perm_p(b, null), "null": "donor 개별 175",
                         "beta_treated": prim.loc["000660", "beta"]})
    neg = pd.DataFrame(neg_rows)
    neg["same_sign"] = np.sign(neg.beta_neg) == np.sign(neg.beta_treated)
    neg["ratio_neg_to_treated"] = neg.beta_neg / neg.beta_treated
    neg.to_csv(TAB / "rq1_negative_control.csv", index=False, encoding="utf-8-sig")

    # ---- (c) 상장 전 평행추세: 상장 전 8개 구간 계수 제곱합의 순열 검정
    s_null = null_es[pre_bins].pow(2).sum(axis=1)
    pt_rows = []
    for code in prim.index:
        coefs = es[es.code == code].set_index("bin")["coef"]
        coefs.index = [f"b{b:+d}" for b in coefs.index]
        s = M.pretrend_stat(coefs, pre_bins)
        pt_rows.append({"code": code, "stat_sumsq": s, "perm_p": float((1 + (s_null >= s).sum()) / (1 + len(s_null))),
                        "mean_pre_coef": float(coefs[pre_bins].mean()), "n_pos_pre": int((coefs[pre_bins] > 0).sum()),
                        "rejected_5pct": None})
    pt = pd.DataFrame(pt_rows)
    pt["rejected_5pct"] = pt.perm_p < 0.05
    pt.to_csv(TAB / "rq1_pretrend.csv", index=False, encoding="utf-8-sig")

    # ---- 05-22 사전 반응
    ant_null = pd.DataFrame([_anticipation(panel, j, [c for c in donors if c != j]) for j in donors], index=donors)
    an_rows = []
    for code in prim.index:
        b = _anticipation(panel, code, donors)
        bin_m1 = es[(es.code == code) & (es.bin == -1)].iloc[0]
        an_rows.append({"code": code, "beta_0522_0526": b["D_ant"], "perm_p_0522_0526": M.perm_p(b["D_ant"], ant_null["D_ant"]),
                        "beta_post_joint": b["D_post"], "bin_m1_coef": bin_m1.coef, "bin_m1_perm_p": bin_m1.perm_p})
    an = pd.DataFrame(an_rows)
    an.to_csv(TAB / "rq1_anticipation.csv", index=False, encoding="utf-8-sig")

    # ---- 판정표 (config decision_rule 그대로 적용)
    jd = []
    for code in prim.index:
        a_ok = bool(prim.loc[code, "holm_p"] < 0.05)
        nrow = neg[neg["for"] == code].iloc[0]
        b_flag = bool(nrow.same_sign and nrow.perm_p_neg < 0.05)
        c_rej = bool(pt.set_index("code").loc[code, "rejected_5pct"])
        verdict = ("판정 보류 (사전추세 기각)" if c_rej else
                   "효과 있음" if (a_ok and not b_flag) else "상장 효과의 증거 없음")
        jd.append({"code": code, "a_holm_p": prim.loc[code, "holm_p"], "a_pass": a_ok,
                   "b_neg_beta": nrow.beta_neg, "b_neg_p": nrow.perm_p_neg, "b_neg_significant_same_sign": b_flag,
                   "c_pretrend_p": pt.set_index("code").loc[code, "perm_p"], "c_rejected": c_rej,
                   "mde_log": prim.loc[code, "mde_log"], "verdict": verdict})
    jdf = pd.DataFrame(jd)
    jdf.to_csv(TAB / "rq1_judgment.csv", index=False, encoding="utf-8-sig")

    for r in neg[neg["for"].isin(["005930", "000660"])].itertuples():
        LOG.add(id=f"RQ1-J-NEG-{r._1}", rq="RQ1", kind="falsification",
                purpose=f"판정 조건 (b): {r._1} 의 음성통제에 같은 DiD", data="proc.rv_daily rv5, 2025-10-01~2026-07-15",
                model="TWFE DiD", settings={"control": r.control, "null": r.null, "seed": SEED},
                result=f"음성통제 β={r.beta_neg:+.3f}, 순열 p={r.perm_p_neg:.3f} (처리 β={r.beta_treated:+.3f}, 비율 {r.ratio_neg_to_treated:.2f})",
                seed=SEED)
    for r in pt.itertuples():
        LOG.add(id=f"RQ1-J-PRE-{r.code}", rq="RQ1", kind="diagnostic", purpose="판정 조건 (c): 상장 전 −8~−1 구간 계수 결합 검정",
                data="rq1_event_study.csv, rq1_placebo_space_event.csv", model="제곱합 통계량 순열 검정 (donor 175)",
                result=f"제곱합={r.stat_sumsq:.3f}, 순열 p={r.perm_p:.3f}, 상장 전 계수 평균 {r.mean_pre_coef:+.3f} ({r.n_pos_pre}/8 양수)",
                seed=SEED)
    for r in an.itertuples():
        LOG.add(id=f"RQ1-J-ANT-{r.code}", rq="RQ1", kind="diagnostic", purpose="05-22 사전 반응 (RQ2 sup-F 시점)",
                data="proc.rv_daily rv5", model="TWFE DiD, D_ant(05-22~05-26) + D_post(05-27~) 동시 추정",
                result=f"05-22~26 β={r.beta_0522_0526:+.3f} (순열 p={r.perm_p_0522_0526:.3f}); −1 구간 {r.bin_m1_coef:+.3f} (p={r.bin_m1_perm_p:.3f})",
                seed=SEED)
    LOG.add(id="RQ1-J-VERDICT", rq="RQ1", kind="baseline", purpose="사전 지정 판정 규칙 적용 결과",
            data="rq1_judgment.csv", model="decision_rule (a)(b)(c)",
            result="; ".join(f"{r.code}: {r.verdict}" for r in jdf.itertuples()),
            interpretation="(b) 의 '비슷한 크기'는 config 에서 수치로 정하지 않았다. 판정표에는 '같은 방향이고 음성통제 자체가 순열 p<0.05' 를 플래그로만 기록했다.",
            seed=SEED)


def stage_robustness():
    """4단계: 보조·강건성. 결론은 바꾸지 않고, 주 결론이 사양을 바꿔도 버티는지 본다."""
    from src.rq1.main_study import robustness as R
    from src.rq2.core.data import load_cfg as rq2_cfg
    panel = D.panel_long("rv5")
    donors = D.groups()["donor"]
    prim = pd.read_csv(TAB / "rq1_primary_did.csv", dtype={"code": str})
    rows = [{"group": "주 분석", "spec": "main (rv5, donor 175)", "code": r.code, "beta": r.beta, "p": r.perm_p,
             "p_type": "donor 순열 (175)"} for r in prim.itertuples()]

    def add(group, spec, res):
        for r in res:
            rows.append({"group": group, "spec": spec, **r})
        print(f"  - {spec}", flush=True)

    for m in ["rv5_close", "rv_pk"]:
        add("측정치", m, R.did_with_placebo(D.panel_long(m), donors))
    add("측정치", "gk (일별 OHLC)", R.did_with_placebo(D.gk_panel_long(), donors))
    add("표본", "05-22~05-26 제외 (donut)",
        R.did_with_placebo(panel[~panel.date.between("2026-05-22", "2026-05-26")], donors))
    exp = D.expiry_days()
    add("표본", f"만기일 {len(exp)}일 제외", R.did_with_placebo(panel[~panel.date.isin(exp)], donors))
    add("대조군", "매칭 20종목 (상장 전 log RV·시총·베타)", R.matched_did(panel, donors, k=20))
    add("보조", "통합 효과 (두 종목 함께)", [R.pooled_did(panel, donors, n_draw=500, seed=SEED)])
    add("보조", "패널 DiD + 종목별 통제", R.did_controls(panel, donors))
    add("보조", "패널 DiD + 자기 시차 통제 (처리·음성통제 공통)", R.did_lag_controls_with_negatives(panel, donors))
    add("보조", "통합 효과 음성통제판 (005935 + 반도체 7 평균)", [R.pooled_negative(panel, donors, n_draw=500, seed=SEED)])
    cfg2 = rq2_cfg()
    add("보조", "시계열 이벤트 회귀 (C_main + HAR)", R.ts_event_regression(cfg2))
    add("같은 기업", "005930 − 005935 (β=1)", R.within_firm(seed=SEED))
    add("반사실", "RQ2 BSTS (학습 ~05-26)", R.bsts_rq2(cfg2))

    out = pd.DataFrame(rows)
    out["pct"] = np.expm1(out.beta.astype(float))
    out.to_csv(TAB / "rq1_robustness.csv", index=False, encoding="utf-8-sig")
    P.robustness_forest(out, FIG / "rq1_robustness_forest.png")
    for (grp, spec), d in out.groupby(["group", "spec"], sort=False):
        if grp == "주 분석":
            continue
        LOG.add(id=f"RQ1-R-{zlib.crc32(spec.encode()) % 10**6:06d}", rq="RQ1",
                kind="robustness" if grp not in ("보조",) else "extension",
                purpose=f"4단계 {grp}: {spec}", data="proc.rv_daily / proc.stock_daily, 2025-10-01~2026-07-15",
                model=spec, settings={"seed": SEED},
                result="; ".join(f"{r.code}: β={float(r.beta):+.3f}, p={float(r.p):.3f} [{r.p_type}]" for r in d.itertuples()),
                interpretation="강건성 결과로 주 결론을 뒤집지 않는다 (config principle).", seed=SEED)


def stage_exploratory():
    """5단계: 탐색 (GARCH-X, 평균 이동 sup-F). 장 마감 30분 분석은 A 와 담당 조율 전이라 하지 않는다."""
    from src.rq1.main_study import exploratory as X
    codes = D.CFG["treated"] + [D.CFG["negative_control"]]

    rets = X.daily_returns(codes)
    g = pd.DataFrame([X.garch_x(c, rets) for c in codes])
    g.to_csv(TAB / "rq1_garch_x.csv", index=False, encoding="utf-8-sig")
    for r in g.itertuples():
        LOG.add(id=f"RQ1-X-GARCH-{r.code}", rq="RQ1", kind="extension",
                purpose="탐색: 일별 수익률 조건부 분산에 상장 기간 더미 (GJR-GARCH-X)",
                data=f"proc.stock_daily ret_log ×100, 2016-01~2026-07-15, n={r.n}",
                model="r=μ+b·donor평균수익률+ε, σ²=ω+αε²+γε²·1[ε<0]+βσ²+δ·L (직접 MLE)",
                diagnostics=f"수렴={r.converged}, 지속성={r.persistence:.3f}",
                result=f"δ={r.delta:+.3f} (SE {r.se_delta:.3f}, 모형 p {r.p_delta:.3f}); δ/ω={r.delta_over_omega:.1f}",
                interpretation="모형 p 이며 보정하지 않았다. 지속성이 1 에 가까워(0.99+) 장기 분산 환산은 불안정해서 하지 않는다. 005935 와 비교해서만 읽는다.", seed=SEED)

    panel = D.panel_long("rv5")
    w = panel.pivot(index="date", columns="code", values="y")
    don = w[D.groups()["donor"]].mean(axis=1)
    series = {"005930 − donor": w["005930"] - don, "000660 − donor": w["000660"] - don,
              "005935 − donor (음성통제)": w["005935"] - don,
              "반도체7 − donor (음성통제)": w[D.CFG["semis_negative_control"]].mean(axis=1) - don,
              "005930 − 005935 (같은 기업)": w["005930"] - w["005935"]}
    rows, curves = [], {}
    for name, s in series.items():
        r, F = X.sup_f_mean_shift(s)
        rows.append({"series": name, **r})
        curves[name] = F
    sf = pd.DataFrame(rows)
    sf.to_csv(TAB / "rq1_supf.csv", index=False, encoding="utf-8-sig")
    P.supf_curves(pd.DataFrame(curves), D.CFG, FIG / "rq1_supf.png")
    for r in sf.itertuples():
        LOG.add(id=f"RQ1-X-SUPF-{zlib.crc32(r.series.encode()) % 10**6:06d}", rq="RQ1", kind="extension",
                purpose=f"탐색: 평균 이동 시점 미지 검정 ({r.series})", data="proc.rv_daily rv5, 2025-10-01~2026-07-15",
                model="sup-F (Andrews 1993, 15% 절사, 동분산 가정)",
                result=f"sup-F={r.sup_F:.1f} ({r.sig}) at {r.break_date}, 이동 {r.shift_at_break:+.2f}; 상장일 F={r.F_at_listing:.1f} (순위 {r.rank_listing}/{r.n_candidates})",
                interpretation="잔차 자기상관을 무시해 과대 기각 경향. 날짜 위치만 참고한다.", seed=SEED)


STAGES = {"data": stage_data, "primary": stage_primary, "judgment": stage_judgment, "robustness": stage_robustness,
          "exploratory": stage_exploratory}


def write_env():
    import importlib
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ["numpy", "pandas", "statsmodels", "scipy", "linearmodels", "arch", "matplotlib", "duckdb"]:
        try:
            v[m] = getattr(importlib.import_module(m), "__version__", "?")
        except ImportError:
            v[m] = "not installed"
    (OUT / "environment.json").write_text(json.dumps(v, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(STAGES))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows cp949 콘솔에서 '−' 등 출력 오류 방지
    OUT.mkdir(parents=True, exist_ok=True)
    old = OUT / "experiment_log.csv"
    if old.exists():   # 이전 단계 기록 유지: 같은 id 는 새 결과로 교체
        LOG.rows = pd.read_csv(old, dtype=str).fillna("").to_dict("records")
    write_env()
    for st in a.stages.split(","):
        t0 = time.time()
        print(f"[RQ1] {st} ...", flush=True)
        n_before = len(LOG.rows)
        STAGES[st]()
        new_ids = {r["id"] for r in LOG.rows[n_before:]}
        LOG.rows = [r for r in LOG.rows[:n_before] if r["id"] not in new_ids] + LOG.rows[n_before:]
        LOG.save()
        print(f"[RQ1] {st} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
