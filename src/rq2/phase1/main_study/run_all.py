"""RQ2 전체 실험 실행.  py -m src.rq2.phase1.main_study.run_all [--stages primary,models,...] [--fast]

단계 (순서대로, 각 단계는 결과 표를 results/tables 에 저장하고 실험 로그에 기록):
  primary     사전 지정 주 분석: BSTS(C_main, 학습≤07-15) — 005930, 000660 + 음성통제 005935, Holm 보정
  models      대안 반사실 모형 비교 (같은 설계, 같은 효과 창)
  withinfirm  같은 기업 비교: 005930 vs 005935 (차이 시계열 ITS + 005935 를 통제로 쓴 BSTS/OLS)
  diagnostics 사전구간 회귀 진단 + 정상성 + 전체표본 ITS (Oster δ, Cinelli-Hazlett RV)
  placebo_time 분봉 사전구간 가짜 개입일 (모형별 위양성률·예측편향)
  placebo_space donor 종목에 같은 개입일을 준 placebo (OLS, SCM, BSTS)
  longhist    F1 2015~ Garman-Klass: 긴 사전구간 placebo 분포 + GJR-GARCH 고유변동성
  robustness  사양 민감도 격자 (통제·donor·측정치·사건일·학습창·donut·이상치·사전분포)
  figures     그림
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.rq2.phase1.main_study import diagnostics as D
from src.rq2.phase1.main_study import plots as P
from src.rq2.core.data import build_controls, donor_codes, load_long_daily, load_panel
from src.rq2.core.explog import ExpLog
from src.rq2.core.models import NDRAW, stationary_bootstrap_paths
from src.rq2.core.pipeline import (CFG, TL, effect_windows, holm, listing_dummy, rows_df, run_spec,
                              winsorize_transform, within_firm_control_X)

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "results"
TAB, FIG = OUT / "tables", OUT / "figures"
CODES = ["005930", "000660"]
NEG = "005935"
ALL3 = CODES + [NEG]
BSTS_MAIN = dict(niter=CFG["bsts"]["niter"], burn=CFG["bsts"]["burn"],
                 expected_model_size=CFG["bsts"]["expected_model_size"],
                 prior_level_sd=CFG["bsts"]["prior_level_sd"], prior_level_df=CFG["bsts"]["prior_level_df"])
BSTS_FAST = dict(BSTS_MAIN, niter=2000, burn=500)
LOG = ExpLog(OUT / "experiment_log.csv")
KEYW = ["transition", "post_effective"]


def save(df: pd.DataFrame, name: str):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False, encoding="utf-8-sig")


def fmt(df, cols=("code", "window", "effect_log", "ci_lo", "ci_hi", "p_two_sided")):
    d = df[df.window.isin(KEYW)][list(cols)]
    return "; ".join(f"{r.code}/{r.window}: {r.effect_log:+.3f} [{r.ci_lo:+.3f},{r.ci_hi:+.3f}] p={r.p_two_sided:.3f}"
                     for r in d.itertuples())


# ====================================================================== primary
def stage_primary():
    rows, paths = [], []
    for code in ALL3:
        cf, y, r = run_spec(dict(code=code, model="bsts", model_kw=BSTS_MAIN))
        rows.append(rows_df(r, code=code, model="bsts"))
        lo, hi = np.quantile(cf.draws, [0.025, 0.975], axis=0)
        p = pd.DataFrame({"date": y.index, "code": code, "y": y.values})
        p["fit"] = cf.fitted_pre.reindex(y.index).values
        p.loc[p.date.isin(cf.pred.index), "cf"] = cf.pred.values
        p.loc[p.date.isin(cf.pred.index), "lo"] = lo
        p.loc[p.date.isin(cf.pred.index), "hi"] = hi
        paths.append(p)
        LOG.add(id=f"RQ2-B01-{code}", rq="RQ2", kind="baseline" if code != NEG else "falsification",
                purpose="사전 지정 주 분석: 규제가 없었을 때의 log RV 반사실 (CausalImpact 방식)"
                if code != NEG else "음성통제: 상품이 없는 삼성전자우에 같은 분석 — 효과가 나오면 규제 귀속 불가",
                data=f"proc.rv_daily rv5 {code}, 2025-10-01~2026-10-30, 학습≤07-15 (190일)",
                model="BSTS local level + spike-and-slab (src/rq2/bsts.py)",
                settings={"controls": CFG["controls"]["C_main"], **BSTS_MAIN, "seed": CFG["seed"]},
                result=fmt(rows[-1]), diagnostics=f"inclusion={cf.info['inclusion']}")
    df = pd.concat(rows)
    fam = df[df.code.isin(CODES) & df.window.isin(KEYW)].copy()
    fam["p_holm"] = holm(fam.p_two_sided).values
    df = df.merge(fam[["code", "window", "p_holm"]], on=["code", "window"], how="left")
    save(df, "rq2_primary_bsts")
    save(pd.concat(paths), "rq2_primary_paths")
    LOG.add(id="RQ2-B01-FAMILY", rq="RQ2", kind="baseline", purpose="주 가설 검정 family (2종목×2창) 다중검정 보정",
            data="RQ2-B01", model="Holm-Bonferroni", settings={"family": "005930/000660 × transition/post_effective"},
            result="; ".join(f"{r.code}/{r.window}: p={r.p_two_sided:.3f}→Holm {r.p_holm:.3f}" for r in fam.itertuples()))
    return df


# ====================================================================== models
MODEL_DESC = {
    "ols": "ITS-OLS (사전 회귀 + HAC, 계수·잔차 block bootstrap)",
    "bsts": "BSTS (주 분석)", "uc": "상태공간 UC: local level + 회귀 (MLE)",
    "sarimax": "SARIMAX: ARMA + exog (AIC 선택)", "harx": "HAR-RV-X (재귀 반사실 시차)",
    "ridge": "Ridge (TimeSeriesSplit CV)", "enet": "ElasticNet (TimeSeriesSplit CV)",
    "lgbm": "LightGBM (얕은 트리, OOF 잔차 bootstrap)", "xgb": "XGBoost (얕은 트리, OOF 잔차 bootstrap)",
    "scm": "합성통제 (절편 허용, 가중치≥0 합1, donor log RV)",
}


def stage_models():
    out = []
    for m in MODEL_DESC:
        for code in ALL3:
            t0 = time.time()
            try:
                cf, y, r = run_spec(dict(code=code, model=m, model_kw=BSTS_MAIN if m == "bsts" else {}))
                d = rows_df(r, code=code, model=m)
                d["info"] = json.dumps(cf.info, ensure_ascii=False, default=str)[:400]
                out.append(d)
                status, res = "ok", fmt(d)
            except Exception as e:  # noqa: BLE001
                status, res = "failed", repr(e)[:300]
            LOG.add(id=f"RQ2-E-{m.upper()}-{code}", rq="RQ2", kind="extension", purpose="모형 의존성 확인 (같은 설계·같은 창)",
                    data=f"rv5 {code}, 학습≤07-15", model=MODEL_DESC[m], settings={"controls": "C_main"},
                    result=res, status=status, interpretation=f"{time.time()-t0:.0f}s")
    LOG.add(id="RQ2-E-FOUNDATION", rq="RQ2", kind="extension", status="skipped",
            purpose="TimesFM/Chronos 등 시계열 foundation model 반사실",
            model="TimesFM / Chronos (미설치)", data="-", settings="-",
            result="실행하지 않음: 패키지·가중치가 환경에 없고, 설치에는 외부 모델 가중치 다운로드가 필요하다.",
            interpretation="적합성 판단: 단변량 zero-shot 예측은 사후구간의 시장 전체 변동성 하락(공통충격)을 통제하지 못해 "
                           "인과 반사실로 부적합. 공변량을 받는 버전이라도 사전 190일로 검증 가능한 이점이 없다. 후속 과제로만 남김.")
    df = pd.concat(out)
    save(df, "rq2_models")
    return df


# ====================================================================== within-firm
def stage_withinfirm():
    lv = load_panel("rv5").loc[CFG["sample"]["start"]:CFG["sample"]["end"]]
    d = (lv["005930"] - lv[NEG]).rename("d")
    idx = d.index
    W = effect_windows(idx)
    dm = pd.DataFrame({"listing": W["listing"], "transition": W["transition"], "post_effective": W["post_effective"]}).astype(float)
    X = build_controls(CFG)[["gap_day"]].reindex(idx)
    m = sm.OLS(d, sm.add_constant(pd.concat([dm, X], axis=1))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    rows = [{"code": "005930", "model": "within_DID_HAC", "window": k, "effect_log": m.params[k],
             "ci_lo": m.conf_int().loc[k, 0], "ci_hi": m.conf_int().loc[k, 1], "p_two_sided": m.pvalues[k]}
            for k in ["listing", "transition", "post_effective"]]
    # 기준: 상장 전 평균 대비. 상장기간 대비로 바꾸면 (규제 효과 = post - listing)
    for k in ["transition", "post_effective"]:
        c = pd.Series(0.0, index=m.params.index)
        c[k], c["listing"] = 1, -1
        t = m.t_test(c.values)
        rows.append({"code": "005930", "model": "within_DID_HAC_vs_listing", "window": k, "effect_log": float(t.effect[0]),
                     "ci_lo": float(t.conf_int()[0, 0]), "ci_hi": float(t.conf_int()[0, 1]), "p_two_sided": float(t.pvalue)})
    wx = within_firm_control_X()
    for mod in ["ols", "bsts"]:
        cf, y, r = run_spec(dict(code="005930", model=mod, controls=["gap_day"], extra_X=wx,
                                 model_kw=BSTS_MAIN if mod == "bsts" else {}))
        for x in r:
            x.update(code="005930", model=f"within_{mod}_ctrl005935")
        rows += r
    df = pd.DataFrame(rows)
    save(df, "rq2_withinfirm")
    LOG.add(id="RQ2-E-WITHIN", rq="RQ2", kind="extension",
            purpose="같은 기업·같은 펀더멘털, 상품만 없는 삼성전자우를 통제로 써 기업·업종 충격 제거",
            data="rv5 005930, 005935", model="차이 시계열 ITS(HAC 5) / 005935 를 회귀변수로 둔 OLS·BSTS",
            settings={"dummies": "listing, transition, post_effective (기준=상장 전)"},
            result="; ".join(f"{r['model']}/{r['window']}: {r['effect_log']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
                             f"p={r.get('p_two_sided', np.nan):.3f}" for r in rows if r["window"] in KEYW + ["listing"]),
            interpretation="SK하이닉스에는 같은 기업 대조가 없다 (우선주 없음).")
    return df


# ====================================================================== within-firm 보정 (calibration)
SUBW = {"transition": (0, 10), "post_effective": (10, 71), "post_eff_20d": (10, 30), "post_eff_21_61d": (30, 71)}


def _emp_p(pl: pd.Series, a: float) -> float:
    """중심화한 양측 경험 p: placebo 평균에서 떨어진 정도로 비교."""
    m = pl.mean()
    return (np.sum(np.abs(pl - m) >= abs(a - m)) + 1) / (len(pl) + 1)


def stage_withinfirm_calib():
    """같은 기업 대조(005930 ~ 005935) 결과의 보정.
    A) rv5 가짜 개입일 placebo: 같은 모형(OLS/BSTS, X = lrv_005935 + gap_day), 가짜일 이후 71일(10+61) 창
    B) 긴 표본 GK placebo: OLS(GK_005930 ~ GK_005935), 학습 250일, 2016~상장 전
    C) 같은 기업 탐색 family (모형 2 × 창 4 = 8 검정) Holm 보정 — 모형 p 와 보정 p 모두
    """
    wx = within_firm_control_X()
    idx = load_panel("rv5").loc[CFG["sample"]["start"]:CFG["sample"]["end"]].dropna(subset=["005930", NEG]).index
    i_ann = idx.get_loc(TL["announce"])
    n_win = 71
    kw = {"ols": {}, "bsts": dict(BSTS_MAIN, niter=1500, burn=500)}
    rows = []
    fakes = list(range(60, i_ann - n_win + 1, 3))
    for i0 in fakes + [i_ann]:
        actual = i0 == i_ann
        tr_end = idx[i0 - 1]
        s_end = idx[min(i0 + n_win - 1, len(idx) - 1)]
        for mod in ["ols", "bsts"]:
            t0 = time.time()
            cf, y, _ = run_spec(dict(code="005930", model=mod, controls=["gap_day"], extra_X=wx,
                                     train_end=tr_end, sample_end=s_end, model_kw=kw[mod]))
            e = (y[y.index > tr_end] - cf.pred).values
            r = {"src": "rv5_placebo_time", "model": f"within_{mod}_ctrl005935", "date": idx[i0], "actual": actual}
            for k, (a, b) in SUBW.items():
                r[k] = float(np.mean(e[a:b])) if len(e) >= b else np.nan
            rows.append(r)
            print(f"  withinfirm_calib A {idx[i0].date()} {mod} {time.time() - t0:.1f}s", flush=True)
    # B) 긴 표본
    L = load_long_daily(CFG).loc["2015-06-01":]
    li = L.index
    j_ann = li.get_loc(TL["announce"])
    n_tr = 250
    df_all = L[["005930", NEG]].dropna()

    def eff_long(i0):
        t0 = li[i0]
        tr = df_all[(df_all.index < t0)].iloc[-n_tr:]
        po = df_all[df_all.index >= t0].iloc[:n_win]
        if len(tr) < n_tr * 0.8 or len(po) < n_win:
            return None
        m = sm.OLS(tr["005930"], sm.add_constant(tr[NEG])).fit()
        e = (po["005930"] - m.predict(sm.add_constant(po[NEG], has_constant="add"))).values
        return {k: float(np.mean(e[a:b])) for k, (a, b) in SUBW.items()}

    last_ok = li.searchsorted(TL["listing"]) - n_win
    for i0 in list(range(li.searchsorted(pd.Timestamp("2016-06-01")), last_ok, 5)) + [j_ann]:
        r = eff_long(i0)
        if r:
            rows.append({"src": "gk_longhist", "model": "within_gk_ols_ctrl005935", "date": li[i0],
                         "actual": i0 == j_ann, **r})
    pl = pd.DataFrame(rows)
    save(pl, "rq2_withinfirm_calib_placebo")
    # 결과 표
    wf = pd.read_csv(TAB / "rq2_withinfirm.csv", dtype={"code": str})
    lg = pl[pl.src == "gk_longhist"]
    out = []
    for mod in ["within_ols_ctrl005935", "within_bsts_ctrl005935"]:
        g = pl[(pl.src == "rv5_placebo_time") & (pl.model == mod)]
        for k in SUBW:
            orig = wf[(wf.model == mod) & (wf.window == k)].iloc[0]
            a_rv = g[g.actual][k].iloc[0]
            plr = g[~g.actual][k].dropna()
            a_gk = lg[lg.actual][k].iloc[0]
            plg = lg[~lg.actual][k].dropna()
            out.append({"model": mod, "window": k, "effect_main": orig.effect_log, "model_p": orig.p_two_sided,
                        "effect_rerun": a_rv, "placebo_rv5_n": len(plr), "placebo_rv5_mean": plr.mean(),
                        "placebo_rv5_sd": plr.std(), "fpr_model_ci_proxy": np.nan,
                        "calib_p_rv5": _emp_p(plr, a_rv),
                        "gk_effect": a_gk, "placebo_gk_n": len(plg), "placebo_gk_mean": plg.mean(),
                        "placebo_gk_sd": plg.std(), "calib_p_gk": _emp_p(plg, a_gk)})
    out = pd.DataFrame(out).drop(columns="fpr_model_ci_proxy")
    for c in ["model_p", "calib_p_rv5", "calib_p_gk"]:
        out[c + "_holm8"] = holm(out[c]).values
    save(out, "rq2_withinfirm_calib")
    b = out[(out.model == "within_bsts_ctrl005935") & (out.window == "post_eff_21_61d")].iloc[0]
    LOG.add(id="RQ2-R-WITHIN-CALIB", rq="RQ2", kind="robustness",
            purpose="같은 기업 대조 결과(특히 BSTS post_eff_21_61d −0.25, 모형 p=0.023)를 다중검정·placebo 로 보정",
            data="A) rv5 2025-10~2026-07-15 가짜 개입일; B) F1 GK 2016~2026-05 가짜 개입일",
            model="A) 주 분석과 같은 within OLS/BSTS(1500/500); B) OLS GK_005930 ~ GK_005935 (학습 250일)",
            settings={"window_days": "10 + 61", "rv5_placebos": len(fakes), "rv5_step": 3, "gk_step": 5,
                      "family": "2 모형 × 4 창 = 8, Holm"},
            result="; ".join(f"{r.model.split('_')[1]}/{r.window}: {r.effect_main:+.3f} p={r.model_p:.3f} "
                             f"(Holm {r.model_p_holm8:.3f}), p_rv5={r.calib_p_rv5:.3f} (Holm {r.calib_p_rv5_holm8:.3f}), "
                             f"GK {r.gk_effect:+.3f} p_gk={r.calib_p_gk:.3f}" for r in out.itertuples()),
            status="ok (A=failed: 보정 불가 판정)",
            interpretation=f"핵심 결과 BSTS 21~61일: 모형 p {b.model_p:.3f} → Holm {b.model_p_holm8:.3f}; "
                           f"GK 긴 표본 같은 설계 {b.gk_effect:+.3f}, p {b.calib_p_gk:.3f} → 유지되지 않음. "
                           "A(rv5 placebo)는 쓸 수 없다: 가짜 창 20개가 71일씩 겹쳐 사실상 독립 표본이 2~3개이고, "
                           "placebo 평균이 +0.26~+0.30 으로 치우쳤다. 원인은 005930~005935 회귀계수 β≈0.77~0.84(<1)인데 "
                           "두 종목 수준이 2025-10~2026-07 에 약 2 로그단위 올라 (1−β)×수준변화 만큼 반사실이 낮게 예측된 것. "
                           "효과가 0 인 OLS 도 같은 최소 p(1/21)를 받는다 → p_rv5 는 효과가 아니라 수준 추세를 잰다. "
                           "결론: β=1 을 강제한 차분 설계(상장기간 대비 +0.02, p 0.79)가 같은 기업 비교의 주 사양이다.")
    print(out.round(3).to_string(), flush=True)
    return out


# ====================================================================== diagnostics
def stage_diagnostics():
    diag, stat, its = [], [], []
    ctrl = CFG["controls"]["C_main"]
    C = build_controls(CFG)
    for code in ALL3:
        lv = load_panel("rv5")[code]
        df = pd.concat([lv.rename("y"), C[ctrl]], axis=1).dropna().loc[CFG["sample"]["start"]:]
        pre = df[df.index <= CFG["sample"]["train_end"]]
        dg = D.regression_diagnostics(pre.y, pre[ctrl], TL["listing"], seed=CFG["seed"])
        dg["code"] = code
        diag.append(dg)
        for col in ["y"] + [c for c in ctrl if c != "gap_day"]:
            s = D.stationarity(pre[col])
            s.update(code=code, series=col, sample="pre")
            stat.append(s)
        e = sm.OLS(pre.y, sm.add_constant(pre[ctrl])).fit().resid
        s = D.stationarity(e)
        s.update(code=code, series="resid", sample="pre")
        stat.append(s)
        # 전체표본 ITS: y = a + τ_L 1[t≥상장] + τ_T transition + τ_P post_effective + Xβ
        # → τ_T, τ_P 는 '상품은 있고 규제는 없던 상장기간' 대비 효과 (주 분석 BSTS 의 추정대상과 같음)
        W = effect_windows(df.index)
        dm = pd.DataFrame({"listing_regime": df.index >= TL["listing"], "transition": W["transition"],
                           "post_effective": W["post_effective"]}, index=df.index).astype(float)
        mh, mo = D.its_regression(df.y, df[ctrl], dm)
        ms_h, ms_o = D.its_regression(df.y, df[[]], dm)
        for k in ["transition", "post_effective"]:
            r = {"code": code, "window": k, "effect_log": mh.params[k], "ci_lo": mh.conf_int().loc[k, 0],
                 "ci_hi": mh.conf_int().loc[k, 1], "p_hac": mh.pvalues[k], "p_ols": mo.pvalues[k],
                 "b_short": ms_o.params[k], "r2_short": ms_o.rsquared, "r2_long": mo.rsquared}
            r.update(D.oster_delta(ms_o.params[k], ms_o.rsquared, mo.params[k], mo.rsquared))
            r.update(D.robustness_value(mo.tvalues[k], int(mo.df_resid)))
            r["RV_hac_q1"] = D.robustness_value(mh.tvalues[k], int(mo.df_resid))["RV_q1"]
            its.append(r)
        dg_full = D.regression_diagnostics(df.y, pd.concat([dm, df[ctrl]], axis=1), TL["listing"], seed=CFG["seed"], nboot=99)
        dg_full["code"] = code + "_ITSfull"
        diag.append(dg_full)
    diag = pd.DataFrame(diag)
    for c in ["vif", "top_influential", "se_ratio_hac_over_ols"]:
        diag[c] = diag[c].map(lambda v: json.dumps(v, ensure_ascii=False))
    save(diag, "rq2_diagnostics")
    save(pd.DataFrame(stat), "rq2_stationarity")
    its = pd.DataFrame(its)
    save(its, "rq2_its_sensitivity")
    LOG.add(id="RQ2-D01", rq="RQ2", kind="diagnostic", purpose="사전구간 회귀 (y~C_main) 가정 점검 — BSTS·OLS·SARIMAX 회귀부 공통",
            data="rv5, 학습≤07-15", model="OLS 잔차 검정 묶음",
            settings="LB(5,10), BG(5), BP, White, JB, VIF, RESET(3), Cook's D, Chow@05-27, sup-F(wild boot 199), CUSUM",
            result="; ".join(f"{r['code']}: LB10 p={r['lb10_p']:.3f}, BP p={r['bp_p']:.3f}, JB p={r['jb_p']:.3f}, "
                             f"VIFmax={r['vif_max']:.1f}, RESET p={r['reset_p']:.3f}, Chow p={r['chow_p']:.3f}, "
                             f"supF p={r['supF_p_boot']:.3f}@{r['supF_date']}" for r in diag.to_dict("records")))
    LOG.add(id="RQ2-D02", rq="RQ2", kind="diagnostic", purpose="정상성 (모형 차분 여부 결정)", data="사전구간",
            model="ADF(AIC) + KPSS(c)", settings="", result={f"{a}/{b}": int(n) for (a, b), n in pd.DataFrame(stat).groupby(["code", "verdict"]).size().items()})
    LOG.add(id="RQ2-D03", rq="RQ2", kind="robustness", purpose="ITS 효과의 비관측 교란 민감도 (Oster δ, Cinelli-Hazlett RV)",
            data="rv5 전 표본", model="ITS-OLS (HAC 5)", settings={"Rmax": "1.3×R2_long"},
            result="; ".join(f"{r.code}/{r.window}: τ={r.effect_log:+.3f} pHAC={r.p_hac:.3f} δ={r.delta_for_zero:.2f} "
                             f"RV={r.RV_q1:.3f} RVα={r.RV_q1_alpha05:.3f}" for r in its.itertuples()))
    return diag, its


# ====================================================================== placebo in time
PT_MODELS = {"ols": {}, "bsts": dict(BSTS_MAIN, niter=1500, burn=500), "uc": {}, "harx": {}, "ridge": {},
             "lgbm": {}, "scm": {}, "sarimax": {"order": (1, 0, 0)}}


def stage_placebo_time():
    idx = load_panel("rv5").loc[CFG["sample"]["start"]:].index
    p = CFG["placebo"]
    cands = idx[(idx >= pd.Timestamp(p["in_time_start"])) & (idx <= pd.Timestamp(p["in_time_end"]))][::p["in_time_step"]]
    H = p["in_time_horizon"]
    rows = []
    for d in cands:
        i = idx.get_loc(d)
        tr_end, s_end = idx[i - 1], idx[min(i + H - 1, len(idx) - 1)]
        if s_end >= TL["listing"]:
            continue
        for code in ALL3:
            for m, kw in PT_MODELS.items():
                t0 = time.time()
                try:
                    cf, y, _ = run_spec(dict(code=code, model=m, train_end=tr_end, sample_end=s_end, model_kw=kw))
                except Exception as e:  # noqa: BLE001
                    rows.append({"date": d, "code": code, "model": m, "error": repr(e)[:100]})
                    print(f"  placebo_time {d.date()} {code} {m} ERROR {repr(e)[:80]}", flush=True)
                    continue
                print(f"  placebo_time {d.date()} {code} {m} {time.time() - t0:.1f}s", flush=True)
                yp = y[y.index > tr_end].values
                for hname, h in [("h10", 10), ("h30", H)]:
                    e = (yp[:h][None, :] - cf.draws[:, :h]).mean(1)
                    pt = float((yp[:h] - cf.pred.values[:h]).mean())
                    lo, hi = np.quantile(e, [.025, .975])
                    rows.append({"date": d, "code": code, "model": m, "h": hname, "effect_log": pt, "ci_lo": lo,
                                 "ci_hi": hi, "reject": not (lo < 0 < hi),
                                 "rmse": float(np.sqrt(np.mean((yp[:h] - cf.pred.values[:h]) ** 2)))})
    df = pd.DataFrame(rows)
    save(df, "rq2_placebo_time")
    s = df.dropna(subset=["effect_log"]).groupby(["model", "h"]).agg(
        n=("effect_log", "size"), mean_effect=("effect_log", "mean"), sd_effect=("effect_log", "std"),
        mean_abs=("effect_log", lambda v: np.mean(np.abs(v))), fpr=("reject", "mean"), rmse=("rmse", "mean")).reset_index()
    save(s, "rq2_placebo_time_summary")
    LOG.add(id="RQ2-P01", rq="RQ2", kind="placebo", purpose="가짜 개입일(상장 전)에서 모형별 위양성률·예측 편향 → 구간 보정 확인",
            data=f"rv5, 가짜 개입일 {len(cands)}개 ({p['in_time_start']}~{p['in_time_end']}, 5일 간격), 사후 10·30일",
            model=",".join(PT_MODELS), settings={"train": "표본시작~가짜일 전날"},
            result="; ".join(f"{r.model}/{r.h}: FPR={r.fpr:.2f}, |eff|={r.mean_abs:.3f}, RMSE={r.rmse:.3f}"
                             for r in s.itertuples()),
            interpretation="FPR 이 0.05 를 크게 넘는 모형의 구간은 과소추정 → 그 모형의 p 값은 해석하지 않는다.")
    return df, s


# ====================================================================== placebo in space
def stage_placebo_space():
    lv = load_panel("rv5").loc[CFG["sample"]["start"]:CFG["sample"]["end"]]
    donors = donor_codes(CFG, "main")
    C = build_controls(CFG)
    ctrl = CFG["controls"]["C_main"]
    tr = lv.index <= CFG["sample"]["train_end"]
    W = effect_windows(lv.index[~tr])
    S = lv[donors].sum(axis=1)
    rows = []
    for j in donors + ALL3:
        y = lv[j]
        X = C[ctrl].copy()
        if j in donors:
            X["don_mean"] = (S - y) / (len(donors) - 1)       # 자기 자신은 donor 평균에서 뺀다
        df = pd.concat([y.rename("y"), X], axis=1).dropna()
        trm = df.index <= CFG["sample"]["train_end"]
        Xc = sm.add_constant(df[ctrl])
        m = sm.OLS(df.y[trm], Xc[trm]).fit()
        e_post = df.y[~trm] - m.predict(Xc[~trm])
        pre_rmse = np.sqrt(np.mean(m.resid ** 2))
        r = {"code": j, "treated": j in ALL3, "pre_rmse": pre_rmse}
        for k in KEYW:
            mk = W[k].reindex(e_post.index).fillna(False).values
            r[f"ols_{k}"] = e_post[mk].mean()
            r[f"ols_{k}_std"] = e_post[mk].mean() / pre_rmse
        rows.append(r)
    df = pd.DataFrame(rows)
    # SCM placebo: post/pre RMSPE 비 (Abadie 2010)
    from src.rq2.core.models import cf_scm
    trs = pd.Series(tr, index=lv.index)
    scm = []
    panel = lv[donors + ALL3].dropna(axis=1)
    for j in [c for c in donors + ALL3 if c in panel.columns]:
        dn = [c for c in donors if c != j and c in panel.columns]
        cf = cf_scm(panel, j, dn, trs, ~trs, seed=0)
        gap = panel[j] - pd.concat([cf.fitted_pre, cf.pred])
        pre = np.sqrt(np.mean(gap[tr] ** 2))
        r = {"code": j, "scm_pre_rmspe": pre}
        for k in KEYW:
            mk = W[k].reindex(gap.index).fillna(False).values
            r[f"scm_{k}"] = gap[mk].mean()
            r[f"scm_{k}_ratio"] = np.sqrt(np.mean(gap[mk] ** 2)) / pre
        scm.append(r)
    df = df.merge(pd.DataFrame(scm), on="code", how="left")
    save(df, "rq2_placebo_space")
    don = df[~df.treated]
    res = []
    for code in ALL3:
        t = df[df.code == code].iloc[0]
        for k in KEYW:
            for col in [f"ols_{k}_std", f"scm_{k}_ratio", f"scm_{k}"]:
                v = don[col].dropna()
                if col.endswith("ratio"):
                    pr = (np.sum(v >= t[col]) + 1) / (len(v) + 1)
                else:
                    pr = (np.sum(np.abs(v) >= abs(t[col])) + 1) / (len(v) + 1)
                res.append({"code": code, "window": k, "stat": col, "value": t[col], "perm_p": pr,
                            "donor_q05": v.quantile(.05), "donor_q95": v.quantile(.95)})
    res = pd.DataFrame(res)
    save(res, "rq2_placebo_space_pvalues")
    LOG.add(id="RQ2-P02", rq="RQ2", kind="placebo", purpose="placebo-in-space: 규제 없는 donor 종목에 같은 개입일 → 처치 종목 효과가 이례적인가",
            data=f"rv5, main donor {len(donors)}종목", model="OLS(C_main, 자기 제외 donor 평균) / SCM RMSPE 비",
            settings="순열 p = (#{|donor| ≥ |treated|}+1)/(N+1)",
            result="; ".join(f"{r.code}/{r.window}/{r.stat}: {r.value:+.3f}, p={r.perm_p:.3f}" for r in res.itertuples()))
    return df, res


# ====================================================================== long history
def stage_longhist():
    L = load_long_daily(CFG)
    L = L.loc["2015-06-01":]
    X_cols = ["don_mean", "us_lvix", "us_abs_sox", "kosdaq_ret_abs"]
    idx = L.index
    i_ann = idx.get_loc(TL["announce"])
    i_eff = idx.get_loc(TL["effective"])
    n_tr, n_tran, n_post = 250, i_eff - i_ann, min(61, len(idx) - i_eff)
    series = {c: L[c] for c in ALL3}
    series["005930-005935"] = L["005930"] - L[NEG]

    def effect_at(ys, i0):
        df = pd.concat([ys.rename("y"), L[X_cols]], axis=1).iloc[i0 - n_tr:i0 + n_tran + n_post].dropna()
        if len(df) < n_tr * 0.8 + n_tran + n_post * 0.8:
            return None
        t0 = idx[i0]
        tr = df.index < t0
        Xc = sm.add_constant(df[X_cols]) if not ys.name or "-" not in str(ys.name) else sm.add_constant(df[X_cols])
        m = sm.OLS(df.y[tr], Xc[tr]).fit()
        e = df.y[~tr] - m.predict(Xc[~tr])
        return {"transition": e.iloc[:n_tran].mean(), "post_effective": e.iloc[n_tran:].mean(),
                "pre_rmse": np.sqrt(m.mse_resid)}

    rows = []
    last_ok = idx.searchsorted(TL["listing"]) - n_tran - n_post
    fake = list(range(idx.searchsorted(pd.Timestamp("2016-06-01")), last_ok, 10))
    for name, ys in series.items():
        ys = ys.rename(name)
        for i0 in fake:
            r = effect_at(ys, i0)
            if r:
                rows.append({"series": name, "date": idx[i0], "actual": False, **r})
        r = effect_at(ys, i_ann)
        rows.append({"series": name, "date": idx[i_ann], "actual": True, **r})
    df = pd.DataFrame(rows)
    save(df, "rq2_longhist_placebo")
    res = []
    for name in series:
        g = df[df.series == name]
        a = g[g.actual].iloc[0]
        pl = g[~g.actual]
        for k in KEYW:
            res.append({"series": name, "window": k, "actual": a[k], "n_placebo": len(pl),
                        "placebo_sd": pl[k].std(), "placebo_q025": pl[k].quantile(.025), "placebo_q975": pl[k].quantile(.975),
                        "emp_p_two_sided": (np.sum(np.abs(pl[k] - pl[k].mean()) >= abs(a[k] - pl[k].mean())) + 1) / (len(pl) + 1),
                        "pct_rank": (pl[k] < a[k]).mean()})
    res = pd.DataFrame(res)
    save(res, "rq2_longhist_pvalues")
    LOG.add(id="RQ2-P03", rq="RQ2", kind="placebo",
            purpose="긴 사전구간(2016~2026.5) 가짜 개입일 분포로 실제 효과의 이례성 평가 (모형 오차의 경험적 귀무분포)",
            data=f"F1 일별 OHLC → log Garman-Klass, donor {len(L.attrs['donors_long'])}종목 평균",
            model="OLS(학습 250일; don_mean, VIX, |SOX|, |KOSDAQ|) 예측오차 평균",
            settings={"n_fake": len(fake), "step": 10, "windows": f"{n_tran}일 / {n_post}일"},
            result="; ".join(f"{r.series}/{r.window}: {r.actual:+.3f}, 경험 p={r.emp_p_two_sided:.3f}, 순위 {r.pct_rank:.2f}"
                             for r in res.itertuples()))
    g = stage_garch(L)
    return df, res, g


def stage_garch(L):
    from arch import arch_model
    rows = []
    i_ann = L.index.get_loc(TL["announce"])
    i_eff = L.index.get_loc(TL["effective"])
    for code in ALL3:
        df = pd.concat([L[f"r_{code}"], L["don_ret"]], axis=1).dropna() * 100
        df.columns = ["r", "m"]
        df = df.loc["2016-01-01":]
        tr = df.index <= CFG["sample"]["train_end"]
        mm = sm.OLS(df.r[tr], sm.add_constant(df.m[tr])).fit()
        e = df.r - mm.predict(sm.add_constant(df.m))
        am = arch_model(e[tr], mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t")
        res = am.fit(disp="off")
        H = len(df) - tr.sum()
        fc = res.forecast(horizon=H, method="simulation", simulations=NDRAW // 2, reindex=False,
                          random_state=np.random.RandomState(CFG["seed"]))
        sims = fc.simulations.values[-1]  # (nsim, H)
        post = e[~tr]
        n_tran = sum((post.index >= TL["announce"]) & (post.index < TL["effective"]))
        for k, sl in [("transition", slice(0, n_tran)), ("post_effective", slice(n_tran, H))]:
            act = np.log(np.mean(post.values[sl] ** 2))
            sim = np.log(np.mean(sims[:, sl] ** 2, axis=1))
            rows.append({"code": code, "window": k, "effect_log": act - np.median(sim),
                         "ci_lo": act - np.quantile(sim, .975), "ci_hi": act - np.quantile(sim, .025),
                         "p_two_sided": 2 * min((sim >= act).mean(), (sim <= act).mean()),
                         "beta_mkt": mm.params.iloc[1], **{k2: round(v, 4) for k2, v in res.params.items()}})
    g = pd.DataFrame(rows)
    save(g, "rq2_garch")
    LOG.add(id="RQ2-E-GARCH", rq="RQ2", kind="extension",
            purpose="일별 수익률 기반 대안 측정: 시장모형 고유수익률의 GJR-GARCH 다기간 예측 대비 실제 고유분산",
            data="F1 ret_log 2016~, 시장요인=donor 평균수익률", model="GJR-GARCH(1,1)-t, 시뮬레이션 예측",
            settings={"train_end": "2026-07-15", "nsim": NDRAW // 2},
            result=fmt(g), interpretation="GARCH 다기간 예측은 비조건 분산으로 회귀 → 사전 고변동기의 영향이 크다. 공통충격은 고유수익률로만 제거.")
    return g


# ====================================================================== robustness
def trend_X():
    idx = load_panel("rv5").index
    return pd.DataFrame({"trend": np.arange(len(idx)) / 21.0}, index=idx)


def robustness_specs():
    te = CFG["sample"]["train_end"]
    S = [("main", {}),
         ("ctrl=C0", {"controls": "C0"}), ("ctrl=C1", {"controls": "C1"}), ("ctrl=C2", {"controls": "C2"}),
         ("ctrl=C_us_only", {"controls": "C_us_only"}), ("ctrl=C_contaminated", {"controls": "C_contaminated"}),
         ("pool=all", {"pool": "all"}), ("pool=strict", {"pool": "strict"}),
         ("measure=rv5_close", {"measure": "rv5_close"}), ("measure=rv_pk", {"measure": "rv_pk"}),
         ("event=07-24", {"train_end": pd.Timestamp("2026-07-23"), "post_start": TL["early_effective_notice"]}),
         ("event=07-31", {"train_end": pd.Timestamp("2026-07-30"), "post_start": TL["effective"]}),
         ("event=08-05", {"train_end": pd.Timestamp("2026-08-04"), "post_start": TL["original_effective"]}),
         ("event=08-19", {"train_end": pd.Timestamp("2026-08-18"), "post_start": TL["premium_rule"]}),
         ("train=pre-listing only", {"train_end": pd.Timestamp("2026-05-26")}),
         ("train=2026 only", {"train_start": pd.Timestamp("2026-01-02")}),
         ("listing dummy", {"extra_X": listing_dummy()}),
         ("donut 07-28~08-04", {"drop_dates": (pd.Timestamp("2026-07-28"), pd.Timestamp("2026-08-04"))}),
         ("winsorize 1/99", {"transform": winsorize_transform}),
         ("linear trend", {"extra_X": trend_X()})]
    B = [("bsts: level sd 0.1", {"model_kw": dict(BSTS_FAST, prior_level_sd=0.1)}),
         ("bsts: model size 1", {"model_kw": dict(BSTS_FAST, expected_model_size=1)}),
         ("bsts: model size 6", {"model_kw": dict(BSTS_FAST, expected_model_size=6)}),
         ("bsts: no level (static reg)", {"model_kw": dict(BSTS_FAST, level=False)})]
    return S, B, te


def stage_robustness():
    S, B, _ = robustness_specs()
    out = []
    for code in ALL3:
        for model in ["bsts", "ols"]:
            specs = S + (B if model == "bsts" else [])
            for name, kw in specs:
                kw = dict(kw)
                if model == "bsts":
                    kw.setdefault("model_kw", BSTS_FAST)
                try:
                    cf, y, r = run_spec(dict(code=code, model=model, **kw))
                    d = rows_df(r, code=code, model=model, spec=name)
                    out.append(d)
                except Exception as e:  # noqa: BLE001
                    out.append(pd.DataFrame([{"code": code, "model": model, "spec": name, "error": repr(e)[:200]}]))
    df = pd.concat(out, ignore_index=True)
    # 기준 대비 변화
    base = df[df.spec == "main"][["code", "model", "window", "effect_log", "p_two_sided"]].rename(
        columns={"effect_log": "base_effect", "p_two_sided": "base_p"})
    df = df.merge(base, on=["code", "model", "window"], how="left")
    df["delta"] = df.effect_log - df.base_effect
    df["sign_flip"] = np.sign(df.effect_log) != np.sign(df.base_effect)
    df["sig05"] = df.p_two_sided < .05
    df["base_sig05"] = df.base_p < .05
    save(df, "rq2_robustness")
    for (code, model), g in df[df.window == "post_effective"].groupby(["code", "model"]):
        LOG.add(id=f"RQ2-R-{model.upper()}-{code}", rq="RQ2", kind="robustness",
                purpose="사양 민감도: 통제·donor·측정치·사건일·학습창·donut·이상치(·BSTS 사전분포)",
                data=f"rv5 등 {code}", model=model, settings={"n_specs": len(g)},
                result=f"post_effective 효과 범위 [{g.effect_log.min():+.3f}, {g.effect_log.max():+.3f}], "
                       f"부호 반전 {int(g.sign_flip.sum())}건, p<.05 {int(g.sig05.sum())}/{len(g)}",
                robustness="; ".join(f"{r.spec}: {r.effect_log:+.3f} (p={r.p_two_sided:.2f})" for r in g.itertuples()))
    return df


# ====================================================================== BSTS placebo in space + pre-trend
def stage_placebo_space_bsts():
    """주 모형(BSTS)의 보정된 추론: donor 마다 같은 설계로 BSTS → 처치 종목 효과의 순열 p."""
    from src.rq2.core.models import cf_bsts
    lv = load_panel("rv5").loc[CFG["sample"]["start"]:CFG["sample"]["end"]]
    donors = donor_codes(CFG, "main")
    C = build_controls(CFG)
    ctrl = CFG["controls"]["C_main"]
    S = lv[donors].sum(axis=1)
    kw = dict(BSTS_MAIN, niter=1200, burn=400)
    rows = []
    for j in donors + ALL3:
        X = C[ctrl].copy()
        if j in donors:
            X["don_mean"] = (S - lv[j]) / (len(donors) - 1)
        df = pd.concat([lv[j].rename("y"), X], axis=1).dropna()
        tr = pd.Series(df.index <= CFG["sample"]["train_end"], index=df.index)
        cf = cf_bsts(df.y, df[ctrl], tr, ~tr, seed=CFG["seed"], **kw)
        yp = df.y[~tr]
        W = effect_windows(yp.index)
        r = {"code": j, "treated": j in ALL3}
        for k in KEYW:
            mk = W[k].values
            e = (yp.values[mk][None, :] - cf.draws[:, mk]).mean(1)
            r[k] = float((yp.values[mk] - cf.pred.values[mk]).mean())
            r[f"{k}_z"] = r[k] / e.std()
        rows.append(r)
    df = pd.DataFrame(rows)
    save(df, "rq2_placebo_space_bsts")
    don = df[~df.treated]
    res = []
    for code in ALL3:
        t = df[df.code == code].iloc[0]
        for k in KEYW:
            for col in [k, f"{k}_z"]:
                v = don[col]
                res.append({"code": code, "window": k, "stat": col, "value": t[col],
                            "perm_p": (np.sum(np.abs(v) >= abs(t[col])) + 1) / (len(v) + 1),
                            "donor_mean": v.mean(), "donor_q025": v.quantile(.025), "donor_q975": v.quantile(.975),
                            "donor_share_reject_z196": float((v.abs() > 1.96).mean()) if col.endswith("_z") else np.nan})
    res = pd.DataFrame(res)
    save(res, "rq2_placebo_space_bsts_pvalues")
    LOG.add(id="RQ2-P04", rq="RQ2", kind="placebo",
            purpose="주 모형(BSTS)의 placebo-in-space 순열 추론 — 모형 구간의 과소추정(placebo_time FPR)을 보정",
            data=f"rv5, main donor {len(donors)}종목 + 처치·음성통제", model="BSTS C_main (niter 1200, burn 400)",
            settings="perm p = (#{|donor| ≥ |treated|}+1)/(N+1); z = 효과/사후 sd",
            result="; ".join(f"{r.code}/{r.stat}: {r.value:+.3f}, perm p={r.perm_p:.3f}" for r in res.itertuples()),
            diagnostics="donor 중 |z|>1.96 비율 = 모형 구간의 실제 위양성률: " + "; ".join(
                f"{r.window} {r.donor_share_reject_z196:.2f}" for r in res[res.stat.str.endswith('_z') & (res.code == '005930')].itertuples()))
    return df, res


def stage_pretrend():
    """사전 추세: (처치 log RV − donor 평균) 의 시간 추세. 분봉(2025-10~07-15)과 F1 GK 긴 표본."""
    lv = load_panel("rv5").loc[CFG["sample"]["start"]:CFG["sample"]["train_end"]]
    don = lv[donor_codes(CFG, "main")].mean(axis=1)
    L = load_long_daily(CFG)
    rows = []
    for code in ALL3 + ["005930-005935"]:
        if code == "005930-005935":
            g = lv["005930"] - lv[NEG]
            gl = (L["005930"] - L[NEG]).loc["2024-01-01":CFG["sample"]["train_end"]]
        else:
            g = lv[code] - don
            gl = (L[code] - L["don_mean"]).loc["2024-01-01":CFG["sample"]["train_end"]]
        for lab, s in [("rv5 2025-10~2026-07-15", g), ("GK 2024-01~2026-07-15", gl.dropna())]:
            for sub, ss in [("전체", s), ("상장 전", s[s.index < TL["listing"]])]:
                t = np.arange(len(ss)) / 21.0     # 월 단위 기울기
                m = sm.OLS(ss.values, sm.add_constant(t)).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
                rows.append({"series": code, "data": lab, "sample": sub, "n": len(ss), "slope_per_month": m.params[1],
                             "ci_lo": m.conf_int()[1, 0], "ci_hi": m.conf_int()[1, 1], "p_hac": m.pvalues[1]})
    df = pd.DataFrame(rows)
    save(df, "rq2_pretrend")
    LOG.add(id="RQ2-D04", rq="RQ2", kind="diagnostic", purpose="사전 추세 (상대 변동성 격차의 추세) — 반사실 외삽의 위험 점검",
            data="rv5 분봉 / F1 GK", model="OLS 추세 + HAC(10)", settings="격차 = 종목 log 변동성 − donor 평균",
            result="; ".join(f"{r.series}/{r.data}/{r.sample}: {r.slope_per_month:+.3f}/월 (p={r.p_hac:.3f})" for r in df.itertuples()))
    return df


# ====================================================================== figures
def stage_figures():
    FIG.mkdir(parents=True, exist_ok=True)
    if (TAB / "rq2_primary_paths.csv").exists():
        P.counterfactual(TAB / "rq2_primary_paths.csv", TL, FIG / "rq2_counterfactual_bsts.png")
    if (TAB / "rq2_models.csv").exists():
        m = pd.read_csv(TAB / "rq2_models.csv", dtype={"code": str})
        m["label"] = m.model
        if (TAB / "rq2_foundation.csv").exists():
            fm = pd.read_csv(TAB / "rq2_foundation.csv", dtype={"code": str})
            fm["label"] = fm.model + " (placebo 보정)"
            m = pd.concat([m, fm[["code", "window", "label", "effect_log", "ci_lo", "ci_hi"]]])
        P.forest(m, "label", FIG / "rq2_models_forest.png",
                 "모형별 효과 (같은 설계·학습≤07-15, 95% 구간; FM 은 placebo 오차분포로 만든 구간)", ref_label="bsts")
    if (TAB / "rq2_robustness.csv").exists():
        r = pd.read_csv(TAB / "rq2_robustness.csv", dtype={"code": str})
        r = r[r.model == "bsts"].copy()
        P.forest(r, "spec", FIG / "rq2_robustness_forest_bsts.png", "BSTS 사양 민감도 (95% 모형 구간)", ref_label="main")
    if (TAB / "rq2_placebo_space_bsts.csv").exists():
        ps = pd.read_csv(TAB / "rq2_placebo_space_bsts.csv", dtype={"code": str})
        don = ps[~ps.treated]
        plac = pd.concat([don.assign(code=c) for c in ALL3])
        P.placebo_hist(plac, ps[ps.treated], ["transition", "post_effective"], FIG / "rq2_placebo_space.png",
                       f"placebo-in-space: donor {len(don)}종목에 같은 BSTS 를 적용한 효과 분포 (주황=해당 종목의 실제 효과)",
                       xlabels=["transition 효과 (log RV)", "post_effective 효과 (log RV)"])
    if (TAB / "rq2_longhist_placebo.csv").exists():
        lh = pd.read_csv(TAB / "rq2_longhist_placebo.csv", dtype={"series": str})
        lh = lh[lh.series.isin(ALL3)].copy()
        lh["code"] = lh.series
        P.placebo_hist(lh[~lh.actual], lh[lh.actual], ["transition", "post_effective"], FIG / "rq2_longhist_placebo.png",
                       "긴 사전구간 placebo (2016~2026.5, 가짜 개입일 237개), 일별 GK OLS (주황=실제 2026-07-16)",
                       xlabels=["transition 10일 평균 오차 (log GK)", "post_effective 61일 평균 오차 (log GK)"])
    L = load_long_daily(CFG)
    P.gap_history(L.loc["2015-06-01":], TL, FIG / "rq2_gap_history.png")


def stage_longhist_usonly():
    """국내 통제가 오염됐을 가능성(D05)에 대한 보정 (RQ2-R-USONLY).
    긴 표본 GK, 미국 변수(VIX, |SOX|)만 통제 → 국내 공통충격을 통제하지 않는 설계의 귀무분포.
    donor 평균(don_mean)도 결과변수로 넣어, 시행 후 국내 시장 전체 변동성 하락이 과거 분포에서 이례적인지 본다."""
    L = load_long_daily(CFG).loc["2015-06-01":]
    X_cols = ["us_lvix", "us_abs_sox"]
    idx = L.index
    i_ann, i_eff = idx.get_loc(TL["announce"]), idx.get_loc(TL["effective"])
    n_tr, n_tran, n_post = 250, i_eff - i_ann, min(61, len(idx) - i_eff)
    series = {c: L[c] for c in ALL3 + ["don_mean"]}

    def effect_at(ys, i0):
        df = pd.concat([ys.rename("y"), L[X_cols]], axis=1).iloc[i0 - n_tr:i0 + n_tran + n_post].dropna()
        if len(df) < n_tr * 0.8 + n_tran + n_post * 0.8:
            return None
        tr = df.index < idx[i0]
        Xc = sm.add_constant(df[X_cols])
        m = sm.OLS(df.y[tr], Xc[tr]).fit()
        e = df.y[~tr] - m.predict(Xc[~tr])
        return {"transition": e.iloc[:n_tran].mean(), "post_effective": e.iloc[n_tran:].mean()}

    last_ok = idx.searchsorted(TL["listing"]) - n_tran - n_post
    fake = list(range(idx.searchsorted(pd.Timestamp("2016-06-01")), last_ok, 10))
    rows = []
    for name, ys in series.items():
        for i0 in fake + [i_ann]:
            r = effect_at(ys, i0)
            if r:
                rows.append({"series": name, "date": idx[i0], "actual": i0 == i_ann, **r})
    df = pd.DataFrame(rows)
    save(df, "rq2_longhist_usonly_placebo")
    res = []
    for name in series:
        g = df[df.series == name]
        a, pl = g[g.actual].iloc[0], g[~g.actual]
        for k in KEYW:
            res.append({"series": name, "window": k, "actual": a[k], "n_placebo": len(pl), "placebo_mean": pl[k].mean(),
                        "placebo_sd": pl[k].std(), "emp_p_two_sided": _emp_p(pl[k], a[k]), "pct_rank": (pl[k] < a[k]).mean()})
    res = pd.DataFrame(res)
    # 상대 비교: 처치 종목 − donor 평균 (같은 가짜일에서), 그 분포
    piv = df.pivot_table(index=["date", "actual"], columns="series", values="post_effective").reset_index()
    save(res, "rq2_longhist_usonly_pvalues")
    LOG.add(id="RQ2-R-USONLY", rq="RQ2", kind="robustness",
            purpose="국내 통제변수 오염 가능성(D05) 대응: 미국 변수만 통제한 반사실의 긴 표본 보정 + donor 평균 자체의 이례성",
            data="F1 GK 2016~2026-10", model="OLS(학습 250일; VIX, |SOX|) 예측오차 평균",
            settings={"n_fake": len(fake), "step": 10},
            result="; ".join(f"{r.series}/{r.window}: {r.actual:+.3f}, 경험 p={r.emp_p_two_sided:.3f}, 순위 {r.pct_rank:.2f}"
                             for r in res.itertuples()))
    print(res.round(3).to_string(), flush=True)
    return res


# ====================================================================== 통제변수 오염 점검
def stage_control_check():
    """규제 후에도 쓰는 국내 통제변수(donor 평균·PCA·KOSDAQ)가 규제 영향을 받았는지 점검 (D05).
    국내 통제 ~ 상수 + 미국 변수(VIX, |SOX|) + gap_day + 상장/발표/시행 더미 (HAC 5).
    더미 계수가 크면 반사실 예측에 쓴 통제변수 자체가 이동한 것 → 효과가 통제로 흡수될 위험.
    한계: 같은 시기의 다른 국내 충격과 구분할 수 없으므로 '오염 없음'의 증명이 아니라 경고 신호 점검이다."""
    C = build_controls(CFG).loc[CFG["sample"]["start"]:CFG["sample"]["end"]].dropna()
    W = effect_windows(C.index)
    dm = pd.DataFrame({"listing_regime": C.index >= TL["listing"], "transition": W["transition"],
                       "post_effective": W["post_effective"]}, index=C.index).astype(float)
    base = C[["us_lvix", "us_abs_sox", "gap_day"]]
    tr = C.index <= CFG["sample"]["train_end"]
    rows = []
    for v in ["don_mean", "don_pc2", "don_pc3", "kosdaq_lrv"]:
        X = sm.add_constant(pd.concat([base, dm], axis=1))
        m = sm.OLS(C[v], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        sd_pre = C.loc[tr, v].std()
        for k in ["transition", "post_effective"]:
            c = pd.Series(0.0, index=m.params.index)
            c[k], c["listing_regime"] = 1, -1          # 상장기간 대비 이동 (주 분석 추정대상과 같은 기준)
            t = m.t_test(c.values)
            rows.append({"control": v, "window": k, "shift_vs_listing": float(t.effect[0]),
                         "ci_lo": float(t.conf_int()[0, 0]), "ci_hi": float(t.conf_int()[0, 1]),
                         "p_hac": float(t.pvalue), "pre_sd": sd_pre, "shift_in_pre_sd": float(t.effect[0]) / sd_pre})
    df = pd.DataFrame(rows)
    save(df, "rq2_control_check")
    LOG.add(id="RQ2-D05", rq="RQ2", kind="diagnostic",
            purpose="사후에도 쓰는 국내 통제변수가 규제 영향을 받지 않았다는 가정 점검",
            data="build_controls 2025-10~2026-10", model="통제변수 ~ US 변수 + gap_day + 상장/전환/시행 더미, HAC 5",
            result="; ".join(f"{r.control}/{r.window}: {r.shift_vs_listing:+.3f} (사전 sd 의 {r.shift_in_pre_sd:+.2f}배) p={r.p_hac:.3f}"
                             for r in df.itertuples()),
            interpretation="국내 통제변수의 이동이 크면 C_us_only 사양(국내 통제 제외)과 비교해야 한다. 다른 국내 충격과 구분 불가.")
    print(df.round(3).to_string(), flush=True)
    return df


# ====================================================================== main
STAGES = {"primary": stage_primary, "models": stage_models, "withinfirm": stage_withinfirm,
          "withinfirm_calib": stage_withinfirm_calib,
          "diagnostics": stage_diagnostics, "placebo_time": stage_placebo_time,
          "placebo_space": stage_placebo_space, "placebo_space_bsts": stage_placebo_space_bsts,
          "pretrend": stage_pretrend, "control_check": stage_control_check, "longhist_usonly": stage_longhist_usonly, "longhist": stage_longhist,
          "robustness": stage_robustness, "figures": stage_figures}


def write_env():
    import importlib
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ["numpy", "pandas", "statsmodels", "scipy", "sklearn", "lightgbm", "xgboost", "arch", "matplotlib", "duckdb"]:
        v[m] = getattr(importlib.import_module(m), "__version__", "?")
    (OUT / "environment.json").write_text(json.dumps(v, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(STAGES))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    old = OUT / "experiment_log.csv"
    if old.exists():   # 이전 단계 기록 유지: 같은 id 는 새 결과로 교체
        prev = pd.read_csv(old, dtype=str).fillna("")
        LOG.rows = prev.to_dict("records")
    write_env()
    for st in a.stages.split(","):
        t0 = time.time()
        print(f"[RQ2] {st} ...", flush=True)
        n_before = len(LOG.rows)
        STAGES[st]()
        new_ids = {r["id"] for r in LOG.rows[n_before:]}
        LOG.rows = [r for r in LOG.rows[:n_before] if r["id"] not in new_ids] + LOG.rows[n_before:]
        LOG.save()
        print(f"[RQ2] {st} done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
