"""ML 확장 하위실험 전체 실행. 1차 연구 결과(src/rq2/phase1/main_study/results)는 읽기만 한다.

  py -m src.rq2.phase1.ml_extension.run_all                       # 전 단계 (캐시가 있으면 건너뜀)
  py -m src.rq2.phase1.ml_extension.run_all --stages tables,figures
단계: bench, fm, tune, ensemble, spill, leaky, ablation, diag_fit, tables, figures
  계산 단계(bench~diag_fit)는 CACHE(data/interim/rq_cache/ml_ext) 에 저장하고 완료 과제는 다시 하지 않는다.
  tables 는 캐시에서 모든 표·실험 로그를 다시 만든다 (몇 분). figures 는 표에서 그림을 만든다.
데이터를 늘리면(예: 2027-01 까지) config.yaml 의 표본·창 값을 바꾸고 CACHE 를 지운 뒤 다시 실행한다.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from src.rq2.core.explog import ExpLog
from src.rq2.phase1.ml_extension import benchmark as B
from src.rq2.phase1.ml_extension import data_pipeline as DP
from src.rq2.phase1.ml_extension import diagnostics as DG
from src.rq2.phase1.ml_extension import ensemble as EN
from src.rq2.phase1.ml_extension import real_effect as RE
from src.rq2.phase1.ml_extension import synthetic_effect as S
from src.rq2.phase1.ml_extension import tuning as TU
from src.rq2.phase1.ml_extension.models import COMPLEXITY, family

E = DP.ECFG
RES = DP.RES
TAB = RES / "tables"
FIG = RES / "figures"
LOG = ExpLog(RES / "experiment_log.csv")
BASE = E["statistical"] + E["ml"] + E["extra"]
FM = E["foundation"]
TUNED = [f"{k}_t{o}" for k in E["tuning"]["models"] for o in ("A", "B")]
ENSM = EN.ENS
ALL = BASE + FM + TUNED + ENSM
SPILL_MODELS = ["did_gap", "ols", "ridge", "bsts", "uc", "harx", "enet", "rf", "lgbm", "xgb"]
ABL_MODELS = ["ols", "bsts", "lgbm", "xgb", "rf"]
ABL_GROUPS = {"no_donor": ["don_mean", "don_pc2", "don_pc3"], "no_kosdaq": ["kosdaq_lrv"],
              "no_us": ["us_lvix", "us_abs_sox"], "no_controls": ["don_mean", "don_pc2", "don_pc3", "kosdaq_lrv",
                                                                   "us_lvix", "us_abs_sox", "gap_day"]}


def save(df: pd.DataFrame, name: str):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False, encoding="utf-8-sig", float_format="%.5g")


def present(ds, models):
    return [m for m in models if B.cache_path(ds, m, "win").exists()]


# ================================================================ 계산 단계
def stage_bench():
    for ds in ["rv5", "gk", "space"]:
        B.run_models(ds, BASE)
    # GK 실제 개입일 (실험 5 의 일별 GK 반복). 벤치마크 캐시와 섞지 않도록 tag=real
    i0 = DP.actual_index("gk")
    B.run_models("gk", BASE, tasks=[(c, i0) for c in E["datasets"]["gk"]["codes"]], tag="real")


def stage_fm():
    py_fm = DP.ROOT / ".venv-fm" / "Scripts" / "python.exe"
    for ds in ["rv5", "gk", "space"]:
        if all(B.cache_path(ds, m, "win").exists() for m in FM):
            continue
        subprocess.run([sys.executable, "-m", "src.rq2.phase1.ml_extension.foundation_bridge", "export", ds], check=True, cwd=DP.ROOT)
        subprocess.run([str(py_fm), "-m", "src.rq2.phase1.ml_extension.foundation_bridge", "forecast", ds], check=True, cwd=DP.ROOT)
        subprocess.run([sys.executable, "-m", "src.rq2.phase1.ml_extension.foundation_bridge", "collect", ds], check=True, cwd=DP.ROOT)


def stage_tune():
    for ds in E["tuning"]["datasets"]:
        TU.run_tuning(ds)
    # 튜닝 모형의 space(실제 개입일) 적합: rv5 의 07-15 이전 마지막 가짜 개입일 대신,
    # 실제 과제 학습구간 안에서 같은 절차로 다시 튜닝한다 (처치 3 + donor 전부는 비용이 커서 처치 3 종목 + donor 40 무작위)
    rng = np.random.default_rng(E["seed"])
    don = DP.donor_codes(DP.CFG, "main")
    units = ["005930", "000660", "005935"] + list(rng.choice(don, 40, replace=False))
    i0 = DP.actual_index("space")
    for kind in E["tuning"]["models"]:
        for ob in ("A", "B"):
            name = f"{kind}_t{ob}"
            _, w0 = B.load_cache("space", name)
            done = set() if w0 is None else set(zip(w0.code, w0.i0))
            todo = [(u, i0) for u in units if (u, i0) not in done]
            if not todo:
                continue
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=E["n_jobs"])(delayed(TU._tune_one)("space", c, i, kind, ob, E["seed"]) for c, i in todo)
            D = [r[0] for r in res if r[0] is not None]
            W = [r[1] for r in res if r[1] is not None]
            d0, w0 = B.load_cache("space", name)
            pd.concat(([d0] if d0 is not None else []) + D).to_parquet(B.cache_path("space", name, "daily"))
            pd.concat(([w0] if w0 is not None else []) + W).to_parquet(B.cache_path("space", name, "win"))
            R = pd.DataFrame([r[2] for r in res])
            fr = DP.CACHE / "space" / f"tuning_{name}.parquet"
            (pd.concat([pd.read_parquet(fr), R]) if fr.exists() else R).to_parquet(fr)
            print(f"  [space/{name}] {len(todo)}", flush=True)


def stage_ensemble():
    for ds in ["rv5", "gk", "space"]:
        EN.build(ds)


def stage_spill():
    S.run_spillover(SPILL_MODELS, "rv5")


def stage_leaky():
    S.leaky_demo("rv5")


def stage_ablation():
    for g, cols in ABL_GROUPS.items():
        B.run_models("rv5", ABL_MODELS, x_shift={"drop": cols}, tag=f"abl_{g}")


def stage_diag_fit():
    DG.pca_leak()


# ================================================================ 표
def stage_tables():
    t0 = time.time()
    out = {}
    # ---- 실험 1
    summ, regimes, dms = [], [], []
    for ds in ["rv5", "gk"]:
        ms = present(ds, ALL)
        D, W = B.collect(ds, ms)
        summ.append(B.summary_table(ds, D, W, "all"))
        if ds == "rv5":
            summ.append(B.summary_table(ds, D, W, "prelisting"))
        regimes.append(B.regime_table(ds, W))
        dms.append(B.dm_tests(ds, W))
    S1 = pd.concat(summ)
    S1["family"] = S1.model.map(family)
    S1["complexity"] = S1.model.map(lambda m: COMPLEXITY.get(m, 7 if m.startswith("ens_") else 5))
    save(S1, "exp1_forecast_by_code")
    agg = S1.groupby(["dataset", "subset", "horizon", "model", "family", "complexity"]).agg(
        n_cut=("n_cut", "sum"), MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), bias=("bias", "mean"),
        cov90=("cov90", "mean"), eff_bias=("eff_bias", "mean"), eff_rmse=("eff_rmse", "mean"),
        eff_mae=("eff_mae", "mean"), native_fpr=("native_fpr", "mean"), fit_sec=("fit_sec", "mean"),
        pre_rmse=("pre_rmse", "mean")).reset_index()
    # 상대 지표: naive_mean·OLS 대비
    for ref in ["ols", "naive_mean"]:
        r = agg[agg.model == ref].set_index(["dataset", "subset", "horizon"])
        agg[f"RMSE_rel_{ref}"] = agg.RMSE / agg.set_index(["dataset", "subset", "horizon"]).index.map(r.RMSE).values
        agg[f"effRMSE_rel_{ref}"] = agg.eff_rmse / agg.set_index(["dataset", "subset", "horizon"]).index.map(r.eff_rmse).values
    save(agg, "exp1_forecast_summary")
    save(pd.concat(regimes), "exp1_regime")
    DM = pd.concat(dms)
    save(DM, "exp1_dm_tests")
    out["S1"] = agg

    # ---- 실험 2
    syn, pw, mdes, Ws = [], [], [], {}
    for ds in ["rv5", "gk"]:
        ms = present(ds, ALL)
        W = S.prepare(ds, ms)
        Ws[ds] = W
        syn.append(S.inject_eval(ds, W))
        P = S.power_curve(ds, W)
        pw.append(P)
        mdes.append(S.mde_table(ds, W, P))
    SY = pd.concat(syn)
    SY["family"] = SY.model.map(family)
    save(SY, "exp2_synthetic_by_code")
    SYa = SY.groupby(["dataset", "scenario", "effect_pct", "window", "model", "family"]).agg(
        bias=("bias", "mean"), rmse=("rmse", "mean"), mae=("mae", "mean"), truth=("truth_mean", "mean"),
        native_reject=("native_reject", "mean"), time_reject=("time_reject", "mean"),
        native_cov=("native_cov", "mean"), time_cov=("time_cov", "mean"),
        native_width=("native_width", "mean"), time_width=("time_width", "mean")).reset_index()
    save(SYa, "exp2_synthetic_summary")
    PW = pd.concat(pw)
    save(PW, "exp2_power_curve")
    MD = pd.concat(mdes)
    MD["family"] = MD.model.map(family)
    save(MD, "exp2_mde")
    sp = S.space_eval(present("space", ALL))
    sp["family"] = sp.model.map(family)
    save(sp, "exp2_space_permutation")
    # 튜닝 모형은 donor 40 개만 적합했으므로, 모든 모형을 같은 donor 40 개로 다시 비교 (공정 비교)
    _, Wt = B.collect("space", ["lgbm_tA"])
    if not Wt.empty:
        u40 = sorted(set(Wt.code))
        sp40 = S.space_eval(present("space", ALL), units=u40)
        sp40["family"] = sp40.model.map(family)
        save(sp40, "exp2_space_permutation_common40")
    spl = S.spillover_eval(SPILL_MODELS, "rv5")
    save(spl, "exp2_spillover")
    lk = S.leaky_summary(S.leaky_demo("rv5"))
    save(lk, "exp2_leaky_lag_demo")

    # ---- 실험 3
    sel_window = RE.select_model(SY, MD)["window_used"]   # 선정 규칙이 실제로 쓴 창 (실험 5 에서 다시 호출해도 같은 결과)
    tr = []
    for ds in E["tuning"]["datasets"]:
        R = TU.tuning_records(ds)
        R["dataset"] = ds
        tr.append(R)
    TR = pd.concat(tr)
    save(TR, "exp3_tuning_records")
    save(TU.param_stability(TR), "exp3_param_stability")
    comp = []
    base_of = {"lgbm": "lgbm", "xgb": "xgb", "enet": "enet", "rf": "rf"}
    for ds in E["tuning"]["datasets"]:
        W = Ws[ds]
        Wh = W[W.window == "h30"]
        Mde = MD[(MD.dataset == ds) & (MD.window == "h30")]
        Sy0 = SY[(SY.dataset == ds) & (SY.window == "h30") & (SY.scenario == "A_immediate")]
        Tr = TR[TR.dataset == ds]
        # time 검정이 모든 과제에서 가능한 창 (rv5 h30 은 겹침 보정 후 검정 불가 → 선정 규칙이 쓴 창)
        tw = "h30" if ds != E["selection_rule"]["dataset"] else sel_window
        Mtw = MD[(MD.dataset == ds) & (MD.window == tw)]
        Stw = SY[(SY.dataset == ds) & (SY.window == tw) & (SY.scenario == "A_immediate")]
        for kind in E["tuning"]["models"]:
            names = [base_of[kind], f"{kind}_tA", f"{kind}_tB"]
            common = None
            for n in names:   # 공통 과제만 비교 (gk 튜닝은 개입일 1/3)
                s = set(zip(Wh[Wh.model == n].code, Wh[Wh.model == n].i0))
                common = s if common is None else common & s
            for n in names:
                g = Wh[(Wh.model == n) & pd.Series(list(zip(Wh.code, Wh.i0)), index=Wh.index).isin(common)]
                D, _ = B.collect(ds, [n])
                if D.empty or not len(g):
                    continue
                D = D[D.h <= 30]
                D = D[pd.Series(list(zip(D.code, D.i0)), index=D.index).isin(common)]
                comp.append({"dataset": ds, "kind": kind, "model": n,
                             "variant": "default" if n == kind else ("tuned_A_forecast" if n.endswith("A") else "tuned_B_effect"),
                             "n_tasks": len(g), "daily_rmse_h30": float(np.sqrt(np.mean((D.y - D.pred) ** 2))),
                             "effect_rmse_h30": float(np.sqrt(np.mean(g.err ** 2))), "effect_bias_h30": float(g.err.mean()),
                             "time_fpr_h30": float(Sy0[(Sy0.model == n) & (Sy0.effect_pct == 0)].time_reject.mean()),
                             "time_power_m20_h30": float(Sy0[(Sy0.model == n) & (Sy0.effect_pct == -20)].time_reject.mean()),
                             "mde_time_pct_h30": float(Mde[Mde.model == n].mde_time_analytic_pct.mean()),
                             "test_window": tw,
                             "time_fpr_tw": float(Stw[(Stw.model == n) & (Stw.effect_pct == 0)].time_reject.mean()),
                             "time_power_m20_tw": float(Stw[(Stw.model == n) & (Stw.effect_pct == -20)].time_reject.mean()),
                             "mde_time_pct_tw": float(Mtw[Mtw.model == n].mde_time_analytic_pct.mean()),
                             "fit_sec": float(g.fit_sec.mean()),
                             "tune_sec": float(Tr[(Tr.kind == kind) & (Tr.objective == n[-1])].tune_sec.mean()) if n != kind else 0.0,
                             "inner_gain": float(Tr[(Tr.kind == kind) & (Tr.objective == n[-1])].inner_gain.mean()) if n != kind else np.nan})
    TC = pd.DataFrame(comp)
    save(TC, "exp3_tuning_before_after")

    # ---- 실험 4
    ecor = []
    for ds in ["rv5", "gk"]:
        C = EN.error_correlation(ds, present(ds, BASE + FM))
        C.index.name = "model"
        c = C.reset_index()
        c.insert(0, "dataset", ds)
        ecor.append(c)
    save(pd.concat(ecor), "exp4_error_correlation_h30")
    ew = []
    for ds in ["rv5", "gk", "space"]:
        f = DP.CACHE / ds / "ensemble_weights.parquet"
        if f.exists():
            w = pd.read_parquet(f)
            ew.append(w.drop(columns=["code", "i0"]).groupby(["dataset", "ensemble"]).mean(numeric_only=True).reset_index()
                      .assign(fallback_share=w.groupby(["dataset", "ensemble"]).fallback_equal.mean().values))
    save(pd.concat(ew), "exp4_ensemble_weights")

    # ---- 실험 5
    sel = RE.select_model(SY, MD)
    out["selected"] = sel["selected"]
    R5 = RE.estimate(present("space", ALL), sel["selected"], "space", "rv5")
    R5["family"] = R5.model.map(family)
    save(R5, "exp5_real_effect_rv5")
    save(RE.within_firm(R5), "exp5_within_firm_rv5")
    gk_sel = sel["selected"] if sel["selected"] in BASE else None
    R5g = RE.estimate([m for m in BASE if B.cache_path("gk", f"{m}@real", "win").exists()], gk_sel, "gk", "gk", tag="real")
    if len(R5g):
        R5g["family"] = R5g.model.map(family)
    save(R5g, "exp5_real_effect_gk")

    # ---- 진단
    dg = {}
    dg["d01_invariance"] = B.invariance_test("rv5", ["naive_mean", "did_gap", "ols", "ridge", "enet", "bsts", "sarimax",
                                                     "uc", "harx", "lgbm", "xgb", "rf", "lgbm_har"])
    pl = DG.pca_leak()
    dg["d01_pca_fullsample"] = pl.groupby("model").agg(rmse_trainonly=("rmse_trainonly", "mean"),
                                                       rmse_fullpca=("rmse_fullpca", "mean"),
                                                       effrmse_trainonly=("err_trainonly", lambda v: np.sqrt(np.mean(v ** 2))),
                                                       effrmse_fullpca=("err_fullpca", lambda v: np.sqrt(np.mean(v ** 2)))).reset_index()
    ov, bt, ac, sh, ps, sb = [], [], [], [], [], []
    for ds in ["rv5", "gk"]:
        ms = present(ds, BASE + FM + ENSM)
        _, W = B.collect(ds, ms)
        ov.append(DG.overfitting(ds, W[W.model.isin(BASE)]))
        bt.append(DG.bias_test(ds, W))
        ac.append(DG.resid_autocorr(ds, ms))
        sh.append(DG.distribution_shift(ds, W))
        ps.append(DG.period_stability(ds, W))
        sb.append(DG.selection_bias(ds, W[W.model.isin(BASE + FM)]))
        sb.append(DG.selection_bias(ds, W[W.model.isin(BASE + FM)], "h10"))
    dg["d02_overfitting"] = pd.concat(ov)
    dg["d03_bias"] = pd.concat(bt)
    dg["d04_resid_autocorr"] = pd.concat(ac)
    dg["d05_distribution_shift"] = pd.concat(sh)
    cal = SYa[(SYa.effect_pct == 0) & (SYa.scenario == "A_immediate")][
        ["dataset", "window", "model", "family", "native_cov", "time_cov", "native_width", "time_width"]]
    d1 = S1[S1.subset == "all"].groupby(["dataset", "horizon", "model"]).cov90.mean().rename("daily_cov90").reset_index()
    dg["d06_calibration"] = cal.merge(d1, left_on=["dataset", "window", "model"], right_on=["dataset", "horizon", "model"],
                                      how="left").drop(columns="horizon")
    f7 = SYa[(SYa.effect_pct == 0) & (SYa.scenario == "A_immediate")][["dataset", "window", "model", "family",
                                                                       "native_reject", "time_reject"]]
    sp0 = sp[sp.effect_pct == 0][["window", "model", "space_reject"]]
    dg["d07_fpr"] = f7
    dg["d07_fpr_space"] = sp0
    dg["d08_hp_sensitivity"] = TR.groupby(["dataset", "kind", "objective"]).agg(
        inner_gain=("inner_gain", "mean"), best_is_default=("best_trial", lambda v: np.mean(v == 0)),
        rel_spread=("inner_median", "mean")).reset_index()
    dg["d09_period_stability"] = pd.concat(ps)
    nc = R5[R5.code == "005935"][["model", "window", "effect", "time_p", "space_p"]]
    dg["d11_negative_control"] = nc
    dg["d12_selection_bias"] = pd.concat(sb)
    # 변수 제거 (선택)
    abl = []
    _, W0 = B.collect("rv5", ABL_MODELS)
    W0 = W0[W0.window == "h30"].assign(err=lambda d: d.ybar - d.predbar)
    base_r = W0.groupby("model").err.apply(lambda v: np.sqrt(np.mean(v ** 2)))
    for g in ABL_GROUPS:
        _, W1 = B.collect("rv5", [f"{m}@abl_{g}" for m in ABL_MODELS])
        if W1.empty:
            continue
        W1 = W1[W1.window == "h30"].assign(err=lambda d: d.ybar - d.predbar, base=lambda d: d.model.str.split("@").str[0])
        r = W1.groupby("base").err.apply(lambda v: np.sqrt(np.mean(v ** 2)))
        for m in r.index:
            abl.append({"model": m, "dropped": g, "eff_rmse_full": base_r[m], "eff_rmse_dropped": r[m],
                        "change": r[m] / base_r[m] - 1})
    dg["opt_feature_ablation"] = pd.DataFrame(abl)
    for k, v in dg.items():
        save(v, k)
    try:
        save(shap_table(), "opt_shap_lgbm_real")
    except Exception as e:  # noqa: BLE001
        LOG.add(id="RQ2-ML-OPT-SHAP", rq="RQ2", kind="diagnostic", purpose="SHAP", status="failed", result=repr(e)[:300])
    write_log(agg, SYa, MD, TC, sel, R5, sp)
    env()
    print(f"tables {time.time() - t0:.0f}s, selected={sel['selected']}", flush=True)


def shap_table() -> pd.DataFrame:
    """실제 개입일 LGBM(처치 2 종목) 사후 61일 예측의 평균 |SHAP| (선택 분석)."""
    import shap
    from src.rq2.phase1.ml_extension.models import make_estimator
    rows = []
    for code in ["005930", "000660"]:
        y, X, tr, po = DP.task_data("space", code, DP.actual_index("space"))
        m = make_estimator("lgbm", {}, E["seed"]).fit(X[tr], y[tr])
        sv = shap.TreeExplainer(m).shap_values(X[po])
        st = shap.TreeExplainer(m).shap_values(X[tr])
        for j, c in enumerate(X.columns):
            rows.append({"code": code, "feature": c, "mean_abs_shap_post": float(np.abs(sv[:, j]).mean()),
                         "mean_shap_post": float(sv[:, j].mean()), "mean_shap_train": float(st[:, j].mean()),
                         "shift_contribution": float(sv[:, j].mean() - st[:, j].mean())})
    return pd.DataFrame(rows)


def env():
    import importlib
    v = {"python": platform.python_version(), "platform": platform.platform(), "seed": E["seed"],
         "n_jobs": E["n_jobs"], "cpu": os.cpu_count()}
    for p in ["numpy", "pandas", "scipy", "statsmodels", "sklearn", "lightgbm", "xgboost", "optuna", "shap", "joblib"]:
        try:
            v[p] = importlib.import_module(p).__version__
        except Exception:  # noqa: BLE001
            v[p] = "n/a"
    try:
        r = subprocess.run([str(DP.ROOT / ".venv-fm" / "Scripts" / "python.exe"), "-c",
                            "import torch, chronos, timesfm, sys; print(torch.__version__, sys.version.split()[0])"],
                           capture_output=True, text=True, timeout=120)
        v["fm_env"] = r.stdout.strip()
    except Exception as e:  # noqa: BLE001
        v["fm_env"] = repr(e)
    (RES / "environment.json").write_text(json.dumps(v, indent=1), encoding="utf-8")


def write_log(agg, SYa, MD, TC, sel, R5, sp):
    cfg = {k: E[k] for k in ["seed", "datasets", "windows", "bsts_bench"]}
    errs = []
    for ds in ["rv5", "gk", "space"]:
        f = DP.CACHE / ds / "errors.csv"
        if f.exists():
            e = pd.read_csv(f)
            errs.append(e.assign(dataset=ds))
    E_ = pd.concat(errs) if errs else pd.DataFrame()
    for ds in ["rv5", "gk"]:
        a = agg[(agg.dataset == ds) & (agg.subset == "all") & (agg.horizon == "h30")].sort_values("eff_rmse")
        LOG.add(id=f"RQ2-ML-E1-{ds.upper()}", rq="RQ2", kind="extension",
                purpose="반사실 예측 벤치마크 (가짜 개입일 rolling-origin)", data=ds, model=list(a.model),
                settings=cfg, seed=E["seed"],
                result={"h30_effect_rmse_top5": a.head(5)[["model", "eff_rmse"]].round(4).to_dict("records"),
                        "ols": float(a[a.model == "ols"].eff_rmse.iloc[0]) if (a.model == "ols").any() else None},
                diagnostics={"n_failed_fits": int(len(E_[E_.dataset == ds])) if len(E_) else 0})
        tw = "h10" if ds == "rv5" else "h30"   # rv5 h30 은 겹침 보정 후 time 검정 불가 (FAIL-TIMENULL-OVERLAP)
        m = MD[(MD.dataset == ds) & (MD.window == tw)].groupby("model").mde_time_analytic_pct.mean().dropna().sort_values(ascending=False)
        LOG.add(id=f"RQ2-ML-E2-{ds.upper()}", rq="RQ2", kind="extension", purpose="합성 개입 회복·검정력·MDE",
                data=ds, model="all", settings={"effects": E["effects_pct"], "scenarios": E["scenarios"]},
                seed=E["seed"], result={"window": tw, "mde_time_pct_best5": m.head(5).round(1).to_dict()})
    LOG.add(id="RQ2-ML-E2-SPACE", rq="RQ2", kind="extension", purpose="donor 순열 검정의 FPR·검정력 (실제 개입일)",
            data="space", model="all", result=sp[sp.effect_pct.isin([0, -20])].groupby(["model", "effect_pct"]).space_reject.mean().round(3).unstack().to_dict())
    LOG.add(id="RQ2-ML-E3", rq="RQ2", kind="extension", purpose="Optuna 중첩 시계열 CV 튜닝 전후", data="rv5, gk",
            model=E["tuning"]["models"], settings=E["tuning"], seed=E["seed"],
            result=TC[["dataset", "model", "effect_rmse_h30", "mde_time_pct_h30"]].round(4).to_dict("records"))
    LOG.add(id="RQ2-ML-E4", rq="RQ2", kind="extension", purpose="앙상블·오차 상관", data="rv5, gk, space",
            model=ENSM, settings=E["ensemble"], seed=E["seed"],
            result=agg[(agg.model.str.startswith("ens_")) & (agg.horizon == "h30") & (agg.subset == "all")][
                ["dataset", "model", "eff_rmse"]].round(4).to_dict("records"))
    p = R5[R5.primary]
    LOG.add(id="RQ2-ML-E5", rq="RQ2", kind="extension", purpose="실제 규제 효과 재추정 (사전 지정 모형)",
            data="rv5 실제 개입일 (space 과제)", model=sel["selected"], settings=sel["rule"], seed=E["seed"],
            result=p[["code", "window", "effect", "time_p", "space_p", "primary_p", "primary_p_source",
                      "primary_p_holm"]].round(4).to_dict("records"),
            interpretation="모형 선택은 가짜 개입일 결과만으로 사전에 정함. 다른 모형은 민감도.")
    if len(E_):
        for (ds, m), g in E_.groupby(["dataset", "model"]):
            LOG.add(id=f"RQ2-ML-FAIL-{ds}-{m}", rq="RQ2", kind="extension", purpose="적합 실패 기록", data=ds, model=m,
                    status="failed", result={"n": len(g), "example": g.error.iloc[0], "tasks": g[["code", "i0"]].head(10).values.tolist()})
    # 방법상 실패와 수정 (REPORT §5)
    LOG.add(id="RQ2-ML-FAIL-TIMENULL-OVERLAP", rq="RQ2", kind="extension", status="failed",
            purpose="time-보정 귀무분포가 겹치는 가짜 개입 창(rv5 3일 간격)을 독립으로 취급",
            data="rv5, gk", model="all",
            result={"before": "rv5 h30 time FPR 0.2~0.4 (BSTS 005935 0.42), 실제 BSTS 005930 post time_p 0.001",
                    "fix": "비겹침 유효 창 수 n_eff 로 자유도·예측구간, n_eff<3 이면 검정 불가",
                    "after": "rv5 h10 FPR 0.02~0.12, h30·h61·post_effective time 검정 불가"},
            interpretation="사전 지정 선정 창 h30 이 검정 불가가 되어 fallback_windows 로 h10 사용 (규칙 이탈, 공개)")
    LOG.add(id="RQ2-ML-FAIL-SPACE-DONOR-MISMATCH", rq="RQ2", kind="extension", status="failed",
            purpose="튜닝 모형만 donor 40개로 space 순열검정 → 기본 모형(donor 175개)과 검정력 비교 불공정",
            data="space", model="*_tA, *_tB",
            result={"before": "-40% 검정력 튜닝 0.45~0.60 vs 기본 ML 0.21~0.26",
                    "after_common40": "같은 donor 40개에서 모든 모형 0.45~0.78, 튜닝 이점 없음"},
            interpretation="exp2_space_permutation_common40 을 비교 기준으로 사용")
    LOG.add(id="RQ2-ML-FAIL-SPACE-TUNE-HANG", rq="RQ2", kind="extension", status="failed",
            purpose="space enet_tB 튜닝이 joblib 작업자 대기로 2시간 무진행",
            data="space", model="enet_tB",
            result={"action": "해당 실행 프로세스 트리만 종료 후 tune 단계 재실행 (캐시 이어받기)", "rerun": "정상 완료"},
            interpretation="결과에는 영향 없음 (완료된 과제만 캐시에 저장되는 구조)")
    LOG.save()


def stage_figures():
    from src.rq2.phase1.ml_extension import plots
    plots.make_all()


STAGES = {"bench": stage_bench, "fm": stage_fm, "tune": stage_tune, "ensemble": stage_ensemble, "spill": stage_spill,
          "leaky": stage_leaky, "ablation": stage_ablation, "diag_fit": stage_diag_fit, "tables": stage_tables,
          "figures": stage_figures}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(STAGES))
    a = ap.parse_args()
    for s in a.stages.split(","):
        t0 = time.time()
        print(f"== {s}", flush=True)
        STAGES[s]()
        print(f"== {s} done {time.time() - t0:.0f}s", flush=True)
