"""진단 1~12. 각 함수는 표(DataFrame)를 돌려주고, run_all 이 저장·보고한다.

 1 leakage          사후 y 불변성 + 시차 누출 시연 + 통제 PCA 를 전체표본으로 추정했을 때의 차이
 2 overfitting      학습 적합 RMSE 대비 표본외 RMSE 비율, 튜닝 안쪽 개선 대비 바깥 개선
 3 bias             창평균 오차 평균 (HAC t) — 체계적 과대/과소 예측
 4 residual autocorr 사후 일별 오차의 1차 자기상관, Ljung-Box(10) 기각률
 5 distribution shift 학습 대비 사후 통제 분포 이동(KS)과 오차 크기의 관계, 상장 경계 통과 여부
 6 calibration      native 일별 90%·창 95% coverage vs time 보정 coverage
 7 FPR inflation    효과 0 에서 native / time / space 기각률
 8 HP sensitivity   가짜 개입일 간 최적 하이퍼파라미터 변동, trial 점수 분산
 9 sample period    rv5 앞/뒤 절반, gk 시기별 모형 순위 안정성
10 placebo          (7) 과 같은 자료의 placebo-in-time·space 요약
11 negative control 005935 실제 효과와 같은 기업 차분
12 selection bias   앞 절반에서 최고 모형을 골라 뒤 절반에서 평가 (winner's curse)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from src.rq2.ml_extension import benchmark as B
from src.rq2.ml_extension import data_pipeline as DP

E = DP.ECFG


# 1 ------------------------------------------------------------------------
def _pca_leak_one(code, i0, models, seed):
    import warnings
    warnings.filterwarnings("ignore")
    from src.rq2.ml_extension.models import registry
    reg = registry(dict(E["bsts_bench"], niter=600, burn=200))
    y, X, tr, po = DP.task_data("rv5", code, i0, H=30)
    Xf = DP.rv5_controls(pd.Timestamp(DP.CFG["sample"]["train_end"]))[X.columns].reindex(X.index)
    out = []
    for m in models:
        a = reg[m](y, X, tr, po, seed=seed).pred.values
        b = reg[m](y, Xf, tr, po, seed=seed).pred.values
        yp = y[po].values
        out.append({"code": code, "i0": i0, "model": m, "err_trainonly": yp.mean() - a.mean(),
                    "err_fullpca": yp.mean() - b.mean(),
                    "rmse_trainonly": np.sqrt(np.mean((yp - a) ** 2)), "rmse_fullpca": np.sqrt(np.mean((yp - b) ** 2))})
    return out


def pca_leak(models=("ols", "lgbm", "xgb", "rf"), every=3) -> pd.DataFrame:
    f = DP.CACHE / "rv5" / "diag_pca_leak.parquet"
    if f.exists():
        return pd.read_parquet(f)
    tasks = [t for t in B.tasks_for("rv5") if t[1] + 30 <= 190][::every]
    r = Parallel(n_jobs=E["n_jobs"])(delayed(_pca_leak_one)(c, i0, list(models), E["seed"]) for c, i0 in tasks)
    d = pd.DataFrame([x for xx in r for x in xx])
    d.to_parquet(f)
    return d


# 2 ------------------------------------------------------------------------
def overfitting(ds, W) -> pd.DataFrame:
    W = W[W.window == "h30"].copy()
    D, _ = B.collect(ds, sorted(W.model.unique()))
    D = D[D.h <= 30].assign(e2=lambda d: (d.y - d.pred) ** 2)
    oos = D.groupby(["model", "code", "i0"]).e2.mean().pow(.5).rename("oos_rmse").reset_index()
    M = W[["model", "code", "i0", "pre_rmse"]].merge(oos, on=["model", "code", "i0"])
    g = M.groupby("model").agg(in_sample_rmse=("pre_rmse", "mean"), oos_rmse=("oos_rmse", "mean")).reset_index()
    g["ratio_oos_in"] = g.oos_rmse / g.in_sample_rmse
    g["dataset"] = ds
    return g


# 3 ------------------------------------------------------------------------
def bias_test(ds, W) -> pd.DataFrame:
    W = B.add_flags(ds, W)
    rows = []
    step = E["datasets"][ds].get("cutoff_step", 1)
    for (m, code, w), g in W[W.window.isin(E["horizons"])].groupby(["model", "code", "window"]):
        e = g.sort_values("i0").err.values
        h = DP.WIN[w][1]
        lag = max(int(np.ceil(h / step)) - 1, 0)
        n = len(e)
        u = e - e.mean()
        L = min(lag, n - 2)
        v = u @ u / n + 2 * sum((1 - k / (L + 1)) * (u[k:] @ u[:-k] / n) for k in range(1, L + 1))
        t = e.mean() / np.sqrt(max(v, 1e-12) / n)
        rows.append({"dataset": ds, "model": m, "code": code, "window": w, "n": n, "bias": e.mean(),
                     "hac_t": t, "p": 2 * stats.t.sf(abs(t), n - 1)})
    return pd.DataFrame(rows)


# 4 ------------------------------------------------------------------------
def resid_autocorr(ds, models, hmax=30) -> pd.DataFrame:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    D, _ = B.collect(ds, models)
    D = D[D.h <= hmax].assign(e=lambda d: d.y - d.pred)
    rows = []
    for (m, code, i0), g in D.groupby(["model", "code", "i0"]):
        e = g.sort_values("h").e.values
        if len(e) < 20:
            continue
        e0 = e - e.mean()
        r1 = (e0[1:] @ e0[:-1]) / (e0 @ e0)
        lb = acorr_ljungbox(e, lags=[10], return_df=True).lb_pvalue.iloc[0]
        rows.append({"model": m, "code": code, "i0": i0, "acf1": r1, "lb_p": lb})
    d = pd.DataFrame(rows)
    g = d.groupby("model").agg(acf1_mean=("acf1", "mean"), lb_reject_rate=("lb_p", lambda p: np.mean(p < .05)),
                               n=("acf1", "size")).reset_index()
    g["dataset"] = ds
    return g


# 5 ------------------------------------------------------------------------
def distribution_shift(ds, W, window="h30") -> pd.DataFrame:
    W = B.add_flags(ds, W)
    W = W[W.window == window]
    ks = []
    for code, i0 in W[["code", "i0"]].drop_duplicates().itertuples(index=False):
        y, X, tr, po = DP.task_data(ds, code, i0)
        x = X["don_mean"]
        ks.append({"code": code, "i0": i0, "ks_don_mean": stats.ks_2samp(x[tr].values[-120:], x[po].values[:30]).statistic,
                   "shift_don_mean_sd": (x[po].values[:30].mean() - x[tr].mean()) / x[tr].std()})
    K = pd.DataFrame(ks)
    M = W.merge(K, on=["code", "i0"])
    rows = []
    for m, g in M.groupby("model"):
        rho = stats.spearmanr(np.abs(g.shift_don_mean_sd), np.abs(g.err)).statistic
        big = np.abs(g.shift_don_mean_sd) > 1
        rows.append({"dataset": ds, "model": m, "spearman_absshift_abserr": rho,
                     "rmse_small_shift": np.sqrt(np.mean(g.err[~big] ** 2)),
                     "rmse_large_shift": np.sqrt(np.mean(g.err[big] ** 2)) if big.any() else np.nan,
                     "n_large": int(big.sum()),
                     "rmse_cross_listing": np.sqrt(np.mean(g.err[g.crosses_listing] ** 2)) if g.crosses_listing.any() else np.nan,
                     "rmse_pre_listing": np.sqrt(np.mean(g.err[~g.crosses_listing] ** 2))})
    return pd.DataFrame(rows)


# 9 ------------------------------------------------------------------------
def period_stability(ds, W, window="h30") -> pd.DataFrame:
    W = B.add_flags(ds, W)
    W = W[W.window == window].copy()
    if ds == "rv5":
        med = np.median(sorted(W.i0.unique()))
        W["period"] = np.where(W.i0 <= med, "first_half", "second_half")
    else:
        y = pd.to_datetime(W.cutoff).dt.year
        W["period"] = pd.cut(y, [2015, 2018, 2021, 2023, 2027], labels=["2016-18", "2019-21", "2022-23", "2024-26"]).astype(str)
    g = W.groupby(["period", "model"]).err.apply(lambda v: np.sqrt(np.mean(v ** 2))).rename("eff_rmse").reset_index()
    g["rank"] = g.groupby("period").eff_rmse.rank()
    g["dataset"] = ds
    return g


# 12 -----------------------------------------------------------------------
def selection_bias(ds, W, window="h30") -> pd.DataFrame:
    W = W[W.window == window].assign(err=lambda d: d.ybar - d.predbar)
    cuts = sorted(W.i0.unique())
    half = len(cuts) // 2
    A = W[W.i0.isin(cuts[:half])]
    Bh = W[W.i0.isin(cuts[half:])]
    ra = A.groupby("model").err.apply(lambda v: np.sqrt(np.mean(v ** 2)))
    rb = Bh.groupby("model").err.apply(lambda v: np.sqrt(np.mean(v ** 2)))
    best = ra.idxmin()
    rows = [{"dataset": ds, "window": window, "selected_on_first_half": best, "first_half_rmse": ra[best],
             "second_half_rmse": rb[best], "second_half_rank": int(rb.rank()[best]), "n_models": len(rb),
             "second_half_best": rb.idxmin(), "second_half_best_rmse": rb.min(),
             "rank_corr_halves": stats.spearmanr(ra.reindex(rb.index), rb).statistic}]
    return pd.DataFrame(rows)
