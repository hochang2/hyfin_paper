"""실험 1: 반사실 예측 벤치마크 (가짜 개입일 rolling-origin).

1) run_models(): (dataset, model) 별로 모든 과제(code × 가짜 개입일)를 적합해 캐시한다.
   캐시 = CACHE/<ds>/<model>_daily.parquet (일별 y·예측·90% 경로구간)
        + CACHE/<ds>/<model>_win.parquet   (평가창별 y 평균, 예측 평균, 반사실 draw 창평균의 분위수)
   이미 있는 과제는 다시 적합하지 않는다 (재실행 시 완료 결과 재사용).
2) metrics(): 지평별 MAE·RMSE·편향, 창평균 오차(=효과 추정오차), 구간 coverage, 계산시간.
3) dm_tests(): 창평균 손실 차이의 Diebold-Mariano (HLN 소표본 보정, 중첩창 길이만큼 Newey-West 시차).
4) invariance_test(): 사후 y 를 바꿔도 반사실 예측이 그대로인지 (사후 정보 누출 검사).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from src.rq2.phase1.ml_extension import data_pipeline as DP
from src.rq2.phase1.ml_extension.models import registry

ECFG = DP.ECFG
SEED = ECFG["seed"]


def cache_path(ds: str, model: str, kind: str) -> Path:
    p = DP.CACHE / ds
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{model}_{kind}.parquet"


def load_cache(ds: str, model: str):
    fd, fw = cache_path(ds, model, "daily"), cache_path(ds, model, "win")
    if fd.exists() and fw.exists():
        return pd.read_parquet(fd), pd.read_parquet(fw)
    return None, None


def summarize_cf(y, po, cf, code, i0, model, extra=None):
    yp = y[po].values
    H = len(yp)
    pred = cf.pred.values[:H]
    D = cf.draws[:, :H] if cf.draws is not None else None
    dates = y.index[po]
    daily = pd.DataFrame({"code": code, "i0": i0, "cutoff": dates[0], "h": np.arange(1, H + 1), "date": dates,
                          "y": yp, "pred": pred})
    if D is not None:
        daily["lo90"], daily["hi90"] = np.quantile(D, [0.05, 0.95], axis=0)
    else:
        daily["lo90"], daily["hi90"] = np.nan, np.nan
    rows = []
    for w, (a, b) in DP.WIN.items():
        if b > H:
            continue
        r = {"code": code, "i0": i0, "cutoff": dates[0], "window": w, "a": a, "b": b,
             "ybar": yp[a:b].mean(), "predbar": pred[a:b].mean()}
        if D is not None:
            mm = D[:, a:b].mean(1)
            r.update(cf_sd=mm.std(), cf_q025=np.quantile(mm, .025), cf_q975=np.quantile(mm, .975),
                     cf_q05=np.quantile(mm, .05), cf_q95=np.quantile(mm, .95))
        rows.append(r)
    win = pd.DataFrame(rows)
    pre = np.nan
    if cf.fitted_pre is not None:
        f = cf.fitted_pre.dropna()
        yt = y.reindex(f.index)
        pre = float(np.sqrt(np.mean((yt - f) ** 2)))
    win["pre_rmse"] = pre
    win["H"] = H
    if extra:
        for k, v in extra.items():
            win[k] = v
    daily["model"] = model
    win["model"] = model
    return daily, win


def _one(ds, code, i0, model, base, kw, x_shift):
    import warnings
    warnings.filterwarnings("ignore")
    fun = registry(ECFG["bsts_bench"])[base] if base in registry({}) else None
    t0 = time.time()
    y, X, tr, po = DP.task_data(ds, code, i0)
    if x_shift and x_shift.get("drop"):   # 변수 제거 실험 (feature ablation)
        X = X.drop(columns=[c for c in x_shift["drop"] if c in X.columns])
        if X.shape[1] == 0:
            X = pd.DataFrame({"const0": np.zeros(len(X))}, index=X.index)
    elif x_shift:   # 시나리오 F: 사후 통제변수에 효과 일부가 번짐
        X = X.copy()
        H = int(po.sum())
        tau = np.log1p(x_shift["pct"] / 100.0)
        for c in x_shift["controls"]:
            if c in X.columns:
                X.loc[po, c] = X.loc[po, c].values + x_shift["share"] * tau * np.ones(H)
    try:
        if kw.get("_tuned"):
            from src.rq2.phase1.ml_extension.models import cf_sk
            cf = cf_sk(y, X, tr, po, seed=SEED, kind=kw["_kind"], params=kw["_params"])
        else:
            cf = fun(y, X, tr, po, seed=SEED, **kw)
    except Exception as e:  # noqa: BLE001
        return None, pd.DataFrame([{"code": code, "i0": i0, "model": model, "error": repr(e)[:200]}])
    d, w = summarize_cf(y, po, cf, code, i0, model, {"fit_sec": time.time() - t0})
    return d, w


def tasks_for(ds: str, codes=None, cut_list=None):
    if ds == "space":
        return [(c, DP.actual_index("space")) for c in (codes or DP.space_units())]
    codes = codes or ECFG["datasets"][ds]["codes"]
    cut_list = cut_list if cut_list is not None else DP.cutoffs(ds)
    return [(c, i0) for c in codes for i0 in cut_list]


def run_models(ds: str, models: list[str], tasks=None, kw_by_model=None, x_shift=None, tag=None,
               n_jobs=None, verbose=True):
    """모형별 캐시 갱신. tag 가 있으면 캐시 이름 = f"{model}@{tag}" (변형 실험)."""
    n_jobs = n_jobs or ECFG["n_jobs"]
    tasks = tasks or tasks_for(ds)
    kw_by_model = kw_by_model or {}
    for m in models:
        name = f"{m}@{tag}" if tag else m
        d0, w0 = load_cache(ds, name)
        done = set() if w0 is None else set(zip(w0.code, w0.i0))
        todo = [t for t in tasks if t not in done]
        if not todo:
            continue
        t0 = time.time()
        base = m.split("_t")[0] if kw_by_model.get(m, {}).get("_tuned") else m
        res = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_one)(ds, c, i0, name, base, kw_by_model.get(m, {}) if not callable(kw_by_model.get(m))
                          else kw_by_model[m](c, i0), x_shift) for c, i0 in todo)
        ds_, ws_, errs = [], [], []
        for d, w in res:
            if d is None:
                errs.append(w)
            else:
                ds_.append(d)
                ws_.append(w)
        if errs:
            e = pd.concat(errs)
            ef = DP.CACHE / ds / "errors.csv"
            e.to_csv(ef, mode="a", header=not ef.exists(), index=False)
            if verbose:
                print(f"  [{ds}/{name}] {len(e)} 실패: {e.error.iloc[0]}", flush=True)
        if ds_:
            D = pd.concat(([d0] if d0 is not None else []) + ds_)
            W = pd.concat(([w0] if w0 is not None else []) + ws_)
            D.to_parquet(cache_path(ds, name, "daily"))
            W.to_parquet(cache_path(ds, name, "win"))
        if verbose:
            print(f"  [{ds}/{name}] {len(todo)} 과제 {time.time() - t0:.0f}s", flush=True)


def collect(ds: str, models: list[str]):
    D, W = [], []
    for m in models:
        d, w = load_cache(ds, m)
        if d is not None:
            D.append(d)
            W.append(w)
    return (pd.concat(D, ignore_index=True) if D else pd.DataFrame(),
            pd.concat(W, ignore_index=True) if W else pd.DataFrame())


# ---------------------------------------------------------------- 평가 지표
def add_flags(ds: str, W: pd.DataFrame) -> pd.DataFrame:
    W = W.copy()
    W["err"] = W.ybar - W.predbar       # 창평균 오차 = 효과 0 일 때의 효과 추정치
    if ds == "rv5":
        lst = ECFG["datasets"]["rv5"]["listing_index"]
        W["crosses_listing"] = (W.i0 + W.b) > lst
    else:
        W["crosses_listing"] = False
    if "cf_q975" in W:
        rej = (~((W.ybar - W.cf_q975 < 0) & (0 < W.ybar - W.cf_q025))).astype(float)
        W["native_reject"] = rej.where(W.cf_q975.notna())
    else:
        W["native_reject"] = np.nan
    return W


def daily_metrics(D: pd.DataFrame) -> pd.DataFrame:
    """지평 h 까지의 일별 오차 지표 (가짜 개입일별로 계산 후 평균)."""
    out = []
    cov = ((D.y >= D.lo90) & (D.y <= D.hi90)).astype(float).where(D.lo90.notna())
    D = D.assign(e=D.y - D.pred, cov=cov)
    for hname in ECFG["horizons"]:
        h = DP.WIN[hname][1]
        g = D[D.h <= h]
        full = g.groupby(["model", "code", "i0"]).h.transform("max") == h   # 지평 전체가 있는 개입일만
        g = g[full]
        per = g.groupby(["model", "code", "i0"]).agg(mae=("e", lambda v: np.mean(np.abs(v))),
                                                     rmse=("e", lambda v: np.sqrt(np.mean(v ** 2))),
                                                     bias=("e", "mean"), cov90=("cov", "mean")).reset_index()
        per["horizon"] = hname
        out.append(per)
    return pd.concat(out, ignore_index=True)


def summary_table(ds: str, D: pd.DataFrame, W: pd.DataFrame, sub: str = "all") -> pd.DataFrame:
    W = add_flags(ds, W)
    per = daily_metrics(D)
    if sub == "prelisting":
        keep = W[(W.window.isin(ECFG["horizons"])) & ~W.crosses_listing][["code", "i0", "window"]]
        per = per.merge(keep.rename(columns={"window": "horizon"}), on=["code", "i0", "horizon"])
        W = W[~W.crosses_listing]
    s1 = per.groupby(["model", "code", "horizon"]).agg(n_cut=("i0", "size"), MAE=("mae", "mean"),
                                                         RMSE=("rmse", "mean"), bias=("bias", "mean"),
                                                         cov90=("cov90", "mean")).reset_index()
    Wh = W[W.window.isin(ECFG["horizons"])].rename(columns={"window": "horizon"})
    s2 = Wh.groupby(["model", "code", "horizon"]).agg(
        eff_bias=("err", "mean"), eff_sd=("err", "std"), eff_rmse=("err", lambda v: np.sqrt(np.mean(v ** 2))),
        eff_mae=("err", lambda v: np.mean(np.abs(v))), native_fpr=("native_reject", "mean"),
        fit_sec=("fit_sec", "mean"), pre_rmse=("pre_rmse", "mean")).reset_index()
    s = s1.merge(s2, on=["model", "code", "horizon"], how="outer")
    s["dataset"], s["subset"] = ds, sub
    return s


def regime_table(ds: str, W: pd.DataFrame, window="h30") -> pd.DataFrame:
    W = add_flags(ds, W)
    W = W[W.window == window]
    R = DP.regime_labels(ds, sorted(W.i0.unique()))
    W = W.merge(R[["i0", "vol_regime", "jump_regime"]], on="i0")
    out = []
    for col in ["vol_regime", "jump_regime"]:
        g = W.groupby(["model", "code", col]).agg(n=("err", "size"), eff_bias=("err", "mean"),
                                                   eff_rmse=("err", lambda v: np.sqrt(np.mean(v ** 2)))).reset_index()
        g = g.rename(columns={col: "regime"})
        g["regime_type"] = col
        out.append(g)
    o = pd.concat(out)
    o["dataset"], o["window"] = ds, window
    return o


# ---------------------------------------------------------------- Diebold-Mariano
def dm_hln(d: np.ndarray, lag: int) -> tuple[float, float]:
    """손실차 d 의 평균이 0 인지. Newey-West(lag) 분산 + Harvey-Leybourne-Newbold 보정, t(n-1)."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 8:
        return np.nan, np.nan
    m = d.mean()
    u = d - m
    lag = min(lag, n - 2)
    g0 = u @ u / n
    v = g0 + 2 * sum((1 - k / (lag + 1)) * (u[k:] @ u[:-k] / n) for k in range(1, lag + 1))
    if v <= 0:
        return np.nan, np.nan
    dm = m / math.sqrt(v / n)
    h = lag + 1
    corr = math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-6))
    t = dm * corr
    return t, 2 * stats.t.sf(abs(t), n - 1)


def dm_tests(ds: str, W: pd.DataFrame, refs=("ols", "bsts"), loss="window_se") -> pd.DataFrame:
    """창평균 오차 제곱(효과 추정 손실) 차이. 음(−)의 t = 비교모형이 기준보다 정확."""
    W = add_flags(ds, W)
    step = ECFG["datasets"][ds].get("cutoff_step", 1)
    rows = []
    for hname in ECFG["horizons"]:
        h = DP.WIN[hname][1]
        g = W[W.window == hname]
        piv = g.pivot_table(index=["code", "i0"], columns="model", values="err")
        for ref in refs:
            if ref not in piv:
                continue
            for m in piv.columns:
                if m == ref:
                    continue
                for code in piv.index.get_level_values(0).unique():
                    p = piv.loc[code].sort_index()
                    d = (p[m] ** 2 - p[ref] ** 2).values
                    t, pv = dm_hln(d, lag=max(math.ceil(h / step) - 1, 0))
                    rows.append({"dataset": ds, "horizon": hname, "ref": ref, "model": m, "code": code,
                                 "n": int(np.isfinite(d).sum()), "mean_loss_diff": np.nanmean(d), "dm_t": t, "p": pv})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 누출 검사
def invariance_test(ds: str, models: list[str], n_tasks: int = 3, shift: float = -0.5) -> pd.DataFrame:
    """사후 y 를 shift 만큼 바꿔 다시 적합 → 반사실 예측이 같으면 사후 정보가 모형에 들어가지 않은 것."""
    import warnings
    warnings.filterwarnings("ignore")
    from src.rq2.phase1.ml_extension.models import cf_gbm_har
    reg = registry(dict(ECFG["bsts_bench"], niter=400, burn=100))
    tasks = tasks_for(ds)
    rng = np.random.default_rng(SEED)
    pick = [tasks[i] for i in rng.choice(len(tasks), n_tasks, replace=False)]
    rows = []
    for code, i0 in pick:
        y, X, tr, po = DP.task_data(ds, code, i0)
        y2 = y.copy()
        y2[po] = y2[po] + shift
        for m in models + ["lgbm_lag_leaky"]:
            if m == "lgbm_lag_leaky":
                a = cf_gbm_har(y, X, tr, po, seed=SEED, leaky=True).pred.values
                b = cf_gbm_har(y2, X, tr, po, seed=SEED, leaky=True).pred.values
            else:
                a = reg[m](y, X, tr, po, seed=SEED).pred.values
                b = reg[m](y2, X, tr, po, seed=SEED).pred.values
            rows.append({"dataset": ds, "code": code, "i0": i0, "model": m,
                         "max_abs_pred_change": float(np.max(np.abs(a - b))),
                         "passed": bool(np.max(np.abs(a - b)) < 1e-8)})
    return pd.DataFrame(rows)
