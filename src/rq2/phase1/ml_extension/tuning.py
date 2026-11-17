"""실험 3: Optuna 하이퍼파라미터 튜닝 (중첩 시계열 CV) 전후 비교.

바깥 루프 = 실험 1 의 가짜 개입일 과제. 안쪽 루프 = 그 과제의 학습구간 안에서만 만든 검증 fold.
  안쪽 fold k: 학습구간의 마지막 4 개 연속 블록(길이 L=20, 학습이 짧으면 max(5,(n-40)//4)) 중 k 번째를
             검증으로, 그 앞 전부를 학습으로 쓴다 (확장창). 바깥 사후구간은 튜닝에 전혀 쓰지 않는다.
  목적 A (forecast) : 검증 블록 일별 MSE 평균
  목적 B (effect)   : 검증 블록 평균오차²의 평균 (= 블록을 가짜 사후창으로 본 효과추정 MSE)
TPE (seed 고정), 30 trial, 앞 10 개 무작위. trial 0 = 기본 하이퍼파라미터 (기본값 대비 안쪽 개선 = 튜닝 낙관치 측정용).
최적값으로 학습구간 전체를 다시 적합 → cf_sk 와 같은 방식(OOF 잔차 bootstrap)으로 반사실 → benchmark 캐시 형식 저장.
모형 이름: <kind>_tA, <kind>_tB.  튜닝 기록: CACHE/<ds>/tuning_<kind>_t<obj>.parquet
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.rq2.phase1.ml_extension import benchmark as B
from src.rq2.phase1.ml_extension import data_pipeline as DP

E = DP.ECFG
T = E["tuning"]

DEFAULTS = {
    "lgbm": dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=15, subsample=0.8,
                 colsample_bytree=0.8, reg_lambda=1.0),
    "xgb": dict(n_estimators=300, learning_rate=0.03, max_depth=2, min_child_weight=1.0, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0),
    "enet": dict(alpha=0.01, l1_ratio=0.5),
    "rf": dict(n_estimators=300, max_depth=4, min_samples_leaf=10, max_features=0.6),
}


def space(kind: str, t) -> dict:
    if kind == "lgbm":
        return dict(n_estimators=t.suggest_int("n_estimators", 50, 600, log=True),
                    learning_rate=t.suggest_float("learning_rate", 0.005, 0.2, log=True),
                    num_leaves=t.suggest_int("num_leaves", 2, 31),
                    min_child_samples=t.suggest_int("min_child_samples", 5, 50),
                    subsample=t.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True))
    if kind == "xgb":
        return dict(n_estimators=t.suggest_int("n_estimators", 50, 600, log=True),
                    learning_rate=t.suggest_float("learning_rate", 0.005, 0.2, log=True),
                    max_depth=t.suggest_int("max_depth", 1, 5),
                    min_child_weight=t.suggest_float("min_child_weight", 1.0, 20.0, log=True),
                    subsample=t.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True))
    if kind == "enet":
        return dict(alpha=t.suggest_float("alpha", 1e-4, 1.0, log=True),
                    l1_ratio=t.suggest_float("l1_ratio", 0.0, 1.0))
    if kind == "rf":
        return dict(n_estimators=t.suggest_int("n_estimators", 50, 200, log=True),   # 계산비용 제한 (RF 는 트리 수에 둔감)
                    max_depth=t.suggest_int("max_depth", 2, 10),
                    min_samples_leaf=t.suggest_int("min_samples_leaf", 3, 40),
                    max_features=t.suggest_float("max_features", 0.3, 1.0))
    raise ValueError(kind)


def inner_folds(n: int):
    L = T["inner_val_len"]
    k = T["inner_folds"]
    if n - k * L < 40:
        L = max(5, (n - 40) // k)
    starts = [n - (k - j) * L for j in range(k)]
    return [(np.arange(s), np.arange(s, s + L)) for s in starts]


def inner_score(kind, params, Xt, yt, folds, objective, seed):
    from src.rq2.phase1.ml_extension.models import make_estimator
    se, be = [], []
    for a, b in folds:
        m = make_estimator(kind, params, seed)
        m.fit(Xt.iloc[a], yt.iloc[a])
        e = yt.iloc[b].values - m.predict(Xt.iloc[b])
        se.append(np.mean(e ** 2))
        be.append(np.mean(e) ** 2)
    return float(np.mean(se) if objective == "A" else np.mean(be)), float(np.mean(se)), float(np.mean(be))


def _tune_one(ds, code, i0, kind, objective, seed):
    import warnings
    warnings.filterwarnings("ignore")
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from src.rq2.phase1.ml_extension.models import cf_sk
    t0 = time.time()
    y, X, tr, po = DP.task_data(ds, code, i0)
    Xt, yt = X[tr], y[tr]
    folds = inner_folds(len(yt))
    aux = {}

    def obj(t):
        p = space(kind, t)
        s, a, b = inner_score(kind, p, Xt, yt, folds, objective, seed)
        aux[t.number] = (a, b)
        return s

    st = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=T["n_startup"]))
    st.enqueue_trial(DEFAULTS[kind])
    st.optimize(obj, n_trials=T["n_trials"])
    best = st.best_params
    tune_sec = time.time() - t0
    vals = np.array([tt.value for tt in st.trials if tt.value is not None])
    rec = {"code": code, "i0": i0, "kind": kind, "objective": objective, "params": str(best),
           "inner_default": float(st.trials[0].value), "inner_best": float(st.best_value),
           "inner_median": float(np.median(vals)), "inner_gain": float(st.trials[0].value - st.best_value),
           "best_trial": int(st.best_trial.number), "tune_sec": tune_sec,
           "inner_daily_mse_best": aux[st.best_trial.number][0], "inner_blockmean_mse_best": aux[st.best_trial.number][1]}
    name = f"{kind}_t{objective}"
    try:
        cf = cf_sk(y, X, tr, po, seed=seed, kind=kind, params=best)
    except Exception as e:  # noqa: BLE001
        return None, None, dict(rec, error=repr(e)[:200])
    d, w = B.summarize_cf(y, po, cf, code, i0, name, {"fit_sec": time.time() - t0, "tune_sec": tune_sec})
    return d, w, rec


def tuning_tasks(ds):
    tasks = B.tasks_for(ds)
    if ds == "gk":
        cuts = sorted({i0 for _, i0 in tasks})[::T["gk_cutoff_every"]]
        tasks = [t for t in tasks if t[1] in set(cuts)]
    return tasks


def run_tuning(ds: str, kinds=None, objectives=("A", "B"), n_jobs=None):
    kinds = kinds or T["models"]
    n_jobs = n_jobs or E["n_jobs"]
    for kind in kinds:
        for ob in objectives:
            name = f"{kind}_t{ob}"
            d0, w0 = B.load_cache(ds, name)
            done = set() if w0 is None else set(zip(w0.code, w0.i0))
            todo = [t for t in tuning_tasks(ds) if t not in done]
            if not todo:
                continue
            t0 = time.time()
            res = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_tune_one)(ds, c, i0, kind, ob, E["seed"]) for c, i0 in todo)
            D = [r[0] for r in res if r[0] is not None]
            W = [r[1] for r in res if r[1] is not None]
            R = pd.DataFrame([r[2] for r in res])
            fr = DP.CACHE / ds / f"tuning_{name}.parquet"
            if fr.exists():
                R = pd.concat([pd.read_parquet(fr), R])
            R.to_parquet(fr)
            if D:
                pd.concat(([d0] if d0 is not None else []) + D).to_parquet(B.cache_path(ds, name, "daily"))
                pd.concat(([w0] if w0 is not None else []) + W).to_parquet(B.cache_path(ds, name, "win"))
            print(f"  [{ds}/{name}] {len(todo)} 과제 {time.time() - t0:.0f}s", flush=True)


def tuning_records(ds: str) -> pd.DataFrame:
    fs = sorted((DP.CACHE / ds).glob("tuning_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def param_stability(R: pd.DataFrame) -> pd.DataFrame:
    """가짜 개입일 사이 최적 하이퍼파라미터의 변동 (sensitivity / 불안정성)."""
    import ast
    rows = []
    for (kind, ob, code), g in R.groupby(["kind", "objective", "code"]):
        P = pd.DataFrame([ast.literal_eval(p) for p in g.params])
        for c in P.columns:
            v = P[c].astype(float)
            rows.append({"kind": kind, "objective": ob, "code": code, "param": c, "median": v.median(),
                         "iqr_over_median": (v.quantile(.75) - v.quantile(.25)) / (abs(v.median()) + 1e-12),
                         "min": v.min(), "max": v.max()})
    return pd.DataFrame(rows)
