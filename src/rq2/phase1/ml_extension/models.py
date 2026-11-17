"""ML 확장 실험의 모형 레지스트리.

1차 연구 추정기(src.rq2.core.models: OLS·ridge·enet·BSTS·SARIMAX·UC·HAR-X·LGBM·XGB)를 그대로 쓰고,
다음만 새로 더한다 (모두 같은 CF 형식: pred, draws(ndraw×H), info, fitted_pre):
  naive_mean  학습 마지막 60일 평균 (통제 없음, 가장 단순한 기준)
  did_gap     donor 평균 + 학습 마지막 60일의 평균 격차 (β=1 고정, 차분 설계의 반사실)
  rf          Random Forest (OOF 잔차 bootstrap 구간)
  lgbm_har    LightGBM + HAR 시차(1·5·22일). 사후에는 시차에 자기 반사실 예측을 재귀적으로 넣는다 (사후 y 미사용)
  sk_tuned    튜닝된 하이퍼파라미터로 만든 sklearn/LGBM/XGB 추정기 (tuning.py 가 사용)
불확실성은 core 와 같은 방식(사전구간 시계열 CV 잔차의 stationary block bootstrap)이다.
모든 트리 모형은 n_jobs=1 (병렬 프로세스 안에서 스레드 과다 방지; 결과에는 영향 없음).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.rq2.core import models as CM
from src.rq2.core.models import CF, NDRAW, _oof_resid, stationary_bootstrap_paths

LOCAL_N = 60


def _resid_draws(pred: np.ndarray, resid: np.ndarray, H: int, seed: int) -> np.ndarray:
    return pred[None, :] + stationary_bootstrap_paths(resid, H, NDRAW, np.random.default_rng(seed))


def cf_naive_mean(y, X, tr, po, seed=0, **kw) -> CF:
    yt = y[tr].values
    m = yt[-LOCAL_N:].mean()
    H = int(po.sum())
    # 잔차: 직전 60일 이동평균으로 한 칸 앞을 예측했을 때의 오차 (평균 예측의 표본외 오차)
    s = pd.Series(yt).rolling(LOCAL_N, min_periods=20).mean().shift(1)
    res = (pd.Series(yt) - s).dropna().values
    pred = np.full(H, m)
    return CF(pd.Series(pred, index=y.index[po]), _resid_draws(pred, res, H, seed), {"level": float(m)},
              pd.Series(s.values, index=y.index[tr]))


def cf_did_gap(y, X, tr, po, seed=0, **kw) -> CF:
    d = (y - X["don_mean"])
    g = d[tr].values[-LOCAL_N:].mean()
    pred = X["don_mean"][po].values + g
    s = pd.Series(d[tr].values).rolling(LOCAL_N, min_periods=20).mean().shift(1)
    res = (pd.Series(d[tr].values) - s).dropna().values
    return CF(pd.Series(pred, index=y.index[po]), _resid_draws(pred, res, int(po.sum()), seed), {"gap": float(g)},
              pd.Series(X["don_mean"][tr].values + s.values, index=y.index[tr]))


def make_estimator(kind: str, params: dict | None = None, seed: int = 0):
    """튜닝·기본 공용 추정기 생성기."""
    params = dict(params or {})
    if kind == "lgbm":
        import lightgbm as lgb
        p = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=15, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=1)
        p.update(params)
        return lgb.LGBMRegressor(**p)
    if kind == "xgb":
        import xgboost as xgb
        p = dict(n_estimators=300, learning_rate=0.03, max_depth=2, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=1.0, random_state=seed, n_jobs=1)
        p.update(params)
        return xgb.XGBRegressor(**p)
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        p = dict(n_estimators=300, max_depth=4, min_samples_leaf=10, max_features=0.6, random_state=seed, n_jobs=1)
        p.update(params)
        return RandomForestRegressor(**p)
    if kind in ("enet", "ridge"):
        from sklearn.linear_model import ElasticNet, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if kind == "ridge":
            return make_pipeline(StandardScaler(), Ridge(alpha=params.get("alpha", 1.0)))
        return make_pipeline(StandardScaler(), ElasticNet(alpha=params.get("alpha", 0.01),
                                                          l1_ratio=params.get("l1_ratio", 0.5), max_iter=20000))
    raise ValueError(kind)


def cf_sk(y, X, tr, po, seed=0, kind="rf", params=None, **kw) -> CF:
    """일반 회귀 추정기 반사실 + OOF 잔차 bootstrap 구간 (core.cf_gbm 과 같은 방식)."""
    make = lambda: make_estimator(kind, params, seed)  # noqa: E731
    m = make()
    m.fit(X[tr], y[tr])
    pred = m.predict(X[po])
    oof = _oof_resid(make, X[tr], y[tr])
    return CF(pd.Series(pred, index=y.index[po]), _resid_draws(pred, oof, int(po.sum()), seed),
              {"oof_rmse": float(np.sqrt(np.mean(oof ** 2))), "params": params or "default"},
              pd.Series(m.predict(X[tr]), index=y.index[tr]))


def _har(yv: np.ndarray, t: int) -> list[float]:
    return [yv[t - 1], yv[t - 5:t].mean(), yv[t - 22:t].mean()]


def cf_gbm_har(y, X, tr, po, seed=0, leaky=False, y_obs_post=None, **kw) -> CF:
    """LightGBM + HAR 시차.
    leaky=False: 사후 시차는 자기 반사실 예측으로 재귀 (타당).
    leaky=True : 사후 시차에 '관측된' 사후 y 를 넣는다 (처치된 결과가 반사실로 새는 잘못된 설계 — 시연용).
                 y_obs_post 로 주입된 사후 y 를 받을 수 있다.
    """
    yv = y.values.astype(float).copy()
    tr_idx = np.flatnonzero(tr.values)
    po_idx = np.flatnonzero(po.values)
    if y_obs_post is not None:
        yv[po_idx] = y_obs_post
    rows = [t for t in tr_idx if t >= 22]
    F = np.array([_har(yv, t) + list(X.values[t]) for t in rows])
    make = lambda: make_estimator("lgbm", {}, seed)  # noqa: E731
    m = make()
    m.fit(F, yv[rows])
    yy = yv.copy()
    out = np.empty(len(po_idx))
    for h, t in enumerate(po_idx):
        z = np.array(_har(yy, t) + list(X.values[t]))[None, :]
        out[h] = m.predict(z)[0]
        if not leaky:
            yy[t] = out[h]
    Fdf = pd.DataFrame(F)
    oof = _oof_resid(make, Fdf, pd.Series(yv[rows]))
    return CF(pd.Series(out, index=y.index[po_idx]), _resid_draws(out, oof, len(po_idx), seed),
              {"leaky": leaky}, pd.Series(m.predict(F), index=y.index[rows]))


def registry(bsts_kw: dict) -> dict:
    return {
        "naive_mean": cf_naive_mean,
        "did_gap": cf_did_gap,
        "ols": CM.cf_ols,
        "ridge": CM.cf_ridge,
        "enet": lambda *a, **k: CM.cf_ridge(*a, kind="enet", **k),
        "bsts": lambda *a, **k: CM.cf_bsts(*a, **{**bsts_kw, **k}),
        "sarimax": CM.cf_sarimax,
        "uc": CM.cf_uc,
        "harx": CM.cf_harx,
        "lgbm": lambda *a, **k: CM.cf_gbm(*a, lib="lgbm", params={"n_jobs": 1}, **k),
        "xgb": lambda *a, **k: CM.cf_gbm(*a, lib="xgb", params={"n_jobs": 1}, **k),
        "rf": lambda *a, **k: cf_sk(*a, kind="rf", **k),
        "lgbm_har": cf_gbm_har,
    }


# 모형 복잡도 (보고용): 대략적 자유도·구조
COMPLEXITY = {"naive_mean": 1, "did_gap": 1, "ols": 2, "ridge": 2, "enet": 2, "uc": 3, "sarimax": 3, "harx": 3,
              "bsts": 4, "rf": 5, "lgbm": 5, "xgb": 5, "lgbm_har": 5, "chronos2_cov": 6, "timesfm_xreg": 6,
              "chronos2_uni": 6, "timesfm_uni": 6}
FAMILY = {"naive_mean": "naive", "did_gap": "naive", "ols": "statistical", "ridge": "statistical",
          "bsts": "statistical", "sarimax": "statistical", "uc": "statistical", "harx": "statistical",
          "enet": "ML", "rf": "ML", "lgbm": "ML", "xgb": "ML", "lgbm_har": "ML",
          "chronos2_cov": "foundation", "timesfm_xreg": "foundation", "chronos2_uni": "foundation",
          "timesfm_uni": "foundation"}


def family(m: str) -> str:
    base = m.split("@")[0]
    if base.startswith("ens_"):
        return "ensemble"
    if "_tA" in base or "_tB" in base:
        return "ML-tuned"
    return FAMILY.get(base, "other")
