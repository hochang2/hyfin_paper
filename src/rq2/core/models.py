"""RQ2 반사실(counterfactual) 추정 모형.

모든 추정기는 같은 형식으로 돌려준다:
    CF(pred = 사후구간 반사실 점예측 (Series), draws = (ndraw, H) 반사실 경로 표본, info = dict)
학습은 train 구간(사전)만 쓴다. 사후 실제 y 는 어떤 추정기에도 들어가지 않는다 (HAR 의 자기 시차도 반사실 예측값으로 대체).

불확실성:
- BSTS: 사후분포
- UC / SARIMAX: 상태공간 모형 시뮬레이션 (모수 불확실성 제외 → 구간이 좁을 수 있음, 보고서에 명시)
- OLS / Ridge / GBM / SCM: 사전구간 (교차검증) 잔차의 stationary block bootstrap 경로 + (OLS 는 계수 표본)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.rq2.core.bsts import fit_bsts

NDRAW = 2000


@dataclass
class CF:
    pred: pd.Series
    draws: np.ndarray
    info: dict = field(default_factory=dict)
    fitted_pre: pd.Series | None = None


# ----------------------------------------------------------------- bootstrap helpers
def stationary_bootstrap_paths(resid: np.ndarray, H: int, ndraw: int, rng, mean_block: float = 10.0) -> np.ndarray:
    """Politis-Romano stationary bootstrap: 잔차 자기상관을 보존한 길이 H 경로 ndraw 개."""
    resid = np.asarray(resid) - np.mean(resid)   # 구간 폭만 반영 (편향은 점추정에 남김)
    n = len(resid)
    out = np.empty((ndraw, H))
    p = 1.0 / mean_block
    for d in range(ndraw):
        i = rng.integers(n)
        for h in range(H):
            out[d, h] = resid[i]
            i = rng.integers(n) if rng.random() < p else (i + 1) % n
    return out


def _ts_folds(n: int, n_folds: int = 5, min_train: int = 60):
    """확장창 시계열 CV: (train_idx, test_idx)."""
    edges = np.linspace(min_train, n, n_folds + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        yield np.arange(a), np.arange(a, b)


# ----------------------------------------------------------------- models
def cf_bsts(y, X, tr, po, seed=0, niter=4000, burn=1000, level=True, **kw) -> CF:
    r = fit_bsts(y[tr].values, X[tr].values if X.shape[1] else np.zeros((tr.sum(), 0)),
                 X[po].values if X.shape[1] else np.zeros((po.sum(), 0)),
                 niter=niter, burn=burn, seed=seed, names=list(X.columns), level=level, **kw)
    pred = pd.Series(np.median(r.pred_draws, 0), index=y.index[po])
    info = {"inclusion": {k: round(float(v), 3) for k, v in r.inclusion_prob.items()},
            "beta_mean_std": {k: round(float(v), 3) for k, v in zip(r.names, r.beta_draws.mean(0))},
            "s2_level": float(np.mean(r.s2_level)), "s2_obs": float(np.mean(r.s2_obs))}
    fit = pd.Series(r.fit_draws.mean(0), index=y.index[tr])
    return CF(pred, r.pred_draws, info, fit)


def cf_ols(y, X, tr, po, seed=0, hac_lags=5, **kw) -> CF:
    rng = np.random.default_rng(seed)
    Xc = sm.add_constant(X, has_constant="add")
    m = sm.OLS(y[tr], Xc[tr]).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    pred = m.predict(Xc[po])
    H = int(po.sum())
    bs = rng.multivariate_normal(m.params.values, m.cov_params().values, NDRAW)
    mean_paths = bs @ Xc[po].values.T
    eps = stationary_bootstrap_paths(m.resid.values, H, NDRAW, rng)
    return CF(pred, mean_paths + eps, {"params": m.params.round(3).to_dict(), "r2": m.rsquared}, m.fittedvalues)


def cf_ridge(y, X, tr, po, seed=0, kind="ridge", **kw) -> CF:
    from sklearn.linear_model import ElasticNetCV, RidgeCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    cv = TimeSeriesSplit(n_splits=5)
    if kind == "ridge":
        est = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=cv)
    else:
        est = ElasticNetCV(l1_ratio=[.1, .5, .9, 1], alphas=np.logspace(-4, 0, 25), cv=cv, max_iter=20000)
    pipe = make_pipeline(StandardScaler(), est)
    pipe.fit(X[tr], y[tr])
    pred = pd.Series(pipe.predict(X[po]), index=y.index[po])
    oof = _oof_resid(lambda: make_pipeline(StandardScaler(), est.__class__(**est.get_params())), X[tr], y[tr])
    draws = pred.values[None, :] + stationary_bootstrap_paths(oof, po.sum(), NDRAW, rng)
    a = pipe[-1]
    return CF(pred, draws, {"alpha": float(a.alpha_), "coef": dict(zip(X.columns, np.round(a.coef_, 3)))},
              pd.Series(pipe.predict(X[tr]), index=y.index[tr]))


def _oof_resid(make, X, y):
    res = []
    n = len(y)
    for a, b in _ts_folds(n):
        m = make()
        m.fit(X.iloc[a], y.iloc[a])
        res.append(y.iloc[b].values - m.predict(X.iloc[b]))
    return np.concatenate(res)


def cf_gbm(y, X, tr, po, seed=0, lib="lgbm", params=None, **kw) -> CF:
    rng = np.random.default_rng(seed)
    params = params or {}

    def make():
        if lib == "lgbm":
            import lightgbm as lgb
            p = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=15,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                     random_state=seed, verbose=-1)
            p.update(params)
            return lgb.LGBMRegressor(**p)
        import xgboost as xgb
        p = dict(n_estimators=300, learning_rate=0.03, max_depth=2, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=1.0, random_state=seed)
        p.update(params)
        return xgb.XGBRegressor(**p)

    m = make()
    m.fit(X[tr], y[tr])
    pred = pd.Series(m.predict(X[po]), index=y.index[po])
    oof = _oof_resid(make, X[tr], y[tr])
    draws = pred.values[None, :] + stationary_bootstrap_paths(oof, po.sum(), NDRAW, rng)
    return CF(pred, draws, {"oof_rmse": float(np.sqrt(np.mean(oof ** 2)))},
              pd.Series(m.predict(X[tr]), index=y.index[tr]))


def cf_uc(y, X, tr, po, seed=0, **kw) -> CF:
    """frequentist 대응: local level + 회귀 (MLE). BSTS 와 같은 구조, 변수선택 없음."""
    mod = sm.tsa.UnobservedComponents(y[tr].values, level="llevel", exog=X[tr].values if X.shape[1] else None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(disp=False, maxiter=500)
    H = int(po.sum())
    ex = X[po].values if X.shape[1] else None
    sim = res.simulate(H, repetitions=NDRAW, anchor="end", exog=ex,
                       random_state=np.random.default_rng(seed)).reshape(H, NDRAW).T
    fc = res.get_forecast(H, exog=ex).predicted_mean
    return CF(pd.Series(fc, index=y.index[po]), sim,
              {"params": dict(zip(res.param_names, np.round(res.params, 4))), "aic": res.aic},
              pd.Series(res.fittedvalues, index=y.index[tr]))


def cf_sarimax(y, X, tr, po, seed=0, order=None, **kw) -> CF:
    """ARMA(p,q)+exog. order 가 없으면 사전구간 AIC 로 {(1,0),(2,0),(1,1),(2,1)} 중 선택 (d=0: ADF/KPSS 결과 반영)."""
    ex_tr = X[tr].values if X.shape[1] else None
    best = None
    grid = [order] if order else [(1, 0, 0), (2, 0, 0), (1, 0, 1), (2, 0, 1)]
    for o in grid:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = sm.tsa.SARIMAX(y[tr].values, exog=ex_tr, order=o, trend="c").fit(disp=False, maxiter=300)
            if best is None or r.aic < best[1].aic:
                best = (o, r)
        except Exception:  # noqa: BLE001
            continue
    o, res = best
    H = int(po.sum())
    ex = X[po].values if X.shape[1] else None
    sim = res.simulate(H, repetitions=NDRAW, anchor="end", exog=ex,
                       random_state=np.random.default_rng(seed)).reshape(H, NDRAW).T
    fc = res.get_forecast(H, exog=ex).predicted_mean
    return CF(pd.Series(fc, index=y.index[po]), sim, {"order": o, "aic": res.aic},
              pd.Series(res.fittedvalues, index=y.index[tr]))


def _har_feats(yv: np.ndarray, t: int) -> list:
    return [yv[t - 1], yv[t - 5:t].mean(), yv[t - 22:t].mean()]


def cf_harx(y, X, tr, po, seed=0, **kw) -> CF:
    """HAR-RV-X. 사후구간에서는 자기 시차에 반사실 예측값을 넣어 재귀적으로 예측 (실제 사후 y 미사용)."""
    rng = np.random.default_rng(seed)
    yv = y.values.copy()
    tr_idx = np.flatnonzero(tr.values if hasattr(tr, "values") else tr)
    po_idx = np.flatnonzero(po.values if hasattr(po, "values") else po)
    rows = [t for t in tr_idx if t >= 22]
    F = np.array([_har_feats(yv, t) for t in rows])
    Z = np.column_stack([np.ones(len(rows)), F, X.values[rows]]) if X.shape[1] else np.column_stack([np.ones(len(rows)), F])
    b, *_ = np.linalg.lstsq(Z, yv[rows], rcond=None)
    resid = yv[rows] - Z @ b
    fitted = pd.Series(Z @ b, index=y.index[rows])

    def path(eps):
        yy = yv.copy()
        out = np.empty(len(po_idx))
        for h, t in enumerate(po_idx):
            z = np.r_[1.0, _har_feats(yy, t), X.values[t] if X.shape[1] else []]
            yy[t] = z @ b + eps[h]
            out[h] = yy[t]
        return out

    pred = path(np.zeros(len(po_idx)))
    E = stationary_bootstrap_paths(resid, len(po_idx), NDRAW // 4, rng)
    draws = np.array([path(e) for e in E])
    names = ["const", "har_d", "har_w", "har_m"] + list(X.columns)
    persistence = float(b[1] + b[2] + b[3])
    return CF(pd.Series(pred, index=y.index[po_idx]), draws,
              {"coef": dict(zip(names, np.round(b, 3))), "persistence": persistence}, fitted)


def cf_scm(y_wide: pd.DataFrame, treated: str, donors: list[str], tr, po, seed=0, demean=True, **kw) -> CF:
    """합성통제. demean=True 면 Doudchenko-Imbens 식 절편 허용 (가중치 >= 0, 합 1)."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    Y1 = y_wide[treated]
    Y0 = y_wide[donors]
    a, A = Y1[tr].values, Y0[tr].values
    if demean:
        a_c, A_c = a - a.mean(), A - A.mean(0)
    else:
        a_c, A_c = a, A
    J = A.shape[1]
    obj = lambda w: np.mean((a_c - A_c @ w) ** 2)  # noqa: E731
    jac = lambda w: -2 * A_c.T @ (a_c - A_c @ w) / len(a_c)  # noqa: E731
    r = minimize(obj, np.full(J, 1 / J), jac=jac, bounds=[(0, 1)] * J,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP",
                 options={"maxiter": 500, "ftol": 1e-10})
    w = r.x
    icpt = (a.mean() - A.mean(0) @ w) if demean else 0.0
    synth = pd.Series(Y0.values @ w + icpt, index=y_wide.index)
    resid = (Y1 - synth)[tr].values
    pred = synth[po]
    draws = pred.values[None, :] + stationary_bootstrap_paths(resid, po.sum(), NDRAW, rng)
    top = pd.Series(w, index=donors).sort_values(ascending=False).head(8).round(3).to_dict()
    return CF(pred, draws, {"top_weights": top, "n_pos_w": int((w > 1e-3).sum()), "intercept": float(icpt)},
              synth[tr])


# ----------------------------------------------------------------- 효과 요약
def summarize(y_post: pd.Series, cf: CF, windows: dict[str, pd.Series]) -> list[dict]:
    """창(window)별 평균 효과 = mean(y - 반사실). log 단위 → % 는 exp(τ)-1."""
    rows = []
    for name, mask in windows.items():
        m = mask.reindex(y_post.index).fillna(False).values
        if m.sum() == 0:
            continue
        eff_draws = (y_post.values[m][None, :] - cf.draws[:, m]).mean(1)
        point = float((y_post.values[m] - cf.pred.values[m]).mean())
        lo, hi = np.quantile(eff_draws, [0.025, 0.975])
        p_lt0 = float((eff_draws < 0).mean())
        rows.append({"window": name, "n_days": int(m.sum()), "effect_log": point,
                     "ci_lo": float(lo), "ci_hi": float(hi), "effect_pct": float(np.expm1(point) * 100),
                     "ci_lo_pct": float(np.expm1(lo) * 100), "ci_hi_pct": float(np.expm1(hi) * 100),
                     "p_two_sided": float(2 * min(p_lt0, 1 - p_lt0)), "post_prob_negative": p_lt0,
                     "sd_effect": float(eff_draws.std())})
    return rows


def pre_fit_metrics(y_tr: pd.Series, fitted: pd.Series | None) -> dict:
    if fitted is None:
        return {}
    e = (y_tr.reindex(fitted.index) - fitted).dropna()
    return {"pre_rmse": float(np.sqrt(np.mean(e ** 2))), "pre_mae": float(np.mean(np.abs(e)))}
