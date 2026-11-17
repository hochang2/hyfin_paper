"""후속 실험 공용: 점예측 적합기, 창평균 요약, 시계열 의존을 고려한 귀무분포(추론) 변형, 캐시.

적합기는 1차 연구(core.models, ml_extension.models)와 같은 기본값을 쓰되, bootstrap 경로(draws)를 만들지 않아
수만 건을 돌릴 수 있게 한 '점예측 전용' 판이다. 사후 y 는 입력으로 받지 않는다 (harx 는 반사실 재귀).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.rq2.ml_followup.panel import CACHE, FCFG, WIN

MP = FCFG["models"]
ALPHA = 0.05


# ================================================================ 적합기
def make_est(kind: str, seed: int = 0, params: dict | None = None):
    params = dict(params or {})
    if kind in ("lgbm", "lgbm_deep"):
        import lightgbm as lgb
        p = dict(MP[kind], random_state=seed, verbose=-1, n_jobs=1)
        p.update(params)
        return lgb.LGBMRegressor(**p)
    if kind == "xgb":
        import xgboost as xgb
        p = dict(MP["xgb"], random_state=seed, n_jobs=1)
        p.update(params)
        return xgb.XGBRegressor(**p)
    if kind in ("rf", "rf_panel"):
        from sklearn.ensemble import RandomForestRegressor
        p = dict(MP[kind], random_state=seed, n_jobs=1)
        p.update(params)
        return RandomForestRegressor(**p)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "ridge":
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import TimeSeriesSplit
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25), cv=TimeSeriesSplit(5)))
    if kind == "enet":
        from sklearn.linear_model import ElasticNetCV
        from sklearn.model_selection import TimeSeriesSplit
        return make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.1, .5, .9, 1], alphas=np.logspace(-4, 0, 25),
                                                            cv=TimeSeriesSplit(5), max_iter=20000))
    if kind == "ridge_fix":   # 패널(행이 많음)용 고정 alpha
        from sklearn.linear_model import Ridge
        return make_pipeline(StandardScaler(), Ridge(alpha=params.get("alpha", 1.0)))
    raise ValueError(kind)


def _har(yv, t):
    return [yv[t - 1], yv[t - 5:t].mean(), yv[t - 22:t].mean()]


def fit_predict(kind: str, y: np.ndarray, X: np.ndarray, n_tr: int, seed: int = 0, params: dict | None = None,
                native: bool = False) -> dict:
    """y·X: 학습 n_tr 행 + 사후 H 행 (사후 y 는 쓰지 않는다). 반환 {'pred': (H,), 'fit': (n_tr,) 또는 None,
    'draws': (nd, H) 또는 None}. native=True 면 bsts 사후분포·ols 계수+잔차 SB 경로를 남긴다."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    yt, Xt, Xp = y[:n_tr], X[:n_tr], X[n_tr:]
    H = len(Xp)
    out = {"draws": None, "fit": None}
    if kind == "naive_mean":
        out["pred"] = np.full(H, yt[-60:].mean())
    elif kind == "did_gap":
        g = (yt - Xt[:, 0])[-60:].mean()          # X 첫 열 = don_mean
        out["pred"] = Xp[:, 0] + g
    elif kind == "ols":
        Z = np.column_stack([np.ones(n_tr), Xt])
        b, *_ = np.linalg.lstsq(Z, yt, rcond=None)
        out["pred"] = np.column_stack([np.ones(H), Xp]) @ b
        out["fit"] = Z @ b
        if native:
            out["draws"] = _ols_draws(Z, yt, b, np.column_stack([np.ones(H), Xp]), seed)
    elif kind == "harx":
        rows = np.arange(22, n_tr)
        F = np.array([_har(yt, t) for t in rows])
        Z = np.column_stack([np.ones(len(rows)), F, Xt[rows]])
        b, *_ = np.linalg.lstsq(Z, yt[rows], rcond=None)
        yy = np.r_[yt, np.zeros(H)]
        for h in range(H):
            t = n_tr + h
            yy[t] = np.r_[1.0, _har(yy, t), Xp[h]] @ b
        out["pred"] = yy[n_tr:]
    elif kind in ("bsts", "bsts_light"):
        from src.rq2.core.bsts import fit_bsts
        p = dict(MP["bsts_light"])
        p.update(params or {})
        r = fit_bsts(yt, Xt, Xp, seed=seed, **p)
        out["pred"] = np.median(r.pred_draws, 0)
        out["fit"] = r.fit_draws.mean(0)
        if native:
            out["draws"] = r.pred_draws
    else:
        m = make_est(kind, seed, params)
        m.fit(Xt, yt)
        out["pred"] = m.predict(Xp)
        out["fit"] = m.predict(Xt)
    return out


def _ols_draws(Z, yt, b, Zp, seed, nd=400, lags=5):
    """core.cf_ols 와 같은 구성: HAC 계수 표본 + 잔차 stationary bootstrap 경로 (점예측 전용판은 400 경로)."""
    import statsmodels.api as sm
    rng = np.random.default_rng(seed)
    m = sm.OLS(yt, Z).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    bs = rng.multivariate_normal(m.params, m.cov_params(), nd)
    return bs @ Zp.T + sb_paths(m.resid, Zp.shape[0], nd, rng)


def sb_paths(resid, H, nd, rng, mean_block=10.0):
    """벡터화한 Politis-Romano stationary bootstrap (core 와 같은 분포, 루프만 벡터화)."""
    r = np.asarray(resid) - np.mean(resid)
    n = len(r)
    idx = np.empty((nd, H), int)
    idx[:, 0] = rng.integers(n, size=nd)
    for h in range(1, H):
        jump = rng.random(nd) < 1.0 / mean_block
        idx[:, h] = np.where(jump, rng.integers(n, size=nd), (idx[:, h - 1] + 1) % n)
    return r[idx]


# ================================================================ 창평균
def window_rows(y_post: np.ndarray, pred: np.ndarray, draws: np.ndarray | None = None,
                windows: dict | None = None) -> list[dict]:
    windows = windows or WIN
    H = len(pred)
    rows = []
    for w, (a, b) in windows.items():
        if b > H:
            continue
        r = {"window": w, "a": a, "b": b, "ybar": float(np.mean(y_post[a:b])), "predbar": float(np.mean(pred[a:b]))}
        if draws is not None:
            wm = draws[:, a:b].mean(1)
            r["cf_q025"], r["cf_q975"] = np.quantile(wm, [0.025, 0.975])
        rows.append(r)
    return rows


# ================================================================ 캐시
def cpath(exp: str, name: str) -> Path:
    p = CACHE / exp / f"{name}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def cached(exp: str, name: str, fn, force: bool = False) -> pd.DataFrame:
    p = cpath(exp, name)
    if p.exists() and not force:
        return pd.read_parquet(p)
    t = time.time()
    df = fn()
    df.to_parquet(p)
    log_event(exp, name, rows=len(df), sec=round(time.time() - t, 1))
    return df


def log_event(exp: str, step: str, **kw):
    from src.rq2.ml_followup.panel import RES
    f = RES / "experiment_log" / "runs.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    with f.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "exp": exp, "step": step, **kw},
                            ensure_ascii=False, default=str) + "\n")


def save_table(df: pd.DataFrame, name: str, round_: int = 4):
    from src.rq2.ml_followup.panel import RES
    p = RES / "tables" / f"{name}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.round(round_).to_csv(p, index=False, encoding="utf-8-sig")
    return p


# ================================================================ 시계열 의존을 고려한 귀무분포 (실험 5)
def n_eff_greedy(i0s, L) -> int:
    """1차 확장 synthetic_effect.n_eff 와 같은 정의 (앞에서부터 탐욕)."""
    s = np.sort(np.asarray(i0s))
    n, last = 0, -np.inf
    for v in s:
        if v - last >= L:
            n, last = n + 1, v
    return n


def nonoverlap_subset(i0s, L) -> np.ndarray:
    """검정 시점에 가장 가까운(최근) 창부터 거꾸로 서로 겹치지 않는 창만 고른 위치 (bool)."""
    i0s = np.asarray(i0s)
    order = np.argsort(-i0s)
    keep = np.zeros(len(i0s), bool)
    last = np.inf
    for k in order:
        if last - i0s[k] >= L:
            keep[k] = True
            last = i0s[k]
    return keep


def nw_lrv(e, bw):
    e = np.asarray(e) - np.mean(e)
    n = len(e)
    v = e @ e / n
    for k in range(1, min(int(bw), n - 1) + 1):
        v += 2 * (1 - k / (bw + 1)) * (e[k:] @ e[:-k]) / n
    return max(v, 1e-12)


def _block_boot_stat(e, bl, B, rng, stationary):
    """블록 bootstrap 으로 (새 관측 − 표본평균) 분포를 만든다. 반복 수 B 는 분포 근사용일 뿐
    독립 관측치 수가 아니다 (유효 정보량은 여전히 비겹침 창 수 수준)."""
    n = len(e)
    bl = max(int(bl), 1)
    if stationary:
        idx = np.empty((B, n), int)
        idx[:, 0] = rng.integers(n, size=B)
        for t in range(1, n):
            jump = rng.random(B) < 1.0 / bl
            idx[:, t] = np.where(jump, rng.integers(n, size=B), (idx[:, t - 1] + 1) % n)
    else:
        nb = int(np.ceil(n / bl))
        st = rng.integers(0, max(n - bl + 1, 1), size=(B, nb))
        idx = (st[:, :, None] + np.arange(bl)[None, None, :]).reshape(B, -1)[:, :n]
        idx = np.minimum(idx, n - 1)
    s = e[idx]
    return s[np.arange(B), rng.integers(n, size=B)] - s.mean(1)


METHODS = ["naive", "neff", "strict", "hac", "mbb", "sb"]


def null_test(e: np.ndarray, i0s: np.ndarray, L: int, step: int, method: str, min_eff: int = 3,
              B: int = 999, rng=None) -> tuple[float, float, int]:
    """가짜 개입일 오차 e (i0s 순서) 로 만든 귀무 예측구간. 반환 (center m0, crit, n_used). crit=NaN 이면 검정 불가.
    naive : 모든 창을 독립으로 (1차 확장 FAIL-TIMENULL-OVERLAP 이전 방식)
    neff  : m0·s0 는 모든 창, df·sqrt(1+1/n) 은 비겹침 창 수 (1차 확장 현행 방식)
    strict: 최근부터 거꾸로 고른 비겹침 창만으로 m0·s0·df
    hac   : m0 = 평균, Var(새 오차 − m0) = s0² + LRV_NW/n (Bartlett, bw = L/step), 정규 임계값
    mbb/sb: 블록 길이 L/step 의 moving / stationary block bootstrap 예측구간
    """
    ok = np.isfinite(e)
    e, i0s = e[ok], i0s[ok]
    n = len(e)
    ne = n_eff_greedy(i0s, L)
    if method == "naive":
        if n < 3:
            return np.nan, np.nan, n
        s0 = e.std(ddof=1)
        return e.mean(), stats.t.ppf(1 - ALPHA / 2, n - 1) * s0 * np.sqrt(1 + 1 / n), n
    if ne < min_eff:
        return np.nan, np.nan, ne
    if method == "neff":
        s0 = e.std(ddof=1)
        return e.mean(), stats.t.ppf(1 - ALPHA / 2, ne - 1) * s0 * np.sqrt(1 + 1 / ne), ne
    if method == "strict":
        k = nonoverlap_subset(i0s, L)
        es = e[k]
        m = len(es)
        if m < min_eff:
            return np.nan, np.nan, m
        return es.mean(), stats.t.ppf(1 - ALPHA / 2, m - 1) * es.std(ddof=1) * np.sqrt(1 + 1 / m), m
    bw = max(int(np.ceil(L / step)), 1)
    if method == "hac":
        s0 = e.std(ddof=1)
        return e.mean(), stats.norm.ppf(1 - ALPHA / 2) * np.sqrt(s0 ** 2 + nw_lrv(e, bw) / n), ne
    rng = rng or np.random.default_rng(0)
    d = _block_boot_stat(e, bw, B, rng, stationary=(method == "sb"))
    return e.mean(), float(np.quantile(np.abs(d), 1 - ALPHA)), ne


def t_crit(n):
    return stats.t.ppf(1 - ALPHA / 2, np.maximum(np.asarray(n, float) - 1, 1)) * np.sqrt(1 + 1 / np.maximum(n, 1))


def null_p(est: float, e: np.ndarray, i0s: np.ndarray, L: int, step: int, method: str, min_eff: int = 3,
           B: int = 4999, rng=None) -> dict:
    """실제 효과 추정치 est 의 p 값 (null_test 와 같은 귀무분포). bootstrap p 는 반복 수와 무관하게
    유효 정보량이 비겹침 창 수에 묶여 있다는 점을 해석에서 유지한다."""
    m0, crit, n = null_test(e, i0s, L, step, method, min_eff, B=B, rng=rng)
    out = {"m0": m0, "crit": crit, "n_used": n, "p": np.nan}
    if not np.isfinite(crit):
        return out
    ok = np.isfinite(e)
    e, i0s = e[ok], i0s[ok]
    d = abs(est - m0)
    if method in ("naive", "neff", "strict"):
        if method == "strict":
            k = nonoverlap_subset(i0s, L)
            s, nn = e[k].std(ddof=1), k.sum()
        else:
            s, nn = e.std(ddof=1), (len(e) if method == "naive" else n)
        out["p"] = float(2 * stats.t.sf(d / (s * np.sqrt(1 + 1 / nn)), nn - 1))
    elif method == "hac":
        bw = max(int(np.ceil(L / step)), 1)
        out["p"] = float(2 * stats.norm.sf(d / np.sqrt(e.std(ddof=1) ** 2 + nw_lrv(e, bw) / len(e))))
    else:
        bw = max(int(np.ceil(L / step)), 1)
        dd = _block_boot_stat(e, bw, B, rng or np.random.default_rng(0), stationary=(method == "sb"))
        out["p"] = float((np.sum(np.abs(dd) >= d) + 1) / (B + 1))
    return out
