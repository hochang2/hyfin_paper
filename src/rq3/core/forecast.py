"""RQ3 표본외 예측과 검정.

- `oos()`: 예측 원점 t 마다 다시 추정해서 y_next(t) = ln RV_{t+1} 을 예측한다.
  원점 t 에서 알 수 있는 학습 행은 y_next 가 t 이전에 실현된 행, 즉 index < t 인 행뿐이다.
- `clark_west()`: 포함 관계 모형(작은 모형 ⊂ 큰 모형)의 MSPE-adjusted 검정, 한쪽 (큰 모형이 더 좋다).
- `holm()`: Holm 보정.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

MODELS = {
    "A": ["har_d", "har_w", "har_m", "D"],
    "B": ["har_d", "har_w", "har_m", "D", "R"],
    "C": ["har_d", "har_w", "har_m", "D", "R", "X"],
}


def _fit_predict(Xtr: np.ndarray, ytr: np.ndarray, xnew: np.ndarray) -> tuple[float, np.ndarray, float]:
    """상수항 포함 OLS. 학습 구간에서 분산이 0 인 열(예: 상장 후 표본만 쓸 때의 D)은 뺀다.
    반환: 예측값, 계수(뺀 열은 NaN), 학습 잔차 분산(자유도 보정)."""
    keep = Xtr.std(axis=0) > 0
    Z = np.column_stack([np.ones(len(Xtr)), Xtr[:, keep]])
    b, *_ = np.linalg.lstsq(Z, ytr, rcond=None)
    full = np.full(Xtr.shape[1], np.nan)
    full[keep] = b[1:]
    resid = ytr - Z @ b
    s2 = float(resid @ resid / max(len(ytr) - Z.shape[1], 1))
    return float(b[0] + xnew[keep] @ b[1:]), full, s2


def oos(d: pd.DataFrame, cols: list[str], first_origin: pd.Timestamp, window: str = "expanding",
        rolling: int | None = None, start: pd.Timestamp | None = None, target: str = "y_next",
        horizon: int = 1) -> pd.DataFrame:
    """예측 원점별 예측값과 계수.

    반환: index = 원점 t, 열 = y (실현값), yhat, s2, n_train, b_<변수>
    - start: 학습 표본 시작일 (None = 설계 행렬 처음부터)
    - horizon h: 대상이 t+1..t+h 에 걸치면 원점 t 에서 실현된 행은 index ≤ t−h 뿐이다
    """
    d = d.dropna(subset=[target])
    if start is not None:
        d = d[d.index >= start]
    X, y, idx = d[cols].to_numpy(float), d[target].to_numpy(float), d.index
    rows = []
    for i in np.where(idx >= first_origin)[0]:
        hi = i - (horizon - 1)
        lo = 0 if window == "expanding" else max(0, hi - rolling)
        yhat, b, s2 = _fit_predict(X[lo:hi], y[lo:hi], X[i])
        rows.append({"date": idx[i], "y": y[i], "yhat": yhat, "s2": s2, "n_train": hi - lo,
                     **{f"b_{c}": v for c, v in zip(cols, b)}})
    return pd.DataFrame(rows).set_index("date")


def nw_lag(n: int) -> int:
    return int(math.floor(4 * (n / 100) ** (2 / 9)))


def nw_se_mean(f: np.ndarray, lag: int) -> float:
    """평균의 Newey-West 표준오차 (Bartlett)."""
    n = len(f)
    u = f - f.mean()
    s = u @ u / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    return math.sqrt(max(s, 0.0) / n)


def clark_west(y: np.ndarray, f_small: np.ndarray, f_big: np.ndarray, lag: int | None = None) -> dict:
    """H0: 두 모형의 MSPE 같음 (큰 모형의 추가 변수 계수 0). H1: 큰 모형이 더 좋다 (한쪽)."""
    y, f1, f2 = map(np.asarray, (y, f_small, f_big))
    e1, e2 = y - f1, y - f2
    adj = e1 ** 2 - (e2 ** 2 - (f1 - f2) ** 2)
    n = len(adj)
    lag = nw_lag(n) if lag is None else lag
    se = nw_se_mean(adj, lag)
    t = adj.mean() / se if se > 0 else np.nan
    mse1, mse2 = np.mean(e1 ** 2), np.mean(e2 ** 2)
    return {"n": n, "nw_lag": lag, "mse_small": mse1, "mse_big": mse2, "r2_oos": 1 - mse2 / mse1,
            "cw_mean": adj.mean(), "cw_t": t, "p_one_sided": float(1 - stats.norm.cdf(t))}


def dm_test(loss_small: np.ndarray, loss_big: np.ndarray, lag: int | None = None) -> dict:
    """Diebold-Mariano (한쪽: 큰 모형 손실이 작다). 포함 관계에서는 보수적이다."""
    dlt = np.asarray(loss_small) - np.asarray(loss_big)
    lag = nw_lag(len(dlt)) if lag is None else lag
    se = nw_se_mean(dlt, lag)
    t = dlt.mean() / se if se > 0 else np.nan
    return {"dm_mean": dlt.mean(), "dm_t": t, "p_one_sided": float(1 - stats.norm.cdf(t))}


def qlike(rv: np.ndarray, rv_hat: np.ndarray) -> np.ndarray:
    """QLIKE = RV/ĥ − ln(RV/ĥ) − 1 (수준 기준)."""
    r = np.asarray(rv) / np.asarray(rv_hat)
    return r - np.log(r) - 1


def block_shuffle(s: pd.Series, groups: np.ndarray, block: int, rng: np.random.Generator) -> pd.Series:
    """groups(구간) 안에서만 연속 block 일 단위 블록의 순서를 섞는다. 구간 경계를 넘는 블록은 없다."""
    out = s.copy()
    for g in pd.unique(groups):
        pos = np.where(groups == g)[0]
        blocks = [pos[k:k + block] for k in range(0, len(pos), block)]
        order = rng.permutation(len(blocks))
        out.iloc[pos] = np.concatenate([s.iloc[blocks[j]].to_numpy() for j in order])
    return out


def ols_hac(y: np.ndarray, X: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """상수항 포함 OLS 계수와 Newey-West 표준오차."""
    Z = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    u = y - Z @ b
    S = (Z * u[:, None]).T @ (Z * u[:, None])
    for k in range(1, lag + 1):
        G = (Z[k:] * u[k:, None]).T @ (Z[:-k] * u[:-k, None])
        S += (1 - k / (lag + 1)) * (G + G.T)
    inv = np.linalg.inv(Z.T @ Z)
    return b, np.sqrt(np.diag(inv @ S @ inv))


def vif(X: pd.DataFrame) -> pd.Series:
    out = {}
    for c in X.columns:
        others = X.drop(columns=c)
        Z = np.column_stack([np.ones(len(X)), others.to_numpy(float)])
        b, *_ = np.linalg.lstsq(Z, X[c].to_numpy(float), rcond=None)
        r = X[c].to_numpy(float) - Z @ b
        r2 = 1 - r @ r / ((X[c] - X[c].mean()) ** 2).sum()
        out[c] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.Series(out)


def holm(p: dict[str, float]) -> dict[str, float]:
    keys = sorted(p, key=p.get)
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out
