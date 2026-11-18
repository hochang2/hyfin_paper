"""추정 핵심: 고정효과 흡수(교대투영) + 여러 표준오차 + wild cluster bootstrap + 무작위화 추론.

FE 를 FWL 로 흡수한 뒤 잔차화된 y, X 로 OLS. 모든 SE 는 같은 잔차화 행렬에서 계산 → 표준오차 비교가 공정.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def demean(M: np.ndarray, fes: list[np.ndarray], tol=1e-10, maxit=2000) -> np.ndarray:
    """여러 고정효과를 교대투영으로 흡수. M: (n,k). fes: 정수 코드 배열들."""
    M = np.asarray(M, float).copy()
    if M.ndim == 1:
        M = M[:, None]
    if not fes:
        return M
    cnts = [np.bincount(f) for f in fes]
    for _ in range(maxit):
        old = M.copy()
        for f, c in zip(fes, cnts):
            for j in range(M.shape[1]):
                s = np.bincount(f, weights=M[:, j], minlength=len(c))
                M[:, j] -= (s / np.maximum(c, 1))[f]
        if np.max(np.abs(M - old)) < tol:
            break
    return M


def codes(s: pd.Series) -> np.ndarray:
    return pd.factorize(s)[0]


class FE:
    """y ~ X | FE. X 는 처치 변수들 (DataFrame 열 이름 리스트)."""

    def __init__(self, df: pd.DataFrame, y: str, X: list[str], fe: list[str], weights=None):
        d = df.dropna(subset=[y] + X).reset_index(drop=True)
        self.d = d
        self.names = X
        self.fes = [codes(d[f]) for f in fe]
        Z = demean(np.column_stack([d[y].values] + [d[x].values for x in X]), self.fes)
        self.yt, self.Xt = Z[:, 0], Z[:, 1:]
        XtX = self.Xt.T @ self.Xt
        self.XtX_inv = np.linalg.pinv(XtX)
        self.b = self.XtX_inv @ self.Xt.T @ self.yt
        self.e = self.yt - self.Xt @ self.b
        self.n = len(d)
        self.k_fe = sum(len(np.unique(f)) for f in self.fes) - max(len(self.fes) - 1, 0)
        self.df_resid = self.n - self.k_fe - len(X)
        # within R²
        self.r2_within = 1 - (self.e @ self.e) / (self.yt @ self.yt)

    # ---- 분산 추정
    def _meat_cluster(self, g):
        u = self.Xt * self.e[:, None]
        S = pd.DataFrame(u).groupby(g).sum().values
        return S.T @ S, len(np.unique(g))

    def vcov(self, kind="cr1", cluster=None, cluster2=None, lags=None):
        k = self.Xt.shape[1]
        A = self.XtX_inv
        if kind == "iid":
            s2 = self.e @ self.e / self.df_resid
            return s2 * A
        if kind == "hc1":
            u = self.Xt * self.e[:, None]
            return self.n / self.df_resid * A @ (u.T @ u) @ A
        if kind == "cr1":
            g = codes(self.d[cluster])
            meat, G = self._meat_cluster(g)
            c = G / (G - 1) * (self.n - 1) / (self.n - k)
            return c * A @ meat @ A
        if kind == "twoway":
            V1 = self.vcov("cr1", cluster)
            V2 = self.vcov("cr1", cluster2)
            g12 = codes(self.d[cluster].astype(str) + "|" + self.d[cluster2].astype(str))
            meat, G = self._meat_cluster(g12)
            c = G / (G - 1) * (self.n - 1) / (self.n - k)
            V12 = c * A @ meat @ A
            V = V1 + V2 - V12
            w, Q = np.linalg.eigh(V)
            return Q @ np.diag(np.maximum(w, 0)) @ Q.T
        if kind == "dk":   # Driscoll–Kraay: 날짜별 합 → Newey–West
            t = codes(self.d[cluster])           # cluster=날짜 열
            u = self.Xt * self.e[:, None]
            h = pd.DataFrame(u).groupby(t).sum().sort_index().values
            T = h.shape[0]
            L = lags if lags is not None else int(np.floor(4 * (T / 100) ** (2 / 9)))
            S = h.T @ h
            for l in range(1, L + 1):
                w = 1 - l / (L + 1)
                G_ = h[l:].T @ h[:-l]
                S += w * (G_ + G_.T)
            return A @ S @ A
        raise ValueError(kind)

    def summary(self, kind="cr1", cluster="code", cluster2=None, lags=None, dof=None):
        V = self.vcov(kind, cluster, cluster2, lags)
        se = np.sqrt(np.diag(V))
        t = self.b / se
        if dof is None:
            if kind in ("cr1", "twoway"):
                G = self.d[cluster].nunique() if kind == "cr1" else min(self.d[cluster].nunique(),
                                                                         self.d[cluster2].nunique())
                dof = G - 1
            else:
                dof = self.df_resid
        p = 2 * stats.t.sf(np.abs(t), dof)
        q = stats.t.ppf(0.975, dof)
        return pd.DataFrame({"coef": self.b, "se": se, "t": t, "p": p, "ci_low": self.b - q * se,
                             "ci_high": self.b + q * se, "dof": dof}, index=self.names)

    def lincomb(self, w, kind="cr1", cluster="code", cluster2=None):
        w = np.asarray(w, float)
        V = self.vcov(kind, cluster, cluster2)
        est = w @ self.b
        se = float(np.sqrt(w @ V @ w))
        G = self.d[cluster].nunique()
        dof = G - 1 if kind == "cr1" else self.df_resid
        p = 2 * stats.t.sf(abs(est / se), dof)
        q = stats.t.ppf(0.975, dof)
        return dict(coef=est, se=se, p=p, ci_low=est - q * se, ci_high=est + q * se)

    # ---- wild cluster bootstrap (WCR, Webb 6점, 귀무 제약)
    def wild(self, j=0, cluster="code", reps=9999, seed=0, w=None):
        """H0: w'β = 0 (기본: j번째 계수). 귀무 제약 잔차 → Webb 가중 → t* 분포. p 는 대칭 양측."""
        rng = np.random.default_rng(seed)
        k = self.Xt.shape[1]
        if w is None:
            w = np.zeros(k)
            w[j] = 1.0
        w = np.asarray(w, float)
        g = codes(self.d[cluster])
        G = g.max() + 1
        # 제약 추정: β 를 w'β=0 하에서 (제약 OLS)
        A = self.XtX_inv
        b = self.b
        b_r = b - A @ w * (w @ b) / (w @ A @ w)
        e_r = self.yt - self.Xt @ b_r
        fit_r = self.Xt @ b_r
        webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])

        def tstat(y):
            bb = A @ (self.Xt.T @ y)
            e = y - self.Xt @ bb
            u = self.Xt * e[:, None]
            S = np.zeros((G, k))
            np.add.at(S, g, u)
            c = G / (G - 1) * (self.n - 1) / (self.n - k)
            V = c * A @ (S.T @ S) @ A
            return (w @ bb) / np.sqrt(w @ V @ w)

        t0 = tstat(self.yt)
        ts = np.empty(reps)
        for r in range(reps):
            v = webb[rng.integers(0, 6, G)]
            ts[r] = tstat(fit_r + e_r * v[g])
        p = (1 + np.sum(np.abs(ts) >= abs(t0))) / (reps + 1)
        # CI: t* 분위수 이용 (studentized), 근사
        se = (w @ b) / t0 if t0 != 0 else np.nan
        qlo, qhi = np.quantile(np.abs(ts), 0.95), np.quantile(np.abs(ts), 0.95)
        return dict(t=t0, p=p, ci_low=w @ b - qhi * abs(se), ci_high=w @ b + qlo * abs(se), reps=reps, G=G)


def holm(p: dict) -> dict:
    items = sorted(p.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        out[k] = run
    return out


def bh(p: dict) -> dict:
    keys = list(p)
    v = np.array([p[k] for k in keys], float)
    m = len(v)
    o = np.argsort(v)
    q = np.empty(m)
    prev = 1.0
    for rank in range(m, 0, -1):
        i = o[rank - 1]
        prev = min(prev, v[i] * m / rank)
        q[i] = prev
    return dict(zip(keys, np.minimum(q, 1)))
