"""CausalImpact 방식 BSTS (local level + spike-and-slab 회귀), Gibbs sampler.

R `CausalImpact`/`bsts` 기본 설정을 따른다 (Brodersen et al. 2015):
- y, X 를 사전구간 평균·표준편차로 표준화
- 상태: mu_t = mu_{t-1} + eta_t,  eta ~ N(0, s2_level)
- 관측: y_t = mu_t + x_t'beta + eps_t,  eps ~ N(0, s2_obs)
- s2_level 사전분포: sd 추정치 0.01 (표준화 y 기준), 표본크기 32 의 역감마
- beta: spike-and-slab. 포함확률 = expected_model_size / k, slab = Zellner 형
  Omega^{-1} = kappa/n * (w X'X + (1-w) diag(X'X)),  kappa=1, w=0.5
- gamma 는 beta·s2_obs 를 적분한 주변우도로 하나씩 갱신 (collapsed Gibbs)
- mu 는 forward-filtering backward-sampling

사후 예측: 각 draw 에서 mu 를 사후구간으로 랜덤워크 전진 + x'beta + eps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BSTSResult:
    pred_draws: np.ndarray      # (ndraw, T_post)  원래 단위의 반사실 y 예측
    fit_draws: np.ndarray       # (ndraw, T_pre)   사전구간 적합값 (mu + x'b, 원 단위)
    beta_draws: np.ndarray      # (ndraw, k)       표준화 단위
    gamma_draws: np.ndarray     # (ndraw, k)
    s2_obs: np.ndarray
    s2_level: np.ndarray
    names: list

    @property
    def inclusion_prob(self) -> dict:
        return dict(zip(self.names, self.gamma_draws.mean(0)))


def _ffbs_local_level(y, s2e, s2h, rng, a0=0.0, p0=1e6):
    """local level 모형에서 mu_{1:T} 하나를 샘플."""
    T = len(y)
    a = np.empty(T); p = np.empty(T)
    at, pt = a0, p0
    for t in range(T):
        pt_pred = pt + s2h if t > 0 else pt
        k = pt_pred / (pt_pred + s2e)
        at = at + k * (y[t] - at)
        pt = (1 - k) * pt_pred
        a[t], p[t] = at, pt
    mu = np.empty(T)
    mu[-1] = a[-1] + np.sqrt(p[-1]) * rng.standard_normal()
    for t in range(T - 2, -1, -1):
        g = p[t] / (p[t] + s2h)
        m = a[t] + g * (mu[t + 1] - a[t])
        v = p[t] * (1 - g)
        mu[t] = m + np.sqrt(max(v, 1e-12)) * rng.standard_normal()
    return mu


def fit_bsts(y_pre, X_pre, X_post, niter=4000, burn=1000, seed=0, expected_model_size=3,
             prior_level_sd=0.01, prior_level_df=32, names=None, level=True) -> BSTSResult:
    rng = np.random.default_rng(seed)
    y_pre = np.asarray(y_pre, float)
    X_pre = np.asarray(X_pre, float).reshape(len(y_pre), -1)
    X_post = np.asarray(X_post, float)
    X_post = X_post.reshape(X_post.shape[0] if X_post.ndim == 2 else -1, X_pre.shape[1])   # k=0 (C0) 도 허용
    T, k = X_pre.shape
    ym, ys = y_pre.mean(), y_pre.std()
    xm, xs = X_pre.mean(0), X_pre.std(0)
    valid = xs > 1e-12                 # 학습구간에서 상수인 열(예: 짧은 구간의 gap_day)은 선택 대상에서 뺀다
    xs = np.where(valid, xs, 1.0)
    y = (y_pre - ym) / ys
    X = (X_pre - xm) / xs
    Xp = (X_post - xm) / xs

    # 사전분포
    a_e, b_e = 0.01 / 2, 0.01 * (1 - 0.8) / 2   # sigma_obs: R^2 기대 0.8, 표본크기 0.01 (bsts 기본과 같은 약한 사전)
    a_h, b_h = prior_level_df / 2, prior_level_df * prior_level_sd ** 2 / 2
    XtX = X.T @ X
    Om = (0.5 * XtX + 0.5 * np.diag(np.diag(XtX))) / T   # kappa=1
    Om[np.ix_(~valid, ~valid)] += np.eye(int((~valid).sum()))   # 상수열: 계속 제외되므로 값은 무관, 특이행렬만 방지
    pi = min(expected_model_size / max(k, 1), 1.0) if k else 0.0

    gamma = valid.copy()
    beta = np.zeros(k)
    s2e, s2h = 0.5, prior_level_sd ** 2
    mu = np.zeros(T)
    keep = niter - burn
    B = np.zeros((keep, k)); G = np.zeros((keep, k), bool)
    S2E = np.zeros(keep); S2H = np.zeros(keep)
    MU_last = np.zeros(keep); FIT = np.zeros((keep, T))

    def logml(ystar, g):
        idx = np.flatnonzero(g)
        n = len(ystar)
        if len(idx) == 0:
            ssr = ystar @ ystar
            return -(a_e + n / 2) * np.log(b_e + ssr / 2), None
        Xg = X[:, idx]
        L0 = Om[np.ix_(idx, idx)]
        Ln = Xg.T @ Xg + L0
        rhs = Xg.T @ ystar
        cL = np.linalg.cholesky(Ln)
        m = np.linalg.solve(cL.T, np.linalg.solve(cL, rhs))
        ssr = ystar @ ystar - m @ Ln @ m
        ld0 = np.linalg.slogdet(L0)[1]
        ldn = 2 * np.log(np.diag(cL)).sum()
        return 0.5 * ld0 - 0.5 * ldn - (a_e + n / 2) * np.log(b_e + max(ssr, 1e-12) / 2), (idx, Ln, m, ssr)

    for it in range(niter):
        # 1) 상태
        if level:
            mu = _ffbs_local_level(y - X @ beta if k else y, s2e, s2h, rng)
        else:
            mu = np.full(T, (y - (X @ beta if k else 0)).mean())
        ystar = y - mu
        # 2) gamma (collapsed)
        if k:
            for j in rng.permutation(np.flatnonzero(valid)):
                g1 = gamma.copy(); g1[j] = True
                g0 = gamma.copy(); g0[j] = False
                l1, _ = logml(ystar, g1); l0, _ = logml(ystar, g0)
                lp1 = l1 + np.log(pi + 1e-300); lp0 = l0 + np.log(1 - pi + 1e-300)
                pr = 1.0 / (1.0 + np.exp(np.clip(lp0 - lp1, -700, 700)))
                gamma[j] = rng.random() < pr
            _, info = logml(ystar, gamma)
        else:
            info = None
        # 3) s2e, beta
        if info is None:
            ssr = ystar @ ystar
            s2e = 1.0 / rng.gamma(a_e + T / 2, 1.0 / (b_e + ssr / 2))
            beta = np.zeros(k)
        else:
            idx, Ln, m, ssr = info
            s2e = 1.0 / rng.gamma(a_e + T / 2, 1.0 / (b_e + ssr / 2))
            cov = s2e * np.linalg.inv(Ln)
            beta = np.zeros(k)
            beta[idx] = rng.multivariate_normal(m, cov)
        # 4) s2h
        if level:
            d = np.diff(mu)
            s2h = 1.0 / rng.gamma(a_h + (T - 1) / 2, 1.0 / (b_h + d @ d / 2))
        if it >= burn:
            i = it - burn
            B[i] = beta; G[i] = gamma; S2E[i] = s2e; S2H[i] = s2h
            MU_last[i] = mu[-1]
            FIT[i] = mu + (X @ beta if k else 0)

    # 사후 예측
    H = len(Xp)
    eta = rng.standard_normal((keep, H)) * np.sqrt(S2H)[:, None] if level else np.zeros((keep, H))
    mu_path = MU_last[:, None] + np.cumsum(eta, axis=1)
    eps = rng.standard_normal((keep, H)) * np.sqrt(S2E)[:, None]
    pred = mu_path + (B @ Xp.T if k else 0) + eps
    return BSTSResult(pred_draws=pred * ys + ym, fit_draws=FIT * ys + ym, beta_draws=B, gamma_draws=G,
                      s2_obs=S2E * ys ** 2, s2_level=S2H * ys ** 2, names=list(names or range(k)))
