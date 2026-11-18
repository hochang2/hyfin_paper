"""RQ1 5단계: 탐색 분석 (결론의 근거로 쓰지 않음, config pre_registration.exploratory).

1) GJR-GARCH-X: 일별 수익률(2016~2026-07-15), 분산식에 상장 더미.
   r_t = μ + b·r_donor,t + ε_t,  σ²_t = ω + α ε²_{t−1} + γ ε²_{t−1}·1[ε_{t−1}<0] + β σ²_{t−1} + δ·L_t
   arch 패키지는 분산식 외생변수를 지원하지 않아 우도를 직접 최대화한다. SE 는 수치 헤시안 (모형 p, 보고만).
2) 평균 이동 sup-F (Andrews 1993): d_t = log RV_T − donor 평균, 변화 시점 미지. 15% 절사.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from src.rq1.core import data as D
from src.rq2.core import data as rq2data

ANDREWS_15 = {"10%": 7.17, "5%": 8.85, "1%": 12.35}   # 제약 1개, 15% 절사 임계값


def daily_returns(codes: list[str]) -> pd.DataFrame:
    s = D.CFG["sample"]
    f1 = rq2data.raw_f1()
    r = f1.pivot(index="date", columns="code", values="ret_log").sort_index() * 100   # %
    donors = [c for c in D.groups()["donor"] if c in r.columns]
    out = r[codes].copy()
    out["don_ret"] = r[donors].mean(axis=1)
    return out.loc["2016-01-01":s["treat_end"]]


def _garchx_nll(params, r, x, L):
    mu, b, omega, alpha, gamma, beta, delta = params
    e = r - mu - b * x
    n = len(e)
    s2 = np.empty(n)
    s2[0] = np.var(e)
    for t in range(1, n):
        s2[t] = omega + alpha * e[t - 1] ** 2 + gamma * e[t - 1] ** 2 * (e[t - 1] < 0) + beta * s2[t - 1] + delta * L[t]
        if s2[t] <= 1e-8:
            return 1e10
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + e ** 2 / s2)


def _num_hessian(f, x, h=1e-4):
    k = len(x)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.eye(k)[i] * h, np.eye(k)[j] * h
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


def garch_x(code: str, rets: pd.DataFrame) -> dict:
    s = D.CFG["sample"]
    df = rets[[code, "don_ret"]].dropna()
    r, x = df[code].to_numpy(), df["don_ret"].to_numpy()
    L = ((df.index >= s["treat_start"]) & (df.index <= s["treat_end"])).astype(float)
    v = np.var(r)
    x0 = np.array([0.0, 1.0, 0.05 * v, 0.05, 0.05, 0.85, 0.0])
    bounds = [(None, None), (None, None), (1e-6, None), (0, 1), (-0.5, 1), (0, 0.999), (None, None)]
    f = lambda p: _garchx_nll(p, r, x, L)  # noqa: E731
    res = optimize.minimize(f, x0, method="L-BFGS-B", bounds=bounds)
    H = _num_hessian(f, res.x)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        se = np.full(len(res.x), np.nan)
    names = ["mu", "b_don", "omega", "alpha", "gamma", "beta", "delta"]
    p = dict(zip(names, res.x))
    persist = p["alpha"] + p["gamma"] / 2 + p["beta"]
    z = p["delta"] / se[-1] if np.isfinite(se[-1]) and se[-1] > 0 else np.nan
    return {"code": code, "n": len(r), "converged": bool(res.success), **{k: float(vv) for k, vv in p.items()},
            "se_delta": float(se[-1]), "p_delta": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
            "persistence": float(persist), "delta_over_omega": float(p["delta"] / p["omega"])}


def sup_f_mean_shift(d: pd.Series, trim: float = 0.15) -> dict:
    """d_t = a + c·1[t ≥ τ] 의 F 통계량을 모든 τ 에 대해 계산해 최대값과 그 날짜. 동분산·무상관 가정(탐색용)."""
    y = d.dropna().to_numpy()
    idx = d.dropna().index
    n = len(y)
    lo, hi = int(np.floor(trim * n)), int(np.ceil((1 - trim) * n))
    ssr0 = np.sum((y - y.mean()) ** 2)
    F = {}
    for k in range(lo, hi):
        a, b = y[:k], y[k:]
        ssr1 = np.sum((a - a.mean()) ** 2) + np.sum((b - b.mean()) ** 2)
        F[idx[k]] = (ssr0 - ssr1) / (ssr1 / (n - 2))
    F = pd.Series(F)
    tau = F.idxmax()
    listing = D.CFG["timeline"]["listing"]
    return {"sup_F": float(F.max()), "break_date": tau.date().isoformat(),
            "F_at_listing": float(F.get(listing, np.nan)),
            "rank_listing": int((F >= F.get(listing, np.inf)).sum()),
            "n_candidates": len(F),
            "shift_at_break": float(d.loc[tau:].mean() - d.loc[:tau].iloc[:-1].mean()),
            "sig": next((k for k in ["1%", "5%", "10%"] if F.max() > ANDREWS_15[k]), "n.s.")}, F
