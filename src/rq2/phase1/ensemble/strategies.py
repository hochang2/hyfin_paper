"""앙상블 가중치 함수와 온라인(미래 정보 없는) 엔진.

모든 전략은 '과거에 창이 완전히 실현된' 가짜 개입일 행(i0' + b <= i0)의 창평균 오차만 본다.
가중치 합 = 1 이므로 앙상블 오차 = E w − c (E = y − P, c = 편향 보정 이동).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import Ridge

from src.rq2.phase1.ensemble.loader import CFG, POOL, WIN

O = CFG["online"]


# ================================================================ 해법
def simplex_qp(S: np.ndarray, cap: float | None = None, l2: float = 0.0) -> np.ndarray:
    """min w'Sw + l2·||w − 1/k||²  s.t. w >= 0, Σw = 1, w <= cap."""
    k = len(S)
    if k == 1:
        return np.ones(1)
    t = np.full(k, 1 / k)
    H = 2 * (S + l2 * np.eye(k))
    f = lambda w: w @ S @ w + l2 * ((w - t) @ (w - t))
    g = lambda w: H @ w - 2 * l2 * t
    ub = cap if cap and cap * k >= 1 else 1.0
    r = minimize(f, t, jac=g, method="SLSQP", bounds=[(0, ub)] * k,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(k)}],
                 options={"ftol": 1e-12, "maxiter": 200})
    w = np.clip(r.x, 0, None)
    return w / w.sum()


def mse(E):
    return (E ** 2).mean(0)


def n_eff(i0s, b, step):
    """비겹침 창 수 근사: 서로 다른 과거 i0 수 × min(1, step/b)."""
    return len(np.unique(i0s)) * min(1.0, step / max(b, 1))


# ================================================================ 전략 (E: 과거 오차 n×k, ctx: 이름·창·간격 등)
def _onehot(k, j):
    w = np.zeros(k)
    w[j] = 1
    return w


def _topk_eq(E, k):
    j = np.argsort(mse(E))[:k]
    w = np.zeros(E.shape[1])
    w[j] = 1 / len(j)
    return w


def _invmse(E, k=None):
    m = mse(E)
    j = np.argsort(m)[:k] if k else np.arange(len(m))
    w = np.zeros(len(m))
    w[j] = 1 / m[j]
    return w / w.sum()


def _lw(E):
    S, _ = ledoit_wolf(E, assume_centered=True)
    return S


def _greedy(E, steps):
    k = E.shape[1]
    cnt = np.zeros(k)
    cur = np.zeros(E.shape[0])
    for s in range(steps):
        cand = [((cur * s + E[:, j]) / (s + 1)) for j in range(k)]
        j = int(np.argmin([np.mean(c ** 2) for c in cand]))
        cnt[j] += 1
        cur = cand[j]
    return cnt / cnt.sum()


def _pair_lowcorr(E, tol):
    m = mse(E)
    best = int(np.argmin(m))
    ok = [j for j in range(len(m)) if j != best and m[j] <= tol * m[best]]
    w = np.zeros(len(m))
    if not ok:
        w[best] = 1
        return w
    C = np.corrcoef(E.T)
    j = min(ok, key=lambda j: C[best, j])
    w[best] = w[j] = 0.5
    return w


def _old_stack(P, y):
    r = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(P, y)
    w = r.coef_
    return w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])


def _bsts_shrink(E, ctx):
    w = simplex_qp(E.T @ E / len(E))
    a = ctx["neff"] / (ctx["neff"] + O["bsts_shrink_kappa"])
    j = ctx["names"].index("bsts")
    return a * w + (1 - a) * _onehot(len(w), j)


def _bestml_hybrid(E, ctx):
    nm = ctx["names"]
    ml = [nm.index(m) for m in POOL["ml"] if m in nm]
    j = ml[int(np.argmin(mse(E[:, ml])))]
    w = np.zeros(len(nm))
    for i in (nm.index("bsts"), nm.index("ols"), j):
        w[i] += 1 / 3
    return w


# 전략 정의: 이름 -> (구성원 규칙, 함수(E, P, y, ctx) -> w, 편향보정 여부, 묶음)
def _need(*ms):
    return lambda names: [m for m in ms] if all(m in names for m in ms) else None


def _pool(key):
    return lambda names: [m for m in POOL[key] if m in names] or None


FULL = lambda names: list(names)
STRATS = {
    # 기준
    "single_bsts": (_need("bsts"), lambda E, P, y, c: np.ones(1), False, "base"),
    "single_ols": (_need("ols"), lambda E, P, y, c: np.ones(1), False, "base"),
    "eq_all": (FULL, lambda E, P, y, c: np.full(E.shape[1], 1 / E.shape[1]), False, "base"),
    "old_stack_replica": (_pool("old_stack_members"), lambda E, P, y, c: _old_stack(P, y), False, "base"),
    "ftl_single": (FULL, lambda E, P, y, c: _onehot(E.shape[1], int(np.argmin(mse(E)))), False, "base"),
    # A 성능 기반
    "top2": (FULL, lambda E, P, y, c: _topk_eq(E, 2), False, "A"),
    "top3": (FULL, lambda E, P, y, c: _topk_eq(E, 3), False, "A"),
    "top5": (FULL, lambda E, P, y, c: _topk_eq(E, 5), False, "A"),
    "invmse_all": (FULL, lambda E, P, y, c: _invmse(E), False, "A"),
    "invmse_top3": (FULL, lambda E, P, y, c: _invmse(E, 3), False, "A"),
    "bc_invmse_top3": (FULL, lambda E, P, y, c: _invmse(E, 3), True, "A"),
    "bc_bsts": (_need("bsts"), lambda E, P, y, c: np.ones(1), True, "A"),
    # B 다양성
    "mv_lw": (FULL, lambda E, P, y, c: simplex_qp(_lw(E)), False, "B"),
    "mv_lw_top5": (FULL, lambda E, P, y, c: _sub(E, 5, lambda e: simplex_qp(_lw(e))), False, "B"),
    "greedy": (FULL, lambda E, P, y, c: _greedy(E, O["greedy_steps"]), False, "B"),
    "pair_lowcorr": (FULL, lambda E, P, y, c: _pair_lowcorr(E, O["pair_mse_tol"]), False, "B"),
    # C 제약 stacking
    "simplex": (FULL, lambda E, P, y, c: simplex_qp(E.T @ E / len(E)), False, "C"),
    "simplex_cap": (FULL, lambda E, P, y, c: simplex_qp(E.T @ E / len(E), cap=O["cap"]), False, "C"),
    "simplex_l2": (FULL, lambda E, P, y, c: simplex_qp(E.T @ E / len(E), l2=np.mean(mse(E))), False, "C"),
    "bsts_shrink": (FULL, lambda E, P, y, c: _bsts_shrink(E, c), False, "C"),
    # D 혼합
    "h_bsts_ols_tfm": (_need("bsts", "ols", "timesfm_xreg"), lambda E, P, y, c: np.full(3, 1 / 3), False, "D"),
    "h_bsts_ols_chr": (_need("bsts", "ols", "chronos2_cov"), lambda E, P, y, c: np.full(3, 1 / 3), False, "D"),
    "h_bsts_chr_uc": (_need("bsts", "chronos2_cov", "uc"), lambda E, P, y, c: np.full(3, 1 / 3), False, "D"),
    "h_bsts_ols_bestml": (FULL, lambda E, P, y, c: _bestml_hybrid(E, c), False, "D"),
    "stat_simplex": (_pool("stat"), lambda E, P, y, c: simplex_qp(E.T @ E / len(E)), False, "D"),
}
NOLEARN = {"single_bsts", "single_ols", "eq_all", "h_bsts_ols_tfm", "h_bsts_ols_chr", "h_bsts_chr_uc"}


def _sub(E, k, fn):
    j = np.argsort(mse(E))[:k]
    w = np.zeros(E.shape[1])
    w[j] = fn(E[:, j])
    return w


def weights(name, Pp, yp, ctx):
    """과거 (P, y) 로 전략 name 의 (w, c). 행이 부족하면 동일가중 + c=0."""
    _, fn, bc, _ = STRATS[name]
    k = Pp.shape[1]
    if name not in NOLEARN and len(yp) < ctx.get("min_rows", O["min_past_rows"]):
        if name in ("single_bsts", "bc_bsts"):
            return np.ones(1), 0.0, True
        return np.full(k, 1 / k), 0.0, True
    E = yp[:, None] - Pp
    w = fn(E, Pp, yp, ctx)
    c = float(np.mean(E @ w)) if bc and len(yp) else 0.0
    return w, c, False


# ================================================================ 온라인 엔진
class Task:
    """한 (자료, 창) 의 넓은 표를 배열로."""

    def __init__(self, W: pd.DataFrame, window: str, step: int, names: list[str]):
        self.W = W
        self.b = WIN[window][1]
        self.L = WIN[window][1] - WIN[window][0]
        self.step = step
        self.window = window
        self.names = names
        self.y = W.ybar.values
        self.i0 = W.i0.values
        self.code = W.code.values

    def members(self, name):
        return STRATS[name][0](self.names)

    def P(self, ms):
        return self.W[ms].values


def online(task: Task, name: str, scope: str = "global", regime: pd.Series | None = None,
           log_w: list | None = None) -> np.ndarray:
    """행마다 온라인 예측. scope: global(종목 합동) | code | hier | glocal(전역 가중치 + 종목 편향, shrinkage)
    | regime(같은 국면 과거 행, 부족하면 전역). regime: i0 -> 라벨."""
    ms = task.members(name)
    if ms is None:
        return None
    P = task.P(ms)
    y, i0, code = task.y, task.i0, task.code
    out = np.full(len(y), np.nan)
    reg = pd.Series(i0).map(regime).values if regime is not None else None
    for t in np.unique(i0):
        past = i0 + task.b <= t
        rows_t = np.where(i0 == t)[0]
        ctx = {"names": ms, "neff": n_eff(i0[past], task.b, task.step)}
        if True:
            if scope == "regime" and regime is not None:
                lab = regime.get(t)
                same = past & (reg == lab)
                if same.sum() >= O["regime_min_rows"]:
                    ctx_r = {"names": ms, "neff": n_eff(i0[same], task.b, task.step)}
                    wg, cg, fb = weights(name, P[same], y[same], ctx_r)
                else:
                    wg, cg, fb = weights(name, P[past], y[past], ctx)
            else:
                wg, cg, fb = weights(name, P[past], y[past], ctx)
            if log_w is not None:
                log_w.append({"i0": int(t), "strategy": name, "scope": scope, "fallback": fb, "c": cg,
                              **{f"w_{m}": float(v) for m, v in zip(ms, wg)}})
        for r in rows_t:
            if scope in ("global", "regime"):
                w, c = wg, cg
            else:
                pc = past & (code == code[r])
                nc = n_eff(i0[pc], task.b, task.step)
                if scope == "code":
                    if pc.sum() >= O["min_past_rows_code"]:
                        w, c, _ = weights(name, P[pc], y[pc], {"names": ms, "neff": nc, "min_rows": O["min_past_rows_code"]})
                    else:
                        w, c = wg, cg
                elif scope == "hier":
                    lam = nc / (nc + O["hier_kappa"])
                    if pc.sum() >= O["min_past_rows_code"]:
                        wc, cc, _ = weights(name, P[pc], y[pc], {"names": ms, "neff": nc, "min_rows": O["min_past_rows_code"]})
                        w, c = lam * wc + (1 - lam) * wg, lam * cc + (1 - lam) * cg
                    else:
                        w, c = wg, cg
                elif scope == "glocal":
                    lam = nc / (nc + O["hier_kappa"])
                    eb = (y[pc] - P[pc] @ wg - cg).mean() if pc.sum() else 0.0
                    w, c = wg, cg + lam * eb
            out[r] = P[r] @ w + c
    return out


def fixed_weights(task: Task, name: str, rows: np.ndarray, min_rows: int | None = None):
    """주어진 학습 행으로 한 번 학습한 (w, c, members)."""
    ms = task.members(name)
    if ms is None:
        return None
    P = task.P(ms)
    ctx = {"names": ms, "neff": n_eff(task.i0[rows], task.b, task.step)}
    if min_rows is not None:
        ctx["min_rows"] = min_rows
    w, c, fb = weights(name, P[rows], task.y[rows], ctx)
    return w, c, ms


def meta_ftl(pred: dict[str, np.ndarray], task: Task, cands: list[str]) -> tuple[np.ndarray, list]:
    """과거 실현 행에서 온라인 MSE 가 가장 작은 전략을 고른다 (전략 선택 자체도 온라인)."""
    y, i0 = task.y, task.i0
    E = {s: y - pred[s] for s in cands if pred.get(s) is not None}
    out = np.full(len(y), np.nan)
    picks = []
    for t in np.unique(i0):
        past = i0 + task.b <= t
        sc = {s: np.nanmean(e[past] ** 2) for s, e in E.items() if past.sum() >= O["min_past_rows"] and np.isfinite(e[past]).all()}
        s = min(sc, key=sc.get) if sc else "eq_all"
        rows = i0 == t
        out[rows] = pred[s][rows]
        picks.append({"i0": int(t), "pick": s})
    return out, picks
