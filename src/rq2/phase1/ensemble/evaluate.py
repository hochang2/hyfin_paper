"""평가 공용: 평가 구간, 지표, 의존성을 고려한 재표집 (가짜 개입일 블록), DM-HAC, White reality check."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.rq2.phase1.ensemble import loader as L
from src.rq2.phase1.ensemble import online_run as R

OOS = L.CFG["oos"]


def eval_rows(D: pd.DataFrame, ds: str) -> pd.DataFrame:
    """예열 제외 평가 행. gk·gkd: 2018-01-01 이후 가짜 개입일, rv5·rv5d: i0 >= 120."""
    if ds in ("gk", "gkd"):
        return D[D.cutoff > pd.Timestamp(OOS["gk_burnin_end"])]
    return D[D.i0 >= OOS["rv5_eval_from"]]


def ref_old(ds):
    return "old_stack" if ds in ("gk", "rv5") else "old_stack_replica"


def metrics(D: pd.DataFrame, ds: str) -> pd.DataFrame:
    """(창, 전략) 별 RMSE·MAE·편향, BSTS·기존 stacking 대비 상대 RMSE, i0 단위 승률."""
    rows = []
    old = ref_old(ds)
    for w, g in D.groupby("window"):
        se = g.assign(se=g.err ** 2).pivot_table(index="i0", columns="strategy", values="se", aggfunc="mean")
        rb = np.sqrt(g[g.strategy == "single_bsts"].err.pow(2).mean())
        ro = np.sqrt(g[g.strategy == old].err.pow(2).mean())
        for s, h in g.groupby("strategy"):
            e = h.err.values
            rm = np.sqrt(np.mean(e ** 2))
            rows.append({"dataset": ds, "window": w, "strategy": s, "family": h.family.iat[0], "n": len(e),
                         "n_i0": h.i0.nunique(), "rmse": rm, "mae": np.mean(np.abs(e)), "bias": np.mean(e),
                         "rel_bsts": rm / rb, "rel_old": rm / ro,
                         "win_vs_bsts": float((se[s] < se["single_bsts"]).mean()),
                         "win_vs_old": float((se[s] < se[old]).mean())})
    T = pd.DataFrame(rows)
    T["rank"] = T.groupby("window").rmse.rank()
    return T


def mean_rel(T: pd.DataFrame, windows) -> pd.Series:
    """창 평균 상대 RMSE (선정 기준)."""
    return T[T.window.isin(windows)].groupby("strategy").rel_bsts.mean()


def _loss_by_i0(g, s):
    return g[g.strategy == s].assign(se=lambda d: d.err ** 2).groupby("i0").se.mean()


def block_boot_diff(D, s, ref, b, step, B=2000, seed=0):
    """MSE 차이 (s − ref) 의 i0 블록 bootstrap. 블록 길이 = 겹침 길이 ceil(b/step) (i0 격자 단위, 원형)."""
    d = (_loss_by_i0(D, s) - _loss_by_i0(D, ref)).dropna().sort_index().values
    n = len(d)
    if n < 5:
        return dict(dmse=np.nan, lo=np.nan, hi=np.nan, p_boot=np.nan, dm_t=np.nan, dm_p=np.nan, n_i0=n)
    l = max(1, int(np.ceil(b / step)))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / l))
    st = rng.integers(0, n, size=(B, nb))
    idx = (st[:, :, None] + np.arange(l)[None, None, :]).reshape(B, -1)[:, :n] % n
    bm = d[idx].mean(1)
    lo, hi = np.quantile(bm, [0.025, 0.975])
    p = float(np.mean(np.abs(bm - d.mean()) >= abs(d.mean())))   # 중심화한 bootstrap 분포의 양측 p
    # DM-HAC (Bartlett, bw = l)
    lrv = _nw(d, l)
    t = d.mean() / np.sqrt(lrv / n)
    return dict(dmse=d.mean(), lo=lo, hi=hi, p_boot=p, dm_t=t, dm_p=2 * stats.t.sf(abs(t), max(n // l - 1, 1)), n_i0=n)


def _nw(d, bw):
    e = d - d.mean()
    n = len(e)
    v = e @ e / n
    for k in range(1, min(bw, n - 1) + 1):
        v += 2 * (1 - k / (bw + 1)) * (e[k:] @ e[:-k]) / n
    return max(v, 1e-14)


def reality_check(D, cands, ref, b, step, B=2000, seed=0):
    """White(2000) reality check: H0 = 후보 중 어느 것도 ref 보다 MSE 가 작지 않다. 후보 수만큼의 선택 편향을 반영."""
    Lr = _loss_by_i0(D, ref)
    M = pd.DataFrame({s: Lr - _loss_by_i0(D, s) for s in cands}).dropna()
    X = M.values
    n = len(X)
    l = max(1, int(np.ceil(b / step)))
    stat = np.sqrt(n) * X.mean(0).max()
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / l))
    mu = X.mean(0)
    cnt = 0
    for _ in range(B):
        st = rng.integers(0, n, nb)
        idx = (st[:, None] + np.arange(l)).ravel()[:n] % n
        cnt += np.sqrt(n) * (X[idx].mean(0) - mu).max() >= stat
    return {"n_cands": X.shape[1], "best": M.columns[int(np.argmax(mu))], "best_dmse_gain": mu.max(),
            "rc_p": (cnt + 1) / (B + 1), "n_i0": n}


def load_eval(ds):
    D = R.load(ds)
    return eval_rows(D, ds)


def weight_stability(ds):
    Wl = R.weights_log(ds)
    rows = []
    for (w, s), g in Wl.groupby(["window", "strategy"]):
        g = g[~g.fallback].sort_values("i0")
        wc = [c for c in g.columns if c.startswith("w_") and g[c].notna().any()]
        if len(g) < 3 or not wc:
            continue
        M = g[wc].fillna(0).values
        ent = -(np.where(M > 0, M * np.log(np.clip(M, 1e-12, None)), 0)).sum(1)
        top = pd.Series([wc[i][2:] for i in M.argmax(1)])
        rows.append({"dataset": ds, "window": w, "strategy": s, "n_fits": len(g),
                     "turnover": np.abs(np.diff(M, axis=0)).sum(1).mean() / 2, "max_w": M.max(1).mean(),
                     "eff_members": np.exp(ent).mean(), "top_member_mode": top.mode().iat[0],
                     "top_member_share": (top == top.mode().iat[0]).mean(),
                     **{f"mw_{c[2:]}": g[c].fillna(0).mean() for c in wc}})
    return pd.DataFrame(rows)
