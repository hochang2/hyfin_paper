"""실험 4: 표본 효율 (Sample Efficiency).

평가 세트는 고정하고 학습 자료의 양·종류만 바꾼다.
  (a) rv5 시간 길이: 이동창 L ∈ {45, 60, 90, 130} + 확장창(표본 시작부터) — 평가 가짜 개입일 ≥ 130 (19개) 고정.
      190일 학습은 발표일 전 rv5 표본 안에서 가짜 창을 둘 수 없어 불가능 (REPORT 에 명시).
  (b) GK 시간 길이: L ∈ {60 … 2000} 일 — 평가 가짜 개입일(2023-06~2026-05, 10일 간격)·표적 고정.
      default vs tuned (LGBM 격자, 날짜 블록 내부 CV, 8 개 가짜 개입일마다 그 시점 이전 자료로만 재튜닝).
  (c) N(종목) × T(일) 같은 행 예산: 표적 자신의 최근 T 일 + donor (N−1) 개의 같은 T 일, 종목별 표준화(G2) 패널.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.rq2.phase1.ml_followup import common as C
from src.rq2.phase1.ml_followup import panel as P

S_ = P.FCFG["sample"]
NJ = P.FCFG["n_jobs"]
EXP = "exp4"
RV5_MODELS = ["ols", "ridge", "enet", "lgbm", "lgbm_deep", "xgb", "rf", "bsts"]
GRID = [dict(num_leaves=nl, min_child_samples=mc, n_estimators=ne)
        for nl, mc, ne in itertools.product([3, 7, 15], [5, 15, 40], [100, 300, 600])]


def _rows(y, n, pred, meta):
    return [{**meta, **r} for r in C.window_rows(y[n:], pred)]


def _task(y, X, n, models, meta, seed, tuned=None):
    warnings.filterwarnings("ignore")
    rows = []
    for m in models:
        o = C.fit_predict(m, y, X, n, seed=seed)
        rows += _rows(y, n, o["pred"], {**meta, "model": m})
    if tuned is not None:
        o = C.fit_predict("lgbm", y, X, n, seed=seed, params=tuned)
        rows += _rows(y, n, o["pred"], {**meta, "model": "lgbm_tuned"})
    return rows


def tune_lgbm(y, X, n, seed) -> dict:
    """날짜 블록 전진 CV(3 조각, 마지막 40%) 로 격자에서 h 무관 1-step 오차 최소 조합 (사전 자료만)."""
    warnings.filterwarnings("ignore")
    yt, Xt = y[:n], X[:n]
    edges = np.linspace(0.6 * n, n, 4).astype(int)
    best, bp = np.inf, None
    for p in GRID:
        se = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            m = C.make_est("lgbm", seed, p)
            m.fit(Xt[:a], yt[:a])
            se += np.sum((yt[a:b] - m.predict(Xt[a:b])) ** 2)
        if se < best:
            best, bp = se, p
    return bp


# ================================================================ (a) rv5
def eval_targets_rv5():
    from src.rq2.phase1.ml_followup.exp1_global import eval_donors
    return P.UNITS3 + eval_donors()[:15]


def run_rv5_lengths(force=False):
    def go():
        cuts = [c for c in P.rv5_cutoffs() if c >= S_["rv5_eval_first"]]
        tasks = []
        for u in eval_targets_rv5():
            for i0 in cuts:
                H = P.rv5_horizon_max(i0)
                for L in S_["rv5_lengths"] + [0]:
                    y, X, n = P.rv5_task(u, i0, H, train_len=L or None)
                    tasks.append((y.values, X.values, n, {"code": u, "i0": i0, "L": L or i0, "L_kind": "roll" if L else "expand",
                                                         "group": "treated" if u in P.UNITS3 else "donor"}))
        res = Parallel(n_jobs=NJ, batch_size=4, verbose=5)(
            delayed(_task)(y, X, n, RV5_MODELS, m, P.SEED) for y, X, n, m in tasks)
        return pd.DataFrame([r for rr in res for r in rr])
    return C.cached(EXP, "rv5_lengths", go, force)


# ================================================================ (b) GK
def gk_eval_cutoffs():
    idx = P.gk_long().index
    a = max(idx.searchsorted(pd.Timestamp(S_["gk_eval_first_date"])), max(S_["gk_lengths"]))
    lst = idx.searchsorted(P.CFG["timeline"]["listing"])
    return list(range(a, lst - 71 + 1, S_["gk_eval_step"]))


def gk_targets():
    dl, _, _ = P.gk_units()
    rng = np.random.default_rng(P.SEED + 4)
    return P.UNITS3 + sorted(rng.choice(dl, S_["gk_donor_targets"], replace=False).tolist())


def run_gk_lengths(force=False):
    def go():
        cuts = gk_eval_cutoffs()
        tune_pts = cuts[:: S_["tune_every"]]
        # 1) 튜닝: (표적, L∈{250,1000}, 튜닝 시점) — 그 시점 이전 자료만
        tj = []
        for u in gk_targets():
            for L in (250, 1000):
                for tp in tune_pts:
                    y, X, n = P.gk_task(u, tp, L, 0)
                    tj.append((u, L, tp, y.values, X.values, n))
        tuned = Parallel(n_jobs=NJ, verbose=2)(delayed(tune_lgbm)(y, X, n, P.SEED) for _, _, _, y, X, n in tj)
        tmap = {(u, L, tp): p for (u, L, tp, *_), p in zip(tj, tuned)}
        tasks = []
        for u in gk_targets():
            for i0 in cuts:
                for L in S_["gk_lengths"]:
                    y, X, n = P.gk_task(u, i0, L, 71)
                    if n < 0.8 * L or len(y) - n < 61:
                        continue
                    models = [m for m in S_["gk_models"] if m != "bsts" or (L <= 500 and u in P.UNITS3)]
                    tp = max([t for t in tune_pts if t <= i0], default=None)
                    tz = tmap.get((u, L, tp)) if L in (250, 1000) else None
                    tasks.append((y.values, X.values, n, models, tz,
                                  {"code": u, "i0": i0, "L": L, "group": "treated" if u in P.UNITS3 else "donor"}))
        res = Parallel(n_jobs=NJ, batch_size=2, verbose=5)(
            delayed(_task)(y, X, n, ms, m, P.SEED, tz) for y, X, n, ms, tz, m in tasks)
        out = pd.DataFrame([r for rr in res for r in rr])
        pd.DataFrame([{"code": k[0], "L": k[1], "tune_i0": k[2], **v} for k, v in tmap.items()]).to_parquet(
            C.cpath(EXP, "gk_tuned_params"))
        return out
    return C.cached(EXP, "gk_lengths", go, force)


# ================================================================ (c) N × T
def _nt_task(u, i0, B, N, seed, algos=("lgbm", "ridge")):
    warnings.filterwarnings("ignore")
    from src.rq2.phase1.ml_followup.exp1_global import _fit, _predict
    T = B // N
    dl, Y, _ = P.gk_units()
    rng = np.random.default_rng(seed + i0 + 31 * N)
    others = [d for d in dl if d != u]
    mem = [u] + (list(rng.choice(others, N - 1, replace=False)) if N > 1 else [])
    parts = []
    for v in mem:
        y, X, n = P.gk_task(v, i0, T, 71 if v == u else 0)
        df = X.copy()
        df["y"] = y.values
        df["t"] = np.arange(len(df))
        df["code"] = v
        df["tr"] = np.arange(len(df)) < n
        yt = y.values[:n]
        c = np.cov(yt, X.don_mean.values[:n])
        df["u_mu"], df["u_sd"] = yt.mean(), yt.std(ddof=1)
        df["u_beta"] = c[0, 1] / c[1, 1]
        df["u_r2"] = c[0, 1] ** 2 / (c[0, 0] * c[1, 1])
        df["u_lmcap"] = 0.0
        parts.append(df)
    D = pd.concat(parts, ignore_index=True)
    D["z"] = (D.y - D.u_mu) / D.u_sd
    import src.rq2.phase1.ml_followup.exp1_global as G1
    feats = list(D.columns.intersection(["don_mean", "us_lvix", "us_abs_sox", "kosdaq_ret_abs"]))
    old = G1.FE
    G1.FE = feats
    rows = []
    try:
        tr = D[D.tr]
        du = D[D.code == u]
        n_u = int(du.tr.sum())
        for algo in algos:
            m, _ = _fit(algo, "G2_feat", tr, seed)
            p = _predict(m, "G2_feat", du)
            for r in C.window_rows(du.y.values[n_u:], p[n_u:]):
                rows.append({"code": u, "i0": i0, "B": B, "N": N, "T": T, "algo": algo, "n_rows": len(tr), **r})
    finally:
        G1.FE = old
    return rows


def run_nt(force=False):
    def go():
        cuts = gk_eval_cutoffs()[::2]
        tasks = [(u, i0, B, N) for u in P.UNITS3 for i0 in cuts for B in S_["nt_budgets"] for N in S_["nt_units"]
                 if B // N >= 20]
        res = Parallel(n_jobs=NJ, batch_size=4, verbose=5)(delayed(_nt_task)(u, i0, B, N, P.SEED) for u, i0, B, N in tasks)
        return pd.DataFrame([r for rr in res for r in rr])
    return C.cached(EXP, "gk_nt", go, force)


# ================================================================ 분석
def _acc(df, keys):
    return df.groupby(keys).agg(rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))), mae=("err", lambda e: e.abs().mean()),
                                bias=("err", "mean"), n=("err", "size"), n_codes=("code", "nunique")).reset_index()


def analyze():
    W = ["h10", "h20", "h30", "h61"]
    out = {}
    # (a) rv5 시간 길이
    R = run_rv5_lengths()
    R = R[R.window.isin(W)].assign(err=lambda d: d.ybar - d.predbar)
    a = _acc(R, ["group", "L_kind", "L", "model", "window"])
    # 같은 평가 세트 보장: L 마다 (code,i0) 수가 같아야 한다
    C.save_table(a, "exp4_rv5_length_curve")
    out["rv5"] = a
    # (b) GK 시간 길이 (+ tuned)
    if C.cpath(EXP, "gk_lengths").exists():
        Gd = run_gk_lengths()
        Gd = Gd[Gd.window.isin(W)].assign(err=lambda d: d.ybar - d.predbar)
        # 모든 L 이 있는 (code,i0) 만 — 길이마다 평가 세트가 달라지지 않게
        full = Gd.groupby(["code", "i0"]).L.nunique()
        keep = full[full == len(S_["gk_lengths"])].index
        Gd = Gd.set_index(["code", "i0"]).loc[keep].reset_index()
        g = _acc(Gd, ["group", "L", "model", "window"])
        base = g[g.model == "ols"][["group", "L", "window", "rmse"]].rename(columns={"rmse": "rmse_ols"})
        g = g.merge(base, on=["group", "L", "window"])
        g["ratio_vs_ols_same_L"] = g.rmse / g.rmse_ols
        C.save_table(g, "exp4_gk_length_curve")
        # Q5: 복잡도별 최소 길이 — 같은 L 의 OLS 보다 RMSE 가 낮아지는 최소 L
        q5 = []
        for (grp, m, w), q in g.groupby(["group", "model", "window"]):
            q = q.sort_values("L")
            ok = q[q.ratio_vs_ols_same_L < 1]
            q5.append({"group": grp, "model": m, "window": w, "min_L_beats_ols": ok.L.min() if len(ok) else np.nan,
                       "best_L": q.loc[q.rmse.idxmin(), "L"], "rmse_at_best": q.rmse.min()})
        C.save_table(pd.DataFrame(q5), "exp4_q5_min_length")
        out["gk"] = g
    # (c) N × T
    if C.cpath(EXP, "gk_nt").exists():
        N = run_nt()
        N = N[N.window.isin(W)].assign(err=lambda d: d.ybar - d.predbar)
        n = N.groupby(["B", "N", "T", "algo", "window"]).agg(rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))),
                                                            bias=("err", "mean"), n=("err", "size")).reset_index()
        C.save_table(n, "exp4_nt_budget")
        out["nt"] = n
    return out


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["rv5", "gk", "nt"]
    if "rv5" in what:
        print(run_rv5_lengths().shape)
    if "nt" in what:
        print(run_nt().shape)
    if "gk" in what:
        print(run_gk_lengths().shape)
    if "analysis" in what:
        analyze()
