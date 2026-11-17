"""실험 7: Donor pool 민감도.

pool 은 표적(처치 3 종목)·가짜 개입일 i0 마다 i0 이전 자료만으로 정한다.
  all / main / strict          : 1차 연구 정의 (고정 목록)
  vol_sim                      : main 중 사전 평균 log rv5 가 가장 가까운 k 개
  trend_sim                    : main 중 사전 rv5 상관(동조성)이 가장 높은 k 개
  size_sim                     : main 중 사전 마지막 60일 평균 log 시총이 가장 가까운 k 개
  industry_plus                : main 에서 사전 rv5 상관 상위 25% (동종 산업·파급 대리지표) 제외
모형: OLS, BSTS(light), Local LGBM, Global LGBM(G2: pool 구성원 사전 행으로 학습, 종목별 표준화).
평가: 32 가짜 개입일 반사실 오차(정확도), 같은 표본 LOO neff/strict FPR(1차 확장 방식), 실제 발표일 효과·시간 p.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.rq2.phase1.ml_followup import common as C
from src.rq2.phase1.ml_followup import panel as P

DO = P.FCFG["donor"]
NJ = P.FCFG["n_jobs"]
EXP = "exp7"
LOCAL = ["ols", "bsts", "lgbm"]


def pool_members(name: str, code: str, i0: int) -> tuple:
    if name in ("all", "main", "strict"):
        return tuple(P.pool(name))
    lv = P.rv5_wide()
    main = P.pool("main")
    pre = lv.iloc[:i0]
    k = DO["k_similar"]
    if name == "vol_sim":
        d = (pre[main].mean() - pre[code].mean()).abs()
        return tuple(d.nsmallest(k).index)
    if name == "trend_sim":
        c = pre[main].corrwith(pre[code])
        return tuple(c.nlargest(k).index)
    if name == "size_sim":
        mc = P.mktcap_wide()
        d_end = lv.index[i0 - 1]
        lm = np.log(mc.loc[:d_end].tail(60).mean())
        cand = [m for m in main if m in lm.index and np.isfinite(lm[m])]
        return tuple((lm[cand] - lm[code]).abs().nsmallest(k).index)
    if name == "industry_plus":
        c = pre[main].corrwith(pre[code])
        thr = c.quantile(DO["industry_corr_quantile"])
        return tuple(c[c < thr].index)
    raise ValueError(name)


def _task(code, i0, pname, seed):
    warnings.filterwarnings("ignore")
    from src.rq2.phase1.ml_followup import exp1_global as G1
    mem = pool_members(pname, code, i0)
    H = P.rv5_horizon_max(i0) if i0 < P.real_index() else 71
    y, X, n = P.rv5_task(code, i0, H, members=mem)
    meta = {"code": code, "i0": i0, "pool": pname, "n_donors": len(mem), "is_real": i0 == P.real_index()}
    rows = []
    for m in LOCAL:
        o = C.fit_predict(m, y.values, X.values, n, seed=seed)
        rows += [{**meta, "model": m, **r} for r in C.window_rows(y.values[n:], o["pred"])]
    # Global LGBM (G2): pool 구성원(자기 제외 통제)의 사전 행 + 표적 사전 행
    D = G1.panel_at(i0, H, list(mem) + [code], members=mem)
    tr = D[D.t < i0]
    mdl, _ = G1._fit("lgbm", "G2_feat", tr, seed)
    du = D[D.code == code]
    p = G1._predict(mdl, "G2_feat", du)
    nu = int((du.t < i0).sum())
    rows += [{**meta, "model": "global_lgbm", **r} for r in C.window_rows(du.y.values[nu:], p[nu:])]
    return rows


def run(force=False):
    def go():
        cuts = P.rv5_cutoffs() + [P.real_index()]
        tasks = [(c, i0, p) for p in DO["pools"] for c in P.UNITS3 for i0 in cuts]
        res = Parallel(n_jobs=NJ, batch_size=2, verbose=5)(delayed(_task)(c, i0, p, P.SEED) for c, i0, p in tasks)
        return pd.DataFrame([r for rr in res for r in rr])
    return C.cached(EXP, "pools", go, force)


def analyze(R=None):
    from src.rq2.phase1.ml_followup.exp5_audit import loo_audit
    R = run() if R is None else R
    R = R.assign(err=R.ybar - R.predbar)
    ps = R[~R.is_real]
    acc = ps[ps.window.isin(P.HZ + ["transition", "post_effective"])].groupby(["pool", "model", "window"]).agg(
        rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))), mae=("err", lambda e: e.abs().mean()), bias=("err", "mean"),
        n=("err", "size"), n_donors=("n_donors", "mean")).reset_index()
    C.save_table(acc, "exp7_pool_accuracy")
    fpr = []
    for p, g in ps.groupby("pool"):
        f = loo_audit(g.assign(model=g.model), P.FCFG["rv5"]["cutoff_step"], methods=["neff", "strict"], label=p)
        fpr.append(f.assign(pool=p))
    fpr = pd.concat(fpr)
    C.save_table(fpr, "exp7_pool_fpr_loo")
    # 실제 발표일 효과: 시간 null(neff, 모든 가짜 창) p
    rng = np.random.default_rng(P.SEED)
    rows = []
    real = R[R.is_real]
    for (p, m, c, w), g in real.groupby(["pool", "model", "code", "window"]):
        if w not in ("transition", "post_effective", "h10", "h20", "h30", "h61"):
            continue
        est = float(g.err.iloc[0])
        q = ps[(ps.pool == p) & (ps.model == m) & (ps.code == c) & (ps.window == w)]
        a, b = P.WIN[w]
        r = {"pool": p, "model": m, "code": c, "window": w, "effect_pct": 100 * (np.exp(est) - 1),
             "n_donors": int(g.n_donors.iloc[0])}
        for meth in ("neff", "strict"):
            o = C.null_p(est, q.err.values, q.i0.values, b - a, P.FCFG["rv5"]["cutoff_step"], meth, 3, 4999, rng)
            r[f"p_{meth}"] = o["p"]
        rows.append(r)
    real_t = pd.DataFrame(rows)
    C.save_table(real_t, "exp7_pool_real_effect")
    # pool 간 실제 효과 범위 (모형·창별)
    rng_t = real_t.groupby(["model", "code", "window"]).effect_pct.agg(["min", "max", "median"]).reset_index()
    C.save_table(rng_t, "exp7_pool_effect_range")
    # pool 구성 겹침 (실제 발표일)
    i0 = P.real_index()
    ov = []
    for c in P.UNITS3:
        ms = {p: set(pool_members(p, c, i0)) for p in DO["pools"]}
        for p in DO["pools"]:
            ov.append({"code": c, "pool": p, "n": len(ms[p]), "jaccard_vs_main": len(ms[p] & ms["main"]) / len(ms[p] | ms["main"])})
    C.save_table(pd.DataFrame(ov), "exp7_pool_overlap")
    return acc, fpr, real_t


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["run", "analysis"]
    if "test" in what:
        import time
        for p in DO["pools"]:
            t = time.time()
            r = pd.DataFrame(_task("005930", 150, p, P.SEED))
            print(p, r.n_donors.iloc[0], round(time.time() - t, 1),
                  r[r.window == "h20"].assign(e=lambda d: d.ybar - d.predbar).set_index("model").e.round(3).to_dict())
    if "run" in what:
        print(run().shape)
    if "analysis" in what:
        analyze()
