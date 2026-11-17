"""실험 2: Transfer Learning (GK → RV5).

평가: rv5 가짜 개입일 32 개 + 실제 발표일, 표적 = 처치 3 + (rv5 main ∩ GK donors_long) 무작위 20.
공통 통제 4 개 [don_mean, us_lvix, us_abs_sox, kosdaq_ret_abs] — don_mean 은 영역별(GK/rv5) 자기 제외 평균.
영역 적응(moment matching): 각 영역·종목마다 자기 학습 행으로 y·X 를 표준화해 학습하고, rv5 예측은 rv5 사전 평균·표준편차로 되돌린다.
GK 사전학습 행은 rv5 가짜 개입일 날짜 '이전' GK 만 쓴다 (평가 시점 이후 GK 금지).

변형
  rv5_only|{ols,lgbm}                 : rv5 학습구간만 (같은 공통 통제)
  {self,multi}_{L}|{lin,lgbm}_zs      : GK 사전학습 모형을 rv5 에 그대로 (zero-shot, 표준화만)
  {self,multi}_{L}|lin_resid          : zero-shot 선형 예측 + rv5 잔차 RidgeCV
  {self,multi}_{L}|lgbm_recal         : rv5 에서 a + b·zero-shot 재보정
  {self,multi}_{L}|lgbm_boost         : 사전학습 LGBM 예측을 init_score 로 rv5 소형 부스팅 추가
  L ∈ {same(rv5 학습구간과 같은 날짜), 250, 500, 1000} — same 은 '더 긴 기간 없이 영역만 추가' 대조군.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.rq2.ml_followup import common as C
from src.rq2.ml_followup import panel as P

T_ = P.FCFG["transfer"]
NJ = P.FCFG["n_jobs"]
EXP = "exp2"
CF = T_["common_features"]
LENS = ["same", 250, 500, 1000]


def targets():
    dl, _, _ = P.gk_units()
    both = sorted(set(P.pool("main")) & set(dl))
    rng = np.random.default_rng(P.SEED + 2)
    return P.UNITS3 + sorted(rng.choice(both, T_["donor_targets"], replace=False).tolist())


def _z(y, X, n):
    my, sy = y[:n].mean(), y[:n].std() + 1e-12
    mx, sx = X[:n].mean(0), X[:n].std(0) + 1e-12
    return (y - my) / sy, (X - mx) / sx, my, sy


def _gk_rows(code, d0, L, n_same_start=None):
    """d0(rv5 개입일) 이전 GK 행. L='same' 이면 rv5 학습 시작일~d0."""
    idx = P.gk_long().index
    j0 = idx.searchsorted(d0)                                     # idx[j0] >= d0 → 행 [.., j0) 은 d0 이전
    if L == "same":
        js = idx.searchsorted(n_same_start)
        n = j0 - js
    else:
        n = L
    y, X, nt = P.gk_task(code, j0, n, 0)
    X = X[CF].values
    return y.values[:nt], X[:nt]


def _lgbm(seed, n=None):
    return C.make_est("lgbm", seed, {"n_estimators": T_.get("pre_trees", 300)} if n is None else n)


def run_cutoff(i0, tg, seed):
    warnings.filterwarnings("ignore")
    import lightgbm as lgb
    from sklearn.linear_model import RidgeCV
    lv = P.rv5_wide()
    d0 = lv.index[i0]
    H = P.rv5_horizon_max(i0) if i0 < P.real_index() else 71
    dl, _, _ = P.gk_units()
    rows = []
    rv = {}
    for u in tg:
        y, X, n = P.rv5_task(u, i0, H, feats=CF)
        rv[u] = (y, X.values, n)
    start = min(rv[u][0].index[0] for u in tg)
    # multi-stock 사전학습 (종목별 표준화, 길이별 한 번) — donors_long 전체 + 처치 3 의 d0 이전 GK
    multi = {}
    for L in LENS:
        Zy, ZX = [], []
        for v in P.UNITS3 + dl:
            yg, Xg = _gk_rows(v, d0, L, start)
            if len(yg) < 40:
                continue
            zy, zx, _, _ = _z(yg, Xg, len(yg))
            Zy.append(zy)
            ZX.append(zx)
        Zy, ZX = np.concatenate(Zy), np.vstack(ZX)
        ml = _lgbm(seed).fit(ZX, Zy)
        b = np.linalg.lstsq(np.column_stack([np.ones(len(Zy)), ZX]), Zy, rcond=None)[0]
        multi[L] = (ml, b, len(Zy))
    for u in tg:
        y, X, n = rv[u]
        yv = y.values
        meta = {"code": u, "i0": i0, "H": H, "n_rv5": n, "group": "treated" if u in P.UNITS3 else "donor",
                "is_real": i0 == P.real_index()}

        def add(name, pred, extra=None):
            for r in C.window_rows(yv[n:], pred):
                rows.append({**meta, "model": name, **(extra or {}), **r})

        for m in ("ols", "lgbm"):
            add(f"rv5_only|{m}", C.fit_predict(m, yv, X, n, seed=seed)["pred"])
        zy, zx, my, sy = _z(yv, X, n)
        Z1 = np.column_stack([np.ones(len(zx)), zx])
        for L in LENS:
            srcs = {}
            yg, Xg = _gk_rows(u, d0, L, y.index[0])
            if len(yg) >= 40:
                gy, gx, _, _ = _z(yg, Xg, len(yg))
                srcs["self"] = (_lgbm(seed).fit(gx, gy), np.linalg.lstsq(np.column_stack([np.ones(len(gy)), gx]), gy, rcond=None)[0], len(gy))
            srcs["multi"] = multi[L]
            for s, (ml, b, npre) in srcs.items():
                tag = f"{s}_{L}"
                ex = {"src": s, "L": str(L), "n_pre": npre}
                lin = Z1 @ b
                gb = ml.predict(zx)
                add(f"{tag}|lin_zs", my + sy * lin[n:], ex)
                add(f"{tag}|lgbm_zs", my + sy * gb[n:], ex)
                rc = RidgeCV(alphas=np.logspace(-2, 3, 15)).fit(zx[:n], zy[:n] - lin[:n])
                add(f"{tag}|lin_resid", my + sy * (lin[n:] + rc.predict(zx[n:])), ex)
                A = np.column_stack([np.ones(n), gb[:n]])
                cb = np.linalg.lstsq(A, zy[:n], rcond=None)[0]
                add(f"{tag}|lgbm_recal", my + sy * (cb[0] + cb[1] * gb[n:]), ex)
                bo = lgb.LGBMRegressor(n_estimators=T_["boost_trees"], learning_rate=0.03, num_leaves=4,
                                       min_child_samples=10, subsample=0.8, subsample_freq=1, random_state=seed,
                                       n_jobs=1, verbose=-1)
                bo.fit(zx[:n], zy[:n], init_score=gb[:n])
                add(f"{tag}|lgbm_boost", my + sy * (gb[n:] + bo.predict(zx[n:])), ex)
    return rows


def run(force=False):
    def go():
        tg = targets()
        cuts = P.rv5_cutoffs() + [P.real_index()]
        res = Parallel(n_jobs=NJ, verbose=5)(delayed(run_cutoff)(i0, tg, P.SEED) for i0 in cuts)
        return pd.DataFrame([r for rr in res for r in rr])
    return C.cached(EXP, "transfer", go, force)


def analyze(R=None):
    from src.rq2.ml_followup.exp5_audit import loo_audit
    R = run() if R is None else R
    R = R.assign(err=R.ybar - R.predbar)
    ps = R[~R.is_real]
    W = P.HZ + ["transition", "post_effective"]
    acc = ps[ps.window.isin(W)].groupby(["group", "model", "window"]).agg(
        rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))), mae=("err", lambda e: e.abs().mean()), bias=("err", "mean"),
        n=("err", "size")).reset_index()
    base = acc[acc.model == "rv5_only|ols"][["group", "window", "rmse"]].rename(columns={"rmse": "rmse_rv5_ols"})
    acc = acc.merge(base, on=["group", "window"])
    acc["ratio_vs_rv5_ols"] = acc.rmse / acc.rmse_rv5_ols
    C.save_table(acc, "exp2_transfer_accuracy")
    # 표적별 승률 (rv5_only|ols 대비 제곱오차가 작은 비율) + 종목 군집 부트스트랩 CI
    piv = ps.pivot_table(index=["code", "i0", "window"], columns="model", values="err")
    rng = np.random.default_rng(P.SEED)
    rows = []
    codes = np.array(sorted(ps.code.unique()))
    for w in W:
        pw = piv.xs(w, level="window")
        for m in pw.columns:
            d = (pw[m] ** 2 - pw["rv5_only|ols"] ** 2).groupby(level="code").mean()
            bs = [d.reindex(rng.choice(codes, len(codes))).mean() for _ in range(999)]
            rows.append({"model": m, "window": w, "d_mse_vs_rv5_ols": d.mean(), "ci_lo": np.quantile(bs, 0.025),
                         "ci_hi": np.quantile(bs, 0.975), "share_codes_better": float((d < 0).mean())})
    C.save_table(pd.DataFrame(rows), "exp2_transfer_dmse_cluster_ci")
    fpr = loo_audit(ps[ps.group == "treated"], P.FCFG["rv5"]["cutoff_step"], methods=["neff", "strict"], label="treated")
    C.save_table(fpr, "exp2_transfer_fpr_loo")
    real = R[R.is_real & R.window.isin(["transition", "post_effective", "h10", "h20", "h30", "h61"])]
    real = real.assign(effect_pct=100 * (np.exp(real.err) - 1))[["code", "model", "window", "effect_pct"]]
    C.save_table(real[real.code.isin(P.UNITS3)], "exp2_transfer_real_effect")
    return acc


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["run", "analysis"]
    if "test" in what:
        import time
        t = time.time()
        r = pd.DataFrame(run_cutoff(150, targets()[:4], P.SEED))
        print(len(r), round(time.time() - t, 1))
        print(r[r.window == "h20"].assign(e=lambda d: (d.ybar - d.predbar).abs()).groupby("model").e.mean().round(3).to_string())
    if "run" in what:
        print(run().shape)
    if "analysis" in what:
        analyze()
