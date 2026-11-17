"""실험 1: Global / Pooled 학습 (donor 175 종목 패널).

비교 (모두 같은 가짜 개입일·같은 사후 창·같은 평가 대상):
  Local      : 종목 하나의 사전 자료만 (1차 확장과 같은 모형, 평가 대상마다 새로 적합)
  G0_raw     : donor 패널, 원 수준 y, 통제 X 만
  G1_norm    : 종목별 사전 평균·sd 로 표준화한 y (변동성 수준 차이 제거)
  G2_feat    : G1 + 종목 특성 (u_mu, u_sd, u_beta, u_r2, u_lmcap; 모두 사전구간으로 계산)
  G2_id      : G2 + 종목 ID 범주 (LightGBM, 표적 종목의 사전 행 포함 = INCL 에서만 의미)
  G3_*       : G2 LOSO 예측 위에 표적 종목 사전 자료로 국소 적응 (recal: a+b·ŷ, resid: 잔차 ridge, boost: 잔차 LGBM 60그루)
학습 모드:
  LOSO : 표적 종목을 뺀 donor 로만 학습 (처치 종목은 원래 pool 밖이라 전체 donor; donor 표적은 5-fold 종목 분할)
  INCL : 표적 종목의 사전 행까지 포함 (사후 y 는 절대 포함하지 않는다)
  POST : (민감도) donor 의 사후 행(i0 ~ i0+H−1)까지 학습에 포함. 표적 자신의 사후 행은 포함하지 않는다.
         실제 발표일에서는 donor 사후 행이 규제 영향을 받았을 수 있다 → 처치 효과 추정치가 얼마나 바뀌는지 보고.
정보 경계: 통제 PCA·종목 특성·표준화는 모두 학습구간(< i0) 으로만. 처치 종목 사후 y 는 어떤 학습에도 없다.
공통 충격·패널 상관: 같은 날 여러 종목 행은 독립이 아니다 → 하이퍼파라미터 CV 는 날짜 블록 분할, 패널 잔차 상관으로
설계효과(design effect) = 1 + (N−1)·ρ̄ 와 유효 종목 수를 보고한다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.rq2.phase1.ml_followup import common as C
from src.rq2.phase1.ml_followup import panel as P

G = P.FCFG["global"]
NJ = P.FCFG["n_jobs"]
EXP = "exp1"
UF = G["unit_features"]
FE = P.FEATS
LOCAL_MODELS = ["did_gap", "ols", "ridge", "enet", "lgbm", "xgb", "rf", "harx", "bsts"]


# ================================================================ 패널
def panel_at(i0: int, H: int, units: list[str], pool_name="main", members=None) -> pd.DataFrame:
    lv = P.rv5_wide()
    idx = lv.index
    out = []
    for u in units:
        X = P.unit_X(i0, u, pool_name, members)[FE]
        df = pd.concat([lv[u].rename("y"), X], axis=1).iloc[:i0 + H]
        df["t"] = np.arange(len(df))
        df = df.dropna()
        n_tr = int((df.t < i0).sum())
        if n_tr < 40:
            continue
        f = P.unit_features(df["y"], df[FE], n_tr, u)
        for k, v in f.items():
            df[k] = v
        df["code"] = u
        out.append(df)
    D = pd.concat(out)
    D["date"] = idx[D.t.values]
    D["z"] = (D.y - D.u_mu) / D.u_sd
    D["u_lmcap"] = D.u_lmcap.fillna(D.u_lmcap.median())
    return D.reset_index(drop=True)


def date_splits(t: np.ndarray, k: int = 4):
    """날짜 블록 전진 분할 (같은 날 행은 같은 쪽)."""
    ud = np.unique(t)
    edges = np.linspace(len(ud) * 0.4, len(ud), k + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        tr, te = np.isin(t, ud[:a]), np.isin(t, ud[a:b])
        yield np.flatnonzero(tr), np.flatnonzero(te)


def make_global(algo: str, seed: int, t_train: np.ndarray):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if algo == "ridge":
        from sklearn.linear_model import RidgeCV
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 22), cv=list(date_splits(t_train))))
    if algo == "enet":
        from sklearn.linear_model import ElasticNetCV
        return make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.1, .5, .9, 1], alphas=np.logspace(-4, 0, 15),
                                                            cv=list(date_splits(t_train)), max_iter=5000))
    if algo == "rf":
        return C.make_est("rf_panel", seed)
    return C.make_est(algo, seed)


def feats_for(variant: str) -> list[str]:
    return FE if variant in ("G0_raw", "G1_norm") else FE + UF


def target_for(variant: str) -> str:
    return "y" if variant == "G0_raw" else "z"


def _fit(algo, variant, Dtr, seed, cat_id=False):
    f = feats_for(variant)
    Xtr = Dtr[f].values
    if cat_id:
        import lightgbm as lgb
        codes = pd.Categorical(Dtr.code)
        Xdf = Dtr[f].copy()
        Xdf["uid"] = codes
        m = C.make_est("lgbm", seed)
        m.fit(Xdf, Dtr[target_for(variant)].values, categorical_feature=["uid"])
        return m, codes.categories
    m = make_global(algo, seed, Dtr.t.values)
    m.fit(Xtr, Dtr[target_for(variant)].values)
    return m, None


def _predict(m, variant, Dte, cats=None):
    f = feats_for(variant)
    if cats is not None:
        Xdf = Dte[f].copy()
        Xdf["uid"] = pd.Categorical(Dte.code, categories=cats)
        p = m.predict(Xdf)
    else:
        p = m.predict(Dte[f].values)
    return p if variant == "G0_raw" else Dte.u_mu.values + Dte.u_sd.values * p


def _adapt(kind, zhat_tr, z_tr, X_tr, zhat_po, X_po, seed):
    """표적 종목의 사전 행으로 G2 LOSO 예측을 국소 보정 (z 척도)."""
    if kind == "recal":
        Z = np.column_stack([np.ones(len(zhat_tr)), zhat_tr])
        b = np.linalg.lstsq(Z, z_tr, rcond=None)[0]
        return b[0] + b[1] * zhat_po
    r = z_tr - zhat_tr
    if kind == "resid":
        m = C.make_est("ridge", seed)
    else:
        a = G["adapt"]
        m = C.make_est("lgbm", seed, {"n_estimators": a["boost_trees"], "learning_rate": a["boost_lr"],
                                      "num_leaves": a["boost_leaves"], "min_child_samples": 10})
    m.fit(X_tr, r)
    return zhat_po + m.predict(X_po)


def _wrows(Dte_u, pred, extra):
    n_tr = int((Dte_u.t < extra["i0"]).sum())
    yp = Dte_u.y.values[n_tr:]
    return [{**extra, **r} for r in C.window_rows(yp, pred[n_tr:])]


def run_cutoff(i0: int, H: int, eval_donors: list[str], algos: list[str], seed: int, post_sens=True) -> list[dict]:
    warnings.filterwarnings("ignore")
    donors = P.pool("main")
    D = panel_at(i0, H, donors + P.UNITS3)
    rng = np.random.default_rng(seed + i0)
    perm = rng.permutation(sorted(donors))
    fold = {u: k % 5 for k, u in enumerate(perm)}
    pre = D[D.t < i0]
    don_pre = pre[pre.code.isin(donors)]
    targets = P.UNITS3 + list(eval_donors)
    rows = []
    Dt = {u: D[D.code == u] for u in targets}
    meta = {"i0": i0, "H": H}
    for algo in algos:
        for variant in ["G0_raw", "G1_norm", "G2_feat"]:
            m_all, _ = _fit(algo, variant, don_pre, seed)
            m_inc, _ = _fit(algo, variant, pre[pre.code.isin(donors + P.UNITS3)], seed)
            m_fold = {}
            for k in range(5):
                m_fold[k], _ = _fit(algo, variant, don_pre[don_pre.code.map(fold) != k], seed)
            for u in targets:
                du = Dt[u]
                if u in P.UNITS3:
                    loso, inc = m_all, m_inc
                else:
                    loso, inc = m_fold[fold[u]], m_all
                p_loso = _predict(loso, variant, du)
                p_inc = _predict(inc, variant, du)
                base = {**meta, "code": u, "algo": algo, "group": "treated" if u in P.UNITS3 else "donor"}
                rows += _wrows(du, p_loso, {**base, "model": f"{variant}|LOSO|{algo}"})
                rows += _wrows(du, p_inc, {**base, "model": f"{variant}|INCL|{algo}"})
                if variant == "G2_feat":
                    tr = (du.t < i0).values
                    zhat = (p_loso - du.u_mu.values) / du.u_sd.values
                    Xu = du[FE].values
                    for kind in ["recal", "resid", "boost"]:
                        zp = _adapt(kind, zhat[tr], du.z.values[tr], Xu[tr], zhat, Xu, seed)
                        rows += _wrows(du, du.u_mu.values + du.u_sd.values * zp,
                                       {**base, "model": f"G3_{kind}|LOSO|{algo}"})
            # POST 민감도: donor 사후 행(i0 ~ i0+H-1) 포함. 표적 자신의 행은 제외(fold 또는 처치)
            if post_sens and variant == "G2_feat" and algo in ("lgbm", "ridge"):
                dpost = D[D.code.isin(donors)]
                mp_all, _ = _fit(algo, variant, dpost, seed)
                mp_fold = {k: _fit(algo, variant, dpost[dpost.code.map(fold) != k], seed)[0] for k in range(5)}
                for u in targets:
                    du = Dt[u]
                    mm = mp_all if u in P.UNITS3 else mp_fold[fold[u]]
                    rows += _wrows(du, _predict(mm, variant, du),
                                   {**meta, "code": u, "algo": algo, "group": "treated" if u in P.UNITS3 else "donor",
                                    "model": f"G2_feat|POST|{algo}"})
    # 종목 ID (LGBM, INCL)
    if "lgbm" in algos:
        m_id, cats = _fit("lgbm", "G2_feat", pre[pre.code.isin(donors + P.UNITS3)], seed, cat_id=True)
        for u in targets:
            du = Dt[u]
            rows += _wrows(du, _predict(m_id, "G2_feat", du, cats),
                           {**meta, "code": u, "algo": "lgbm", "group": "treated" if u in P.UNITS3 else "donor",
                            "model": "G2_id|INCL|lgbm"})
    # 패널 잔차 상관 (G2 LGBM, donor 사전 in-sample) → 설계효과
    if "lgbm" in algos:
        m2, _ = _fit("lgbm", "G2_feat", don_pre, seed)
        res = don_pre.assign(r=don_pre.y - _predict(m2, "G2_feat", don_pre))
        R = res.pivot_table(index="t", columns="code", values="r")
        cm = R.corr().values
        rho = np.nanmean(cm[np.triu_indices_from(cm, 1)])
        rows.append({"i0": i0, "model": "_design", "rho_bar": rho, "n_units": R.shape[1],
                     "design_effect": 1 + (R.shape[1] - 1) * rho, "n_rows": len(don_pre)})
    return rows


def _local_task(code, i0, H, models, seed):
    warnings.filterwarnings("ignore")
    y, X, n = P.rv5_task(code, i0, H)
    rows = []
    for m in models:
        o = C.fit_predict(m, y.values, X.values, n, seed=seed)
        for r in C.window_rows(y.values[n:], o["pred"]):
            rows.append({"i0": i0, "H": H, "code": code, "algo": m,
                         "group": "treated" if code in P.UNITS3 else "donor", "model": f"Local|-|{m}", **r})
    return rows


def eval_donors() -> list[str]:
    rng = np.random.default_rng(P.SEED)
    return sorted(rng.choice(P.pool("main"), P.FCFG["rv5"]["donor_eval_n"], replace=False).tolist())


def run_pseudo(force=False) -> pd.DataFrame:
    def go():
        ed = eval_donors()
        cuts = P.rv5_cutoffs()
        res = Parallel(n_jobs=NJ, verbose=5)(
            delayed(run_cutoff)(i0, P.rv5_horizon_max(i0), ed, G["algos"], P.SEED) for i0 in cuts)
        return pd.DataFrame([r for rr in res for r in rr])
    return C.cached(EXP, "global_pseudo", go, force)


def run_local_pseudo(force=False) -> pd.DataFrame:
    def go():
        ed = eval_donors()
        tasks = [(u, i0, P.rv5_horizon_max(i0)) for i0 in P.rv5_cutoffs() for u in P.UNITS3 + ed]
        res = Parallel(n_jobs=NJ, batch_size=4, verbose=5)(
            delayed(_local_task)(u, i0, H, LOCAL_MODELS, P.SEED) for u, i0, H in tasks)
        return pd.DataFrame([r for rr in res for r in rr])
    return C.cached(EXP, "local_pseudo", go, force)


def run_real(force=False) -> pd.DataFrame:
    """실제 발표일: donor 175 + 처치 3 모두 표적 (space 순열용). 학습은 발표일 전, POST 는 donor 사후 행 포함 민감도."""
    def go():
        i0 = P.real_index()
        donors = P.pool("main")
        # donor 175 를 11 조각으로 나눠 병렬 (조각마다 같은 전역 모형을 다시 적합 — 시드 고정이라 결과 동일)
        chunks = np.array_split(np.array(donors), NJ)
        res = Parallel(n_jobs=NJ, verbose=5)(
            delayed(run_cutoff)(i0, 71, list(ch), ["ridge", "lgbm", "xgb"], P.SEED, True) for ch in chunks)
        rows = [r for rr in res for r in rr]
        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=[c for c in ["model", "code", "window"] if c in df], keep="first")
        loc = Parallel(n_jobs=NJ, batch_size=8, verbose=5)(
            delayed(_local_task)(u, i0, 71, ["did_gap", "ols", "lgbm", "harx", "ridge", "xgb"], P.SEED)
            for u in donors + P.UNITS3)
        return pd.concat([df, pd.DataFrame([r for rr in loc for r in rr])], ignore_index=True)
    return C.cached(EXP, "global_real", go, force)


# ================================================================ 데이터 양 vs 공유 (K donor, 행 수 맞춤)
def _k_cutoff(i0, H, ks, draws, seed, algos=("lgbm", "ridge")):
    warnings.filterwarnings("ignore")
    donors = P.pool("main")
    D = panel_at(i0, H, donors + P.UNITS3)
    pre = D[D.t < i0]
    rng = np.random.default_rng(seed + 7 * i0)
    rows = []
    # 유사 종목: 처치 2 종목 평균 log RV 와의 사전 상관 상위 K
    piv = pre.pivot_table(index="t", columns="code", values="y")
    tm = piv[P.TREATED].mean(1)
    sim = piv[donors].corrwith(tm).sort_values(ascending=False).index.tolist()
    n_local = int((pre.code == "005930").sum())
    specs = []
    for K in ks:
        for d in range(draws if K < len(donors) else 1):
            specs.append((f"rand{K}", d, list(rng.choice(donors, K, replace=False))))
        if K < len(donors):
            specs.append((f"sim{K}", 0, sim[:K]))
    for d in range(G["rowmatch_draws"]):
        specs.append(("rowmatch175", d, None))
    for name, d, mem in specs:
        if mem is None:
            sub = pre[pre.code.isin(donors)]
            sub = sub.iloc[rng.choice(len(sub), n_local, replace=False)]
        else:
            sub = pre[pre.code.isin(mem)]
        for algo in algos:
            m, _ = _fit(algo, "G2_feat", sub, seed)
            for u in P.UNITS3:
                du = D[D.code == u]
                rows += _wrows(du, _predict(m, "G2_feat", du),
                               {"i0": i0, "H": H, "code": u, "algo": algo, "group": "treated", "spec": name,
                                "draw": d, "n_rows": len(sub), "K": len(set(sub.code)),
                                "model": f"G2_feat|{name}|{algo}"})
    return rows


def run_k_subsets(force=False) -> pd.DataFrame:
    def go():
        res = Parallel(n_jobs=NJ, verbose=5)(
            delayed(_k_cutoff)(i0, P.rv5_horizon_max(i0), P.FCFG["sample"]["global_k"], P.FCFG["sample"]["global_k_draws"],
                               P.SEED) for i0 in P.rv5_cutoffs())
        return pd.DataFrame([r for rr in res for r in rr])
    return C.cached(EXP, "global_k", go, force)


# ================================================================ 분석


def _acc(df, keys):
    return df.groupby(keys).agg(rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))), mae=("err", lambda e: e.abs().mean()),
                                bias=("err", "mean"), n=("err", "size")).reset_index()


def cluster_dmse(df, base_model, keys=("group", "window"), B=999, seed=0):
    """제곱오차 차이(모형 − 기준)를 종목별 평균 → 종목 군집 부트스트랩 CI. 겹치는 가짜 창을 독립으로 세지 않는다."""
    rng = np.random.default_rng(seed)
    piv = df.pivot_table(index=list(keys) + ["code", "i0"], columns="model", values="err")
    rows = []
    for k, g in piv.groupby(level=list(keys)):
        g = g.droplevel(list(keys))
        codes = np.array(sorted(g.index.get_level_values("code").unique()))
        for m in g.columns:
            d = (g[m] ** 2 - g[base_model] ** 2).groupby(level="code").mean().dropna()
            if len(d) == 0:
                continue
            bs = [d.reindex(rng.choice(d.index, len(d))).mean() for _ in range(B)] if len(d) > 2 else [np.nan]
            rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), "model": m, "base": base_model,
                         "d_mse": d.mean(), "ci_lo": np.nanquantile(bs, 0.025), "ci_hi": np.nanquantile(bs, 0.975),
                         "share_codes_better": float((d < 0).mean()), "n_codes": len(d)})
    return pd.DataFrame(rows)


def analyze():
    from src.rq2.phase1.ml_followup.exp5_audit import loo_audit
    W = P.HZ + ["transition", "post_effective"]
    g = run_pseudo()
    design = g[g.model == "_design"]
    g = g[g.model != "_design"]
    loc = run_local_pseudo()
    A = pd.concat([g, loc], ignore_index=True)
    A = A[A.window.isin(W)].assign(err=lambda d: d.ybar - d.predbar)
    acc = _acc(A, ["group", "model", "window"])
    base = acc[acc.model == "Local|-|ols"][["group", "window", "rmse"]].rename(columns={"rmse": "rmse_local_ols"})
    acc = acc.merge(base, on=["group", "window"])
    acc["ratio_vs_local_ols"] = acc.rmse / acc.rmse_local_ols
    C.save_table(acc, "exp1_global_accuracy")
    C.save_table(cluster_dmse(A, "Local|-|ols", seed=P.SEED), "exp1_global_dmse_vs_local_ols")
    # donor 표적: 종목별로 Global 이 Local 보다 나은지 (Q3: 특정 종목에서만?)
    dd = A[A.window.isin(["h20", "h61"])]
    piv = dd.pivot_table(index=["window", "code", "i0"], columns="model", values="err")
    per = []
    for (w, u), q in piv.groupby(level=["window", "code"]):
        for m in ["G2_feat|LOSO|lgbm", "G2_feat|LOSO|ridge", "G3_recal|LOSO|lgbm", "G2_id|INCL|lgbm"]:
            if m in q:
                per.append({"window": w, "code": u, "model": m, "group": "treated" if u in P.UNITS3 else "donor",
                            "d_mse_vs_local_ols": float(np.mean(q[m] ** 2 - q["Local|-|ols"] ** 2)),
                            "d_mse_vs_local_best_fixed_lgbm": float(np.mean(q[m] ** 2 - q["Local|-|lgbm"] ** 2))})
    C.save_table(pd.DataFrame(per), "exp1_global_per_code")
    # FPR (1차 확장 방식 LOO, 처치 3 종목) — neff vs strict
    fpr = loo_audit(A[A.group == "treated"], P.FCFG["rv5"]["cutoff_step"], methods=["neff", "strict"], label="treated")
    C.save_table(fpr, "exp1_global_fpr_loo")
    # 설계효과
    C.save_table(design[["i0", "rho_bar", "n_units", "design_effect", "n_rows"]].assign(
        n_eff_units=lambda d: d.n_units / d.design_effect), "exp1_panel_design_effect")
    # K 곡선 (데이터 양 vs 종목 공유)
    K = run_k_subsets()
    K = K[K.window.isin(W)].assign(err=lambda d: d.ybar - d.predbar)
    kc = K.groupby(["spec", "algo", "window"]).agg(K=("K", "mean"), n_rows=("n_rows", "mean"),
                                                   rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))),
                                                   bias=("err", "mean")).reset_index()
    C.save_table(kc, "exp1_global_k_curve")
    # 실제 발표일
    R = run_real()
    R = R[R.model != "_design"].assign(err=lambda d: d.ybar - d.predbar)
    R = R[R.window.isin(W)]
    rng = np.random.default_rng(P.SEED)
    rows = []
    donors = set(P.pool("main"))
    for (m, w), q in R.groupby(["model", "window"]):
        dn = q[q.code.isin(donors)]
        for u in P.UNITS3:
            t = q[q.code == u]
            if t.empty:
                continue
            est = float(t.err.iloc[0])
            r = {"model": m, "window": w, "code": u, "effect_pct": 100 * (np.exp(est) - 1), "n_space": len(dn),
                 "p_space": (1 + np.sum(np.abs(dn.err) >= abs(est))) / (1 + len(dn)) if len(dn) else np.nan}
            ps = A[(A.model == m) & (A.code == u) & (A.window == w)]
            a, b = P.WIN[w]
            for meth in ("neff", "strict"):
                if len(ps) and w != "post_effective":
                    o = C.null_p(est, ps.err.values, ps.i0.values, b - a, P.FCFG["rv5"]["cutoff_step"], meth, 3, 4999, rng)
                    r[f"p_time_{meth}"] = o["p"]
            rows.append(r)
    real = pd.DataFrame(rows)
    C.save_table(real, "exp1_global_real_effect")
    # POST 민감도 (실제 발표일): donor 사후 행 포함 여부에 따른 처치 효과 차이
    pv = real[real.model.isin(["G2_feat|LOSO|lgbm", "G2_feat|POST|lgbm", "G2_feat|LOSO|ridge", "G2_feat|POST|ridge"])]
    C.save_table(pv, "exp1_post_sensitivity_real")
    return acc


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["local", "pseudo", "k", "real"]
    if "local" in what:
        print(run_local_pseudo().shape)
    if "pseudo" in what:
        print(run_pseudo().shape)
    if "k" in what:
        print(run_k_subsets().shape)
    if "real" in what:
        print(run_real().shape)
    if "analysis" in what:
        analyze()
