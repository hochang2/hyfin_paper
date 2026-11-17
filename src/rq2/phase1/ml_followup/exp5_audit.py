"""실험 5: 통계적 추론 감사 (Statistical Inference Audit).

질문: 1차 ML 확장의 time-보정 귀무분포(겹치는 가짜 창 + n_eff), MDE(2.8·s0), h10 기준 모형 선정,
      donor 175 space 순열 검정이 실제로 명목 수준(5%)을 지키는가?

rv5 표본(261일)만으로는 이 질문에 답할 수 없다 (가짜 개입일 32개가 사실상 한 덩어리). 그래서
F1 일별 GK(2015~)에서 rv5 와 같은 구조를 수십 번 재현한다 ('조밀 GK 모사'):
  - 가짜 개입일 3일 간격, 이동창 학습 140일(rv5 가짜 개입일 학습 길이 평균), 사후 경로 71일.
  - '블록' = 연속 가짜 개입일 32개 + 그 다음 격자점 1개(= 모사 실제 개입일 i*). 블록 안 가짜 창은 i* 전에 끝나는 것만 쓴다
    (rv5 에서 h61 창이 i0 ≤ 129 에서만 생기는 것과 같은 절단).
  - 모사 실제 개입일에는 효과가 없다 → |오차 − m0| > crit 비율 = 실제 FPR. 효과 δ 를 더하면 검정력.
블록은 30일씩 밀며 반복한다. 이웃 블록은 학습·평가 자료를 공유하므로 블록 수가 독립 표본 수는 아니다
(REPORT 에 유효 블록 수 = 61일 창 기준 비겹침 블록 수를 함께 적는다).

space 감사: (a) rv5 실제 표본의 32 가짜 개입일 + 실제일에 donor 175 + 처치 3 전부 적합, (b) GK 에서 30일마다 donors_long 131 + 3.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from src.rq2.phase1.ml_followup import common as C
from src.rq2.phase1.ml_followup import panel as P

A = P.FCFG["audit"]
NJ = P.FCFG["n_jobs"]
EXP = "exp5"


# ================================================================ 적합 (캐시)
def _fit_task(code, i0, y, X, n_tr, models, seed):
    warnings.filterwarnings("ignore")
    rows = []
    yp = y[n_tr:]
    for m in models:
        o = C.fit_predict(m, y, X, n_tr, seed=seed, native=m in ("ols", "bsts"))
        pre = float(np.std(y[:n_tr] - o["fit"], ddof=1)) if o["fit"] is not None else np.nan
        for r in C.window_rows(yp, o["pred"], o["draws"]):
            rows.append({"model": m, "code": code, "i0": i0, "n_tr": n_tr, "H": len(yp), "pre_rmse": pre, **r})
    return rows


def gk_dense_cutoffs() -> list[int]:
    idx = P.gk_long().index
    a = max(idx.searchsorted(pd.Timestamp(A["gk_first_date"])), A["gk_train_len"])
    lst = idx.searchsorted(P.CFG["timeline"]["listing"])
    return list(range(a, lst - 71 + 1, A["gk_step"]))


def run_gk_dense(force=False) -> pd.DataFrame:
    def go():
        tasks = []
        for code in P.UNITS3:
            for i0 in gk_dense_cutoffs():
                y, X, n = P.gk_task(code, i0, A["gk_train_len"], 71)
                if n < 100 or len(y) - n < 61:
                    continue
                tasks.append((code, i0, y.values, X.values, n))
        res = Parallel(n_jobs=NJ, batch_size=4, verbose=2)(
            delayed(_fit_task)(c, i, y, X, n, A["gk_models"], P.SEED) for c, i, y, X, n in tasks)
        return pd.DataFrame([r for rr in res for r in rr])
    return C.cached(EXP, "gk_dense", go, force)


SPACE_MODELS = ["did_gap", "ols", "harx", "lgbm"]


def run_gk_space(force=False) -> pd.DataFrame:
    """GK: 30일마다 donors_long 131 + 처치 3 의 OLS 등 (donor 는 자기 제외 don_mean)."""
    def go():
        dl, Y, _ = P.gk_units()
        cut = gk_dense_cutoffs()[:: A["space_every"] // A["gk_step"]]
        tasks = []
        for code in P.UNITS3 + dl:
            for i0 in cut:
                y, X, n = P.gk_task(code, i0, A["gk_train_len"], 71)
                if n < 100 or len(y) - n < 61:
                    continue
                tasks.append((code, i0, y.values, X.values, n))
        res = Parallel(n_jobs=NJ, batch_size=16, verbose=1)(
            delayed(_fit_task)(c, i, y, X, n, SPACE_MODELS, P.SEED) for c, i, y, X, n in tasks)
        return pd.DataFrame([r for rr in res for r in rr])
    return C.cached(EXP, "gk_space", go, force)


def run_rv5_space(force=False) -> pd.DataFrame:
    """rv5: 32 가짜 개입일(사후 창은 발표일 전 절단) + 실제 발표일, donor 175 + 처치 3."""
    def go():
        lv = P.rv5_wide()
        tasks = []
        units = P.pool("main") + P.UNITS3
        for i0 in P.rv5_cutoffs() + [P.real_index()]:
            H = P.rv5_horizon_max(i0) if i0 < P.real_index() else 71
            for code in units:
                y, X, n = P.rv5_task(code, i0, H)
                if len(y) - n < 5:
                    continue
                tasks.append((code, i0, y.values, X.values, n))
        res = Parallel(n_jobs=NJ, batch_size=16, verbose=1)(
            delayed(_fit_task)(c, i, y, X, n, SPACE_MODELS, P.SEED) for c, i, y, X, n in tasks)
        return pd.DataFrame([r for rr in res for r in rr])
    return C.cached(EXP, "rv5_space", go, force)


# ================================================================ 분석
DELTA_PCT = np.arange(-5, -65, -5)
WINS_ALL = ["h5", "h10", "h20", "h30", "h61", "transition", "post_effective"]


def _err(D):
    return D.assign(err=D.ybar - D.predbar)


def block_audit(D: pd.DataFrame, methods=None) -> pd.DataFrame:
    """조밀 GK 의 각 블록에서 '모사 실제 개입일' 오차를 가짜 창 귀무분포로 검정."""
    methods = methods or C.METHODS
    D = _err(D)
    cl = np.array(sorted(D.i0.unique()))
    nb, sk = A["block_n_cut"], A["block_slide"] // A["gk_step"]
    rng = np.random.default_rng(P.SEED)
    rows = []
    for (m, code, w), g in D.groupby(["model", "code", "window"]):
        s = g.set_index("i0")
        a, b = P.WIN[w]
        L = b - a
        for k0 in range(0, len(cl) - nb, sk):
            ps, it = cl[k0:k0 + nb], cl[k0 + nb]
            if it not in s.index:
                continue
            use = ps[(ps + b <= it) & np.isin(ps, s.index)]
            e = s.err.reindex(use).values
            r0 = s.loc[it]
            base = {"model": m, "code": code, "window": w, "block": int(k0), "i_test": int(it),
                    "e_star": float(r0.err), "n_ps": len(use), "s0_all": float(np.std(e, ddof=1)) if len(e) > 2 else np.nan}
            for meth in methods:
                m0, crit, n = C.null_test(e, use, L, A["gk_step"], meth, A["min_eff"], A["bootstrap_reps"], rng)
                rows.append({**base, "method": meth, "m0": m0, "crit": crit, "n_used": n})
            if "cf_q025" in s and np.isfinite(r0.get("cf_q025", np.nan)):
                # native: 효과 구간 [ybar−q975, ybar−q025] (+δ). 0 이 밖이면 기각 ⇔ |mid + δ| > 반폭
                # 공통식 |e_star + δ − m0| > crit 에 맞추려고 m0 = e_star − mid, crit = 반폭
                lo, hi = r0.ybar - r0.cf_q975, r0.ybar - r0.cf_q025
                rows.append({**base, "method": "native", "m0": float(r0.err) - (lo + hi) / 2,
                             "crit": (hi - lo) / 2, "n_used": 1})
    return pd.DataFrame(rows)


def summarize_audit(R: pd.DataFrame) -> pd.DataFrame:
    R = R.copy()
    R["testable"] = np.isfinite(R.crit)
    d = R.e_star - R.m0
    R["rej0"] = np.where(R.testable, np.abs(d) > R.crit, np.nan)
    for p in DELTA_PCT:
        R[f"pw{p}"] = np.where(R.testable, np.abs(d + np.log1p(p / 100)) > R.crit, np.nan)
    rows = []
    for (m, w, meth), g in R.groupby(["model", "window", "method"]):
        t = g[g.testable]
        a, b = P.WIN[w]
        nb = g.block.nunique()
        r = {"model": m, "window": w, "method": meth, "n_tests": len(g), "testable_share": g.testable.mean(),
             "fpr": t.rej0.mean() if len(t) else np.nan, "coverage": 1 - t.rej0.mean() if len(t) else np.nan,
             "n_blocks": nb, "n_eff_tests": round(2 * nb * min(1.0, A["block_slide"] / max(b - a, 1)), 1),
             "mean_n_used": t.n_used.mean() if len(t) else np.nan,
             "sd_true": (t.e_star - t.m0).std(ddof=1) if len(t) > 2 else np.nan,
             "s0_mean": t.s0_all.mean() if len(t) else np.nan,
             "crit_mean": t.crit.mean() if len(t) else np.nan}
        for p in DELTA_PCT:
            r[f"power{p}"] = t[f"pw{p}"].mean() if len(t) else np.nan
        hit = [abs(p) for p in DELTA_PCT if len(t) and t[f"pw{p}"].mean() >= 0.8]
        r["mde_emp_pct"] = min(hit) if hit else np.nan
        r["mde_analytic_pct"] = -np.expm1(-2.8 * r["s0_mean"]) * 100 if np.isfinite(r["s0_mean"]) else np.nan
        r["mde_required_pct"] = -np.expm1(-(r["crit_mean"] + 0.84 * r["sd_true"])) * 100 if np.isfinite(r["sd_true"]) else np.nan
        rows.append(r)
    S = pd.DataFrame(rows)
    se = np.sqrt(S.fpr * (1 - S.fpr) / S.n_eff_tests.clip(lower=1))
    S["fpr_lo"], S["fpr_hi"] = (S.fpr - 1.96 * se).clip(lower=0), (S.fpr + 1.96 * se).clip(upper=1)
    return S


def loo_audit(W: pd.DataFrame, step: int, methods=None, label="") -> pd.DataFrame:
    """1차 확장 방식(같은 표본 안에서 각 가짜 개입일을 나머지 비겹침 창으로 검정)의 FPR. 방법별 비교용."""
    methods = methods or C.METHODS
    W = _err(W)
    rng = np.random.default_rng(P.SEED)
    rows = []
    for (m, code, w), g in W.groupby(["model", "code", "window"]):
        i0 = g.i0.values
        e = g.err.values
        a, b = P.WIN[w]
        L = b - a
        for k in range(len(g)):
            keep = (np.abs(i0 - i0[k]) >= L) & np.isfinite(e)
            for meth in methods:
                m0, crit, n = C.null_test(e[keep], i0[keep], L, step, meth, A["min_eff"], 499, rng)
                rows.append({"model": m, "code": code, "window": w, "i0": i0[k], "method": meth, "e_star": e[k],
                             "m0": m0, "crit": crit, "n_used": n, "s0_all": np.nan, "block": i0[k]})
    R = pd.DataFrame(rows)
    R["testable"] = np.isfinite(R.crit)
    R["rej0"] = np.where(R.testable, np.abs(R.e_star - R.m0) > R.crit, np.nan)
    out = R.groupby(["model", "window", "method"]).agg(testable_share=("testable", "mean"), fpr=("rej0", "mean"),
                                                         n_used=("n_used", "mean")).reset_index()
    out["sample"] = label
    return out


# ---------------------------------------------------------------- 선정 규칙 감사
def selection_audit(D: pd.DataFrame, sel_window="h10", models=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """블록마다 1차 확장 규칙(가짜 창 FPR ≤ 0.10 중 2.8·s0 최소)을 sel_window 로 적용해 모형을 고르고,
    모사 실제 개입일의 h10/h30/h61/post_effective 오차를 고정 모형·사후 최적과 비교."""
    D = _err(D)
    models = models or sorted(D.model.unique())
    cl = np.array(sorted(D.i0.unique()))
    nb, sk = A["block_n_cut"], A["block_slide"] // A["gk_step"]
    a, b = P.WIN[sel_window]
    L = b - a
    piv = D.pivot_table(index=["code", "window", "i0"], columns="model", values="err")
    rows = []
    for code in sorted(D.code.unique()):
        for k0 in range(0, len(cl) - nb, sk):
            ps, it = cl[k0:k0 + nb], cl[k0 + nb]
            use = ps[ps + b <= it]
            cand = {}
            for m in models:
                try:
                    e = piv.loc[(code, sel_window)].reindex(use)[m].values
                except KeyError:
                    continue
                ok = np.isfinite(e)
                e2, i2 = e[ok], use[ok]
                if C.n_eff_greedy(i2, L) < A["min_eff"]:
                    continue
                rej = []
                for k in range(len(e2)):
                    keep = np.abs(i2 - i2[k]) >= L
                    m0, crit, n = C.null_test(e2[keep], i2[keep], L, A["gk_step"], "neff", A["min_eff"])
                    if np.isfinite(crit):
                        rej.append(abs(e2[k] - m0) > crit)
                fpr = np.mean(rej) if rej else np.nan
                cand[m] = (fpr, 2.8 * np.std(e2, ddof=1))
            ok = {m: v for m, v in cand.items() if np.isfinite(v[0]) and v[0] <= 0.10}
            sel = min(ok, key=lambda m: ok[m][1]) if ok else None
            rank_s0 = pd.Series({m: v[1] for m, v in cand.items()})
            for w in ["h10", "h30", "h61", "post_effective"]:
                try:
                    et = piv.loc[(code, w, it)]
                except KeyError:
                    continue
                r = {"code": code, "block": int(k0), "i_test": int(it), "window": w, "selected": sel,
                     "e_sel": et.get(sel, np.nan) if sel else np.nan, "e_best": et.abs().min(),
                     "best_model": et.abs().idxmin()}
                for m in models:
                    r[f"e_{m}"] = et.get(m, np.nan)
                common = rank_s0.index.intersection(et.dropna().index)
                r["spearman_s0_vs_abs_err"] = stats.spearmanr(rank_s0[common], et[common].abs())[0] if len(common) > 3 else np.nan
                rows.append(r)
    S = pd.DataFrame(rows)
    summ = []
    for w, g in S.groupby("window"):
        r = {"window": w, "n": len(g), "rmse_selected": np.sqrt(np.nanmean(g.e_sel ** 2)),
             "rmse_oracle": np.sqrt(np.mean(g.e_best ** 2)), "spearman_mean": g.spearman_s0_vs_abs_err.mean()}
        for m in models:
            r[f"rmse_{m}"] = np.sqrt(np.nanmean(g[f"e_{m}"] ** 2))
        summ.append(r)
    sel_freq = S[S.window == "h10"].selected.value_counts(normalize=True).rename("share").reset_index()
    return pd.DataFrame(summ), sel_freq, S


def rank_consistency(D: pd.DataFrame) -> pd.DataFrame:
    """모형별 전체 과제 RMSE 를 창마다 계산하고 h10 순위와 다른 창 순위의 Spearman."""
    D = _err(D)
    T = D.groupby(["window", "model"]).err.apply(lambda v: np.sqrt(np.mean(v ** 2))).unstack(0)
    out = T.copy()
    for w in T.columns:
        out.loc["spearman_vs_h10", w] = stats.spearmanr(T["h10"], T[w])[0]
    return out.reset_index()


# ---------------------------------------------------------------- space 순열 감사
def space_audit(S: pd.DataFrame, donors: list[str], treated=P.UNITS3, exclude_i0=()) -> pd.DataFrame:
    """treated 단위의 space 순열 p (원 오차, pre-RMSE 표준화) 와 donor 중 |오차| 백분위. 효과 없는 시점이면 FPR."""
    S = _err(S)
    so = S[S.model == "ols"].set_index(["code", "i0"]).pre_rmse.groupby(level=[0, 1]).first()
    S = S.join(so.rename("scale"), on=["code", "i0"])
    S["z"] = S.err / S.scale
    rows = []
    for (m, i0, w), g in S[~S.i0.isin(exclude_i0)].groupby(["model", "i0", "window"]):
        d = g[g.code.isin(donors)]
        if len(d) < 20:
            continue
        for v, col in [("raw", "err"), ("stud", "z")]:
            ed = d[col].values
            c = ed.mean()
            for u in treated:
                eu = g.loc[g.code == u, col]
                if eu.empty:
                    continue
                eu = float(eu.iloc[0])
                p = (np.sum(np.abs(ed - c) >= abs(eu - c)) + 1) / (len(ed) + 1)
                pct = np.mean(np.abs(ed - c) < abs(eu - c))
                rows.append({"model": m, "i0": i0, "window": w, "variant": v, "code": u, "p": p, "abs_pct": pct})
    return pd.DataFrame(rows)


def space_summary(T: pd.DataFrame) -> pd.DataFrame:
    return T.groupby(["model", "window", "variant", "code"]).agg(
        n=("p", "size"), fpr=("p", lambda v: np.mean(v < 0.05)), mean_abs_pct=("abs_pct", "mean")).reset_index()


def exchangeability(real_i0: int) -> pd.DataFrame:
    """실제 발표일 학습구간 기준 단위 특성: 처치 단위가 donor 분포에서 어디에 있는가."""
    rows = []
    for code in P.pool("main") + P.UNITS3:
        y, X, n = P.rv5_task(code, real_i0, 1)
        f = P.unit_features(y, X, n, code)
        f["code"] = code
        rows.append(f)
    F = pd.DataFrame(rows)
    don = F[~F.code.isin(P.UNITS3)]
    out = []
    for u in P.UNITS3:
        r = {"code": u}
        for c in ["u_mu", "u_sd", "u_beta", "u_r2", "u_lmcap"]:
            v = F.loc[F.code == u, c].iloc[0]
            r[c] = v
            r[f"{c}_pctile"] = np.mean(don[c] < v)
        out.append(r)
    return pd.DataFrame(out), F


def err_vs_features(S: pd.DataFrame, F: pd.DataFrame, donors) -> pd.DataFrame:
    """donor 의 가짜 개입일 |오차| 를 단위 특성에 회귀 (Spearman). 교환가능성 위반 정도."""
    S = _err(S)
    S = S[S.code.isin(donors)]
    A_ = S.groupby(["model", "window", "code"]).err.apply(lambda v: np.sqrt(np.mean(v ** 2))).rename("rmse").reset_index()
    A_ = A_.merge(F, on="code")
    rows = []
    for (m, w), g in A_.groupby(["model", "window"]):
        for c in ["u_sd", "u_lmcap", "u_beta", "u_r2", "u_mu"]:
            rho, p = stats.spearmanr(g[c], g.rmse, nan_policy="omit")
            rows.append({"model": m, "window": w, "feature": c, "spearman": rho, "p": p, "n": len(g)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 실제일 재계산 (전/후)
def real_before_after(models: list[str], R_gk: pd.DataFrame | None = None) -> pd.DataFrame:
    """1차 확장 rv5 실제 효과의 time p 를 보정 방법별로 다시 계산. GK 보정 = 조밀 GK 블록에서
    neff 임계값을 몇 배 해야 FPR 5% 가 되는지(κ) 를 같은 모형·창에서 구해 적용한 결과."""
    from src.rq2.phase1.ml_extension import benchmark as B
    _, Wr = B.collect("space", models)
    _, Wp = B.collect("rv5", models)
    Wr, Wp = _err(Wr), _err(Wp)
    donors = set(P.pool("main"))
    rng = np.random.default_rng(P.SEED)
    rows = []
    for m in models:
        for w in ["transition", "post_effective", "h10", "h20", "h30", "h61"]:
            a, b = P.WIN[w]
            L = b - a
            dw = Wr[(Wr.model == m) & (Wr.window == w)]
            if dw.empty:
                continue
            don = dw[dw.code.isin(donors)].err.values
            for u in P.UNITS3:
                ru = dw[dw.code == u]
                ps = Wp[(Wp.model == m) & (Wp.window == w) & (Wp.code == u)].sort_values("i0")
                if ru.empty:
                    continue
                est = float(ru.err.iloc[0])
                r = {"model": m, "window": w, "code": u, "effect_log": est, "effect_pct": np.expm1(est) * 100,
                     "n_pseudo": len(ps)}
                for meth in C.METHODS:
                    o = C.null_p(est, ps.err.values, ps.i0.values, L, 3, meth, A["min_eff"], rng=rng) if len(ps) else {"p": np.nan, "n_used": 0, "m0": np.nan, "crit": np.nan}
                    r[f"p_{meth}"], r[f"n_{meth}"] = o["p"], o["n_used"]
                    if meth == "neff":
                        r["m0"], r["crit_neff"] = o["m0"], o["crit"]
                # GK 보정: 같은 모형이 조밀 GK 에 없으면(xgb, ridge) 같은 창의 전 모형 표준화 통계를 합쳐 쓴다
                g = R_gk[(R_gk.method == "neff") & (R_gk.window == w) & np.isfinite(R_gk.crit)] if R_gk is not None else None
                if g is not None and (g.model == m).any():
                    g = g[g.model == m]
                    r["gkcal_source"] = m
                else:
                    r["gkcal_source"] = "pooled"
                if g is not None and len(g) and np.isfinite(r.get("crit_neff", np.nan)):
                    zz = np.abs(g.e_star - g.m0) / g.crit
                    r["kappa_gk"] = float(np.quantile(zz, 0.95))
                    stat = abs(est - r["m0"]) / r["crit_neff"]
                    r["p_gkcal"] = float((np.sum(zz >= stat) + 1) / (len(zz) + 1))
                else:
                    r["kappa_gk"], r["p_gkcal"] = np.nan, np.nan
                c = don.mean()
                r["p_space_raw"] = (np.sum(np.abs(don - c) >= abs(est - c)) + 1) / (len(don) + 1)
                rows.append(r)
    return pd.DataFrame(rows)


def run_analysis():
    D = run_gk_dense()
    R = block_audit(D)
    R.to_parquet(C.cpath(EXP, "block_audit"))
    S = summarize_audit(R)
    C.save_table(S, "exp5_time_null_audit_gk")
    # 1차 확장 방식의 같은-표본 LOO FPR: rv5 (실제 표본) 과 GK 조밀 (rv5 크기 블록 하나 = 첫 블록)
    from src.rq2.phase1.ml_extension import benchmark as B
    mods = ["naive_mean", "did_gap", "ols", "enet", "lgbm", "rf", "harx", "bsts"]
    _, Wr = B.collect("rv5", mods)
    L1 = loo_audit(Wr[Wr.window.isin(WINS_ALL)], 3, label="rv5_loo")
    C.save_table(L1, "exp5_time_null_loo_rv5")
    sel, freq, Sx = selection_audit(D)
    C.save_table(sel, "exp5_selection_audit_h10")
    C.save_table(freq, "exp5_selection_freq_h10")
    sel20, _, _ = selection_audit(D, "h20")
    C.save_table(sel20, "exp5_selection_audit_h20")
    C.save_table(rank_consistency(D), "exp5_rank_consistency_gk")
    # space
    Srv = run_rv5_space()
    real = P.real_index()
    Tp = space_audit(Srv, P.pool("main"), exclude_i0=(real,))
    C.save_table(space_summary(Tp), "exp5_space_fpr_rv5_pseudo")
    Sg = run_gk_space()
    dl, _, _ = P.gk_units()
    Tg = space_audit(Sg, dl)
    C.save_table(space_summary(Tg), "exp5_space_fpr_gk")
    ex, F = exchangeability(real)
    C.save_table(ex, "exp5_exchangeability_treated_pctile")
    C.save_table(err_vs_features(Srv[Srv.i0 != real], F, P.pool("main")), "exp5_space_err_vs_features_rv5")
    # 실제 발표일 전/후
    rm = ["harx", "ols", "bsts", "did_gap", "naive_mean", "enet", "lgbm", "rf", "xgb", "ridge"]
    BA = real_before_after(rm, R)
    C.save_table(BA, "exp5_real_effect_before_after")
    # 실제일 space: 원 오차 vs 표준화 (rv5_space 의 새 적합: did_gap/ols/harx/lgbm)
    Treal = space_audit(Srv[Srv.i0 == real], P.pool("main"))
    C.save_table(Treal, "exp5_real_space_raw_vs_stud")
    return S


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["dense", "gkspace", "rv5space"]
    if "analysis" in what:
        print(run_analysis().head())
    if "rv5space" in what:
        print(run_rv5_space().shape)
    if "gkspace" in what:
        print(run_gk_space().shape)
    if "dense" in what:
        print(run_gk_dense().shape)
