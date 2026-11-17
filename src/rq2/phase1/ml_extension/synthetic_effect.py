"""실험 2: 합성 개입 (알려진 효과를 주입해 회복·검정력·위양성 측정).

주입: 가짜 개입일 이후 관측 y 에 tau_t (log) 를 더한다. tau_t 는 시나리오(A 즉시·B 점진·C 일시·D 이질·E 고변동 한정)로 정한다.
핵심 성질 (불변성 검정으로 확인): 모든 타당한 모형은 사후 y 를 입력으로 쓰지 않는다
  → 반사실 예측이 주입과 무관 → 효과 추정치 = (ybar + taubar) − predbar = err + taubar
  → 효과 추정오차는 주입 크기·시나리오와 무관하게 err 이다 (편향·RMSE 는 tau 에 따라 변하지 않음).
  모형 사이의 차이는 (1) err 분포, (2) 추론 보정, (3) 통제변수 번짐(F), (4) 시차 누출 설계에서만 생긴다.
  그래서 (1)(2) 는 캐시된 err 로 해석적으로 계산하고, (3)(4) 만 다시 적합한다.

추론 3종:
  native : 모형 자체의 창평균 반사실 분포(draws) 95% 구간. FM 은 없음.
  time   : 같은 모형·종목·창의 다른 가짜 개입일(창이 겹치지 않는 것)의 err 를 귀무분포로,
           평균 중심 t 예측구간 |est − m0| > t_{n−1,.975}·s0·sqrt(1+1/n) 이면 기각.
  space  : 실제 개입일의 donor 175 종목 err 로 순열 검정 (주입 대상 donor 를 뺀 나머지와 비교).
MDE: time 보정 기준 2.8·s0 (해석적, 80% 검정력·5% 양측) 와, 세밀한 효과 격자의 경험 검정력 곡선이 0.8 을 넘는 최소 |효과|.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from src.rq2.phase1.ml_extension import benchmark as B
from src.rq2.phase1.ml_extension import data_pipeline as DP

E = DP.ECFG
ALPHA = E["alpha"]


# ---------------------------------------------------------------- 주입 효과의 창평균
def taubar_table(ds: str, W: pd.DataFrame, scn_name: str, pct: float) -> pd.Series:
    """과제(code,i0,window) 별 창평균 주입효과 (log). D/E 는 과제별 z_t 필요."""
    scn = E["scenarios"][scn_name]
    keys = W[["code", "i0", "a", "b", "H"]].drop_duplicates()
    if scn["kind"] in ("state", "highvol"):
        zc = {}
        out = []
        for r in keys.itertuples():
            k = (r.code, r.i0)
            if k not in zc:
                zc[k] = DP.state_z(ds, r.code, r.i0).values
            tp = DP.tau_path(scn, pct, int(r.H), zc[k])
            out.append(tp[r.a:r.b].mean())
        keys["taubar"] = out
    else:
        Hmax = int(keys.H.max())
        tp = DP.tau_path(scn, pct, Hmax)
        keys["taubar"] = [tp[a:b].mean() for a, b in zip(keys.a, keys.b)]
    return W.merge(keys, on=["code", "i0", "a", "b", "H"], how="left")["taubar"].values


# ---------------------------------------------------------------- time-보정 귀무분포
def n_eff(i0s, L) -> int:
    """창 길이 L 인 가짜 개입일들 중 서로 겹치지 않는 창의 최대 개수 (유효 독립 표본 수).
    가짜 개입일 간격(rv5 3일, gk 10일)이 창보다 짧으면 이웃 창의 오차는 거의 같은 값이라
    표본 수를 그대로 쓰면 귀무 sd 의 불확실성을 과소평가한다 (FAIL-TIMENULL-OVERLAP 참고)."""
    s = np.sort(np.asarray(i0s))
    n, last = 0, -np.inf
    for v in s:
        if v - last >= L:
            n, last = n + 1, v
    return n


def time_null(W: pd.DataFrame, min_eff: int = None) -> pd.DataFrame:
    """각 과제·창에 대해 창이 겹치지 않는 다른 가짜 개입일 err 의 평균·sd 와 유효 표본 수.
    m0·s0 는 겹치지 않는 모든 과제로 추정하되, t 분포 자유도와 sqrt(1+1/n) 에는
    서로 겹치지 않는 창의 개수 n0(=n_eff) 를 쓴다. n0 < min_eff 이면 검정 불가(NaN)."""
    min_eff = min_eff or E.get("time_null_min_eff", 3)
    out = []
    for (m, code, w), g in W.groupby(["model", "code", "window"]):
        i0 = g.i0.values
        e = g.err.values
        L = (g.b - g.a).values
        for k in range(len(g)):
            keep = (np.abs(i0 - i0[k]) >= np.maximum(L, L[k])) & np.isfinite(e)
            ee = e[keep]
            ne = n_eff(i0[keep], L[k]) if keep.any() else 0
            ok = ne >= min_eff and len(ee) > 2
            out.append((g.index[k], ee.mean() if ok else np.nan, ee.std(ddof=1) if ok else np.nan, ne, len(ee)))
    o = pd.DataFrame(out, columns=["ix", "m0", "s0", "n0", "n_raw"]).set_index("ix")
    return W.join(o)


def crit_t(n0):
    n0 = np.asarray(n0, float)
    return stats.t.ppf(1 - ALPHA / 2, np.maximum(n0 - 1, 1)) * np.sqrt(1 + 1 / np.maximum(n0, 1))


def prepare(ds: str, models: list[str]) -> pd.DataFrame:
    D, W = B.collect(ds, models)
    W = B.add_flags(ds, W)
    W = time_null(W)
    W["crit"] = crit_t(W.n0) * W.s0
    return W


# ---------------------------------------------------------------- 주입 평가
def inject_eval(ds: str, W: pd.DataFrame, effects=None, scenarios=None, windows=None) -> pd.DataFrame:
    effects = effects if effects is not None else E["effects_pct"]
    scenarios = scenarios or list(E["scenarios"])
    windows = windows or (E["horizons"] if ds != "space" else ["transition", "post_effective"])
    W = W[W.window.isin(windows)].copy()
    rows = []
    for scn in scenarios:
        for pct in effects:
            tb = taubar_table(ds, W, scn, pct)
            est = W.err.values + tb
            nat_lo = W.ybar.values + tb - W.cf_q975.values
            nat_hi = W.ybar.values + tb - W.cf_q025.values
            has_nat = np.isfinite(nat_lo)
            nat_rej = np.where(has_nat, (nat_lo > 0) | (nat_hi < 0), np.nan)
            nat_cov = np.where(has_nat, (nat_lo <= tb) & (tb <= nat_hi), np.nan)
            t_rej = np.where(np.isfinite(W.crit), np.abs(est - W.m0) > W.crit, np.nan)
            t_cov = np.where(np.isfinite(W.crit), np.abs(est - W.m0 - tb) <= W.crit, np.nan)
            d = pd.DataFrame({"model": W.model.values, "code": W.code.values, "window": W.window.values,
                              "i0": W.i0.values, "truth": tb, "est": est, "err": est - tb,
                              "err_pct": np.expm1(est) - np.expm1(tb),
                              "nat_rej": nat_rej, "nat_cov": nat_cov, "t_rej": t_rej, "t_cov": t_cov,
                              "nat_width": W.cf_q975.values - W.cf_q025.values, "t_width": 2 * W.crit.values,
                              "t_ok": np.isfinite(W.crit.values).astype(float), "n0": W.n0.values,
                              "crosses_listing": W.crosses_listing.values})
            g = d.groupby(["model", "code", "window"]).agg(
                n=("err", "size"), bias=("err", "mean"), rmse=("err", lambda v: np.sqrt(np.mean(v ** 2))),
                mae=("err", lambda v: np.mean(np.abs(v))), bias_pct=("err_pct", "mean"),
                truth_mean=("truth", "mean"), native_reject=("nat_rej", "mean"), native_cov=("nat_cov", "mean"),
                time_reject=("t_rej", "mean"), time_cov=("t_cov", "mean"),
                native_width=("nat_width", "mean"), time_width=("t_width", "mean"),
                time_testable=("t_ok", "mean"), n_eff=("n0", "mean")).reset_index()
            g["scenario"], g["effect_pct"], g["dataset"] = scn, pct, ds
            rows.append(g)
    return pd.concat(rows, ignore_index=True)


def power_curve(ds: str, W: pd.DataFrame, windows=None, grid=None) -> pd.DataFrame:
    """시나리오 A(상수) 세밀 격자 검정력 (time 보정·native)."""
    grid = grid if grid is not None else -np.arange(0, 61, 1.0)
    windows = windows or E["horizons"]
    W = W[W.window.isin(windows)]
    rows = []
    for pct in grid:
        tau = np.log1p(pct / 100)
        est = W.err.values + tau
        t_rej = np.where(np.isfinite(W.crit), np.abs(est - W.m0) > W.crit, np.nan)
        lo = W.ybar.values + tau - W.cf_q975.values
        hi = W.ybar.values + tau - W.cf_q025.values
        n_rej = np.where(np.isfinite(lo), (lo > 0) | (hi < 0), np.nan)
        d = pd.DataFrame({"model": W.model.values, "code": W.code.values, "window": W.window.values,
                          "t": t_rej, "n": n_rej})
        g = d.groupby(["model", "code", "window"]).agg(time_power=("t", "mean"), native_power=("n", "mean")).reset_index()
        g["effect_pct"] = pct
        rows.append(g)
    P = pd.concat(rows, ignore_index=True)
    P["dataset"] = ds
    return P


def mde_table(ds: str, W: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    target = E["power_target"]
    rows = []
    for (m, code, w), g in P.groupby(["model", "code", "window"]):
        g = g.sort_values("effect_pct", ascending=False)   # 0, -1, -2, ...
        def first(col):
            ok = g[g[col] >= target]
            ok = ok[ok.effect_pct < 0]
            return float(ok.effect_pct.iloc[0]) if len(ok) else np.nan
        sub = W[(W.model == m) & (W.code == code) & (W.window == w)]
        s0 = np.sqrt(np.nanmean(sub.s0 ** 2))
        rows.append({"dataset": ds, "model": m, "code": code, "window": w,
                     "mde_time_emp_pct": first("time_power"), "mde_native_emp_pct": first("native_power"),
                     "mde_time_analytic_log": 2.8 * s0, "mde_time_analytic_pct": float(np.expm1(-2.8 * s0) * 100),
                     "null_sd": s0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- space 순열 (실제 개입일, donor)
def space_eval(models: list[str], effects=None, units=None) -> pd.DataFrame:
    """units 를 주면 그 donor 만 쓴다 (튜닝 모형은 donor 40 개만 적합 → 같은 donor 집합에서 비교)."""
    effects = effects if effects is not None else E["effects_pct"]
    D, W = B.collect("space", models)
    W = W[W.window.isin(["transition", "post_effective"])].copy()
    W["err"] = W.ybar - W.predbar
    donors = set(DP.donor_codes(DP.CFG, "main"))
    if units is not None:
        donors &= set(units)
    W = W[W.code.isin(donors)]
    rows = []
    for (m, w), g in W.groupby(["model", "window"]):
        e = g.err.values
        n = len(e)
        for pct in effects:
            tau = np.log1p(pct / 100)
            rej, err = [], []
            for j in range(n):
                others = np.delete(e, j)
                c = others.mean()
                p = (np.sum(np.abs(others - c) >= abs(e[j] + tau - c)) + 1) / n
                rej.append(p <= ALPHA)
                err.append(e[j])
            rows.append({"dataset": "space", "model": m, "window": w, "effect_pct": pct, "n_units": n,
                         "bias": float(np.mean(err)), "rmse": float(np.sqrt(np.mean(np.square(err)))),
                         "space_reject": float(np.mean(rej)), "null_sd": float(np.std(e, ddof=1))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 시나리오 F: 통제변수 번짐 (재적합)
def run_spillover(models: list[str], ds="rv5"):
    sp = E["spillover"]
    for share in sp["share"]:
        for pct in sp["effects_pct"]:
            B.run_models(ds, models, x_shift={"pct": pct, "share": share, "controls": sp["controls"]},
                         tag=f"spill{int(share * 100)}_{abs(int(pct))}")


def spillover_eval(models: list[str], ds="rv5") -> pd.DataFrame:
    sp = E["spillover"]
    _, W0 = B.collect(ds, models)
    W0 = B.add_flags(ds, W0)[["model", "code", "i0", "window", "err", "ybar", "predbar"]]
    rows = []
    for share in sp["share"]:
        for pct in sp["effects_pct"]:
            tag = f"spill{int(share * 100)}_{abs(int(pct))}"
            _, W1 = B.collect(ds, [f"{m}@{tag}" for m in models])
            if W1.empty:
                continue
            W1["base"] = W1.model.str.split("@").str[0]
            tau = np.log1p(pct / 100)
            M = W1.merge(W0, left_on=["base", "code", "i0", "window"], right_on=["model", "code", "i0", "window"],
                         suffixes=("", "_0"))
            M = M[M.window.isin(sp["windows"])]
            M["est"] = M.ybar + tau - M.predbar          # 번진 통제로 다시 만든 반사실
            M["est_clean"] = M.ybar_0 - M.predbar_0 + tau   # 번짐이 없을 때
            M["attenuation"] = (M.predbar - M.predbar_0) / tau   # 효과 중 반사실이 흡수한 비율
            g = M.groupby(["base", "code", "window"]).agg(
                est_mean=("est", "mean"), clean_mean=("est_clean", "mean"),
                bias=("est", lambda v: np.nan), attenuation=("attenuation", "mean")).reset_index()
            g["bias"] = g.est_mean - tau
            g["effect_pct"], g["share"], g["truth_log"] = pct, share, tau
            rows.append(g.rename(columns={"base": "model"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------- 시차 누출 설계 시연
def _leaky_one(ds, code, i0, pcts, seed):
    import warnings
    warnings.filterwarnings("ignore")
    from src.rq2.phase1.ml_extension.models import cf_gbm_har
    y, X, tr, po = DP.task_data(ds, code, i0)
    yp = y[po].values
    H = len(yp)
    out = []
    for leaky in (False, True):
        for pct in pcts:
            tau = np.log1p(pct / 100)
            yobs = yp + tau
            cf = cf_gbm_har(y, X, tr, po, seed=seed, leaky=leaky, y_obs_post=yobs)
            for w in E["horizons"]:
                a, b = DP.WIN[w]
                if b > H:
                    continue
                out.append({"code": code, "i0": i0, "window": w, "leaky": leaky, "effect_pct": pct, "truth": tau,
                            "est": yobs[a:b].mean() - cf.pred.values[a:b].mean()})
    return out


def leaky_demo(ds="rv5", pcts=(0, -20, -40)):
    f = DP.CACHE / ds / "leaky_demo.parquet"
    if f.exists():
        return pd.read_parquet(f)
    tasks = B.tasks_for(ds)
    res = Parallel(n_jobs=E["n_jobs"])(delayed(_leaky_one)(ds, c, i0, list(pcts), E["seed"]) for c, i0 in tasks)
    d = pd.DataFrame([r for rr in res for r in rr])
    d.to_parquet(f)
    return d


def leaky_summary(d: pd.DataFrame) -> pd.DataFrame:
    d = d.assign(err=d.est - d.truth)
    g = d.groupby(["leaky", "window", "effect_pct"]).agg(truth=("truth", "mean"), est_mean=("est", "mean"),
                                                         bias=("err", "mean"),
                                                         rmse=("err", lambda v: np.sqrt(np.mean(v ** 2)))).reset_index()
    g["recovered_share"] = g.est_mean / g.truth.replace(0, np.nan)
    return g
