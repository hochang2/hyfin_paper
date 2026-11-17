"""실험 6: 비선형 DGP 와 합성 효과 복원.

처치는 사후 y 에 상수를 더하는 방식이 아니라 DGP 의 잠재 상태를 바꾼다:
  y_t = m(X_t; 시나리오) + h_t,  h_t = rho_h h_{t-1} + e_t                         (y0)
  y1_t = y0_t + g_t,  g_t = rho_g g_{t-1} + (1-rho_g) kappa (1 + s·z_t)            (사후, g_{-1}=0)
  → 효과가 점차 쌓이고(이행기), 상태(z_t = 표준화 X0)에 따라 크기가 달라진다. kappa = log(1+δ).
  truth(창) = mean(y1 - y0) — 같은 충격으로 두 잠재결과를 동시에 생성하므로 정확히 안다.
spillover: 처치가 donor 에도 spill·g_t 만큼 번지고, donor 평균인 X0 의 사후 값이 오염된다 (y0 의 X0 는 오염 전).

가족
  theory : X = AR(1) 요인 4 개 (rho 0.95), 계수 고정.
  market : X = 실제 GK 통제(don_mean, us_lvix, us_abs_sox, kosdaq_ret_abs)의 무작위 연속 구간,
           계수·잔차 AR·잡음 크기 = 005930 GK 의 상장 전 자료로 보정. 비선형 변환은 보정된 선형지표 위에 얹음.

추론: (1) oracle 보정 — 효과 0 반복의 절반(보정용)으로 |오차| 95% 분위 → 나머지 절반에서 FPR·검정력·포함률.
      (2) 실행 가능한 공간 placebo — 같은 DGP 의 미처치 donor 19 개로 순위 p (일부 모델·크기만).
모델은 사후 y 를 보지 않으므로 spillover 가 아니면 효과 추정 오차 = 반사실 예측 오차 → 모델은 반복마다 한 번만 적합하고
모든 효과 크기에 같은 예측을 쓴다 (spillover 는 X 가 효과에 따라 바뀌므로 효과마다 재적합).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.rq2.ml_followup import common as C
from src.rq2.ml_followup import panel as P

D_ = P.FCFG["dgp"]
NJ = P.FCFG["n_jobs"]
EXP = "exp6"
HW = {k: v for k, v in P.WIN.items() if k in P.HZ}
RHO_H, RHO_G, BURN = 0.6, 0.8, 200
BSTS_MAX_N, BSTS_REPS = 500, 40
PLACEBO_J, PLACEBO_MODELS, PLACEBO_SIZES, PLACEBO_REPS = 19, ["ols", "ridge", "lgbm"], [190, 1000], 60
SPILL_PLACEBO_EFFECTS = (0, -20, -40)


# ================================================================ 보정 (market)
def market_calib() -> dict:
    y, X, n = P.gk_task("005930", P.gk_long().index.searchsorted(P.CFG["timeline"]["listing"]), 3000, 0)
    Z = np.column_stack([np.ones(n), X.values])
    b, *_ = np.linalg.lstsq(Z, y.values, rcond=None)
    r = y.values - Z @ b
    rho = float(np.corrcoef(r[1:], r[:-1])[0, 1])
    return {"Xall": X.values, "b": b, "rho": rho, "sd_e": float(np.std(r[1:] - rho * r[:-1])),
            "r2": float(1 - r.var() / y.values.var()), "sd_y": float(y.values.std())}


THEORY_B = np.array([0.0, 0.6, 0.3, 0.2, 0.15])


def _theory_X(T, rng):
    X = np.zeros((T + BURN, 4))
    e = rng.standard_normal((T + BURN, 4)) * np.sqrt(1 - 0.95 ** 2)
    for t in range(1, T + BURN):
        X[t] = 0.95 * X[t - 1] + e[t]
    return X[BURN:]


# ================================================================ DGP
def simulate(scen: str, fam: str, n: int, H: int, rng, cal: dict | None, J: int = 0):
    """반환: X (n+H,k), y0 (J+1, n+H) — 0 행이 처치 표적, z (H,) 상태지표, rho_h."""
    T = n + H
    assert cal is not None
    if fam == "theory":
        X = _theory_X(T, rng)
        b = THEORY_B
        rho, sd_lin = RHO_H, 1.0
    else:
        A = cal["Xall"]
        s = rng.integers(0, len(A) - T)
        X = A[s:s + T].copy()
        b = cal["b"]
        rho = cal["rho"]
    lin = X @ b[1:]
    mu, sd = lin[:n].mean(), lin[:n].std() + 1e-12
    s_ = (lin - mu) / sd                                         # 표준화된 선형지표 (사전 기준)
    # 신호 크기 고정: 선형지표 1 sd, 잡음 sd 는 R2≈0.6 이 되게 (market 은 보정값)
    r2 = 0.6 if fam == "theory" else min(max(cal["r2"], 0.3), 0.8)
    sd_u = np.sqrt((1 - r2) / r2)
    sd_e = sd_u * np.sqrt(1 - rho ** 2)
    scale = cal["sd_y"] * np.sqrt(r2)                            # 신호 sd = 실제 GK sd·sqrt(R2) → 효과 크기(log)가 실제 척도와 비교 가능

    def m_fun(sv, Xs, k):
        if scen in ("linear", "volclust", "spillover"):
            return sv
        if scen == "nonlinear":
            return 0.7 * sv + 0.6 * (np.maximum(sv - 0.5, 0) ** 2) - 0.3 * np.tanh(2 * sv)
        if scen == "interaction":
            z1 = (Xs[:, 0] - Xs[:n, 0].mean()) / (Xs[:n, 0].std() + 1e-12)
            z2 = (Xs[:, 1] - Xs[:n, 1].mean()) / (Xs[:n, 1].std() + 1e-12)
            return 0.7 * sv + 0.5 * z1 * z2
        if scen == "regime":
            st = np.zeros(T, int)
            u = rng.random(T)
            for t in range(1, T):
                st[t] = st[t - 1] if u[t] < 0.98 else 1 - st[t - 1]
            return np.where(st == 0, 0.4 * sv, 1.4 * sv + 0.5)
        if scen == "break_shift":
            tb = int(n * rng.uniform(0.7, 0.9))                   # 사전 표본 마지막 10~30% 에서 계수·수준 변화
            out = sv.copy()
            out[tb:] = 1.5 * sv[tb:] + 0.5
            return out
        raise ValueError(scen)

    Y = np.zeros((J + 1, T))
    for j in range(J + 1):
        if j == 0:
            sv = s_
        else:                                                     # donor: 계수 교란 (같은 X)
            bj = b[1:] * (1 + 0.3 * rng.standard_normal(len(b) - 1))
            lj = X @ bj
            sv = (lj - lj[:n].mean()) / (lj[:n].std() + 1e-12)
        m = m_fun(sv, X, j)
        e = rng.standard_normal(T + BURN)
        if scen == "volclust":                                    # GARCH(1,1) 잡음, 무조건 분산 1
            sig2, u_ = np.ones(T + BURN), np.zeros(T + BURN)
            for t in range(1, T + BURN):
                sig2[t] = 0.05 + 0.10 * u_[t - 1] ** 2 + 0.85 * sig2[t - 1]
                u_[t] = np.sqrt(sig2[t]) * e[t]
            e = u_
        h = np.zeros(T + BURN)
        for t in range(1, T + BURN):
            h[t] = rho * h[t - 1] + sd_e * e[t]
        Y[j] = scale * (m + h[BURN:])
    z = np.clip(s_[n:], -2, 2)
    return X, Y, z


def effect_path(kappa, z, slope):
    g = np.zeros(len(z))
    prev = 0.0
    for t in range(len(z)):
        prev = RHO_G * prev + (1 - RHO_G) * kappa * (1 + slope * z[t])
        g[t] = prev
    return g


# ================================================================ 한 반복
def _models_for(n, rep):
    return [m for m in D_["models"] if m != "bsts" or (n <= BSTS_MAX_N and rep < BSTS_REPS)]


def one_rep(scen, fam, n, rep, seed, cal):
    warnings.filterwarnings("ignore")
    H = D_["post_len"]
    rng = np.random.default_rng([seed, rep, n, D_["scenarios"].index(scen), 0 if fam == "theory" else 1])
    do_pl = n in PLACEBO_SIZES and rep < PLACEBO_REPS
    X, Y, z = simulate(scen, fam, n, H, rng, cal, J=PLACEBO_J if do_pl else 0)
    rows = []
    effs = D_["effects_pct"]
    spill = scen == "spillover"
    for d in effs:
        kap = np.log1p(d / 100)
        g = effect_path(kap, z, D_["hetero_slope"])
        if not spill and d != effs[0]:
            continue                                               # 예측은 효과와 무관 — 아래에서 효과별로 펼침
        Xo = X.copy()
        if spill:                                                  # 관측 X0(donor 평균 대용) 사후 오염
            Xo[n:, 0] = X[n:, 0] + D_["spill_share"] * g
        for m in _models_for(n, rep):
            pred = C.fit_predict(m, Y[0], Xo, n, seed=seed + rep)["pred"]
            pl = None
            if do_pl and m in PLACEBO_MODELS and (not spill or d in SPILL_PLACEBO_EFFECTS):
                pl = []
                for j in range(1, PLACEBO_J + 1):
                    yj = Y[j].copy()
                    if spill:
                        yj[n:] += D_["spill_share"] * g
                    pj = C.fit_predict(m, yj, Xo, n, seed=seed + rep)["pred"]
                    pl.append({w: float(np.mean(yj[n + a:n + b] - pj[a:b])) for w, (a, b) in HW.items()})
            dl = [d] if spill else effs
            for dd in dl:
                gg = effect_path(np.log1p(dd / 100), z, D_["hetero_slope"])
                y1 = Y[0, n:] + gg
                for w, (a, b) in HW.items():
                    r = {"family": fam, "scenario": scen, "n": n, "rep": rep, "model": m, "effect_pct": dd, "window": w,
                         "cf_err": float(np.mean(Y[0, n + a:n + b] - pred[a:b])),
                         "truth": float(np.mean(gg[a:b])),
                         "est": float(np.mean(y1[a:b] - pred[a:b]))}
                    if pl is not None:
                        # 처치 표적 효과 추정치 vs donor placebo 추정치 (|·| 순위)
                        pv = np.array([q[w] for q in pl])
                        r["p_space"] = float((1 + np.sum(np.abs(pv) >= abs(r["est"]))) / (1 + len(pv)))
                    rows.append(r)
    return rows


def run_sim(force=False):
    def go():
        cal = market_calib()
        tasks = [(s, f, n, r) for f in D_["families"] for s in D_["scenarios"] for n in D_["train_sizes"]
                 for r in range(D_["reps"])]
        res = Parallel(n_jobs=NJ, batch_size=2, verbose=5)(
            delayed(one_rep)(s, f, n, r, P.SEED, cal) for s, f, n, r in tasks)
        return pd.DataFrame([x for rr in res for x in rr])
    return C.cached(EXP, "sim", go, force)


# ================================================================ 분석
def analyze(R: pd.DataFrame | None = None):
    R = run_sim() if R is None else R
    R = R.copy()
    R["err"] = R.est - R.truth
    R["half"] = np.where(R.rep % 2 == 0, "cal", "eval")
    keys = ["family", "scenario", "n", "model", "window"]
    # 1) 예측 정확도 (효과 0, spillover 포함 — spillover 효과 0 이면 오염 없음)
    z0 = R[R.effect_pct == 0]
    acc = z0.groupby(keys).agg(cf_rmse=("cf_err", lambda e: np.sqrt(np.mean(e ** 2))), cf_mae=("cf_err", lambda e: e.abs().mean()),
                               cf_bias=("cf_err", "mean"), n_rep=("rep", "nunique")).reset_index()
    # 2) 효과 복원 (효과별)
    rec = R.groupby(keys + ["effect_pct"]).agg(eff_bias=("err", "mean"), eff_rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))),
                                               truth_mean=("truth", "mean")).reset_index()
    # 3) oracle 보정 추론: cal 반쪽의 효과 0 |est| 95% 분위
    crit = z0[z0.half == "cal"].groupby(keys).est.apply(lambda e: np.quantile(np.abs(e), 0.95)).rename("crit").reset_index()
    E = R[R.half == "eval"].merge(crit, on=keys)
    E["rej"] = np.abs(E.est) > E.crit
    E["cover"] = np.abs(E.est - E.truth) <= E.crit
    inf = E.groupby(keys + ["effect_pct"]).agg(rej=("rej", "mean"), coverage=("cover", "mean"), crit=("crit", "first")).reset_index()
    inf["crit_pct"] = 100 * (np.exp(inf.crit) - 1)
    # 4) 실행 가능한 공간 placebo
    sp = R.dropna(subset=["p_space"]).assign(rej_space=lambda d: d.p_space <= 0.05) \
        .groupby(keys + ["effect_pct"]).agg(rej_space=("rej_space", "mean"), n_rep=("rep", "nunique")).reset_index()
    C.save_table(acc, "exp6_cf_accuracy")
    C.save_table(rec, "exp6_effect_recovery")
    C.save_table(inf, "exp6_oracle_power_coverage")
    C.save_table(sp, "exp6_space_placebo_power")
    # 5) 모델 대비 요약: h20·h61, 효과 −20%, ols 기준 비율
    # 같은 rep 끼리만 비교 (bsts 는 n≤500·rep<40 에서만 적합 → ols 도 그 rep 로 제한)
    wk = ["family", "scenario", "n", "window"]
    zw = z0[z0.window.isin(["h20", "h61", "post_effective"])]
    pv = zw.pivot_table(index=wk + ["rep"], columns="model", values="cf_err")
    rows = []
    for m in pv.columns:
        d = pd.DataFrame({"m": pv[m], "b": pv["ols"]}).dropna()
        for k, x in d.groupby(level=wk):
            rows.append({**dict(zip(wk, k)), "model": m, "cf_rmse": np.sqrt((x.m ** 2).mean()),
                         "ols_rmse": np.sqrt((x.b ** 2).mean()), "n_rep": len(x)})
    rel = pd.DataFrame(rows)
    rel["rmse_ratio_vs_ols"] = rel.cf_rmse / rel.ols_rmse
    C.save_table(rel, "exp6_rmse_ratio_vs_ols")
    return acc, rec, inf, sp, rel


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["sim", "analysis"]
    if "test" in what:
        cal = market_calib()
        print({k: v for k, v in cal.items() if k != "Xall"})
        import time
        for s in D_["scenarios"]:
            t = time.time()
            r = pd.DataFrame(one_rep(s, "market", 190, 0, P.SEED, cal))
            print(s, len(r), round(time.time() - t, 1),
                  r[(r.window == "h20") & (r.effect_pct == -20)].groupby("model")[["truth", "est"]].mean().round(3).to_dict("index"))
    if "sim" in what:
        print(run_sim().shape)
    if "analysis" in what:
        analyze()
