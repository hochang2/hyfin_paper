"""Exp3. 관측되지 않은 교란 민감도 (RQ2-S3-*).

(a) 같은 기업 차분 (005930 − 005935, 상장기간 대비 HAC DID):
    관측 θ̂ = 실제 효과 τ + 차별적 충격 δ (보통주에만, 우선주에는 없는 규제 외 충격).
    필요 δ: 부호 전환 δ = θ̂, 효과 x% 결론 δ = θ̂ − log(1+x), 유의한 감소 δ = θ̂ + crit.
    역사적 비교: (i) GK 긴 표본의 같은 구조 차분 변화 분포 (가짜 개입일, 34일 기준창 → [10,71) 창)
                 (ii) 차분의 사전 추세 (월 기울기 × 기준창~사후창 중심 거리)
                 (iii) 시장 민감도 차이: d = a + γ·don_mean (사전구간) → δ_mkt = γ·Δdon_mean (실제 사후 변화)
                 (iv) don_mean·kosdaq 을 차분 회귀에 직접 넣은 재추정 (관측 교란 통제 시 θ̂ 변화)
(b) BSTS 숨은 교란 U_t (사후구간에만 처치 종목 y 에 더해짐, 통제와 무관):
    θ̂ = τ + mean(U). 모양: constant / temporary(첫 10일) / growing(0→c 선형) / market_interaction(c·z_t, z=표준화 don_mean).
    필요 크기 c 를 모양별로 구하고, '관측 통제 하나만큼 강한 교란' 기준(leave-one-control-out 재적합의 θ̂ 변화)과 비교.
    Oster δ·Cinelli-Hazlett RV 는 OLS 의 R² 변화 논리라 BSTS(상태공간·변수선택)에 그대로 쓰지 않는다 — 대신 같은 질문을
    '관측 통제 k 를 뺄 때의 θ̂ 이동' 이라는 기준으로 바꿔 답한다 (OLS 판 Cinelli-Hazlett 는 참고로만 같이 보고).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.rq2.core.data import load_long_daily, load_panel
from src.rq2.core.pipeline import effect_windows
from src.rq2.phase1.sensitivity import common as C
from src.rq2.phase1.sensitivity import figs as F
from src.rq2.phase1.sensitivity.bias_sensitivity import diff_simple

KC = C.CFG["confounding"]
TARGETS = [0, -5, -10, -20, -30]


def _d_rv5():
    lv = load_panel("rv5").loc[C.RQ2["sample"]["start"]:C.RQ2["sample"]["end"]]
    return (lv["005930"] - lv[C.NEG]).rename("d")


def did(extra: pd.DataFrame | None = None, d: pd.Series | None = None) -> dict:
    """주 분석 within_DID_HAC_vs_listing 과 같은 회귀. extra 는 추가 통제."""
    d = _d_rv5() if d is None else d
    idx = d.index
    W = effect_windows(idx)
    dm = pd.DataFrame({k: W[k] for k in ["listing", "transition", "post_effective"]}).astype(float)
    X = C.base_controls()[["gap_day"]].reindex(idx)
    if extra is not None:
        X = X.join(extra.reindex(idx))
    Z = pd.concat([dm, X], axis=1).dropna()
    m = sm.OLS(d.reindex(Z.index), sm.add_constant(Z)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    c = pd.Series(0.0, index=m.params.index)
    c["post_effective"], c["listing"] = 1, -1
    t = m.t_test(c.values)
    return {"theta": float(t.effect[0]), "se": float(np.ravel(t.sd)[0]), "p": float(np.ravel(t.pvalue)[0]), "params": m.params}


def diff_part() -> tuple[pd.DataFrame, pd.DataFrame]:
    th = C.theta("DIFF_005930")
    null = C.null_dist("diff")
    crit = C.crit95(null)
    need = {f"effect_{t}pct" if t else "sign_flip": th - float(C.lg(t)) for t in TARGETS}
    need["sig_negative"] = th + crit
    # (i) 역사적 차별 충격 분포
    LD = load_long_daily(C.RQ2).loc[:"2026-07-15"]
    H = diff_simple(LD["005930"], LD[C.NEG], KC["diff_listing_days"], 1)
    H = H[H.cutoff + pd.Timedelta(days=110) < pd.Timestamp("2026-07-16")]
    lh = pd.read_csv(C.MAIN / "rq2_longhist_placebo.csv")
    lh = lh[(lh.series == "005930-005935") & ~lh.actual].post_effective.to_numpy()
    # (ii) 사전 추세
    pt = pd.read_csv(C.MAIN / "rq2_pretrend.csv")
    pt = pt[pt.series == "005930-005935"]
    lst, ann, eff = (pd.Timestamp(C.RQ2["timeline"][k]) for k in ["listing", "announce", "effective"])
    gap_m = ((eff + (pd.Timestamp(C.RQ2["sample"]["end"]) - eff) / 2) - (lst + (ann - lst) / 2)).days / 30.44
    # (iii) 시장 민감도 차이
    d = _d_rv5()
    X = C.base_controls()
    pre = d.index <= C.TRAIN_END
    cc = pd.read_csv(C.MAIN / "rq2_control_check.csv")
    shift = cc[cc.window == "post_effective"].set_index("control").shift_vs_listing
    mk = []
    for v in ["don_mean", "kosdaq_lrv"]:
        Z = pd.concat([d, X[v]], axis=1).dropna()
        Zp = Z[Z.index <= C.TRAIN_END]
        m = sm.OLS(Zp.d, sm.add_constant(Zp[v])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        g, se = m.params[v], m.bse[v]
        dx = shift.get(v, np.nan)
        mk.append(dict(control=v, gamma_pre=g, gamma_se=se, post_shift_vs_listing=dx, delta_mkt=g * dx,
                       delta_lo=(g - 1.96 * se) * dx, delta_hi=(g + 1.96 * se) * dx))
    # GK 긴 표본 γ (don_mean)
    Lg = pd.concat([(LD["005930"] - LD[C.NEG]).rename("d"), LD["don_mean"]], axis=1).dropna()
    m = sm.OLS(Lg.d, sm.add_constant(Lg.don_mean)).fit(cov_type="HAC", cov_kwds={"maxlags": 20})
    mk.append(dict(control="don_mean (GK 2015~, 장기 γ)", gamma_pre=m.params["don_mean"], gamma_se=m.bse["don_mean"],
                   post_shift_vs_listing=shift["don_mean"], delta_mkt=m.params["don_mean"] * shift["don_mean"],
                   delta_lo=(m.params["don_mean"] - 1.96 * m.bse["don_mean"]) * shift["don_mean"],
                   delta_hi=(m.params["don_mean"] + 1.96 * m.bse["don_mean"]) * shift["don_mean"]))
    MK = pd.DataFrame(mk)
    # (iv) 관측 교란을 직접 통제한 재추정
    re = [dict(spec="주 분석 (gap_day)", **{k: v for k, v in did().items() if k != "params"})]
    for v in [["don_mean"], ["kosdaq_lrv"], ["don_mean", "kosdaq_lrv", "us_lvix"]]:
        r = did(X[v])
        re.append(dict(spec="+ " + ", ".join(v), theta=r["theta"], se=r["se"], p=r["p"]))
    RE = pd.DataFrame(re)
    rows = []
    for k, dl in need.items():
        r = dict(tipping=k, delta_needed=dl, implied_true_pct=None)
        r["P_hist_gk_simple(D>=δ)"] = float(np.mean(H.e >= dl))
        r["P_longhist_placebo(D>=δ)"] = float(np.mean(lh >= dl))
        for _, p in pt.iterrows():
            r[f"pretrend_{p.data[:3]}_{p['sample']}"] = p.slope_per_month * gap_m
        for _, q in MK.iterrows():
            r[f"mkt_{q.control}"] = q.delta_mkt
        rows.append(r)
    T = pd.DataFrame(rows).drop(columns="implied_true_pct")
    hist = dict(n=len(H), mean=H.e.mean(), sd=H.e.std(), q01=H.e.quantile(.01), q05=H.e.quantile(.05),
                q95=H.e.quantile(.95), q99=H.e.quantile(.99), max=H.e.max(), gap_months=gap_m,
                n_eff=C.n_eff(len(H), 1))
    return T, MK, RE, pd.DataFrame([hist]), H


# ------------------------------------------------------------------ BSTS 숨은 교란
def shape_factor(shape: str) -> float:
    """사후 61일 창에서 mean(U)/c."""
    L = C.POST_L
    if shape == "constant":
        return 1.0
    if shape == "temporary":
        return KC["temporary_days"] / L
    if shape == "growing":
        return float(np.mean(np.arange(L) / (L - 1)))
    if shape == "market_interaction":
        X = C.base_controls()
        pre = X.index <= C.TRAIN_END
        z = (X.don_mean - X.don_mean[pre].mean()) / X.don_mean[pre].std()
        return float(z[X.index >= C.EFFECTIVE].mean())
    raise KeyError(shape)


def loco() -> pd.DataFrame:
    """leave-one-control-out: 관측 통제 k 를 뺀 재적합 θ̂ 이동 = 'k 만큼 강한 교란'이 만드는 이동의 기준."""
    rows = []
    ctr = C.RQ2["controls"]["C_main"]
    for code in C.TREAT:
        for m in ["bsts", "ols"]:
            base = C.fit_effect(code, m)["effects"]["post_effective"]
            for k in ctr:
                e = C.fit_effect(code, m, controls=[c for c in ctr if c != k], tag=f"loco_{k}")["effects"]["post_effective"]
                rows.append(dict(code=code, model=m, dropped=k, theta_full=base, theta_drop=e, delta=e - base))
    return pd.DataFrame(rows)


def ols_rv(code: str) -> dict:
    """참고: OLS 판 Cinelli-Hazlett robustness value (사후 더미를 처치로 둔 ITS 회귀). BSTS 에는 적용하지 않는다."""
    y = load_panel("rv5")[code].loc[C.RQ2["sample"]["start"]:C.RQ2["sample"]["end"]]
    X = C.base_controls()[C.RQ2["controls"]["C_main"]]
    W = effect_windows(y.index)
    D = pd.DataFrame({"transition": W["transition"], "post": W["post_effective"]}).astype(float)
    Z = pd.concat([y.rename("y"), D, X], axis=1).dropna()
    m = sm.OLS(Z.y, sm.add_constant(Z.drop(columns="y"))).fit()
    t, dof = m.tvalues["post"], m.df_resid
    f = abs(t) / np.sqrt(dof)
    rv = 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)
    pr2 = t ** 2 / (t ** 2 + dof)
    return dict(code=code, ols_post=m.params["post"], t=t, partial_R2_treat=pr2, RV_q1=rv, dof=dof)


def bsts_part(LO: pd.DataFrame):
    rows = []
    for code in C.TREAT:
        des = f"BSTS_{code}"
        th = C.theta(des)
        crit = C.crit95(C.null_dist("bsts"))
        need = {f"effect_{t}pct" if t else "sign_flip": th - float(C.lg(t)) for t in TARGETS}
        need["sig_negative"] = th + crit
        lo = LO[(LO.code == code) & (LO.model == "bsts")]
        bench = lo.delta.abs().max()
        bk = lo.loc[lo.delta.abs().idxmax(), "dropped"]
        for shape in KC["bsts_shapes"]:
            fct = shape_factor(shape)
            for k, mu in need.items():
                c_need = mu / fct if fct != 0 else np.inf
                rows.append(dict(design=des, shape=shape, mean_factor=fct, tipping=k, meanU_needed=mu, c_needed=c_need,
                                 benchmark_max_loco=bench, benchmark_control=bk,
                                 multiple_of_strongest_control=abs(mu) / bench if bench > 0 else np.inf))
    return pd.DataFrame(rows)


def fig(T, H, MK, B, LO):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), layout="constrained")
    ax = axs[0]
    ax.hist(H.e, bins=60, color=F.CAT[6], alpha=0.45, density=True, label="GK 2015~ 같은 구조 차분 변화 (가짜 개입일, 일 단위)")
    for _, r in T.iterrows():
        ax.axvline(r.delta_needed, color=F.INK2, lw=1, ls="--")
        ax.text(r.delta_needed, ax.get_ylim()[1] * 0.92, r.tipping.replace("effect_", "").replace("pct", "%"),
                rotation=90, fontsize=7, va="top", ha="right")
    for j, (_, q) in enumerate(MK.iterrows()):
        ax.plot([q.delta_lo, q.delta_hi], [0.25 + 0.2 * j] * 2, color=F.CAT[j], lw=3, alpha=0.6)
        ax.plot(q.delta_mkt, 0.25 + 0.2 * j, "o", color=F.CAT[j], label=f"시장민감도 차이 δ: {q.control}")
    ax.set_xlabel("차별적 충격 δ (log) — 오른쪽일수록 '실제로는 감소' 결론에 필요한 충격")
    ax.set_title("(a) 같은 기업 차분: 필요한 차별 충격 vs 역사적 분포", fontsize=9, loc="left")
    ax.legend(fontsize=6.5, loc="upper left")
    ax = axs[1]
    shapes = KC["bsts_shapes"]
    keys = ["effect_-10pct", "effect_-20pct", "sig_negative"]
    x = np.arange(len(shapes))
    for j, des in enumerate(["BSTS_005930", "BSTS_000660"]):
        for i, k in enumerate(keys):
            g = B[(B.design == des) & (B.tipping == k)].set_index("shape").reindex(shapes)
            ax.scatter(x + (j - 0.5) * 0.3, g.multiple_of_strongest_control, marker="osD"[i], color=F.COL[des],
                       s=40, label=f"{des} · {k}" if True else None)
    ax.axhline(1, color=F.INK2, lw=1, ls="--")
    ax.text(-0.4, 1.05, "가장 강한 관측 통제 1개만큼", fontsize=7, color=F.INK2)
    ax.set_xticks(x, shapes)
    ax.set_yscale("log")
    ax.set_ylabel("필요한 mean(U) / max|LOCO θ̂ 이동|  (배수)")
    ax.set_title("(b) BSTS 숨은 교란: 관측 통제 대비 몇 배 강해야 하나 (사후 평균 기준, 모양별 c 는 표 참조)",
                 fontsize=9, loc="left")
    ax.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1))
    F.save(fig, "s3_confounding")


def run():
    T, MK, RE, HS, H = diff_part()
    LO = loco()
    B = bsts_part(LO)
    RV = pd.DataFrame([ols_rv(c) for c in C.TREAT])
    C.save_table(T, "s3_diff_tipping")
    C.save_table(MK, "s3_diff_market_sensitivity")
    C.save_table(RE, "s3_diff_with_observed_controls")
    C.save_table(HS, "s3_diff_hist_shocks_summary")
    C.save_table(LO, "s3_bsts_loco")
    C.save_table(B, "s3_bsts_hidden_confounder")
    C.save_table(RV, "s3_ols_cinelli_hazlett_reference")
    fig(T, H, MK, B, LO)
    C.LOG.add(id="RQ2-S3-DIFF", rq="RQ2", kind="robustness", purpose="같은 기업 차분의 차별적 충격 임계점",
              data="rv5 d=005930−005935, GK 2015~ 차분", model="HAC DID (상장기간 대비)",
              settings={"listing_days": KC["diff_listing_days"]},
              result=T[["tipping", "delta_needed", "P_hist_gk_simple(D>=δ)"]].round(3).to_dict("records"),
              diagnostics={"market_delta": MK[["control", "delta_mkt"]].round(3).to_dict("records"),
                           "with_controls": RE[["spec", "theta"]].round(3).to_dict("records")}, seed=C.SEED)
    C.LOG.add(id="RQ2-S3-BSTS", rq="RQ2", kind="robustness", purpose="BSTS 숨은 교란 모양별 필요 크기 + LOCO 기준",
              data="rv5 주 분석 표본", model="BSTS 주 설정 (LOCO 14회 재적합) + OLS",
              settings={"shapes": KC["bsts_shapes"]},
              result=B[B["shape"] == "constant"][["design", "tipping", "meanU_needed", "multiple_of_strongest_control"]]
              .round(3).to_dict("records"),
              diagnostics=LO[LO.model == "bsts"][["code", "dropped", "delta"]].round(3).to_dict("records"),
              interpretation="Oster/Cinelli-Hazlett 는 BSTS 에 기계적으로 적용하지 않음 (OLS 참고값만)", seed=C.SEED)
    return dict(T=T, MK=MK, RE=RE, HS=HS, LO=LO, B=B, RV=RV)


if __name__ == "__main__":
    out = run()
    C.LOG.save()
    pd.set_option("display.width", 250)
    for k, v in out.items():
        print("==", k)
        print(v.round(4).to_string())
