"""Exp2. 통제 오염·spillover 민감도 (RQ2-S2-*).

단위 정의: 처치 종목의 실제 효과 τ (log RV, post_effective 61일 평균). spillover 강도 s 는 '영향받는 통제 단위의
log RV 가 같은 날 s·τ 만큼 움직인다' 는 뜻 (같은 log 단위). s=0 무오염, s=1 통제도 처치 종목과 같은 크기로 움직임.
  A  donor 평균 오염      : don_mean 만 s·τ 이동, PCA 점수는 그대로 (점수만 바꾸는 단순 시나리오)
  A2 donor 전체 주입      : 모든 donor log RV 에 s·τ 주입 → don_mean 재계산 + 학습구간 고유벡터로 PCA 재투영
  B  KOSDAQ 오염          : kosdaq_lrv 만 s·τ
  C  결합 오염            : A2 + B
  D  부분 donor 오염      : 사전 상관 상위 30% donor 만 s·τ (나머지 0) → 재투영 (don_mean 은 0.3·s·τ 이동)
  E  요인 수준 오염       : donor j 에 s·τ·β_j (β_j = 사전구간 처치 종목에 대한 j 의 회귀계수, 평균 1 로 정규화) → 재투영
  F  음성통제 오염 (DIFF) : 005935 log RV 가 s·τ → 같은 기업 차분 추정치 감쇠 = s (해석적, 정확)

계산: 오염은 사후구간에만 들어가므로 학습 적합이 같다 (같은 seed → 같은 Gibbs 표본). 따라서 '고정 예측함수 + 바뀐 X' =
'전체 절차 재적합' 이고, 예측은 X_post 에 (거의) 선형이다. 단위 충격 u 로 한 번 재적합해 반응계수
A = Δ예측/u 를 얻고 θ̂(τ,s) = θ̂_0 + τ − A·s·τ 로 확장한다. 선형성(u=−0.1 대비)과 실제 주입 재적합으로 검증한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.rq2.phase1.sensitivity import common as C
from src.rq2.phase1.sensitivity import figs as F

CC = C.CFG["contamination"]
SCEN = {"A_mean_only": "donor 평균만 (PCA 점수 고정)", "A2_donor_all": "donor 전체 주입 + PCA 재투영",
        "B_kosdaq": "KOSDAQ 지수만", "C_joint": "donor 전체 + KOSDAQ", "D_partial30": "사전상관 상위 30% donor",
        "E_factor": "요인 노출(β) 비례 donor 주입"}


def _donor_weights(code: str, kind: str) -> pd.Series:
    D = C.donor_panel()
    y = C.load_panel("rv5")[code].reindex(D.index)
    tm = D.index <= C.TRAIN_END
    Dt, yt = D[tm], y[tm]
    if kind == "partial":
        r = Dt.corrwith(yt)
        top = r.nlargest(int(np.ceil(CC["partial_top_frac"] * len(r)))).index
        return pd.Series(np.where(D.columns.isin(top), 1.0, 0.0), index=D.columns)
    beta = Dt.apply(lambda col: col.cov(yt) / yt[col.notna() & yt.notna()].var())
    return beta / beta.mean()


def scenario_X(scen: str, code: str, u: float) -> pd.DataFrame:
    if scen == "A_mean_only":
        return C.controls_shifted(u, 0.0, mean_only=True)
    if scen == "A2_donor_all":
        return C.controls_shifted(u, 0.0)
    if scen == "B_kosdaq":
        return C.controls_shifted(0.0, u)
    if scen == "C_joint":
        return C.controls_shifted(u, u)
    if scen == "D_partial30":
        return C.controls_shifted(u * _donor_weights(code, "partial"), 0.0)
    if scen == "E_factor":
        return C.controls_shifted(u * _donor_weights(code, "beta"), 0.0)
    raise KeyError(scen)


def response() -> pd.DataFrame:
    """모형·종목·시나리오별 반응계수 A (창별)."""
    rows = []
    u = CC["unit_shock"]
    for code in C.TREAT:
        for m in CC["models"]:
            base = C.fit_effect(code, m)["effects"]
            for scen in SCEN:
                e = C.fit_effect(code, m, X=scenario_X(scen, code, u), tag=f"{scen}_{u}")["effects"]
                for w in ["transition", "post_effective", "post_eff_20d", "post_eff_21_61d"]:
                    rows.append(dict(code=code, model=m, scenario=scen, window=w, u=u, theta0=base[w],
                                     theta_u=e[w], A=(base[w] - e[w]) / u))
            # 선형성: C_joint 를 u/2 로 한 번 더
            e2 = C.fit_effect(code, m, X=scenario_X("C_joint", code, u / 2), tag=f"C_joint_{u/2}")["effects"]
            w = "post_effective"
            rows.append(dict(code=code, model=m, scenario="C_joint@half", window=w, u=u / 2, theta0=base[w],
                             theta_u=e2[w], A=(base[w] - e2[w]) / (u / 2)))
    return pd.DataFrame(rows)


def design_A(R: pd.DataFrame) -> pd.DataFrame:
    """설계별 A: BSTS = bsts, SIMPLEX = Σ w_m A_m (실제 post_effective 가중치), DIFF = 005935 오염시 1 (해석적)."""
    w = C.stat_simplex_weights()["weights"]
    P = R[R.window == "post_effective"]
    rows = []
    for code in C.TREAT:
        for scen in SCEN:
            p = P[(P.code == code) & (P.scenario == scen)].set_index("model").A
            rows.append(dict(design=f"BSTS_{code}", scenario=scen, A=p["bsts"]))
            rows.append(dict(design=f"SIMPLEX_{code}", scenario=scen, A=sum(w[m] * p[m] for m in w)))
    for scen in SCEN:
        rows.append(dict(design="DIFF_005930", scenario=scen, A=0.0))
    rows.append(dict(design="DIFF_005930", scenario="F_negctrl_005935", A=1.0))
    return pd.DataFrame(rows)


def grid(DA: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in DA.iterrows():
        kind = C.CFG["designs"][r.design]["kind"]
        null = C.null_dist(kind)
        e = null - null.mean()
        c95 = C.crit95(null)
        th = C.theta(r.design)
        for tp in CC["true_effects_pct"] + [0]:
            tau = float(C.lg(tp))
            for s in CC["shares"]:
                est = tau * (1 - r.A * s)
                rows.append(dict(design=r.design, scenario=r.scenario, A=r.A, true_pct=tp, tau=tau, s=s,
                                 donor_shift_log=s * tau, est_log=est, est_pct=C.pct(est),
                                 attenuation=r.A * s, bias_from_contam=est - tau,
                                 sign_change=bool(tau != 0 and est * tau < 0),
                                 near_zero=bool(abs(est) < CC["near_zero"]),
                                 power=float(np.mean(np.abs(est + e) > c95)),
                                 coverage=float(np.mean(np.abs(est + e - tau) <= c95)),
                                 fpr=float(np.mean(np.abs(e) > c95)) if tp == 0 else np.nan,
                                 implied_tau_pct=C.pct(th / (1 - r.A * s)) if abs(1 - r.A * s) > 1e-6 else np.nan))
    return pd.DataFrame(rows)


def thresholds(DA: pd.DataFrame) -> pd.DataFrame:
    """실제 −10~−30% 효과가 0 근처(|θ̂|<0.02)로 추정되려면 필요한 s, 부호가 뒤집히려면 필요한 s, 현실성 상한."""
    lh = pd.read_csv(C.MAIN / "rq2_synthesis_main.csv", dtype={"code": str})
    us = lh[lh.design.str.contains("미국 변수만") & (lh.code == "don_mean") & (lh.window == "post_effective")].iloc[0]
    don_lo = us.effect_log - 1.96 * us.null_sd       # 미국 변수만 반사실 대비 donor 평균 사후 변화의 95% 하한
    rows = []
    for _, r in DA.iterrows():
        for tp in CC["true_effects_pct"]:
            tau = float(C.lg(tp))
            s_zero = (1 - CC["near_zero"] / abs(tau)) / r.A if r.A > 0 else np.inf
            s_flip = 1 / r.A if r.A > 0 else np.inf
            s_real = min(abs(don_lo) / abs(tau), 1.0) if r.scenario != "F_negctrl_005935" else np.nan
            rows.append(dict(design=r.design, scenario=r.scenario, A=r.A, true_pct=tp,
                             s_needed_near_zero=s_zero, s_needed_sign_flip=s_flip,
                             within_0_1_near_zero=bool(s_zero <= 1), donor_shift_needed_log=s_zero * tau,
                             realistic_s_max=s_real, donor_change_lower95=don_lo,
                             near_zero_feasible=bool(np.isfinite(s_real) and s_zero <= s_real)))
    return pd.DataFrame(rows)


def refit_check(R: pd.DataFrame) -> pd.DataFrame:
    """실제 주입 재적합 (y 에 τ, donor+KOSDAQ 에 s·τ) vs 선형 근사 θ̂0 + τ − A s τ."""
    rows = []
    P = R[(R.window == "post_effective") & (R.scenario == "C_joint")]
    for code in C.TREAT:
        X0 = C.base_controls()
        post = X0.index[X0.index >= C.EFFECTIVE]
        for s, tau in CC["refit_checks"]:
            ysh = pd.Series(tau, index=post)
            for m in ["bsts", "ols"]:
                e = C.fit_effect(code, m, X=scenario_X("C_joint", code, s * tau), y_shift=ysh,
                                 tag=f"inject_{s}_{tau}")["effects"]["post_effective"]
                a = P[(P.code == code) & (P.model == m)].iloc[0]
                lin = a.theta0 + tau - a.A * s * tau
                rows.append(dict(code=code, model=m, s=s, tau=tau, refit=e, linear=lin, diff=e - lin))
    return pd.DataFrame(rows)


def fig_curves(G: pd.DataFrame, DA: pd.DataFrame):
    import matplotlib.pyplot as plt
    des = ["BSTS_005930", "BSTS_000660", "SIMPLEX_005930", "SIMPLEX_000660", "DIFF_005930"]
    fig, axs = plt.subplots(1, len(des), figsize=(19, 4.8), layout="constrained", sharey=True)
    for ax, d in zip(axs, des):
        for k, scen in enumerate(["C_joint", "A_mean_only", "B_kosdaq", "D_partial30"] if not d.startswith("DIFF")
                                 else ["F_negctrl_005935", "C_joint"]):
            for tp, ls in zip(CC["true_effects_pct"], ["-", "--", ":"]):
                g = G[(G.design == d) & (G.scenario == scen) & (G.true_pct == tp)]
                ax.plot(g.s * 100, g.est_pct, ls, color=F.CAT[k], lw=1.6,
                        label=f"{scen} · 실제 {tp}%" if tp == -20 or d == des[0] and tp == -10 else None)
        ax.axhline(0, color=F.INK2, lw=1)
        ax.set_title(d, fontsize=9, loc="left")
        ax.set_xlabel("spillover 강도 s (%)")
        ax.legend(fontsize=6)
    axs[0].set_ylabel("추정될 효과 (%)  [실선 −10, 파선 −20, 점선 −30%]")
    fig.suptitle("통제 오염 시 실제 감소 효과가 어떻게 추정되는가 (bias 0 가정, A = 모형 반응계수)", fontsize=10, x=0.01, ha="left")
    F.save(fig, "s2_contamination_curves")
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    piv = DA.pivot(index="scenario", columns="design", values="A").reindex(list(SCEN) + ["F_negctrl_005935"])
    x = np.arange(len(piv.index))
    wdt = 0.16
    for j, d in enumerate(des):
        ax.bar(x + (j - 2) * wdt, piv[d].values, wdt, color=F.COL[d], label=d)
    ax.set_xticks(x, piv.index, rotation=20, fontsize=8)
    ax.axhline(1, color=F.INK2, lw=1, ls="--")
    ax.set_ylabel("반응계수 A (통제 1 log 이동 → 반사실 이동)")
    ax.set_title("시나리오별 감쇠 계수 A: s=100% 일 때 감쇠율 = A", fontsize=9, loc="left")
    ax.legend(fontsize=7)
    F.save(fig, "s2_attenuation_coefficients")


def run():
    R = response()
    DA = design_A(R)
    G = grid(DA)
    T = thresholds(DA)
    RC = refit_check(R)
    C.save_table(R, "s2_response_by_model")
    C.save_table(DA, "s2_attenuation_by_design")
    C.save_table(G, "s2_contamination_grid")
    C.save_table(T, "s2_contamination_thresholds")
    C.save_table(RC, "s2_refit_check")
    fig_curves(G, DA)
    lin = R[R.scenario == "C_joint@half"].merge(
        R[(R.scenario == "C_joint") & (R.window == "post_effective")][["code", "model", "A"]], on=["code", "model"],
        suffixes=("_half", "_full"))
    for d, g in DA.groupby("design"):
        C.LOG.add(id=f"RQ2-S2-{d}", rq="RQ2", kind="robustness",
                  purpose="통제 오염(spillover) 시 감쇠·부호·검정력. 0 근처 추정에 필요한 오염 강도",
                  data="rv5 주 분석 표본, C_main 통제", model=d,
                  settings={"shares": CC["shares"], "unit_shock": CC["unit_shock"], "scenarios": list(SCEN)},
                  diagnostics={"linearity_max_absdiff_A": float((lin.A_half - lin.A_full).abs().max()),
                               "refit_max_absdiff": float(RC["diff"].abs().max())},
                  result=dict(zip(g.scenario, g.A.round(3))),
                  interpretation="τ=0 검정은 s 에 무관 (FPR 불변); 크기·검정력만 감쇠", seed=C.SEED)
    return dict(R=R, DA=DA, G=G, T=T, RC=RC, lin=lin)


if __name__ == "__main__":
    out = run()
    C.LOG.save()
    pd.set_option("display.width", 250)
    print(out["R"][out["R"].window == "post_effective"].round(3).to_string())
    print(out["DA"].round(3).to_string())
    print(out["T"].round(3).to_string())
    print(out["RC"].round(4).to_string())
