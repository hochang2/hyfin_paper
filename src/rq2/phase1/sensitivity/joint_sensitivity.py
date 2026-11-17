"""Exp5. 결합 민감도 (RQ2-S5-*). 제한 격자: bias b × 통제 오염 s × 초과 교란 δ.

가산 시나리오 (해석적):  θ̂ = τ(1 − A·s) + b + δ   →   함의된 실제 효과 τ = (θ̂ − b − δ) / (1 − A·s)
  - b : 반사실 오차 (Exp1 의 경험적 placebo 오차 원천으로 범위를 정함)
  - δ : '초과' 교란. 경험적 b 는 가짜 개입일의 평범한 교란을 이미 포함하므로, δ 는 그 위에 더해지는 몫만 뜻한다.
        같은 교란을 b 와 δ 로 두 번 빼지 않기 위해 b 의 기준은 가짜 개입일 오차, δ 의 기준은 LOCO/역사적 차분 충격의 '초과분' 으로 둔다.
  - s : 통제 오염 비율, A : Exp2 반응계수 (C_joint, post_effective). 오염은 τ 의 크기만 줄이고 τ=0 검정(FPR)은 바꾸지 않는다.
  - 유의성: 감소가 유의하려면 θ̂ − b − δ < −crit_total (Exp1 의 sig_negative_with_se 와 같은 임계). s 와 무관.
구조 시뮬레이션: 실제 BSTS 에 y(사후) += τ + δ + b, 통제 C_joint 에 s·τ 를 같이 넣고 재적합 → 가산식과 비교 (비선형·상호작용 점검).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.rq2.phase1.sensitivity import common as C
from src.rq2.phase1.sensitivity import figs as F
from src.rq2.phase1.sensitivity.contamination import scenario_X

JC = C.CFG["joint"]
DES = ["BSTS_005930", "BSTS_000660", "SIMPLEX_005930", "SIMPLEX_000660", "DIFF_005930"]
TARGETS = {"sign_flip": 0.0, "-10%": float(C.lg(-10)), "-20%": float(C.lg(-20))}


def _t(name):
    return pd.read_csv(C.TAB / f"{name}.csv", dtype={"code": str})


def inputs() -> pd.DataFrame:
    DA = _t("s2_attenuation_by_design")
    TH = _t("s2_contamination_thresholds")
    TP = _t("s1_tipping_points")
    SR = _t("s1_bias_sources")
    rows = []
    for d in DES:
        scen = "F_negctrl_005935" if d.startswith("DIFF") else "C_joint"
        A = float(DA[(DA.design == d) & (DA.scenario == scen)].A.iat[0])
        s_real = TH[(TH.design == d) & (TH.scenario == "C_joint")].realistic_s_max
        tp = TP[TP.design == d].set_index("tipping")
        src = "gk_diff_long" if d.startswith("DIFF") else "gk_all"
        sr = SR[(SR.design == d) & (SR.source == src)].iloc[0]
        rows.append(dict(design=d, theta=float(tp.theta.iat[0]), A=A, A_scenario=scen,
                         s_realistic_max_tau20=float(s_real.max()) if len(s_real) else 1.0,
                         crit_total=float(tp.loc["sig_negative_with_se", "b_needed"] - tp.theta.iat[0]),
                         b_source=src, b_q05=sr.q05, b_q95=sr.q95, b_mean=sr["mean"]))
    return pd.DataFrame(rows)


def grid(I: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in I.iterrows():
        for b in JC["bias_grid"]:
            for s in JC["share_grid"]:
                for dl in JC["conf_grid"]:
                    att = 1 - r.A * s
                    num = r.theta - b - dl
                    tau = num / att if att > 1e-9 else np.nan
                    rows.append(dict(design=r.design, b=b, s=s, delta=dl, net_shift=b + dl, attenuation=att,
                                     implied_tau=tau, implied_pct=float(C.pct(tau)) if np.isfinite(tau) else np.nan,
                                     sig_negative=bool(num < -r.crit_total),
                                     in_realistic=bool(r.b_q05 <= b <= r.b_q95)))
    return pd.DataFrame(rows)


def frontier(I: pd.DataFrame) -> pd.DataFrame:
    """주어진 s, δ 에서 τ* 결론에 필요한 b:  b = θ̂ − δ − τ*(1 − A s)."""
    rows = []
    ss = np.linspace(0, 1, 21)
    for _, r in I.iterrows():
        for dl in JC["conf_grid"]:
            for nm, ts in TARGETS.items():
                for s in ss:
                    rows.append(dict(design=r.design, delta=dl, target=nm, s=s,
                                     b_needed=r.theta - dl - ts * (1 - r.A * s)))
            rows.append(dict(design=r.design, delta=dl, target="sig_negative", s=np.nan,
                             b_needed=r.theta - dl + r.crit_total))
    return pd.DataFrame(rows)


def summary(I: pd.DataFrame, G: pd.DataFrame) -> pd.DataFrame:
    """현실 범위(b ∈ 경험 q05~q95, s ≤ 현실 상한, δ ∈ conf_grid) 안에서 결론별로 도달 가능한가."""
    rows = []
    for _, r in I.iterrows():
        g = G[(G.design == r.design) & G.in_realistic & (G.s <= r.s_realistic_max_tau20 + 1e-9)]
        g0 = g[g.delta == 0]
        rows.append(dict(design=r.design, theta_pct=float(C.pct(r.theta)), A=r.A, s_max=r.s_realistic_max_tau20,
                         b_range=f"[{r.b_q05:.2f}, {r.b_q95:.2f}]",
                         share_cells_negative=float((g.implied_tau < 0).mean()),
                         share_cells_le_m10=float((g.implied_tau <= TARGETS["-10%"]).mean()),
                         share_cells_le_m20=float((g.implied_tau <= TARGETS["-20%"]).mean()),
                         share_cells_sig_negative=float(g.sig_negative.mean()),
                         delta0_share_le_m10=float((g0.implied_tau <= TARGETS["-10%"]).mean()),
                         min_implied_pct=float(g.implied_pct.min()), max_implied_pct=float(g.implied_pct.max()),
                         n_cells=len(g)))
    return pd.DataFrame(rows)


def structural(I: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code in C.TREAT:
        r = I.set_index("design").loc[f"BSTS_{code}"]
        base = C.fit_effect(code, "bsts")["effects"]["post_effective"]
        X0 = C.base_controls()
        post = X0.index[X0.index >= C.EFFECTIVE]
        for b, s, dl, tau in JC["structural_cells"]:
            ysh = pd.Series(tau + dl + b, index=post)
            e = C.fit_effect(code, "bsts", X=scenario_X("C_joint", code, s * tau), y_shift=ysh,
                             tag=f"joint_{b}_{s}_{dl}_{tau}")["effects"]["post_effective"]
            add = base + tau * (1 - r.A * s) + b + dl
            rec = (e - base - b - dl) / (1 - r.A * s)
            rows.append(dict(code=code, b=b, s=s, delta=dl, tau=tau, theta0=base, refit=e, additive=add,
                             diff=e - add, tau_recovered=rec, tau_error=rec - tau))
    return pd.DataFrame(rows)


def figs(I, G, FR):
    import matplotlib.pyplot as plt
    # 히트맵: δ=0, b × s → implied %
    fig, axs = plt.subplots(1, len(DES), figsize=(20, 4.4), layout="constrained")
    for ax, d in zip(axs, DES):
        g = G[(G.design == d) & (G.delta == 0)].pivot(index="b", columns="s", values="implied_pct") * 100
        im = ax.imshow(g.values, origin="lower", cmap=F.DIV, vmin=-40, vmax=40, aspect="auto")
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g.values[i, j]
                ax.text(j, i, f"{v:.0f}" if np.isfinite(v) else "∞", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(g.shape[1]), [f"{x:.2f}" for x in g.columns])
        ax.set_yticks(range(g.shape[0]), [f"{x:+.2f}" for x in g.index])
        r = I.set_index("design").loc[d]
        ax.set_title(f"{d}\nA={r.A:.2f}, 경험 b {r.b_q05:.2f}~{r.b_q95:.2f}", fontsize=8, loc="left")
        ax.set_xlabel("오염 비율 s")
        ax.grid(False)
    axs[0].set_ylabel("bias b (log)")
    fig.colorbar(im, ax=axs, shrink=0.8, label="함의된 실제 효과 % (δ=0)")
    F.save(fig, "s5_joint_heatmap")
    # frontier
    fig, axs = plt.subplots(1, len(DES), figsize=(20, 4.4), layout="constrained", sharey=True)
    for ax, d in zip(axs, DES):
        r = I.set_index("design").loc[d]
        ax.axhspan(r.b_q05, r.b_q95, color=F.GRID, alpha=0.7, label="경험 b q05~q95")
        ax.axvspan(r.s_realistic_max_tau20, 1, color=F.CAT[7], alpha=0.06, label="s 현실 상한 밖 (τ=−20%)")
        for k, (nm, ls) in enumerate(zip(TARGETS, ["-", "--", ":"])):
            for dl, a in [(0.0, 1.0), (0.1, 0.45)]:
                f = FR[(FR.design == d) & (FR.target == nm) & (FR.delta == dl)]
                ax.plot(f.s, f.b_needed, ls, color=F.CAT[k], alpha=a, lw=1.6,
                        label=f"{nm} (δ={dl})" if d == DES[0] else None)
        sg = FR[(FR.design == d) & (FR.target == "sig_negative") & (FR.delta == 0)].b_needed.iat[0]
        ax.axhline(sg, color=F.CAT[3], lw=1.2, label="유의한 감소 (δ=0)" if d == DES[0] else None)
        ax.set_title(d, fontsize=9, loc="left")
        ax.set_xlabel("오염 비율 s")
    axs[0].set_ylabel("필요한 bias b (log)")
    axs[0].legend(fontsize=6.5, loc="upper left")
    F.save(fig, "s5_joint_frontier")


def run():
    I = inputs()
    G = grid(I)
    FR = frontier(I)
    S = summary(I, G)
    ST = structural(I)
    C.save_table(I, "s5_joint_inputs")
    C.save_table(G, "s5_joint_grid")
    C.save_table(FR, "s5_joint_frontier")
    C.save_table(S, "s5_joint_summary")
    C.save_table(ST, "s5_structural_check")
    figs(I, G, FR)
    for _, r in S.iterrows():
        C.LOG.add(id=f"RQ2-S5-{r.design}", rq="RQ2", kind="robustness", purpose="bias×오염×초과교란 결합 민감도",
                  data="Exp1·Exp2 결과표", model="가산식 τ=(θ̂−b−δ)/(1−A s)",
                  settings={k: JC[k] for k in ["bias_grid", "share_grid", "conf_grid"]},
                  result=r.round(3).to_dict(), seed=C.SEED)
    C.LOG.add(id="RQ2-S5-STRUCT", rq="RQ2", kind="diagnostic", purpose="가산식 vs BSTS 구조 주입 재적합",
              data="rv5", model="BSTS 주 설정", settings={"cells": JC["structural_cells"]},
              result=ST[["code", "b", "s", "delta", "tau", "diff", "tau_error"]].round(4).to_dict("records"), seed=C.SEED)
    return dict(I=I, S=S, ST=ST)


if __name__ == "__main__":
    out = run()
    C.LOG.save()
    pd.set_option("display.width", 250)
    for k, v in out.items():
        print("==", k)
        print(v.round(4).to_string())
