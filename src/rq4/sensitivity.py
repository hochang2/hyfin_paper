"""6단계: 민감도. 세 축을 따로 계산하고 겹쳐 세지 않는다.

S1 평행추세 위반 (HonestDiD 식, Rambachan–Roth 2023): ES-B (ref = 발표 직전 주) 계수 사용.
   목표 θ = 시행 후 주(2~15주) 평균. 사전 δ = 선행계수(무예상 가정), 사후 δ 는 제약집합 안에서 최악.
   · RM(M̄): 사후 연속 변화 |δ_t+1 − δ_t| ≤ M̄ · max 사전 |Δδ|
   · SD(M) : 사후 쪽 2차 차분 |Δ²δ| ≤ M (−1주 이후 구간만; 사전 2차 차분 크기는 별도 보고)
   신뢰구간: β ~ N(b̂, V̂_CR1) 1000회 → 회당 식별집합 [lb, ub] 을 LP 로 → [q2.5(lb), q97.5(ub)] (보수적).
   (R HonestDiD 패키지가 없어 직접 구현. 조건부·FLCI 구간보다 넓다.)
S2 누락 변수 (Oster δ, Cinelli–Hazlett RV): 관측 공변량(변동성×2X)으로 보정. 평행추세 민감도와 별개.
S3 대조군 오염 s: 대조(1X) 변화 중 비율 s 가 규제가 낳은 것 → 보정 β_s = β + s·Δ_C (s=0 원래, s=1 사전·사후 단순비교)
S4 분류·원저화·하루 빼기(LODO)·도넛
S5 결합: s × M̄ 격자 (RM 식별집합을 s·Δ_C 만큼 이동) → 0 배제 여부 heatmap, 강건성 경계
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats

from src.rq4.collapsed import actual_positions, wide, window_effect
from src.rq4.common import CFG, FIG, PAL, INK2, SEED, ExpLog, plt_setup, save_table
from src.rq4.did import fit
from src.rq4.event_study import add_week, es_fit
from src.rq4.panel import main_panel, window

LOG = ExpLog("09_sensitivity")
G = CFG["groups"]
MBARS = [0, 0.25, 0.5, 0.75, 1, 1.5, 2]
MSD = [0, 0.02, 0.05, 0.1, 0.2, 0.3]
SGRID = [0, 0.1, 0.25, 0.5, 0.75, 1]


# ---------- S1 HonestDiD ----------
def es_inputs(fam, mB):
    names = [n for n in mB.names if n.startswith(fam + "_w")]
    weeks = np.array([int(n.split("_w")[1]) for n in names])
    o = np.argsort(weeks)
    names, weeks = [names[i] for i in o], weeks[o]
    idx = [mB.names.index(n) for n in names]
    b = mB.b[idx]
    V = mB.vcov("cr1", "code")[np.ix_(idx, idx)]
    return weeks, b, V


def bounds(weeks, beta, kind, M, target):
    """식별집합 [lb, ub] of θ = mean_{t∈target}(β_t − δ_t). 사전 δ = β (주 < −1), δ_{−1}=0."""
    pre = weeks < -1
    postw = weeks[~pre]                     # 0,1,...,15
    bp = beta[~pre]
    full_w = np.r_[weeks[pre], -1, postw]
    dpre = np.r_[beta[pre], 0.0]            # 사전 δ 고정 (−7..−1)
    n = len(postw)
    tmask = np.isin(postw, target)
    c = np.zeros(n)
    c[tmask] = 1 / tmask.sum()             # θ = c·β_post − c·δ_post
    A, ub = [], []
    if kind == "RM":
        mx = np.max(np.abs(np.diff(dpre))) if len(dpre) > 1 else 0
        lim = M * mx
        # |δ_0 − δ_{−1}| ≤ lim, |δ_{k+1} − δ_k| ≤ lim
        for k in range(n):
            row = np.zeros(n)
            row[k] = 1
            if k > 0:
                row[k - 1] = -1
            A += [row, -row]
            ub += [lim, lim] if k > 0 else [lim + 0.0, lim - 0.0]
    else:  # SD
        # 2차 차분: (δ_{t+1} − δ_t) − (δ_t − δ_{t−1}), t = −1, 0, ..., 마지막−1
        d_m2, d_m1 = dpre[-2], 0.0
        for t in range(-1, n - 1):
            row = np.zeros(n)
            const = 0.0
            # 위치: δ_{t+1} (t+1 ≥ 0 → post idx t+1), δ_t, δ_{t−1}
            row[t + 1] += 1
            if t >= 0:
                row[t] -= 2
            else:
                const += -2 * d_m1
            if t - 1 >= 0:
                row[t - 1] += 1
            elif t - 1 == -1:
                const += d_m1
            else:
                const += d_m2
            A += [row, -row]
            ub += [M - const, M + const]
    A, ub = np.array(A), np.array(ub)
    base = c @ bp
    bnd = [(None, None)] * n
    r1 = optimize.linprog(c, A_ub=A, b_ub=ub, bounds=bnd, method="highs")      # min c·δ
    r2 = optimize.linprog(-c, A_ub=A, b_ub=ub, bounds=bnd, method="highs")     # max c·δ
    if r1.status != 0 or r2.status != 0:
        return np.nan, np.nan
    return base - (-r2.fun), base - r1.fun


def honest(weeks, b, V, kind, M, target, draws):
    lb0, ub0 = bounds(weeks, b, kind, M, target)
    L, U = [], []
    for bb in draws:
        l, u = bounds(weeks, bb, kind, M, target)
        L.append(l)
        U.append(u)
    L, U = np.array(L), np.array(U)
    return lb0, ub0, np.nanquantile(L, 0.025), np.nanquantile(U, 0.975)


# ---------- S2 Oster, Cinelli–Hazlett ----------
def oster(b_s, r_s, b_l, r_l, rmax):
    delta = b_l * (r_l - r_s) / ((b_s - b_l) * (rmax - r_l)) if (b_s - b_l) * (rmax - r_l) != 0 else np.inf
    bstar = b_l - (b_s - b_l) * (rmax - r_l) / (r_l - r_s) if r_l != r_s else np.nan
    return delta, bstar


def ch_rv(t, dof, q=1.0, alpha=None):
    f = q * abs(t) / np.sqrt(dof)
    if alpha is not None:
        tcrit = stats.t.ppf(1 - alpha / 2, dof - 1)
        f = q * abs(t) / np.sqrt(dof) - tcrit / np.sqrt(dof - 1)
        if f < 0:
            return 0.0
    return 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)


def main():
    t0 = time.time()
    plt = plt_setup()
    rng = np.random.default_rng(SEED)
    d = window(main_panel())

    # ---- S1
    w = add_week(d)
    mB, _ = es_fit(w, (w["ew"] == -1).values)
    rows, hd = [], {}
    for fam in ["idx", "semi"]:
        weeks, b, V = es_inputs(fam, mB)
        target = weeks[weeks >= 2]
        draws = rng.multivariate_normal(b, V, size=1000)
        pre = weeks < -1
        dpre = np.r_[b[pre], 0.0]
        mx1 = np.max(np.abs(np.diff(dpre)))
        mx2 = np.max(np.abs(np.diff(dpre, 2)))
        theta = b[np.isin(weeks, target)].mean()
        for kind, grid in [("RM", MBARS), ("SD", MSD)]:
            for M in grid:
                lb, ub, cl, cu = honest(weeks, b, V, kind, M, target, draws)
                rows.append(dict(family=fam, restriction=kind, M=M, theta_hat=theta, id_lb=lb, id_ub=ub,
                                 ci_low=cl, ci_high=cu, excludes0=(cl > 0) or (cu < 0),
                                 max_pre_abs_diff=mx1, max_pre_abs_2nd_diff=mx2))
        hd[fam] = (weeks, b, V, target, draws)
    hon = pd.DataFrame(rows)
    save_table(hon, "t09_honestdid")
    for (fam, kind), g in hon.groupby(["family", "restriction"]):
        ok = g[g.excludes0]
        bd = ok.M.max() if len(ok) else np.nan
        LOG.add(id=f"S1_{fam}_{kind}", phase="6_sensitivity", kind="sensitivity_pt", hypothesis="평행추세 위반",
                purpose=f"HonestDiD {kind}", treated=fam, model="ES-B ref=−1주, 목표=시행 후 주 평균",
                estimand="θ", estimate=g.theta_hat.iloc[0],
                ci_low=g[g.M == g.M.min()].ci_low.iloc[0], ci_high=g[g.M == g.M.min()].ci_high.iloc[0],
                sensitivity=f"0 배제 최대 M = {bd}; 사전 max|Δδ|={g.max_pre_abs_diff.iloc[0]:.3f}, "
                            f"max|Δ²δ|={g.max_pre_abs_2nd_diff.iloc[0]:.3f}",
                interpretation="M=0(RM)도 0 포함 → 위반 없이도 유의 아님" if not g.excludes0.any() else
                f"breakdown M ≈ {bd}")

    # ---- S2 Oster / CH
    from src.rq4.diagnostics import covariates
    dc = covariates(d).dropna(subset=["x_rv", "x_vk"])
    ms_, X = fit(dc)
    ml_, _ = fit(dc, extra=["x_rv", "x_vk", "x_absr"])
    os_rows = []
    wdiff = lambda m: np.array([1.0 if n == "post_semi" else -1.0 if n == "post_idx" else 0 for n in m.names])
    for term in ["post_idx", "post_semi", "diff"]:
        if term == "diff":
            bs, bl = wdiff(ms_) @ ms_.b, wdiff(ml_) @ ml_.b
            lc_i = ml_.lincomb(wdiff(ml_), "iid")
            lc_c = ml_.lincomb(wdiff(ml_), "cr1", "code")
            t_i, t_c = lc_i["coef"] / lc_i["se"], lc_c["coef"] / lc_c["se"]
        else:
            j = ms_.names.index(term)
            bs, bl = ms_.b[j], ml_.b[ml_.names.index(term)]
            si, sc = ml_.summary("iid"), ml_.summary("cr1", cluster="code")
            t_i, t_c = si.loc[term, "t"], sc.loc[term, "t"]
        rs, rl = ms_.r2_within, ml_.r2_within
        rmax = min(1.3 * rl, 1.0)
        dl, bstar = oster(bs, rs, bl, rl, rmax)
        G_ = dc.code.nunique()
        os_rows.append(dict(term=term, beta_short=bs, beta_long=bl, r2_short=rs, r2_long=rl, rmax=rmax,
                            oster_delta=dl, beta_star_delta1=bstar,
                            rv_iid=ch_rv(t_i, ml_.df_resid), rv_iid_a05=ch_rv(t_i, ml_.df_resid, alpha=.05),
                            rv_cr1=ch_rv(t_c, G_ - 1), rv_cr1_a05=ch_rv(t_c, G_ - 1, alpha=.05),
                            partial_r2_iid=t_i ** 2 / (t_i ** 2 + ml_.df_resid)))
    # 벤치마크: 관측 공변량 x_vk 의 부분 R²
    sc_l = ml_.summary("iid")
    bench = {k: sc_l.loc[k, "t"] ** 2 / (sc_l.loc[k, "t"] ** 2 + ml_.df_resid) for k in ["x_rv", "x_vk", "x_absr"]}
    ost = pd.DataFrame(os_rows)
    ost["bench_partial_r2"] = str({k: round(v, 4) for k, v in bench.items()})
    save_table(ost, "t09b_oster_cinelli")
    for _, r in ost.iterrows():
        LOG.add(id=f"S2_{r.term}", phase="6_sensitivity", kind="sensitivity_ovb", hypothesis="누락 변수",
                purpose="Oster δ / Cinelli–Hazlett RV", treated=r.term,
                controls="계열 실현변동성×2X, VKOSPI×2X, |수익률|×2X", estimate=r.beta_long,
                sensitivity=f"Oster δ={r.oster_delta:.2f}, β*(δ=1,Rmax=1.3R²)={r.beta_star_delta1:.3f}; "
                            f"RV(iid)={r.rv_iid:.3f}, RV_α(iid)={r.rv_iid_a05:.3f}, RV(CR1 t, G−1)={r.rv_cr1:.3f}, RV_α(CR1)={r.rv_cr1_a05:.3f}",
                interpretation="R² 는 FE 흡수 후 within. 공변량 R² 기여가 작아 Oster δ 는 불안정할 수 있음")

    # ---- S3 오염
    ib, ip = actual_positions()
    W, V = wide()
    act = window_effect(ib, ip)
    spec = {"idx": (G["B_index_2x"], G["E_index_1x"]), "semi": (G["C_semi_2x"], G["E_semi_1x"])}
    codes = set(d.code)
    dC, dT = {}, {}
    for f, (tr, co) in spec.items():
        tr = [c for c in tr if c in codes]
        co = [c for c in co if c in codes]
        dT[f] = W.iloc[ip][tr].mean().mean() - W.iloc[ib][tr].mean().mean()
        dC[f] = W.iloc[ip][co].mean().mean() - W.iloc[ib][co].mean().mean()
    dist = pd.read_csv(FIG.parent / "tables" / "t03b_placebo_windows_nonoverlap.csv")
    crow = []
    for s in SGRID:
        r = {"s": s}
        for f in ["idx", "semi"]:
            b_s = act[f] + s * dC[f]
            sd = dist[f].std()
            r[f] = b_s
            r[f + "_lo"], r[f + "_hi"] = b_s - 2.07 * sd, b_s + 2.07 * sd
        r["diff"] = r["semi"] - r["idx"]
        crow.append(r)
    cont = pd.DataFrame(crow)
    cont["dT_idx"], cont["dC_idx"], cont["dT_semi"], cont["dC_semi"] = dT["idx"], dC["idx"], dT["semi"], dC["semi"]
    save_table(cont, "t09c_contamination")
    LOG.add(id="S3", phase="6_sensitivity", kind="sensitivity_contam", hypothesis="대조군 오염",
            purpose="1X 대조 변화 중 비율 s 가 규제 기인", model="β_s = β + s·Δ_C",
            estimate=f"Δ_C idx {dC['idx']:+.3f}, semi {dC['semi']:+.3f}",
            sensitivity="; ".join(f"s={r.s}: idx {r.idx:+.3f}, semi {r.semi:+.3f}, diff {r['diff']:+.3f}" for _, r in cont.iterrows()),
            interpretation="반도체 1X 가 사후 크게 줄어(Δ_C<0) s↑ 이면 β_C 가 작아진다. 대조가 규제 이후 오히려 줄어든 것이 "
                           "규제 탓(수요 냉각 확산)이라면 2X 상대 증가는 과대추정")

    # ---- S4 분류(강건성 참조), 원저화, LODO, 도넛
    s4 = []
    for lo, hi in [(0.01, 0.99), (0.05, 0.95)]:
        Wd = W.copy()
        sub = Wd.iloc[np.r_[ib, ip]]
        ql, qh = sub.quantile(lo), sub.quantile(hi)
        Wd.iloc[np.r_[ib, ip]] = sub.clip(ql, qh, axis=1)
        from src.rq4 import collapsed as Cm
        Cm._WIDE["_w"] = (Wd, V)
        r = window_effect(ib, ip, outcome="_w")
        s4.append(dict(test=f"원저화 상품별 {int(lo * 100)}/{int(hi * 100)}%", **{k: r[k] for k in ["pooled", "idx", "semi", "diff"]}))
    lodo = []
    for k in np.r_[ib, ip]:
        r = window_effect(np.setdiff1d(ib, [k]), np.setdiff1d(ip, [k]))
        lodo.append(dict(date=W.index[k], **{kk: r[kk] for kk in ["pooled", "idx", "semi", "diff"]}))
    lodo = pd.DataFrame(lodo)
    save_table(lodo, "t09d_lodo")
    for k in ["pooled", "idx", "semi", "diff"]:
        s4.append(dict(test=f"LODO 범위 {k}", **{kk: np.nan for kk in ["pooled", "idx", "semi", "diff"]},
                       note=f"{lodo[k].min():+.3f} ~ {lodo[k].max():+.3f}"))
    # 주 단위 LOWO
    wk = pd.Series(W.index[np.r_[ib, ip]]).dt.to_period("W")
    allp = np.r_[ib, ip]
    lowo = []
    for wv in wk.unique():
        drop = allp[(wk == wv).values]
        r = window_effect(np.setdiff1d(ib, drop), np.setdiff1d(ip, drop))
        lowo.append(r)
    lowo = pd.DataFrame(lowo)
    for k in ["pooled", "idx", "semi", "diff"]:
        s4.append(dict(test=f"주 빼기(LOWO) 범위 {k}", note=f"{lowo[k].min():+.3f} ~ {lowo[k].max():+.3f}"))
    for kd in [3, 5, 10]:
        r = window_effect(ib[:-kd], ip[kd:])
        s4.append(dict(test=f"도넛 ±{kd}일 (기준 끝·사후 시작 제외)", **{k: r[k] for k in ["pooled", "idx", "semi", "diff"]}))
    s4 = pd.DataFrame(s4)
    save_table(s4, "t09e_other_sensitivity")
    for i, r in s4.iterrows():
        LOG.add(id=f"S4_{i:02d}", phase="6_sensitivity", kind="sensitivity_other", purpose=r["test"],
                estimate=(f"pooled {r['pooled']:+.3f}; idx {r['idx']:+.3f}; semi {r['semi']:+.3f}; diff {r['diff']:+.3f}"
                          if pd.notna(r.get("pooled")) else r.get("note")),
                sensitivity=r.get("note") if isinstance(r.get("note"), str) else "")

    # ---- S5 결합 heatmap (RM) : s × M̄, 사건연구 θ 에 s·Δ_C 이동
    jrows = []
    for fam in ["idx", "semi"]:
        for M in MBARS:
            hr = hon[(hon.family == fam) & (hon.restriction == "RM") & (hon.M == M)].iloc[0]
            cl, cu = hr.ci_low, hr.ci_high       # S1 과 같은 구간 (일관성)
            for s in SGRID:
                sh = s * dC[fam]
                jrows.append(dict(family=fam, M=M, s=s, ci_low=cl + sh, ci_high=cu + sh,
                                  excl0=(cl + sh > 0) or (cu + sh < 0), sign=np.sign(cl + sh) if cl + sh > 0 else
                                  (np.sign(cu + sh) if cu + sh < 0 else 0)))
    J = pd.DataFrame(jrows)
    save_table(J, "t09f_joint_s_M")
    fr = J[J.excl0].groupby(["family", "s"]).M.max().unstack(0).reindex(SGRID)   # NaN = 어떤 M̄ 에서도 0 배제 못함
    save_table(fr.reset_index(), "t09g_frontier")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, fam, lab in zip(axes, ["idx", "semi"], ["지수 2X (β_B)", "반도체 2X (β_C)"]):
        g = J[J.family == fam].pivot(index="s", columns="M", values="ci_low")
        h = J[J.family == fam].pivot(index="s", columns="M", values="ci_high")
        val = np.where(g.values > 0, g.values, np.where(h.values < 0, h.values, 0))
        vmax = np.nanmax(np.abs(val)) or 1
        im = ax.imshow(val, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
        for i in range(val.shape[0]):
            for j in range(val.shape[1]):
                ax.text(j, i, "0포함" if val[i, j] == 0 else f"{val[i, j]:+.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(val[i, j]) > 0.5 * vmax else INK2)
        ax.set_xticks(range(len(g.columns)))
        ax.set_xticklabels(g.columns)
        ax.set_yticks(range(len(g.index)))
        ax.set_yticklabels(g.index)
        ax.set_xlabel("Mbar (RM: 사후 변화 ≤ Mbar × 사전 최대 변화)")
        ax.set_ylabel("오염 s")
        ax.set_title(f"{lab}: 0 에 가까운 CI 끝 (0포함=흰색)", fontsize=9)
    fig.suptitle("그림 7. 평행추세 위반 × 대조군 오염 결합 민감도", fontsize=10)
    fig.savefig(FIG / "f07_joint_heatmap.png")
    plt.close(fig)

    # HonestDiD 그림
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (kind, grid) in zip(axes, [("RM", MBARS), ("SD", MSD)]):
        for i, fam in enumerate(["idx", "semi"]):
            g = hon[(hon.family == fam) & (hon.restriction == kind)]
            x = np.arange(len(g)) + (i - 0.5) * 0.18
            ax.vlines(x, g.ci_low, g.ci_high, color=PAL[i], lw=2, label={"idx": "지수", "semi": "반도체"}[fam])
            ax.scatter(x, (g.id_lb + g.id_ub) / 2, color=PAL[i], s=10)
        ax.set_xticks(range(len(grid)))
        ax.set_xticklabels(grid)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel("Mbar" if kind == "RM" else "M")
        ax.set_title(f"HonestDiD {kind}: 시행 후 평균 θ 의 95% 구간 (점=식별집합 중점)", fontsize=9)
        ax.legend(fontsize=7)
    fig.savefig(FIG / "f08_honestdid.png")
    plt.close(fig)
    LOG.add(id="S5", phase="6_sensitivity", kind="sensitivity_joint", purpose="s × M̄ 결합 (RM)",
            sensitivity="frontier(0 배제 최대 M̄) " + fr.to_string().replace("\n", " | "), runtime_s=time.time() - t0)
    LOG.save()
    print(hon.round(3).to_string())
    print(ost.round(4).to_string())
    print(cont.round(3).to_string())
    print(s4.round(3).to_string())
    print(fr)
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
