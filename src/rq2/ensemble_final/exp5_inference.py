"""Exp5 인과추론 신뢰성 감사 — 앙상블 '선택·가중치 학습 과정까지 포함한' 시간 placebo 검정.

ml_followup Exp5 의 블록 구조를 그대로 쓴다: 연속 가짜 개입일 nb 개 = 귀무 표본, 바로 다음 격자점 = '모사 실제 개입일'(참효과 0).
각 블록에서 앙상블 가중치는 그 블록의 가짜 창(시험일 이전에 실현된 것)만으로 학습한다 (3종목 합동).

귀무분포 두 가지:
  in      : 시험일 가중치와 같은 가중치로 계산한 가짜 창 오차 (가중치가 그 창을 이미 봄 → 과소 분산 위험)
  cf      : 각 가짜 창 j 의 오차를 'j 와 겹치지 않는 다른 가짜 창'으로 다시 학습한 가중치로 계산 (cross-fitted).
            시험일 가중치도 시험 창과 겹치지 않는 창으로 학습되므로 귀무 오차와 시험 오차가 같은 절차에서 나온다.
  sel     : 전략 선택까지 포함 — 시험일엔 블록 안 cf MSE 최소 전략을, 각 귀무 창 j 엔 j 와 겹치지 않는 창의 cf MSE 최소
            전략을 고른다 (선택 단계도 placebo 마다 반복).
검정: ml_followup.common.null_test (neff · strict). 효과 δ = log(1 + p/100) 를 시험 창 오차에 더해 검정력.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.rq2.ensemble_final import figures as FG
from src.rq2.ensemble_final import loader as L
from src.rq2.ensemble_final import strategies as S
from src.rq2.ml_followup import common as C

A = L.CFG["audit"]
O = L.CFG["online"]
EFF = A["effects_pct"]
AUD_GKD = ["single_bsts", "single_ols", "eq_all", "old_stack_replica", "ftl_single", "top3", "invmse_top3",
           "bc_invmse_top3", "bc_bsts", "mv_lw", "greedy", "simplex", "simplex_l2", "bsts_shrink", "stat_simplex",
           "simplex_l2|hier"]
AUD_GK10 = ["single_bsts", "single_ols", "col:timesfm_xreg", "col:old_stack", "eq_all", "old_stack_replica",
            "invmse_top3", "bc_bsts", "simplex", "simplex_l2", "bsts_shrink", "stat_simplex", "h_bsts_ols_tfm",
            "h_bsts_chr_uc", "simplex_l2|hier"]
SEL_EXCL = {"single_bsts", "single_ols", "col:timesfm_xreg", "col:old_stack"}


def _fit(T: S.Task, name, rows):
    """rows 로 학습한 함수: 행 인덱스 -> 예측. hier 는 종목별 shrinkage."""
    if name.startswith("col:"):
        col = T.W[name[4:]].values
        return lambda r: col[r]
    base, _, scope = name.partition("|")
    ms = T.members(base)
    P = T.P(ms)
    y, i0, code = T.y, T.i0, T.code
    ctx = {"names": ms, "neff": S.n_eff(i0[rows], T.b, T.step)}
    wg, cg, _ = S.weights(base, P[rows], y[rows], ctx)
    if scope != "hier":
        return lambda r: P[r] @ wg + cg
    per = {}
    for c in np.unique(code[rows]):
        rc = rows[code[rows] == c]
        nc = S.n_eff(i0[rc], T.b, T.step)
        lam = nc / (nc + O["hier_kappa"])
        if len(rc) >= O["min_past_rows_code"]:
            wc, cc, _ = S.weights(base, P[rc], y[rc], {"names": ms, "neff": nc, "min_rows": O["min_past_rows_code"]})
            per[c] = (lam * wc + (1 - lam) * wg, lam * cc + (1 - lam) * cg)
    return lambda r: P[r] @ per.get(code[r], (wg, cg))[0] + per.get(code[r], (wg, cg))[1]


def _block(T: S.Task, strats, ps, it):
    """한 블록: 전략별 시험 오차·in·cf 귀무 오차 (행 단위)."""
    i0, y = T.i0, T.y
    use = ps[ps + T.b <= it]
    rows = np.where(np.isin(i0, use))[0]
    trow = np.where(i0 == it)[0]
    if len(rows) == 0 or len(trow) == 0:
        return None
    out = {}
    for s in strats:
        if s == "sel":
            continue
        if not s.startswith("col:") and T.members(s.split("|")[0]) is None:
            continue
        f = _fit(T, s, rows)
        e_test = np.array([y[r] - f(r) for r in trow])
        e_in = np.array([y[r] - f(r) for r in rows])
        e_cf = np.full(len(rows), np.nan)
        for j in np.unique(i0[rows]):
            tr = rows[np.abs(i0[rows] - j) >= T.b]
            if len(tr) < 3:
                continue
            fj = _fit(T, s, tr)
            for k in np.where(i0[rows] == j)[0]:
                e_cf[k] = y[rows[k]] - fj(rows[k])
        out[s] = (e_test, e_in, e_cf)
    # 선택 포함 절차
    cands = [s for s in out if s not in SEL_EXCL]
    if cands and "sel" in strats:
        cf = {s: out[s][2] for s in cands}
        ok = lambda m: np.isfinite(m).sum() > 0
        sc = {s: np.nanmean(v ** 2) for s, v in cf.items() if ok(v)}
        st = min(sc, key=sc.get)
        e_sel = np.full(len(rows), np.nan)
        for j in np.unique(i0[rows]):
            far = np.abs(i0[rows] - j) >= T.b
            scj = {s: np.nanmean(v[far] ** 2) for s, v in cf.items() if np.isfinite(v[far]).sum() >= 3}
            if not scj:
                continue
            sj = min(scj, key=scj.get)
            k = i0[rows] == j
            e_sel[k] = cf[sj][k]
        out["sel"] = (out[st][0], np.full(len(rows), np.nan), e_sel, st)
    return rows, trow, out


def _tests(T: S.Task, k0, ps, it, strats, rng_seed):
    r = _block(T, strats, ps, it)
    if r is None:
        return []
    rows, trow, out = r
    rng = np.random.default_rng(rng_seed)
    i0, code = T.i0, T.code
    recs = []
    for s, v in out.items():
        e_test, e_in, e_cf = v[:3]
        pick = v[3] if len(v) > 3 else s
        for nt, en in (("in", e_in), ("cf", e_cf)):
            if not np.isfinite(en).any():
                continue
            for ci, r_t in enumerate(trow):
                c = code[r_t]
                m = code[rows] == c
                e, ii = en[m], i0[rows][m]
                for meth in A["methods"]:
                    m0, crit, n = C.null_test(e, ii, T.L, T.step, meth, A["min_eff"], 999, rng)
                    recs.append({"window": T.window, "strategy": s, "pick": pick, "null": nt, "method": meth,
                                 "block": int(k0), "i_test": int(it), "code": c, "e_star": float(e_test[ci]),
                                 "m0": m0, "crit": crit, "n_used": n,
                                 "s0": float(np.nanstd(e, ddof=1)) if np.isfinite(e).sum() > 2 else np.nan})
    return recs


def audit(ds, strats, nb, sk, windows, n_jobs=4):
    f = L.cpath(f"audit_{ds}")
    if f.exists():
        return pd.read_parquet(f)
    recs = []
    for w in windows:
        W = L.wide(ds, w)
        names = [m for m in L.POOL["full"] if m in W.columns]
        T = S.Task(W, w, L.STEP[ds], names)
        cl = np.array(sorted(np.unique(T.i0)))
        jobs = [(k0, cl[k0:k0 + nb], cl[k0 + nb]) for k0 in range(0, len(cl) - nb, sk)]
        res = Parallel(n_jobs=n_jobs)(delayed(_tests)(T, k0, ps, it, strats, L.SEED + k0) for k0, ps, it in jobs)
        for r in res:
            recs += r
        print(f"  audit {ds} {w}: {len(jobs)} 블록", flush=True)
    R = pd.DataFrame(recs)
    R.to_parquet(f)
    return R


def summarize(R, step, slide):
    R = R.copy()
    R["testable"] = np.isfinite(R.crit)
    d = R.e_star - R.m0
    R["rej0"] = np.where(R.testable, np.abs(d) > R.crit, np.nan)
    for p in EFF:
        R[f"pw{p}"] = np.where(R.testable, np.abs(d + np.log1p(p / 100)) > R.crit, np.nan)
    rows = []
    for (w, s, nt, meth), g in R.groupby(["window", "strategy", "null", "method"]):
        t = g[g.testable]
        a, b = L.WIN[w]
        nbk = g.block.nunique()
        r = {"window": w, "strategy": s, "null": nt, "method": meth, "n_tests": len(g),
             "testable_share": g.testable.mean(), "fpr": t.rej0.mean() if len(t) else np.nan,
             "n_eff_tests": round(2 * nbk * min(1.0, slide * step / max(b - a, 1)), 1),   # ml_followup 와 같은 정의
             "effect_bias_raw": t.e_star.mean(), "effect_rmse_raw": np.sqrt((t.e_star ** 2).mean()),
             "effect_bias_c": (t.e_star - t.m0).mean(), "effect_rmse_c": np.sqrt(((t.e_star - t.m0) ** 2).mean()),
             "crit_mean": t.crit.mean(), "s0_mean": t.s0.mean(),
             "sd_true": (t.e_star - t.m0).std(ddof=1)}
        r["coverage"] = 1 - r["fpr"]
        for p in EFF:
            r[f"power{p}"] = t[f"pw{p}"].mean() if len(t) else np.nan
        hit = [abs(p) for p in EFF if len(t) and t[f"pw{p}"].mean() >= 0.8]
        r["mde_emp_pct"] = min(hit) if hit else np.nan
        r["mde_req_pct"] = -np.expm1(-(r["crit_mean"] + 0.84 * r["sd_true"])) * 100
        if s == "sel":
            r["sel_picks"] = "; ".join(f"{k}:{v}" for k, v in g.drop_duplicates(["block"]).pick.value_counts().head(4).items())
        rows.append(r)
    S_ = pd.DataFrame(rows)
    se = np.sqrt(S_.fpr * (1 - S_.fpr) / S_.n_eff_tests.clip(lower=1))
    S_["fpr_lo"], S_["fpr_hi"] = (S_.fpr - 1.96 * se).clip(lower=0), (S_.fpr + 1.96 * se).clip(upper=1)
    return S_


def fig(Sg, Sk):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (Sx, ttl, w) in zip(axs, [(Sg, "GK 조밀(3일 격자) h20, neff", "h20"), (Sk, "GK 10일 격자(FM 포함) h10, neff", "h10")]):
        s = Sx[(Sx.window == w) & (Sx.method == "neff")]
        strats = list(dict.fromkeys(s[s.null == "cf"].sort_values("fpr").strategy))
        y = np.arange(len(strats))
        for off, nt, col in ((-0.15, "in", FG.CAT[1]), (0.15, "cf", FG.CAT[0])):
            t = s[s.null == nt].set_index("strategy").reindex(strats)
            ax.errorbar(t.fpr, y + off, xerr=[t.fpr - t.fpr_lo, t.fpr_hi - t.fpr], fmt="o", color=col, ms=5,
                        elinewidth=1.5, capsize=0, label={"in": "같은 가중치(in-sample) 귀무", "cf": "cross-fitted 귀무 (선택 포함)"}[nt])
        ax.axvline(0.05, color=FG.INK2, lw=1, ls="--")
        ax.set_yticks(y, strats, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("오기각률 (명목 5%)")
        ax.set_title(ttl, fontsize=9, loc="left")
        ax.legend(fontsize=7, loc="lower right")
    FG.save(fig, "exp5_fpr_in_vs_cf")


def run():
    Rg = audit("gkd", AUD_GKD + ["sel"], A["block_n_cut"], A["block_slide_days"] // L.STEP["gkd"], ["h10", "h20", "h30"])
    Sg = summarize(Rg, L.STEP["gkd"], A["block_slide_days"] // L.STEP["gkd"])
    Rk = audit("gk", AUD_GK10 + ["sel"], A["gk10_block_n_cut"], A["gk10_block_slide"], ["h10", "h30"])
    Sk = summarize(Rk, L.STEP["gk"], A["gk10_block_slide"])
    L.save_table(Sg.assign(dataset="gkd"), "exp5_audit_gkd")
    L.save_table(Sk.assign(dataset="gk10"), "exp5_audit_gk10_fm")
    fig(Sg, Sk)
    return Sg, Sk


if __name__ == "__main__":
    run()
