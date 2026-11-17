"""Exp6 실제 RQ2 적용 (탐색적 강건성 분석 — 사전 지정 주 분석은 그대로 유지).

· 앙상블 구성은 Exp4 에서 GK 선정 구간으로 고정한 것(selection_frozen.json)을 그대로 쓴다. 실제 결과를 본 뒤 바꾸지 않는다.
· 가중치는 처치 3종목의 rv5 가짜 개입일 창 중 실제 발표일(i0=190) 이전에 완전히 실현된 것만으로 학습 (3종목 합동).
  transition([0,10)) <- rv5 h10 가짜 창, post_effective([10,71)) <- rv5 post_effective 가짜 창.
· 실제 효과 = ybar − 예측 (ml_ext space 캐시, i0=190). 규제 이후 실제 변동성은 가중치·국면 정의에 쓰지 않는다.
· 시간 placebo p: 같은 종목의 가짜 창 cross-fitted 오차 (Exp5 cf 와 같은 절차), neff·strict.
· 공간 placebo p: donor 175 종목에 같은 가중치를 적용한 효과 분포 (처치 종목은 donor 와 교환가능하지 않음 → 보수적).
· 음성통제: 005935(삼성전자 우선주, 규제 비대상). 같은 회사 차이 005930 − 005935 도 같은 절차로.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.rq2.phase1.ensemble import figures as FG
from src.rq2.phase1.ensemble import loader as L
from src.rq2.phase1.ensemble import strategies as S
from src.rq2.phase1.ml_followup import common as C

RC = L.CFG["real"]
TREAT = RC["codes"]
O = L.CFG["online"]
MAIN = L.ROOT / "src/rq2/phase1/main_study/results/tables"


def _frozen():
    return json.loads((L.RES / "selection_frozen.json").read_text(encoding="utf-8"))["picks"]


def _best_single(pool):
    """사전 규칙: rv5 발표 전 가짜 개입일 h10 RMSE (3종목 합산) 최소."""
    W = L.wide("rv5", "h10")
    r = {m: np.sqrt(np.mean((W.ybar - W[m]) ** 2)) for m in pool if m in W}
    return min(r, key=r.get), r


def _task(window):
    W = L.wide("rv5", RC["windows"][window])
    return S.Task(W, RC["windows"][window], L.STEP["rv5"], [m for m in L.POOL["full"] if m in W.columns])


def _predictor(T: S.Task, name, rows, code=None, regime=None, reg_now=None):
    """학습 행 rows 로 만든 (w, c, members). code 가 주어지면 hier (종목 shrinkage), regime 이면 같은 국면 행."""
    if name.startswith("m:") or name.startswith("col:"):
        return np.ones(1), 0.0, [name.split(":")[1]]
    base, _, scope = name.partition("|")
    ms = T.members(base)
    P = T.P(ms)
    y, i0 = T.y, T.i0
    if scope.startswith("regime") and regime is not None:
        same = rows[np.isin(i0[rows], [k for k, v in regime.items() if v == reg_now])]
        if len(same) >= O["regime_min_rows"]:
            rows = same
    wg, cg, _ = S.weights(base, P[rows], y[rows], {"names": ms, "neff": S.n_eff(i0[rows], T.b, T.step)})
    if scope == "hier" and code is not None:
        rc = rows[T.code[rows] == code]
        nc = S.n_eff(i0[rc], T.b, T.step)
        if len(rc) >= O["min_past_rows_code"]:
            lam = nc / (nc + O["hier_kappa"])
            wc, cc, _ = S.weights(base, P[rc], y[rc], {"names": ms, "neff": nc, "min_rows": O["min_past_rows_code"]})
            return lam * wc + (1 - lam) * wg, lam * cc + (1 - lam) * cg, ms
    return wg, cg, ms


def _null_errors(T, name, code, regime=None):
    """해당 종목 가짜 창의 cross-fitted 오차 (각 창 j 는 j 와 겹치지 않는 창으로 학습한 가중치)."""
    rows = np.arange(len(T.y))
    out_e, out_i = [], []
    for r in np.where(T.code == code)[0]:
        j = T.i0[r]
        tr = rows[np.abs(T.i0 - j) >= T.b]
        if name.startswith("m:") or name.startswith("old"):
            pr = T.W[name.split(":")[-1] if name.startswith("m:") else name].values[r]
        else:
            reg_now = regime.get(j) if regime is not None else None
            w, c, ms = _predictor(T, name, tr, code, regime, reg_now)
            pr = T.W[ms].values[r] @ w + c
        out_e.append(T.y[r] - pr)
        out_i.append(j)
    return np.array(out_e), np.array(out_i)


def run():
    pk = _frozen()
    best_ml, r_ml = _best_single(L.POOL["ml"])
    best_fm, r_fm = _best_single(L.POOL["fm"])
    best_any, _ = _best_single(L.POOL["full"])
    rows_spec = [  # (표시 이름, 전략 키, 구분)
        ("ml_ext BSTS (같은 space 캐시)", "m:bsts", "기준"),
        (f"최고 단일 ML ({best_ml})", f"m:{best_ml}", "단일"),
        (f"최고 FM ({best_fm})", f"m:{best_fm}", "단일"),
        (f"최고 단일 전체 ({best_any})", f"m:{best_any}", "단일"),
        ("기존 stacking (ml_ext ens_stack)", "old_stack", "기존 앙상블"),
        (f"새 선택적 앙상블 [사전고정: {pk['overall']}]", pk["overall"], "새 앙상블 (주)"),
        (f"종목 적응 [{pk['stock_adaptive']}]", pk["stock_adaptive"], "새 앙상블 (탐색)"),
        (f"국면 적응 [{pk['regime_aware']}]", pk["regime_aware"], "새 앙상블 (탐색)"),
    ]
    reg_rv5 = L.regime_frame("rv5").set_index("i0").trend.to_dict()
    reg_real = L.regime_frame("space").trend.iat[0]
    recs, dons = [], []
    for wname in RC["windows"]:
        T = _task(wname)
        Sp = L.wide("space", wname)
        donors = Sp[~Sp.code.isin(TREAT)]
        allrows = np.arange(len(T.y))   # 모든 가짜 창은 i0+b <= 190 (loader 에서 보장)
        for label, key, kind in rows_spec:
            regime = reg_rv5 if "|regime" in key else None
            eff, nulls = {}, {}
            for c in TREAT:
                if key.startswith("m:") or key == "old_stack":
                    col = key[2:] if key.startswith("m:") else key
                    pred_c = Sp.loc[Sp.code == c, col].iat[0]
                    don_pred = donors[col].values
                else:
                    w, cc, ms = _predictor(T, key, allrows, c, regime, reg_real)
                    pred_c = Sp.loc[Sp.code == c, ms].values[0] @ w + cc
                    wg, cg, _ = _predictor(T, key, allrows, None, regime, reg_real)
                    don_pred = donors[ms].values @ wg + cg
                eff[c] = Sp.loc[Sp.code == c, "ybar"].iat[0] - pred_c
                nulls[c] = _null_errors(T, key if key != "old_stack" else "old_stack", c, regime)
                don_eff = donors.ybar.values - don_pred
            if kind == "기준" or key == pk["overall"]:
                dons.append(pd.DataFrame({"window": wname, "strategy": key, "code": donors.code, "effect": don_eff}))
            for c in TREAT + ["diff_005930_005935"]:
                if c.startswith("diff"):
                    est = eff["005930"] - eff["005935"]
                    e0, i0a = nulls["005930"]
                    e1, i0b = nulls["005935"]
                    m = np.isin(i0a, i0b)
                    en, ii = e0[m] - e1[np.isin(i0b, i0a[m])], i0a[m]
                    sp = np.nan
                    mde_sp = np.nan
                else:
                    est = eff[c]
                    en, ii = nulls[c]
                    sp = (1 + np.sum(np.abs(don_eff) >= abs(est))) / (1 + len(don_eff))
                    mde_sp = np.quantile(np.abs(don_eff), 0.95) + 0.84 * np.std(don_eff, ddof=1)
                r = {"window": wname, "row": label, "kind": kind, "strategy": key, "code": c, "effect_log": est,
                     "effect_pct": 100 * np.expm1(est), "p_space": sp, "mde_space_log": mde_sp,
                     "n_null": len(en)}
                for meth in L.CFG["audit"]["methods"]:
                    pp = C.null_p(est, en, ii, T.L, T.step, meth, L.CFG["audit"]["min_eff"], B=999,
                                  rng=np.random.default_rng(L.SEED))
                    r[f"p_time_{meth}"] = pp["p"]
                    if meth == "neff":
                        r["m0"], r["crit"] = pp["m0"], pp["crit"]
                        r["effect_c"] = est - pp["m0"] if np.isfinite(pp["m0"]) else np.nan
                        r["ci_lo"], r["ci_hi"] = r["effect_c"] - pp["crit"], r["effect_c"] + pp["crit"]
                        s0 = np.std(en, ddof=1) if len(en) > 2 else np.nan
                        r["mde_time_log"] = pp["crit"] + 0.84 * s0 if np.isfinite(pp["crit"]) else np.nan
                recs.append(r)
    R = pd.DataFrame(recs)
    R["mde_time_pct"] = 100 * np.expm1(R.mde_time_log)
    R["mde_space_pct"] = 100 * np.expm1(R.mde_space_log)
    sig = lambda x: np.isfinite(x) and x < 0.05
    R["significant_any"] = [sig(a) or sig(b) for a, b in zip(R.p_time_neff, R.p_space)]
    R["conclusion_kept"] = np.where(R.code == "005935", np.where(R.significant_any, "음성통제 유의(경고)", "음성통제 정상"),
                                    np.where(R.significant_any, "변경 신호(탐색적)", "유지 (유의한 변화 없음)"))
    main = pd.read_csv(MAIN / "rq2_synthesis_main.csv")
    main = main[main.code.astype(str).str.zfill(6).isin(TREAT) | main.design.str.contains("차이|within|DID", regex=True)]
    L.save_table(R, "exp6_real_effect")
    L.save_table(main, "exp6_main_study_reference")
    D = pd.concat(dons, ignore_index=True)
    L.save_table(D, "exp6_space_donor_effects")
    sel = {"best_ml": best_ml, "best_fm": best_fm, "best_any": best_any, "rmse_ml": r_ml, "rmse_fm": r_fm,
           "frozen_picks": pk, "regime_real": reg_real}
    (L.RES / "tables" / "exp6_choices.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1, default=float),
                                                       encoding="utf-8")
    fig(R, D, pk)
    return R, main, sel


def fig(R, D, pk):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(17, 7.5), layout="constrained")
    for ax, w in zip(axs, RC["windows"]):
        s = R[(R.window == w) & R.code.isin(TREAT)]
        labs, est = [], []
        for _, t in s.iterrows():
            labs.append(f"{t.code} · {t.row}")
            est.append(t.effect_log)
        y = np.arange(len(labs))
        dd = D[(D.window == w) & (D.strategy == "m:bsts")].effect
        lo, hi = np.quantile(dd, [0.025, 0.975])
        ax.axvspan(lo, hi, color="#e4e3df", alpha=0.6, lw=0, label="donor 효과 95% 범위 (BSTS)")
        col = [FG.CAT[{"005930": 0, "000660": 2, "005935": 4}[l[:6]]] for l in labs]
        ax.scatter(est, y, c=col, s=30, zorder=3, edgecolors=FG.SURF, linewidths=1)
        ax.axvline(0, color=FG.INK2, lw=1)
        ax.set_yticks(y, labs, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_title(f"{w}: 실제 효과 (log, 원점수) — 파랑 005930 · 초록 000660 · 분홍 005935(음성통제)", fontsize=8.5, loc="left")
        ax.legend(fontsize=7, loc="lower right")
    FG.save(fig, "exp6_real_effects")


if __name__ == "__main__":
    run()
