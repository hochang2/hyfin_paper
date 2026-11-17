"""Exp4 엄격한 표본 밖 검증.

설계 (config.yaml 에서 결과 전 고정):
  GK  : 2016-06~2017 예열 | 2018~2021 선정 구간 | 2022~2026-01 최종 평가 구간.
        선정 기준 = h10·h30·post_effective 의 BSTS 대비 상대 RMSE 평균. 선정 결과를 selection_frozen.json 에 먼저 쓰고
        그 다음에만 최종 구간 지표를 계산한다. 가중치 자체는 모든 구간에서 온라인(과거 실현 창만).
  전이: 같은 고정 선택을 rv5 (처치 3종목, i0>=120) 와 rv5 donor 20종목 (앙상블 개발에 쓰인 적 없는 단면) 에 적용.
한계: GK 기간은 ml_extension 에서 이미 본 자료 (기존 stacking 이 GK 최고였다는 사실을 알고 설계) — 완전히 새로운 외부 자료가 아니다.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.rq2.ensemble_final import evaluate as EV
from src.rq2.ensemble_final import figures as FG
from src.rq2.ensemble_final import loader as L
from src.rq2.ensemble_final import online_run as R

OOS = L.CFG["oos"]
B = OOS["boot_reps"]
FROZEN = L.RES / "selection_frozen.json"
PICK_ORDER = ["best_single", "old_stack", "selective", "stock_adaptive", "regime_aware", "overall"]


def _cands(T):
    s = T.strategy.unique()
    return [x for x in s if not x.startswith("m:") and x not in ("old_stack", "old_equal")]


def select():
    D = R.load("gk")
    S = D[(D.cutoff >= OOS["gk_select"][0]) & (D.cutoff <= OOS["gk_select"][1])]
    T = EV.metrics(S, "gk")
    sc = EV.mean_rel(T, OOS["select_windows"])
    fam = T.drop_duplicates("strategy").set_index("strategy").family
    c = _cands(T)
    pick = {
        "overall": sc[c].idxmin(),
        "best_single": sc[[x for x in sc.index if x.startswith("m:")]].idxmin(),
        "selective": sc[[x for x in c if fam[x] in ("A", "B", "C", "D", "meta")]].idxmin(),
        "stock_adaptive": sc[[x for x in c if "|" in x and x.split("|")[1] in ("code", "hier", "glocal")]].idxmin(),
        "regime_aware": sc[[x for x in c if "|regime" in x]].idxmin(),
        "old_stack": "old_stack",
    }
    out = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rule": "GK 2018-2021 h10·h30·post_effective 상대 RMSE 평균 최소",
           "n_candidates": len(c), "picks": pick, "select_scores": sc.sort_values().round(5).to_dict()}
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    L.save_table(T.assign(period="select"), "exp4_gk_select_period")
    return out


def holdout(fz):
    D = R.load("gk")
    H = D[(D.cutoff >= OOS["gk_holdout"][0])]
    T = EV.metrics(H, "gk")
    L.save_table(T.assign(period="holdout"), "exp4_gk_holdout_all")
    sc_h = EV.mean_rel(T, OOS["select_windows"])
    ss = pd.Series(fz["select_scores"])
    c = [x for x in ss.index if x in sc_h.index]
    wc = pd.DataFrame({"select": ss[c], "holdout": sc_h[c]})
    wc["rank_select"], wc["rank_holdout"] = wc.select.rank(), wc.holdout.rank()
    L.save_table(wc.reset_index(names="strategy"), "exp4_winners_curse")
    rho = wc[["select", "holdout"]].corr("spearman").iat[0, 1]
    return H, T, wc, rho


def compare(D, ds, picks, label):
    """고정 선택들의 지표 + BSTS·기존 stacking 대비 MSE 차이의 블록 bootstrap CI + DM-HAC + reality check."""
    rows, rcs = [], []
    old = "old_stack" if "old_stack" in set(D.strategy) else "old_stack_replica"
    T = EV.metrics(D, ds)
    c_all = _cands(T)
    for w, g in D.groupby("window"):
        b = L.WIN[w][1]
        st = L.STEP[ds]
        mb = g[g.strategy == "single_bsts"].err.pow(2).mean()
        for role, s in list(picks.items()) + [("bsts", "single_bsts")]:
            s = old if s == "old_stack" else s
            if s not in set(g.strategy):
                rows.append({"dataset": ds, "period": label, "window": w, "role": role, "strategy": s, "available": False})
                continue
            t = T[(T.window == w) & (T.strategy == s)].iloc[0]
            r1 = EV.block_boot_diff(g, s, "single_bsts", b, st, B, L.SEED)
            r2 = EV.block_boot_diff(g, s, old, b, st, B, L.SEED + 1)
            rows.append({"dataset": ds, "period": label, "window": w, "role": role, "strategy": s, "available": True,
                         "rmse": t.rmse, "mae": t.mae, "bias": t.bias, "rel_bsts": t.rel_bsts, "rel_old": t.rel_old,
                         "win_vs_bsts": t.win_vs_bsts, "win_vs_old": t.win_vs_old,
                         "dmse_bsts_pct": 100 * r1["dmse"] / mb, "lo_bsts_pct": 100 * r1["lo"] / mb,
                         "hi_bsts_pct": 100 * r1["hi"] / mb, "p_boot_bsts": r1["p_boot"], "dm_p_bsts": r1["dm_p"],
                         "dmse_old_pct": 100 * r2["dmse"] / mb, "lo_old_pct": 100 * r2["lo"] / mb,
                         "hi_old_pct": 100 * r2["hi"] / mb, "p_boot_old": r2["p_boot"], "dm_p_old": r2["dm_p"],
                         "n_i0": r1["n_i0"]})
        rc = EV.reality_check(g, [x for x in c_all if x != "single_bsts"], "single_bsts", b, st, B, L.SEED + 2)
        rcs.append({"dataset": ds, "period": label, "window": w, **rc})
    return pd.DataFrame(rows), pd.DataFrame(rcs)


def yearly(H):
    """연도별 순위 안정성 (GK 전체 평가 구간, 창 평균 상대 RMSE)."""
    D = R.load("gk")
    D = EV.eval_rows(D, "gk").assign(year=lambda d: d.cutoff.dt.year)
    rows = []
    for y, g in D.groupby("year"):
        T = EV.metrics(g, "gk")
        s = EV.mean_rel(T, OOS["select_windows"])
        w = T.groupby("strategy").win_vs_bsts.mean()
        rows += [{"year": y, "strategy": k, "rel_bsts": v, "win_vs_bsts": w[k]} for k, v in s.items()]
    Y = pd.DataFrame(rows)
    P = Y.pivot(index="strategy", columns="year", values="rel_bsts")
    P = P.loc[[x for x in P.index if not x.startswith("m:") or x in ("m:timesfm_xreg", "m:ols")]]
    cr = P.rank().corr("spearman")
    adj = [cr.iat[i, i + 1] for i in range(len(cr) - 1)]
    return Y, pd.DataFrame({"year_pair": [f"{a}-{b}" for a, b in zip(cr.index[:-1], cr.index[1:])], "spearman": adj})


def fig(C):
    import matplotlib.pyplot as plt
    sets = [("gk", "holdout"), ("rv5", "transfer"), ("rv5d", "transfer")]
    roles = ["best_single", "old_stack", "selective", "stock_adaptive", "regime_aware", "overall"]
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharex=False, layout="constrained")
    for ax, (ds, per) in zip(axs, sets):
        s = C[(C.dataset == ds) & (C.period == per) & C.available & C.role.isin(roles)]
        labs, est, lo, hi = [], [], [], []
        for w in L.CFG["datasets"][ds]["windows"]:
            for r in roles:
                t = s[(s.window == w) & (s.role == r)]
                if len(t):
                    t = t.iloc[0]
                    labs.append(f"{w} · {r} ({t.strategy})")
                    est.append(t.dmse_bsts_pct), lo.append(t.lo_bsts_pct), hi.append(t.hi_bsts_pct)
        FG.dot_ci(ax, labs, np.array(est), np.array(lo), np.array(hi), ref=0)
        ax.set_title(f"{ds} {per}: MSE 변화율 vs BSTS (%, 95% 블록 bootstrap)", fontsize=9, loc="left")
        ax.set_xlabel("음수 = BSTS 보다 정확")
    FG.save(fig, "exp4_oos_dmse_ci")


def run():
    fz = select()                                   # 1) 선정 고정 (최종 구간을 보기 전에 파일로 기록)
    picks = fz["picks"]
    H, T, wc, rho = holdout(fz)                     # 2) 최종 구간
    C1, RC1 = compare(H, "gk", picks, "holdout")
    C2, RC2 = compare(EV.load_eval("rv5"), "rv5", picks, "transfer")
    C3, RC3 = compare(EV.load_eval("rv5d"), "rv5d", picks, "transfer")
    C = pd.concat([C1, C2, C3], ignore_index=True)
    RC = pd.concat([RC1, RC2, RC3], ignore_index=True)
    L.save_table(C, "exp4_oos_comparison")
    L.save_table(RC, "exp4_reality_check")
    Y, adj = yearly(H)
    L.save_table(Y, "exp4_gk_yearly")
    L.save_table(adj, "exp4_gk_yearly_rank_stability")
    L.save_table(pd.DataFrame([{"spearman_select_vs_holdout": rho, "n_cands": len(wc),
                                "selected_rank_holdout": wc.loc[picks["overall"], "rank_holdout"]}]), "exp4_winners_curse_summary")
    fig(C)
    return fz, C, RC, rho


if __name__ == "__main__":
    run()
