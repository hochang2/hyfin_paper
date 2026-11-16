"""실험 5: 실제 규제 효과 재추정 (사전 지정 모형 + 전 모형 민감도).

순서 (사후 결과를 보고 모형을 고르지 않도록):
  1) select_model(): 실험 2 의 가짜 개입일 결과만으로 config.selection_rule 을 적용해 주 모형을 고른다.
     (rv5, h30, 처치 2 종목 평균 time-보정 FPR ≤ 0.10 → 그중 time-보정 MDE 최소). 선택 결과를 먼저 파일로 남긴다.
  2) estimate(): 모든 모형의 실제 효과를 계산하되, 주 결과 = 1) 에서 고른 모형, 나머지는 민감도.
효과 = 실제 사후 y 창평균 − 반사실 창평균 (log RV). 창: transition(07-16~07-29), post_effective(07-31~10-30).
p 값 3종:
  native : 모형 자체 창평균 분포 (정규근사 sd). FM 없음.
  time   : 같은 모형·종목의 rv5 가짜 개입일 같은 창 오차로 만든 귀무분포 (평균 중심 t 예측구간).
  space  : 실제 개입일 donor 175 종목의 같은 모형 오차와 비교한 순열 p.
Holm: 주 모형의 {005930, 000660} × {transition, post_effective} 의 주 p (time 가능 시 time, 아니면 space).
GK: 일별 Garman-Klass 로 같은 일을 반복 (긴 표본 귀무분포). rv5 결과와 섞지 않고 따로 보고한다.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from src.rq2.core.pipeline import holm
from src.rq2.ml_extension import benchmark as B
from src.rq2.ml_extension import data_pipeline as DP
from src.rq2.ml_extension import synthetic_effect as S

E = DP.ECFG
WINS = ["transition", "post_effective"]
TREATED = ["005930", "000660", "005935"]


def select_model(syn: pd.DataFrame, mde: pd.DataFrame) -> dict:
    r = E["selection_rule"]
    tried = []
    for win in [r["window"]] + list(r.get("fallback_windows", [])):
        a = syn[(syn.dataset == r["dataset"]) & (syn.window == win) & (syn.scenario == "A_immediate")
                & (syn.effect_pct == 0) & (syn.code.isin(r["codes"]))]
        testable = a.groupby("model").time_testable.min() if "time_testable" in a else pd.Series(1.0, a.model.unique())
        tried.append({"window": win, "min_testable_share": float(testable.max()) if len(testable) else 0.0})
        if not len(testable) or testable.max() < 1.0:
            continue
        fpr = a.groupby("model").time_reject.mean()
        m = mde[(mde.dataset == r["dataset"]) & (mde.window == win) & (mde.code.isin(r["codes"]))]
        md = m.groupby("model").mde_time_analytic_log.mean()
        tab = pd.DataFrame({"time_fpr": fpr, "mde_log": md, "testable": testable}).dropna()
        tab = tab[tab.testable >= 1.0]
        ok = tab[tab.time_fpr <= r["max_calibrated_fpr"]].sort_values("mde_log")
        if len(ok):
            break
    sel = ok.index[0] if len(ok) else None
    out = {"rule": r, "window_used": win, "fallback_used": win != r["window"], "windows_tried": tried,
           "selected": sel, "candidates": tab.sort_values("mde_log").round(4).reset_index().to_dict("records")}
    f = DP.RES / "tables" / "exp5_model_selection.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


def _pseudo_null(ds_null: str, models):
    """같은 창의 가짜 개입일 오차 평균·sd. n0 = 서로 겹치지 않는 창 개수(유효 표본), n_raw = 전체 과제 수.
    n0 < time_null_min_eff 이면 time 검정 불가 (rv5 post_effective 61일 창은 rv5 표본이 짧아 n0=1)."""
    _, W = B.collect(ds_null, models)
    W = W[W.window.isin(WINS)].assign(err=lambda d: d.ybar - d.predbar)
    rows = []
    for (m, c, w), g in W.groupby(["model", "code", "window"]):
        L = int((g.b - g.a).iloc[0])
        ne = S.n_eff(g.i0.values, L)
        ok = ne >= E.get("time_null_min_eff", 3)
        rows.append({"model": m, "code": c, "window": w, "m0": g.err.mean() if ok else np.nan,
                     "s0": g.err.std(ddof=1) if ok else np.nan, "n0": ne, "n_raw": len(g)})
    return pd.DataFrame(rows)


def estimate(models: list[str], selected: str | None, ds_real="space", ds_null="rv5", tag=None) -> pd.DataFrame:
    """ds_real='space' (rv5 실제일, donor 순열 가능) 또는 'gk' (tag='real' 캐시)."""
    names = [f"{m}@{tag}" if tag else m for m in models]
    _, W = B.collect(ds_real, names)
    if W.empty:
        return pd.DataFrame()
    W = W[W.window.isin(WINS)].copy()
    W["model"] = W.model.str.split("@").str[0]
    W["effect"] = W.ybar - W.predbar
    N = _pseudo_null(ds_null, models)
    W = W.merge(N, on=["model", "code", "window"], how="left")
    crit = stats.t.ppf(0.975, W.n0 - 1) * W.s0 * np.sqrt(1 + 1 / W.n0)
    W["time_ci_lo"] = W.effect - W.m0 - crit
    W["time_ci_hi"] = W.effect - W.m0 + crit
    W["time_p"] = 2 * stats.t.sf(np.abs(W.effect - W.m0) / (W.s0 * np.sqrt(1 + 1 / W.n0)), W.n0 - 1)
    W["native_ci_lo"] = W.ybar - W.cf_q975
    W["native_ci_hi"] = W.ybar - W.cf_q025
    W["native_p"] = 2 * stats.norm.sf(np.abs(W.effect) / W.cf_sd)
    W["mde_time_log"] = 2.8 * W.s0
    if ds_real == "space":
        donors = set(DP.donor_codes(DP.CFG, "main"))
        don = W[W.code.isin(donors)]
        sp = []
        for r in W[W.code.isin(TREATED)].itertuples():
            e = don[(don.model == r.model) & (don.window == r.window)].effect.values
            c = e.mean()
            sp.append(((np.sum(np.abs(e - c) >= abs(r.effect - c)) + 1) / (len(e) + 1), np.std(e, ddof=1), c))
        T = W[W.code.isin(TREATED)].copy()
        T["space_p"], T["space_sd"], T["donor_mean_effect"] = [x[0] for x in sp], [x[1] for x in sp], [x[2] for x in sp]
        T["mde_space_log"] = 2.8 * T.space_sd
    else:
        T = W[W.code.isin(TREATED)].copy()
    T["effect_pct"] = np.expm1(T.effect) * 100
    T["primary"] = T.model == selected
    T["dataset"] = ds_real if ds_real == "space" else "gk"
    # 주 p 값: time 검정이 가능하면 time, 아니면 space 순열 (rv5 post_effective 는 유효 창 1개라 time 불가)
    T["primary_p"] = T.time_p
    T["primary_p_source"] = np.where(T.time_p.notna(), "time", "none")
    if "space_p" in T:
        sw = T.time_p.isna() & T.space_p.notna()
        T.loc[sw, "primary_p"] = T.loc[sw, "space_p"]
        T.loc[sw, "primary_p_source"] = "space"
    fam = T[(T.primary) & T.code.isin(["005930", "000660"]) & T.primary_p.notna()]
    if len(fam):
        T.loc[fam.index, "primary_p_holm"] = holm(pd.Series(fam.primary_p.values)).values
    cols = ["dataset", "model", "primary", "code", "window", "effect", "effect_pct", "native_ci_lo", "native_ci_hi",
            "native_p", "m0", "s0", "n0", "n_raw", "time_ci_lo", "time_ci_hi", "time_p", "mde_time_log",
            "primary_p", "primary_p_source", "primary_p_holm"]
    cols += [c for c in ["space_p", "space_sd", "donor_mean_effect", "mde_space_log"] if c in T]
    return T.reindex(columns=cols).sort_values(["code", "window", "model"]).reset_index(drop=True)


def within_firm(T: pd.DataFrame) -> pd.DataFrame:
    """같은 기업 차분: 005930 효과 − 005935 효과 (모형별). 귀무 sd 는 두 귀무 오차 차의 sd 가 아니라
    보수적으로 sqrt(s0_a² + s0_b²) (상관 무시) 와 rv5 가짜 개입일 차분 오차 sd 를 함께 보고."""
    _, W = B.collect("rv5", sorted(T.model.unique()))
    W = W[W.window.isin(WINS)].assign(err=lambda d: d.ybar - d.predbar)
    piv = W.pivot_table(index=["model", "window", "i0"], columns="code", values="err")
    dd = (piv["005930"] - piv["005935"]).dropna().reset_index()
    rows = []
    for (m, w), g in dd.groupby(["model", "window"]):
        a_, b_ = DP.WIN[w]
        ne = S.n_eff(g.i0.values, b_ - a_)
        ok = ne >= E.get("time_null_min_eff", 3)
        rows.append({"model": m, "window": w, "mean": g[0].mean() if ok else np.nan,
                     "std": g[0].std(ddof=1) if ok else np.nan, "size": ne, "n_raw": len(g)})
    nd = pd.DataFrame(rows).set_index(["model", "window"])
    a = T[T.code == "005930"].set_index(["model", "window"])
    b = T[T.code == "005935"].set_index(["model", "window"])
    d = (a.effect - b.effect).rename("diff").to_frame().join(nd)
    d["time_p"] = 2 * stats.t.sf(np.abs(d["diff"] - d["mean"]) / (d["std"] * np.sqrt(1 + 1 / d["size"])), d["size"] - 1)
    return d.reset_index()


def original_results() -> pd.DataFrame:
    """1차 연구 주 결과 (변경하지 않고 읽기만)."""
    p = DP.HERE.parent / "main_study" / "results" / "tables"
    for name in ["rq2_primary_bsts.csv", "rq2_primary.csv", "rq2_models.csv"]:
        if (p / name).exists():
            return pd.read_csv(p / name)
    return pd.DataFrame()
