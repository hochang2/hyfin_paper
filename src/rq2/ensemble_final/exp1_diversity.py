"""Exp1 개별 모형 성능과 오차 다양성 (기존 예측만 사용).

일별 오차(예측일 단위)와 창평균 효과 오차(가짜 개입일 단위)를 구분한다. 참효과 = 0 이므로 창평균 오차 = 효과 추정오차.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.rq2.ensemble_final import figures as FG
from src.rq2.ensemble_final import loader as L

MODELS = L.POOL["full"]
LAB = {"bsts": "BSTS", "ols": "OLS", "ridge": "Ridge", "enet": "ENet", "lgbm": "LGBM", "xgb": "XGB", "rf": "RF",
       "uc": "UC", "sarimax": "SARIMAX", "harx": "HARX", "did_gap": "DiD-gap", "chronos2_cov": "Chronos-2 cov",
       "timesfm_xreg": "TimesFM cov", "old_stack": "기존 stacking", "old_equal": "기존 동일가중"}
MAIN_W = {"gk": "h30", "rv5": "h10", "rv5d": "h10", "gkd": "h10"}


def _stats(e):
    e = np.asarray(e)
    return {"n": len(e), "rmse": np.sqrt(np.mean(e ** 2)), "mae": np.mean(np.abs(e)), "bias": np.mean(e),
            "over_share": np.mean(e < 0), "bias2_share": np.mean(e) ** 2 / np.mean(e ** 2)}


def err_long(ds, windows=None):
    """창평균 오차 long: code, i0, window, model, err (기존 앙상블 포함)."""
    d = L.long(ds).copy()
    if windows:
        d = d[d.window.isin(windows)]
    d = d[d.model.isin(MODELS + ["old_stack", "old_equal"])]
    d["err"] = d.ybar - d.predbar
    return d


def summary_table():
    rows = []
    for ds in ("gk", "rv5", "rv5d", "gkd"):
        d = err_long(ds, ["h5", "h10", "h20", "h30", "h61", "post_effective"])
        # 모든 모형이 있는 (code,i0,window) 만
        k = d.groupby(["code", "i0", "window"]).model.transform("nunique")
        d = d[k == k.max()] if ds != "gk" else d[k >= len([m for m in MODELS if m in set(d.model)])]
        for (w, m), g in d.groupby(["window", "model"]):
            rows.append({"dataset": ds, "window": w, "model": m, **_stats(g.err)})
    T = pd.DataFrame(rows)
    T["rank"] = T.groupby(["dataset", "window"]).rmse.rank()
    ref = T[T.model == "bsts"].set_index(["dataset", "window"]).rmse
    T["rel_bsts"] = T.rmse / T.set_index(["dataset", "window"]).index.map(ref)
    return T


def by_code():
    rows = []
    for ds in ("gk", "rv5", "rv5d"):
        d = err_long(ds, [MAIN_W[ds]])
        for (c, m), g in d.groupby(["code", "model"]):
            rows.append({"dataset": ds, "window": MAIN_W[ds], "code": c, "model": m, **_stats(g.err)})
    T = pd.DataFrame(rows)
    T["rank"] = T.groupby(["dataset", "code"]).rmse.rank()
    return T


def daily_horizon():
    """일별 오차의 예측 시차 구간별 RMSE·편향 (gk·rv5, 실제 발표 이전 날짜만)."""
    bins = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 61), (62, 71)]
    rows, E = [], {}
    for ds in ("gk", "rv5"):
        for m in MODELS + ["ens_stack"]:
            f = L.EXT / ds / f"{m}_daily.parquet"
            if not f.exists():
                continue
            d = pd.read_parquet(f, columns=["code", "i0", "h", "date", "y", "pred"])
            if ds == "rv5":
                d = d[d.date < L.rv5_index()[L.REAL]]
            d["e"] = d.y - d.pred
            mm = "old_stack" if m == "ens_stack" else m
            E[(ds, mm)] = d.set_index(["code", "i0", "h"]).e
            for lo, hi in bins:
                g = d[(d.h >= lo) & (d.h <= hi)]
                rows.append({"dataset": ds, "model": mm, "h_bin": f"{lo}-{hi}", **_stats(g.e)})
    T = pd.DataFrame(rows)
    T["rank"] = T.groupby(["dataset", "h_bin"]).rmse.rank()
    return T, E


def corr_tables(E_daily):
    out = {}
    for ds in ("gk", "rv5", "rv5d"):
        W = L.wide(ds, MAIN_W[ds])
        ms = [m for m in MODELS if m in W]
        Ew = pd.DataFrame({m: W.ybar - W[m] for m in ms})
        out[(ds, "window")] = (Ew.corr(), Ew.cov())
        if ds in ("gk", "rv5"):
            Ed = pd.DataFrame({m: E_daily[(ds, m)] for m in ms if (ds, m) in E_daily}).dropna()
            out[(ds, "daily")] = (Ed.corr(), Ed.cov())
    return out


def pair_gain():
    """BSTS 와 50/50 결합했을 때의 창평균 RMSE 이득 (전체 표본, 사후적 — 설명용) 과 오차상관."""
    rows = []
    for ds in ("gk", "rv5", "rv5d"):
        for w in L.CFG["datasets"][ds]["windows"]:
            W = L.wide(ds, w)
            eb = W.ybar - W.bsts
            rb = np.sqrt(np.mean(eb ** 2))
            for m in [x for x in MODELS if x in W and x != "bsts"]:
                em = W.ybar - W[m]
                e5 = (eb + em) / 2
                # 사후 최적 볼록 가중치 (설명용)
                a = np.clip(((eb - em) @ eb) / max(((eb - em) ** 2).sum(), 1e-12), 0, 1)
                eo = (1 - a) * eb + a * em
                rows.append({"dataset": ds, "window": w, "partner": m, "corr_bsts": np.corrcoef(eb, em)[0, 1],
                             "rmse_partner_rel": np.sqrt(np.mean(em ** 2)) / rb,
                             "rmse_5050_rel": np.sqrt(np.mean(e5 ** 2)) / rb,
                             "w_opt_partner": a, "rmse_opt_rel": np.sqrt(np.mean(eo ** 2)) / rb,
                             "bias_bsts": eb.mean(), "bias_partner": em.mean()})
    return pd.DataFrame(rows)


def regime_table():
    rows = []
    for ds in ("gk", "rv5", "rv5d"):
        w = MAIN_W[ds]
        F = L.regime_frame(ds)[["i0", "vol_t3", "ret_t3", "era"]]
        d = err_long(ds, [w]).merge(F, on="i0")
        for col in ("vol_t3", "ret_t3", "era"):
            for (lab, m), g in d.groupby([col, "model"]):
                rows.append({"dataset": ds, "window": w, "split": col, "regime": lab, "model": m, **_stats(g.err)})
    T = pd.DataFrame(rows)
    T["rank"] = T.groupby(["dataset", "split", "regime"]).rmse.rank()
    return T


def old_stack_diagnosis():
    """기존 stacking 이 기대보다 못했던 이유: (1) 가중치 불안정 (2) 일별 목표 vs 창평균 목표 (3) 공통 편향."""
    rows = []
    for ds in ("gk", "rv5"):
        W = pd.read_parquet(L.EXT / ds / "ensemble_weights.parquet")
        W = W[(W.ensemble == "ens_stack") & ~W.fallback_equal]
        wc = [c for c in W.columns if c.startswith("w_")]
        for c, g in W.groupby("code"):
            g = g.sort_values("i0")
            M = g[wc].fillna(0).values
            ent = -(np.where(M > 0, M * np.log(np.clip(M, 1e-12, None)), 0)).sum(1)
            rows.append({"dataset": ds, "code": c, "n_fits": len(g),
                         "turnover_mean": np.abs(np.diff(M, axis=0)).sum(1).mean() / 2 if len(g) > 1 else np.nan,
                         "max_w_mean": M.max(1).mean(), "eff_n_members": np.exp(ent).mean(),
                         "top_member_mode": pd.Series([wc[i][2:] for i in M.argmax(1)]).mode().iat[0],
                         **{f"mean_{x}": g[x].mean() for x in wc}})
    return pd.DataFrame(rows)


def complementary(PG, S):
    """보완 후보: BSTS 와 상관이 낮고(같은 자료 중앙값 미만) 50/50 결합 이득이 있는 짝 + 각 자료의 최고 단일."""
    rows = []
    for (ds, w), g in PG.groupby(["dataset", "window"]):
        med = g.corr_bsts.median()
        for _, r in g.iterrows():
            rows.append({**r.to_dict(), "low_corr": r.corr_bsts < med, "gain_5050": r.rmse_5050_rel < 1,
                         "candidate": (r.corr_bsts < med) and (r.rmse_5050_rel < 0.99)})
    return pd.DataFrame(rows)


def figs(C, T, PG):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, key, t in zip(axs, [("gk", "window"), ("rv5", "window"), ("gk", "daily")],
                          ["GK h30 창평균 효과오차 상관", "rv5 h10 창평균 효과오차 상관", "GK 일별 예측오차 상관"]):
        Cm = C[key][0]
        lab = [LAB.get(m, m) for m in Cm.columns]
        im = FG.heatmap(ax, Cm.values, lab, t, vmin=0.5, vmax=1)
    fig.colorbar(im, ax=axs, shrink=0.6, label="오차 상관")
    FG.save(fig, "exp1_error_corr_heatmap")

    # 편향 비교 (창평균 효과 = 0 이어야 함)
    sub = T[(T.window.isin(["h10", "h30"])) & (T.dataset.isin(["gk", "rv5", "rv5d"]))]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), sharey=False)
    for ax, ds in zip(axs, ["gk", "rv5", "rv5d"]):
        s = sub[(sub.dataset == ds) & (sub.window == MAIN_W[ds] if ds != "gk" else sub.window == "h30")]
        s = s.sort_values("bias")
        y = np.arange(len(s))
        ax.barh(y, s.bias, color=[FG.CAT[1] if m in ("old_stack", "old_equal") else (FG.CAT[0] if m != "bsts" else FG.CAT[2]) for m in s.model], height=0.6)
        ax.set_yticks(y, [LAB.get(m, m) for m in s.model], fontsize=7)
        ax.axvline(0, color=FG.INK2, lw=1)
        ax.set_title(f"{ds} {'h30' if ds == 'gk' else MAIN_W[ds]}: 평균 효과오차(편향)", fontsize=9, loc="left")
        ax.set_xlabel("평균 (y - 예측), log")
    FG.save(fig, "exp1_bias_comparison")

    # 예측기간별 순위
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, ds in zip(axs, ["gk", "rv5"]):
        s = T[(T.dataset == ds) & T.window.isin(["h5", "h10", "h20", "h30", "h61"]) & T.model.isin(MODELS + ["old_stack"])]
        P = s.pivot(index="window", columns="model", values="rel_bsts").reindex(["h5", "h10", "h20", "h30", "h61"])
        x = np.arange(len(P))
        hl = {"bsts": FG.CAT[2], "ols": FG.CAT[0], "timesfm_xreg": FG.CAT[6], "chronos2_cov": FG.CAT[4],
              "old_stack": FG.CAT[1], "uc": FG.CAT[3]}
        for m in P.columns:
            ax.plot(x, P[m], color=hl.get(m, "#c3c2bd"), lw=2 if m in hl else 1, marker="o" if m in hl else None, ms=4)
            if m in hl:
                ax.annotate(LAB.get(m, m), (x[-1], P[m].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                            fontsize=7, color=FG.INK2, va="center")
        ax.set_xticks(x, P.index)
        ax.axhline(1, color=FG.INK2, lw=1, ls="--")
        ax.set_ylim(0.8, 1.5)
        ax.set_title(f"{ds}: 창 길이별 효과오차 RMSE (BSTS=1)", fontsize=9, loc="left")
    FG.save(fig, "exp1_horizon_rel_rmse")


def run():
    T = summary_table()
    L.save_table(T, "exp1_model_summary")
    L.save_table(by_code(), "exp1_model_by_code")
    H, E = daily_horizon()
    L.save_table(H, "exp1_daily_horizon")
    C = corr_tables(E)
    for (ds, kind), (cr, cv) in C.items():
        L.save_table(cr.reset_index(names="model"), f"exp1_err_corr_{ds}_{kind}", 3)
        L.save_table(cv.reset_index(names="model"), f"exp1_err_cov_{ds}_{kind}", 5)
    PG = pair_gain()
    L.save_table(PG, "exp1_pair_gain_bsts")
    L.save_table(complementary(PG, T), "exp1_complementary_candidates")
    L.save_table(regime_table(), "exp1_regime_era")
    L.save_table(old_stack_diagnosis(), "exp1_old_stack_weights_diag")
    figs(C, T, PG)
    return T


if __name__ == "__main__":
    run()
