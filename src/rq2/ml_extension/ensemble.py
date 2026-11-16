"""실험 4: 앙상블과 모형 간 오차 상관.

앙상블 (반사실 예측의 가중결합):
  ens_equal     통계+ML 10 개 동일가중 (members_all)
  ens_equal_ml  ML 4 개 동일가중
  ens_invmse    과거 실현오차 MSE 역수 가중
  ens_stack     과거 실현일에서 y ~ 예측들 비음수 ridge (절편 없음) 가중
  ens_hybrid    BSTS 0.5 + LGBM 0.25 + XGB 0.25 고정
가중치 학습 자료 = 같은 종목의 '이전' 가짜 개입일 예측 중 실현일(date) < 현재 개입일 인 일별 행만 (미래 정보 없음).
  실현일이 min_past_days 미만이면 동일가중으로 대체 (기록).
space(실제 개입일) = rv5 의 07-15 이전 실현일을 세 종목 합쳐 학습한 가중치를 모든 단위에 쓴다.
창평균 반사실 분포: 구성원 창평균 sd 의 가중합을 sd 로 하는 정규 근사 (상관 1 가정 → 보수적).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.rq2.ml_extension import benchmark as B
from src.rq2.ml_extension import data_pipeline as DP

E = DP.ECFG
C = E["ensemble"]
ENS = ["ens_equal", "ens_equal_ml", "ens_invmse", "ens_stack", "ens_hybrid"]


def _wide(ds, members):
    D, W = B.collect(ds, members)
    P = D.pivot_table(index=["code", "i0", "h", "date", "y"], columns="model", values="pred").reset_index()
    S = W.pivot_table(index=["code", "i0", "window"], columns="model", values="cf_sd")
    return P, S, W


def _weights(kind, past: pd.DataFrame, members: list[str]):
    k = len(members)
    eq = np.full(k, 1 / k)
    if kind in ("ens_equal", "ens_equal_ml"):
        return eq, False
    if kind == "ens_hybrid":
        return np.array([C["hybrid"].get(m, 0.0) for m in members]), False
    past = past.dropna(subset=members)
    if len(past) < C["min_past_days"]:
        return eq, True
    Fm = past[members].values
    yv = past.y.values
    if kind == "ens_invmse":
        mse = np.mean((yv[:, None] - Fm) ** 2, 0)
        w = 1 / mse
        return w / w.sum(), False
    if kind == "ens_stack":
        from sklearn.linear_model import Ridge
        r = Ridge(alpha=C["stack_alpha"], fit_intercept=False, positive=True).fit(Fm, yv)
        w = r.coef_
        if w.sum() <= 0:
            return eq, True
        return w / w.sum(), False   # 합 1 로 정규화 (수준 보존)
    raise ValueError(kind)


def _members(kind):
    if kind == "ens_equal_ml":
        return list(C["members_ml"])
    if kind == "ens_hybrid":
        return list(C["hybrid"])
    return list(C["members_all"])


def build(ds: str):
    allm = sorted(set(C["members_all"]) | set(C["hybrid"]))
    if ds == "space":
        Pref, _, _ = _wide("rv5", allm)
        ref_past = Pref[Pref.date <= pd.Timestamp(DP.CFG["sample"]["train_end"])]
    P, S, W = _wide(ds, allm)
    out_d, out_w, wlog = {k: [] for k in ENS}, {k: [] for k in ENS}, []
    for (code, i0), g in P.groupby(["code", "i0"]):
        g = g.sort_values("h")
        cut = g.date.iloc[0]
        if ds == "space":
            past = ref_past
        else:
            past = P[(P.code == code) & (P.i0 < i0) & (P.date < cut)]
        for kind in ENS:
            mem = _members(kind)
            if g[mem].isna().any().any():
                continue
            w, fb = _weights(kind, past, mem)
            pred = g[mem].values @ w
            yp = g.y.values
            H = len(g)
            d = pd.DataFrame({"code": code, "i0": i0, "cutoff": cut, "h": g.h.values, "date": g.date.values,
                              "y": yp, "pred": pred, "lo90": np.nan, "hi90": np.nan, "model": kind})
            out_d[kind].append(d)
            for wn, (a, b) in DP.WIN.items():
                if b > H:
                    continue
                pb = pred[a:b].mean()
                try:
                    sd = float(np.dot(S.loc[(code, i0, wn), mem].values, w))
                except KeyError:
                    sd = np.nan
                out_w[kind].append({"code": code, "i0": i0, "cutoff": cut, "window": wn, "a": a, "b": b,
                                    "ybar": yp[a:b].mean(), "predbar": pb, "cf_sd": sd,
                                    "cf_q025": pb - 1.96 * sd, "cf_q975": pb + 1.96 * sd,
                                    "cf_q05": pb - 1.645 * sd, "cf_q95": pb + 1.645 * sd,
                                    "pre_rmse": np.nan, "H": H, "fit_sec": np.nan, "model": kind})
            wlog.append({"dataset": ds, "code": code, "i0": i0, "ensemble": kind, "fallback_equal": fb,
                         "n_past_days": len(past), **{f"w_{m}": x for m, x in zip(mem, w)}})
    for kind in ENS:
        if out_d[kind]:
            pd.concat(out_d[kind]).to_parquet(B.cache_path(ds, kind, "daily"))
            pd.DataFrame(out_w[kind]).to_parquet(B.cache_path(ds, kind, "win"))
    wl = pd.DataFrame(wlog)
    wl.to_parquet(DP.CACHE / ds / "ensemble_weights.parquet")
    return wl


def error_correlation(ds: str, models: list[str], window="h30") -> pd.DataFrame:
    """창평균 오차(=효과 추정오차)의 모형 간 상관. 종목별로 평균을 뺀 뒤 합쳐 계산."""
    _, W = B.collect(ds, models)
    W = W[W.window == window].assign(err=lambda d: d.ybar - d.predbar)
    W["err"] = W.err - W.groupby(["model", "code"]).err.transform("mean")
    piv = W.pivot_table(index=["code", "i0"], columns="model", values="err")
    return piv.corr()


def daily_error_correlation(ds: str, models: list[str], hmax=30) -> pd.DataFrame:
    D, _ = B.collect(ds, models)
    D = D[D.h <= hmax].assign(e=lambda d: d.y - d.pred)
    piv = D.pivot_table(index=["code", "i0", "h"], columns="model", values="e")
    return piv.corr()
