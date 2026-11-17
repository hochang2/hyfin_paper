"""Exp2·Exp3 의 온라인 예측을 모든 (자료, 창) 에 대해 한 번 계산해 캐시한다 (가중치 학습 = 과거 실현 창만).

출력 (캐시): online_<ds>.parquet  — code, i0, cutoff, window, ybar, strategy, family, pred, err
            weights_<ds>.parquet — 전역 전략의 시점별 가중치 (안정성 진단)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.rq2.phase1.ensemble import loader as L
from src.rq2.phase1.ensemble import strategies as S

EXP3_BASES = ["simplex_l2", "invmse_top3"]          # Exp3 종목·국면 변형을 붙일 기준 전략 (결과 전 지정)
EXP3_SCOPES = ["code", "hier", "glocal", "regime_vol", "regime_trend"]
SINGLES = L.POOL["full"]


def task_of(ds, w):
    W = L.wide(ds, w)
    names = [m for m in L.POOL["full"] if m in W.columns]
    return S.Task(W, w, L.STEP[ds], names)


def run_task(ds, w, wlog):
    T = task_of(ds, w)
    F = L.regime_frame(ds).set_index("i0")
    pred, fam = {}, {}
    for m in T.names:
        pred[f"m:{m}"], fam[f"m:{m}"] = T.W[m].values, "single"
    for c in ("old_stack", "old_equal"):
        if c in T.W:
            pred[c], fam[c] = T.W[c].values, "old_cache"
    for s, spec in S.STRATS.items():
        p = S.online(T, s, "global", log_w=wlog)
        if p is not None:
            pred[s], fam[s] = p, spec[3]
    for s in EXP3_BASES:
        for sc in EXP3_SCOPES:
            if sc.startswith("regime"):
                reg = F["vol_hi"] if sc == "regime_vol" else F["trend"]
                p = S.online(T, s, "regime", regime=reg)
            else:
                p = S.online(T, s, sc)
            pred[f"{s}|{sc}"], fam[f"{s}|{sc}"] = p, "E3"
    cands = [s for s, f in fam.items() if f in ("A", "B", "C", "D")]
    pred["meta_ftl"], picks = S.meta_ftl(pred, T, cands)
    fam["meta_ftl"] = "meta"
    base = T.W[["code", "i0", "cutoff", "ybar"]].assign(window=w)
    out = [base.assign(strategy=s, family=fam[s], pred=p) for s, p in pred.items()]
    D = pd.concat(out, ignore_index=True)
    D["err"] = D.ybar - D.pred
    for r in wlog:
        r.setdefault("window", w)
    return D, pd.DataFrame(picks).assign(window=w)


def run(ds, force=False):
    f, fw = L.cpath(f"online_{ds}"), L.cpath(f"weights_{ds}")
    if f.exists() and not force:
        return pd.read_parquet(f)
    t0 = time.time()
    Ds, Ws, Ps = [], [], []
    for w in L.CFG["datasets"][ds]["windows"]:
        wl = []
        D, pk = run_task(ds, w, wl)
        Ds.append(D)
        Ws.append(pd.DataFrame(wl).assign(window=w))
        Ps.append(pk)
        print(f"  online {ds} {w}: {D.strategy.nunique()} 전략, {time.time() - t0:.0f}s", flush=True)
    D = pd.concat(Ds, ignore_index=True)
    D.to_parquet(f)
    pd.concat(Ws, ignore_index=True).to_parquet(fw)
    pd.concat(Ps, ignore_index=True).to_parquet(L.cpath(f"metapicks_{ds}"))
    return D


def load(ds):
    return run(ds)


def weights_log(ds):
    run(ds)
    return pd.read_parquet(L.cpath(f"weights_{ds}"))


if __name__ == "__main__":
    for ds in ("rv5", "rv5d", "gk", "gkd"):
        run(ds, force=True)
