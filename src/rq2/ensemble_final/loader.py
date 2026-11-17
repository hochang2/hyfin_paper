"""공통 로더: 기존 캐시의 창평균 예측을 (자료, 창) 별 넓은 표로 모은다. 새 적합은 하지 않는다.

넓은 표 = 행 (code, i0, cutoff) · 열 ybar + 모형별 predbar. 오차 e = ybar − predbar (참효과 0 인 가짜 개입일에서 효과 추정오차).
자료:
  gk    ml_ext gk 캐시 (3종목 × 238, 10일 간격, 이동 250일 학습) — 13개 모형 + 기존 앙상블
  rv5   ml_ext rv5 캐시 (3종목 × 32, 3일 간격) — 같은 모형
  rv5d  ml_followup exp1 local_pseudo (9개) + exp3 fm_zs (FM 2개) 가 모두 있는 donor 20종목
  gkd   ml_followup exp5 gk_dense (3종목, 3일 간격, 이동 140일) — 8개 모형 (Exp5 감사용)
  space ml_ext space 캐시 (실제 개입일 i0=190, donor 175 + 처치 2 + 음성통제 1) — 13개 모형 + 기존 앙상블
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE / "results"
CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
CACHE = ROOT / CFG["cache_dir"]
EXT = ROOT / CFG["ext_cache_dir"]
FU = ROOT / CFG["fu_cache_dir"]
SEED = CFG["seed"]
WIN = {"h5": (0, 5), "h10": (0, 10), "h20": (0, 20), "h30": (0, 30), "h61": (0, 61),
       "transition": (0, 10), "post_effective": (10, 71)}
REAL = CFG["real_last_index"]
POOL = CFG["pool"]
OLD = {"ens_stack": "old_stack", "ens_equal": "old_equal", "ens_invmse": "old_invmse", "ens_hybrid": "old_hybrid"}
STEP = {k: v["step"] for k, v in CFG["datasets"].items()}


@functools.lru_cache(None)
def _ext_long(ds: str) -> pd.DataFrame:
    """ml_ext 캐시의 모든 기본 모형 + 기존 앙상블 창평균 (변형 '@' 제외)."""
    names = POOL["full"] + ["naive_mean", "chronos2_uni", "timesfm_uni"] + list(OLD)
    out = []
    for m in names:
        f = EXT / ds / f"{m}_win.parquet"
        if f.exists():
            d = pd.read_parquet(f, columns=["code", "i0", "cutoff", "window", "ybar", "predbar"])
            out.append(d.assign(model=OLD.get(m, m)))
    return pd.concat(out, ignore_index=True)


@functools.lru_cache(None)
def _rv5d_long() -> pd.DataFrame:
    lp = pd.read_parquet(FU / "exp1" / "local_pseudo.parquet")
    lp = lp[lp.group == "donor"].assign(model=lambda d: d.algo)[["code", "i0", "window", "ybar", "predbar", "model"]]
    fz = pd.read_parquet(FU / "exp3" / "fm_zs.parquet")
    fz = fz[fz.model.isin(POOL["fm"]) & (fz.i0 < REAL)]
    rows = []
    for (c, i0, m), g in fz.groupby(["code", "i0", "model"]):
        g = g.sort_values("h")
        y, p = g.y.values, g.pred.values
        for w, (a, b) in WIN.items():
            if b <= len(g):
                rows.append((c, i0, w, y[a:b].mean(), p[a:b].mean(), m))
    fw = pd.DataFrame(rows, columns=["code", "i0", "window", "ybar", "predbar", "model"])
    codes = sorted(set(lp.code) & set(fw.code) - {"005930", "000660", "005935"})
    d = pd.concat([lp, fw])
    d = d[d.code.isin(codes)]
    idx = rv5_index()
    return d.assign(cutoff=idx[d.i0.values])


@functools.lru_cache(None)
def _gkd_long() -> pd.DataFrame:
    d = pd.read_parquet(FU / "exp5" / "gk_dense.parquet", columns=["model", "code", "i0", "window", "ybar", "predbar"])
    d = d[d.model != "naive_mean"]
    return d.assign(cutoff=gk_index()[d.i0.values])


def long(ds: str) -> pd.DataFrame:
    if ds in ("gk", "rv5", "space"):
        d = _ext_long(ds)
    elif ds == "rv5d":
        d = _rv5d_long()
    elif ds == "gkd":
        d = _gkd_long()
    else:
        raise ValueError(ds)
    if ds in ("rv5", "rv5d"):
        d = d[d.i0 + d.window.map(lambda w: WIN[w][1]) <= REAL]
    return d


@functools.lru_cache(None)
def wide(ds: str, window: str) -> pd.DataFrame:
    """행 (code, i0) · 열 cutoff, ybar, 모형들. 모든 모형이 같은 (code, i0) 를 갖는 행만 남긴다 (평가 시점 일치)."""
    d = long(ds)
    d = d[d.window == window]
    y = d.groupby(["code", "i0"]).agg(cutoff=("cutoff", "first"), ybar=("ybar", "first"))
    # 같은 과제에서 모형 간 ybar 가 일치하는지 확인 (같은 평가 자료)
    chk = d.groupby(["code", "i0"]).ybar.agg(lambda s: s.max() - s.min())
    assert chk.max() < 1e-8, f"{ds} {window}: 모형 간 ybar 불일치 {chk.max()}"
    P = d.pivot_table(index=["code", "i0"], columns="model", values="predbar")
    W = y.join(P).reset_index().sort_values(["i0", "code"]).reset_index(drop=True)
    base = [m for m in POOL["full"] if m in W.columns]
    return W.dropna(subset=base).reset_index(drop=True)


def members(ds: str) -> list[str]:
    W = wide(ds, CFG["datasets"].get(ds, {"windows": ["transition"]})["windows"][0])
    return [m for m in POOL["full"] if m in W.columns]


# ================================================================ 날짜 · 국면
@functools.lru_cache(None)
def rv5_index() -> pd.DatetimeIndex:
    from src.rq2.ml_extension import data_pipeline as DP
    return DP.rv5_panel().index


@functools.lru_cache(None)
def gk_index() -> pd.DatetimeIndex:
    from src.rq2.ml_extension import data_pipeline as DP
    return DP.gk_panel().index


@functools.lru_cache(None)
def _don_mean(kind: str) -> pd.Series:
    from src.rq2.ml_extension import data_pipeline as DP
    if kind == "gk":
        return DP.gk_panel()["don_mean"]
    X = DP.rv5_controls(DP.rv5_panel().index[REAL - 1])
    return X["don_mean"]


@functools.lru_cache(None)
def _kospi() -> pd.Series:
    m = pd.read_parquet(ROOT / "data/interim/rq_cache/rq2_market_daily.parquet")
    return np.log1p(m.set_index("date")["r_kospi"].fillna(0.0))


def regime_frame(ds: str) -> pd.DataFrame:
    """가짜 개입일 i0 마다 그 전날까지 관측 가능한 국면 변수.
    vol_hi : donor 평균 변동성 20일 평균이 그 시점까지의 과거 가짜 개입일 값의 중앙값 초과 (확장 기준, 미래 미사용)
    trend  : KOSPI 20일 누적수익률이 과거 분포 20% 미만 crash / 80% 초과 rally / 그 외 normal
    vol_t3, ret_t3 : 묘사용 전체 삼분위 (가중치 결정에 쓰지 않음)"""
    R = CFG["regime"]
    kind = "gk" if ds in ("gk", "gkd") else "rv5"
    idx = gk_index() if kind == "gk" else rv5_index()
    dm = _don_mean(kind).reindex(idx)
    k = _kospi()
    if ds == "space":
        cuts = [REAL]
    else:
        cuts = sorted(long(ds).i0.unique())
    rows = []
    for i0 in cuts:
        vol = dm.iloc[i0 - R["vol_lookback"]:i0].mean()
        d0 = idx[i0]
        kk = k[k.index < d0].iloc[-R["ret_lookback"]:]
        rows.append({"i0": i0, "cutoff": d0, "pre_vol20": vol, "pre_ret20": kk.sum()})
    F = pd.DataFrame(rows)
    if ds == "space":   # 실제일 국면은 rv5 가짜 개입일 분포 기준
        ref = regime_frame("rv5")
        F["vol_hi"] = F.pre_vol20 > ref.pre_vol20.median()
        lo, hi = ref.pre_ret20.quantile([R["crash_q"], R["rally_q"]])
        F["trend"] = np.where(F.pre_ret20 < lo, "crash", np.where(F.pre_ret20 > hi, "rally", "normal"))
        return F
    v, r = F.pre_vol20.values, F.pre_ret20.values
    vh, tr = [], []
    for j in range(len(F)):
        pv, pr = v[:j + 1], r[:j + 1]          # 현재 시점까지 관측 가능한 값만
        vh.append(bool(v[j] > np.median(pv)) if j >= 5 else False)
        if j >= 10:
            lo, hi = np.quantile(pr, [R["crash_q"], R["rally_q"]])
            tr.append("crash" if r[j] < lo else ("rally" if r[j] > hi else "normal"))
        else:
            tr.append("normal")
    F["vol_hi"], F["trend"] = vh, tr
    F["vol_t3"] = pd.qcut(F.pre_vol20, 3, labels=["low", "mid", "high"]).astype(str)
    F["ret_t3"] = pd.qcut(F.pre_ret20, 3, labels=["down", "flat", "up"]).astype(str)
    if kind == "gk":
        eras = [pd.Timestamp(x) for x in R["gk_eras"]]
        lab = ["2016-19", "2020-21", "2022-23", "2024-26"]
        F["era"] = [lab[max(0, np.searchsorted(eras, d, side="right") - 1)] for d in F.cutoff]
    else:
        F["era"] = np.where(F.i0 + 30 <= 155, "pre_listing", "cross_listing")   # 2026-05-27 상장일
    return F


# ================================================================ 저장
def save_table(df: pd.DataFrame, name: str, round_: int = 4) -> Path:
    p = RES / "tables" / f"{name}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.round(round_).to_csv(p, index=False, encoding="utf-8-sig")
    return p


def cpath(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{name}.parquet"
