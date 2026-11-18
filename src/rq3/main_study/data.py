"""RQ3 예측용 데이터 (hyfin.proc 만 읽음).

- 설정은 src/rq3/config.yaml. 원자료 로드·캐시는 src.rq2.core.data 를 수정 없이 쓴다.
- `design(target, underlying)`: 예측 원점 t 마다 한 행. 설명변수는 모두 t 일 장 마감까지 알 수 있는 값,
  예측 대상은 y_next = ln RV_{t+1}.
- 상장 전(2026-05-27 전)에는 상품이 없으므로 S = X = 0 (참값). 상장 더미 D 도 0.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.rq2.core import data as rq2data

ROOT = Path(__file__).resolve().parents[3]
CFG_PATH = ROOT / "src" / "rq3" / "config.yaml"
UNDERLYING = {"삼성전자": "005930", "SK하이닉스": "000660"}
HAR_W, HAR_M = 5, 22
FIRST_ORIGIN_DAY = 30      # 상장일 포함 30번째 거래일 (config forecast_scheme.first_origin)


def load_cfg() -> dict:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    cfg["timeline"] = {k: pd.Timestamp(v) for k, v in cfg["timeline"].items() if k != "source"}
    cfg["sample"] = {k: pd.Timestamp(v) for k, v in cfg["sample"].items()}
    return cfg


CFG = load_cfg()


def raw_products() -> pd.DataFrame:
    """단일종목 상품 18개의 일별 자료 (DB 캐시)."""
    df = rq2data._cached("rq3_etf_p1", """
        SELECT date, code, name, kind, leverage, index_name, list_date,
               shares, nav_or_iv, aum_calc, aum_won, value, flow_won, indiv_amt
        FROM hyfin.proc.etf_daily WHERE grp = 'P1_단일종목상품'""")
    df["date"] = pd.to_datetime(df["date"])
    df["underlying"] = df["index_name"].map(lambda s: next(c for k, c in UNDERLYING.items() if k in s))
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def trading_days() -> pd.DatetimeIndex:
    s = CFG["sample"]
    rv = rq2data.raw_rv()
    d = rv.loc[rv.code == "005930", "date"].sort_values()
    return pd.DatetimeIndex(d[(d >= s["start"]) & (d <= s["end"])].values)


def stock_daily(code: str) -> pd.DataFrame:
    """기초자산 종가 log 수익률·거래대금(원)."""
    f1 = rq2data.raw_f1()
    out = f1.loc[f1.code == code, ["date", "ret_log", "value"]].set_index("date").sort_index()
    return out.reindex(trading_days())


def scale(underlying: str, size: str = "aum_calc", denom: str = "value") -> pd.Series:
    """S_t = Σ_i L_i(L_i − 1)·AUM_{i,t−1} / 기초자산 거래대금(또는 시가총액)_t.

    - AUM 은 전일 값. 상장일은 전일 값이 없으므로 당일 값.
    - size='aum_won' 이면 그 값이 없는 ETN 은 자동으로 빠진다.
    - 상장 전 = 0.
    """
    days = trading_days()
    listing = CFG["sample"]["listing"]
    p = raw_products()
    p = p[(p.underlying == underlying) & (p.date >= listing)]
    w = p.pivot(index="date", columns="code", values=size).reindex(days[days >= listing])
    if size == "aum_calc" and w.isna().any().any():
        raise ValueError(f"{underlying} {size} 결측: {w.isna().sum()[lambda s: s > 0].to_dict()}")
    lag = w.shift(1)
    lag.iloc[0] = w.iloc[0]
    L = p.drop_duplicates("code").set_index("code")["leverage"].astype(float)
    num = (lag * (L * (L - 1)).reindex(lag.columns)).sum(axis=1, min_count=1)
    if denom == "value":
        den = stock_daily(underlying)["value"]
    elif denom == "mktcap":
        from src.common.db import sql
        f = rq2data.CACHE / f"rq3_mktcap_{underlying}.parquet"
        if not f.exists():
            m = sql(f"SELECT date, mktcap FROM hyfin.proc.stock_daily WHERE code = '{underlying}'")
            m.to_parquet(f)
        m = pd.read_parquet(f)
        den = m.assign(date=pd.to_datetime(m.date)).set_index("date")["mktcap"].reindex(days)
    else:
        raise ValueError(denom)
    s = (num / den.reindex(num.index)).reindex(days).fillna(0.0)
    s[days < listing] = 0.0
    return s.rename("S")


@lru_cache(maxsize=32)
def design(target: str, underlying: str | None = None, measure: str = "rv5",
           size: str = "aum_calc", denom: str = "value") -> pd.DataFrame:
    """예측 원점 t 마다 한 행.

    열: y_next(ln RV_{t+1}), har_d, har_w, har_m, D, R, S, X, r, post_day(상장일=1, 상장 전 0)
    - target: RV 를 예측할 종목. underlying: 리밸런싱 압력을 계산할 기초자산 (기본 = target).
    - R = ln(1 + 100|r_t|) 는 target 의 수익률, X = ln(1 + 100|r_t^u|·S_t^u) 는 underlying 의 수익률·규모.
    - HAR 월 시차(22일)가 다 차지 않은 앞부분 행은 뺀다. 마지막 날(다음 날 RV 없음)은 y_next = NaN 으로 남긴다.
    """
    underlying = underlying or target
    days = trading_days()
    rv = rq2data.raw_rv()
    v = rv.loc[rv.code == target].set_index("date")[measure].reindex(days)
    v = v.where(v > 0)
    if v.isna().any():
        raise ValueError(f"{target} {measure} 결측/0: {list(v[v.isna()].index.date)}")

    r_t = stock_daily(target)["ret_log"]
    r_u = stock_daily(underlying)["ret_log"]
    S = scale(underlying, size, denom)
    listing = CFG["sample"]["listing"]
    post_day = pd.Series(np.cumsum(days >= listing), index=days)

    out = pd.DataFrame({
        "y_next": np.log(v).shift(-1),
        "y_next5": np.log(v.rolling(5).mean().shift(-5)),     # ln 평균 RV_{t+1..t+5} (강건성)
        "har_d": np.log(v),
        "har_w": np.log(v.rolling(HAR_W).mean()),
        "har_m": np.log(v.rolling(HAR_M).mean()),
        "D": (days >= listing).astype(float),
        "R": np.log1p(100 * r_t.abs()),
        "R_u": np.log1p(100 * r_u.abs()),                    # underlying 의 수익률 충격 (교차 탐색용)
        "S": S,
        "X": np.log1p(100 * r_u.abs() * S),
        "r": r_t,
        "post_day": post_day,
    }, index=days)
    out.index.name = "date"
    return out.dropna(subset=["har_m", "R", "X"])


AUX_VARS = ["VR", "IND", "FLOW", "DAUM"]


def aux_vars(underlying: str) -> pd.DataFrame:
    """보조 ETF 변수 (탐색용). 모두 t 일 장 마감 정보, 상장 전 0.

    VR   = ln(1 + 100·Σ 상품 거래대금 / 기초자산 거래대금)
    IND  = 100·Σ 상품 개인 순매수 금액 / 기초자산 거래대금 (부호 있음)
    FLOW = 100·Σ 자금흐름(flow_won) / Σ 전일 AUM (부호 있음, 상장일은 당일 AUM)
    DAUM = Δ ln Σ L(L−1)·AUM (상장일 0)
    결측(상장일 flow 등)은 0 으로 둔다.
    """
    days = trading_days()
    listing = CFG["sample"]["listing"]
    p = raw_products()
    p = p[(p.underlying == underlying) & (p.date >= listing)].copy()
    p["w_aum"] = p.leverage * (p.leverage - 1) * p.aum_calc
    g = p.groupby("date").agg(value=("value", "sum"), indiv=("indiv_amt", "sum"), flow=("flow_won", "sum"),
                              aum=("aum_calc", "sum"), w_aum=("w_aum", "sum"))
    g = g.reindex(days[days >= listing])
    val_u = stock_daily(underlying)["value"].reindex(g.index)
    aum_lag = g.aum.shift(1)
    aum_lag.iloc[0] = g.aum.iloc[0]
    out = pd.DataFrame({
        "VR": np.log1p(100 * g.value / val_u),
        "IND": 100 * g.indiv / val_u,
        "FLOW": 100 * g.flow / aum_lag,
        "DAUM": np.log(g.w_aum).diff(),
    }, index=g.index).fillna(0.0)
    return out.reindex(days).fillna(0.0)


def first_origin() -> pd.Timestamp:
    days = trading_days()
    return days[days >= CFG["sample"]["listing"]][FIRST_ORIGIN_DAY - 1]


def period_of(d: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """규제 구간: pre_listing / listing_announce / announce_effective / post_effective."""
    t = CFG["timeline"]
    d = pd.Series(pd.DatetimeIndex(d))
    return pd.Series(np.select(
        [d < t["listing"], d < t["announce"], d < t["effective"]],
        ["pre_listing", "listing_announce", "announce_effective"], "post_effective"), index=d.index)
