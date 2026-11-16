"""RQ2 분석용 데이터 (proc 만 읽음).

- `load_panel()`: 일별 log RV 패널 (처치 2종, 삼성전자우, donor 주식, 지수) + 시장 통제변수
- `build_design()`: 처치 종목 하나에 대한 y, X (통제변수는 사전구간 정보로만 만든다)
- `load_long_daily()`: F1 일별 OHLC 2015~ → Garman-Klass 변동성 (긴 사전구간 강건성용)

DB 쿼리 결과는 data/interim/rq_cache/ 에 parquet 으로 캐시한다 (Git 제외).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "interim" / "rq_cache"
CFG_PATH = ROOT / "configs" / "rq2.yaml"


def load_cfg() -> dict:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    cfg["timeline"] = {k: pd.Timestamp(v) for k, v in cfg["timeline"].items()}
    for k in ("start", "end", "train_end"):
        cfg["sample"][k] = pd.Timestamp(cfg["sample"][k])
    return cfg


def _cached(name: str, query: str) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    from src.common.db import sql
    df = sql(query)
    df.to_parquet(f)
    return df


def raw_rv() -> pd.DataFrame:
    df = _cached("rq2_rv_daily", """
        SELECT date, code, asset, rv5, rv5_close, rv_pk, ret_cc, ret_oc, ret_overnight,
               n_buckets, n_buckets_traded, volume_krx, volume_all
        FROM hyfin.proc.rv_daily WHERE asset IN ('주식','index')""")
    df["date"] = pd.to_datetime(df["date"])
    return df


def raw_market() -> pd.DataFrame:
    df = _cached("rq2_market_daily", "SELECT * FROM hyfin.proc.market_daily WHERE date >= DATE '2014-12-01'")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def raw_calendar() -> pd.DataFrame:
    df = _cached("rq2_calendar", "SELECT * FROM hyfin.proc.calendar")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def raw_f1() -> pd.DataFrame:
    df = _cached("rq2_stock_daily_ohlc", """
        SELECT date, code, name, open, high, low, close, volume, value, ret_log, halted, grp
        FROM hyfin.proc.stock_daily""")
    df["date"] = pd.to_datetime(df["date"])
    return df


def donor_codes(cfg: dict, pool: str = "main") -> list[str]:
    """donor pool. main = 반도체·삼성·SK 계열·늦은 상장 제외 / all = 처치·우선주만 제외 / strict = main + 전자·IT 추가 제외."""
    rv = raw_rv()
    st = rv[rv.asset == "주식"]
    n = st.groupby("code").size()
    full = set(n[n >= n.max()].index)          # 261일 모두 있는 종목
    base = full - set(cfg["treated"]) - {cfg["within_firm_control"]}
    if pool == "all":
        return sorted(base)
    ex = set().union(*[set(v) for v in cfg["donor_exclude"].values()])
    out = base - ex
    if pool == "strict":
        out -= {"066570", "064400", "022100", "307950", "035420", "035720", "373220", "051910", "066970", "003670"}
    return sorted(out)


@lru_cache(maxsize=4)
def load_panel(measure: str = "rv5") -> pd.DataFrame:
    """date × code 의 log 변동성 (wide). measure ∈ rv5, rv5_close, rv_pk."""
    rv = raw_rv()
    v = rv.pivot(index="date", columns="code", values=measure).sort_index()
    v = v.where(v > 0)
    return np.log(v)


def _pca_scores(Z: pd.DataFrame, train_mask: pd.Series, k: int = 3) -> pd.DataFrame:
    """사전구간(train_mask)으로만 표준화·고유벡터를 추정하고 전체 기간에 투영 (look-ahead 방지)."""
    Zt = Z[train_mask]
    mu, sd = Zt.mean(), Zt.std(ddof=0).replace(0, 1)
    S = ((Z - mu) / sd).fillna(0.0)
    St = S[train_mask]
    _, _, vt = np.linalg.svd(St.values - St.values.mean(0), full_matrices=False)
    vt = vt[:k]
    # 부호 고정: 첫 성분이 평균과 같은 방향이 되도록
    for i in range(k):
        if vt[i].sum() < 0:
            vt[i] = -vt[i]
    sc = S.values @ vt.T
    return pd.DataFrame(sc, index=Z.index, columns=[f"don_pc{i+1}" for i in range(k)])


def build_controls(cfg: dict, measure: str = "rv5", pool: str = "main",
                   train_end: pd.Timestamp | None = None) -> pd.DataFrame:
    """일별 통제변수 표. 모든 열은 그날 장 마감 시점에 관측 가능 (미국 값은 전날 밤 종가)."""
    train_end = train_end or cfg["sample"]["train_end"]
    lv = load_panel(measure)
    lv = lv.loc[cfg["sample"]["start"]:cfg["sample"]["end"]]
    donors = donor_codes(cfg, pool)
    D = lv[donors]
    X = pd.DataFrame(index=lv.index)
    X["don_mean"] = D.mean(axis=1)
    X["don_median"] = D.median(axis=1)
    pcs = _pca_scores(D, D.index <= train_end, k=3)
    X = X.join(pcs)
    X["kosdaq_lrv"] = lv["IDX101"]
    X["kospi_lrv"] = lv["IDX001"]
    X["elec_lrv"] = lv["IDX013"]
    m = raw_market().reindex(X.index)
    X["us_lvix"] = np.log(m["vix"])
    X["us_abs_sox"] = m["r_sox"].abs() * 100
    X["us_abs_ndx"] = m["r_ndx"].abs() * 100
    X["us_r_sox"] = m["r_sox"] * 100
    X["lvkospi"] = np.log(m["vkospi"])
    cal = raw_calendar().reindex(X.index)
    X["gap_day"] = (cal["gap_min"].fillna(0) > 0).astype(float)
    X["late_open"] = cal["late_open"].fillna(False).astype(float)
    X["monday"] = (X.index.dayofweek == 0).astype(float)
    return X


def build_design(cfg: dict, code: str, controls: list[str], measure: str = "rv5", pool: str = "main",
                 train_end: pd.Timestamp | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """y(처치 종목 log RV)와 X(통제변수). 결측 행은 제거."""
    lv = load_panel(measure)
    y = lv[code].loc[cfg["sample"]["start"]:cfg["sample"]["end"]].rename("y")
    X = build_controls(cfg, measure, pool, train_end)[controls] if controls else pd.DataFrame(index=y.index)
    df = pd.concat([y, X], axis=1).dropna()
    return df["y"], df.drop(columns="y")


# ---------------------------------------------------------------- 긴 사전구간 (F1 일별 OHLC)
def garman_klass(o, h, l, c) -> pd.Series:
    """Garman-Klass 일중 분산 (시가→종가, overnight 제외)."""
    hl = np.log(h / l)
    co = np.log(c / o)
    return 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2


def load_long_daily(cfg: dict, pool: str = "main") -> pd.DataFrame:
    """2015~ 일별 log GK 변동성: 처치 2종, 005935, donor 평균. 가격제한(상하한가)·거래정지일은 NaN."""
    f1 = raw_f1()
    f1 = f1[(f1.volume > 0) & (f1.high > f1.low)].copy()
    f1["gk"] = garman_klass(f1.open, f1.high, f1.low, f1.close)
    f1 = f1[f1.gk > 0]
    f1["lgk"] = np.log(f1.gk)
    w = f1.pivot(index="date", columns="code", values="lgk").sort_index()
    r = f1.pivot(index="date", columns="code", values="ret_log").sort_index()
    donors = [c for c in donor_codes(cfg, pool) if c in w.columns]
    # 2015 부터 계속 존재한 donor 만 (구성 변화로 평균이 흔들리지 않게)
    cov = w[donors].loc[:"2015-12-31"].notna().mean()
    donors_long = [c for c in donors if cov.get(c, 0) > 0.95 and w[c].notna().mean() > 0.9]
    out = pd.DataFrame(index=w.index)
    for c in cfg["treated"] + [cfg["within_firm_control"]]:
        out[c] = w[c]
        out[f"r_{c}"] = r[c]
    out["don_mean"] = w[donors_long].mean(axis=1)
    out["don_ret"] = r[donors_long].mean(axis=1)
    out["n_donor"] = w[donors_long].notna().sum(axis=1)
    m = raw_market().reindex(out.index)
    out["us_lvix"] = np.log(m["vix"])
    out["us_abs_sox"] = m["r_sox"].abs() * 100
    out["kosdaq_ret_abs"] = m["r_kosdaq"].abs() * 100
    out.attrs["donors_long"] = donors_long
    return out
