"""RQ1 분석용 데이터 (hyfin.proc 만 읽음).

- 설정은 src/rq1/config.yaml 에서, 대조군 정의(donor pool)는 RQ2 와 같게 configs/rq2.yaml 에서 읽는다.
- 원자료 로드·캐시는 src.rq2.core.data 를 그대로 쓴다 (수정하지 않음).
- `panel_long()`: 종목 × 일 long 패널. 표본은 sample.start ~ sample.treat_end (발표 전날) 까지만.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.rq2.core import data as rq2data

ROOT = Path(__file__).resolve().parents[3]
CFG1_PATH = ROOT / "src" / "rq1" / "config.yaml"
BIN_DAYS = 5          # 이벤트 스터디 구간 길이 (거래일)
REF_BIN_MAX = -9      # 이 구간 이하를 모두 기준(reference)으로 둔다 (config main_design.event_study)


def load_cfg() -> dict:
    cfg = yaml.safe_load(CFG1_PATH.read_text(encoding="utf-8"))
    cfg["timeline"] = {k: pd.Timestamp(v) for k, v in cfg["timeline"].items() if k != "source"}
    cfg["sample"] = {k: pd.Timestamp(v) for k, v in cfg["sample"].items()}
    return cfg


CFG = load_cfg()
RQ2CFG = rq2data.load_cfg()   # donor_exclude 등 대조군 정의만 쓴다


def groups(pool: str = "main") -> dict[str, list[str]]:
    """역할별 종목 목록. 서로 겹치지 않는다."""
    return {
        "treated": list(CFG["treated"]),
        "donor": rq2data.donor_codes(RQ2CFG, pool),
        "neg_same": [CFG["negative_control"]],
        "neg_semis": list(CFG["semis_negative_control"]),
    }


def role_of(pool: str = "main") -> dict[str, str]:
    return {c: role for role, codes in groups(pool).items() for c in codes}


@lru_cache(maxsize=8)
def panel_long(measure: str = "rv5", pool: str = "main") -> pd.DataFrame:
    """종목 × 일 long 패널.

    열: date, code, role, y(log 변동성), post(상장 이후), rel(상장일=0 인 거래일 차이), bin(5일 구간)
    - 표본: sample.start ~ sample.treat_end. 07-16 이후 자료는 들어오지 않는다.
    - 변동성이 0 이하이거나 없는 칸은 y=NaN 으로 두고 행을 뺀다 (0 으로 채우지 않음).
    """
    s = CFG["sample"]
    lv = rq2data.load_panel(measure).loc[s["start"]:s["treat_end"]]
    roles = role_of(pool)
    lv = lv[[c for c in roles if c in lv.columns]]

    dates = lv.index
    listing_pos = int(np.searchsorted(dates.values, s["treat_start"].to_datetime64()))
    rel = pd.Series(np.arange(len(dates)) - listing_pos, index=dates)

    df = lv.stack(future_stack=True).rename("y").reset_index()
    df.columns = ["date", "code", "y"]
    df = df.dropna(subset=["y"])
    df["role"] = df["code"].map(roles)
    df["post"] = (df["date"] >= s["treat_start"]).astype(int)
    df["rel"] = df["date"].map(rel).astype(int)
    df["bin"] = np.floor_divide(df["rel"], BIN_DAYS).astype(int)
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def gk_panel_long(pool: str = "main") -> pd.DataFrame:
    """일별 log Garman-Klass 변동성(F1 OHLC) 으로 같은 모양의 패널. 측정치 강건성용."""
    s = CFG["sample"]
    f1 = rq2data.raw_f1()
    f1 = f1[(f1.volume > 0) & (f1.high > f1.low)].copy()
    f1["gk"] = rq2data.garman_klass(f1.open, f1.high, f1.low, f1.close)
    f1 = f1[f1.gk > 0]
    w = np.log(f1.pivot(index="date", columns="code", values="gk")).sort_index()
    base = panel_long("rv5", pool)[["date", "code", "role", "post", "rel", "bin"]]
    w = w.loc[s["start"]:s["treat_end"]]
    g = w.stack().rename("y").reset_index()
    g.columns = ["date", "code", "y"]
    return base.merge(g, on=["date", "code"], how="inner")


def long_gk_wide(codes: list[str]) -> pd.DataFrame:
    """2015~ 일별 log GK (wide). 긴 표본 placebo 보정용."""
    f1 = rq2data.raw_f1()
    f1 = f1[f1.code.isin(codes) & (f1.volume > 0) & (f1.high > f1.low)].copy()
    f1["gk"] = rq2data.garman_klass(f1.open, f1.high, f1.low, f1.close)
    f1 = f1[f1.gk > 0]
    return np.log(f1.pivot(index="date", columns="code", values="gk")).sort_index()


def match_vars() -> pd.DataFrame:
    """매칭 변수 (상장 전날 기준): log 시가총액, 전년도 베타."""
    from src.common.db import sql
    f = rq2data.CACHE / "rq1_match_vars.parquet"
    if f.exists():
        return pd.read_parquet(f)
    df = sql(f"SELECT code, mktcap, beta_prev FROM hyfin.proc.stock_daily WHERE date = DATE '{CFG['sample']['pre_end']:%Y-%m-%d}'")
    rq2data.CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(f)
    return df


def expiry_days() -> list[pd.Timestamp]:
    """표본 기간의 월 옵션·선물 만기일: 매월 둘째 목요일, 휴장이면 직전 거래일 (규칙으로 계산)."""
    s = CFG["sample"]
    tdays = rq2data.raw_calendar().loc[s["start"]:s["treat_end"]].index
    out = []
    for m in pd.period_range(s["start"], s["treat_end"], freq="M"):
        thu = pd.date_range(m.start_time, m.end_time, freq="W-THU")[1]
        prev = tdays[tdays <= thu]
        if len(prev) and prev[-1].month == thu.month:
            out.append(prev[-1])
    return out


def missing_cells(measure: str = "rv5", pool: str = "main") -> pd.DataFrame:
    """표본 안에서 값이 빠진 (date, code). 데이터 점검표용."""
    s = CFG["sample"]
    lv = rq2data.load_panel(measure).loc[s["start"]:s["treat_end"]]
    roles = role_of(pool)
    lv = lv[[c for c in roles if c in lv.columns]]
    m = lv.isna().stack()
    out = m[m].reset_index()
    out.columns = ["date", "code", "_"]
    out["role"] = out["code"].map(roles)
    return out.drop(columns="_")


def calendar_flags() -> pd.DataFrame:
    """표본 기간의 특수일 (시장 공백일, 개장 지연일)."""
    s = CFG["sample"]
    cal = rq2data.raw_calendar().loc[s["start"]:s["treat_end"]]
    return pd.DataFrame({"gap_day": cal["gap_min"].fillna(0) > 0,
                         "late_open": cal["late_open"].fillna(False).astype(bool)})
