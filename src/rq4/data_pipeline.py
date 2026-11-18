"""1단계 데이터: proc 에서 읽어 parquet 캐시로. 원자료는 읽기만 한다.

캐시 (data/interim/rq_cache/rq4/)
· etf_daily_all.parquet  : proc.etf_daily 전체 (392종, 2015~). 열은 proc 문서(processing.md)에 있는 것만 쓴다.
· market_daily.parquet   : proc.market_daily (지수·VKOSPI·예탁금 등)
· stock_ss.parquet       : proc.stock_daily 중 005930·000660 (거래대금·수익률)
· calendar.parquet       : proc.calendar

`python -m src.rq4.data_pipeline [--force]`
"""
from __future__ import annotations

import sys

import pandas as pd

from src.rq4.common import cached

ETF_COLS = ["date", "code", "fg_code", "name", "close", "nav", "nav_or_iv", "premium_pct", "shares", "volume",
            "value", "aum_calc", "traded", "ret_log", "nav_ret_log", "flow_won", "kind", "leverage",
            "replication", "grp", "grp_desc", "index_name", "manager", "fee_pct", "list_date", "delist_date",
            "indiv_qty", "indiv_amt", "inst_qty", "inst_amt", "fin_inv_qty", "fin_inv_amt", "foreign_qty",
            "foreign_amt", "aum_won"]


def _sql(q):
    from src.common.db import sql
    return sql(q)


def load_etf(force=False) -> pd.DataFrame:
    df = cached("etf_daily_all", lambda: _sql(f"select {', '.join(ETF_COLS)} from hyfin.proc.etf_daily"), force)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_market(force=False) -> pd.DataFrame:
    df = cached("market_daily", lambda: _sql("select * from hyfin.proc.market_daily"), force)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_stock_ss(force=False) -> pd.DataFrame:
    df = cached("stock_ss", lambda: _sql(
        "select date, code, close, volume, value, ret_log from hyfin.proc.stock_daily "
        "where code in ('005930','000660')"), force)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_calendar(force=False) -> pd.DataFrame:
    df = cached("calendar", lambda: _sql("select * from hyfin.proc.calendar"), force)
    df["date"] = pd.to_datetime(df["date"])
    return df


def main(force=False):
    for f in (load_etf, load_market, load_stock_ss, load_calendar):
        d = f(force)
        print(f.__name__, d.shape, d["date"].min().date(), d["date"].max().date())


if __name__ == "__main__":
    main("--force" in sys.argv)
