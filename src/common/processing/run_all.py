"""proc 테이블·뷰 전부 다시 만들기

순서가 중요하다: calendar → (market_daily, kospi200_membership) / intraday_5m → rv_daily

    py -m src.common.processing.run_all
"""
from __future__ import annotations

import time

from src.common.db import connect
from src.common.processing import (
    make_etf_panel,
    make_index_membership,
    make_intraday_bars,
    make_market_controls,
    make_realized_volatility,
    make_standard_views,
    make_stock_panel,
    make_trading_calendar,
)

STEPS = [
    make_trading_calendar,
    make_market_controls,
    make_index_membership,
    make_intraday_bars,
    make_realized_volatility,
    make_stock_panel,
    make_etf_panel,
    make_standard_views,
]


def run() -> None:
    con = connect()
    for m in STEPS:
        t = time.time()
        m.build(con)
        print(f"  ({m.__name__.rsplit('.', 1)[-1]} {time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    run()
