"""raw → proc 전처리. raw = 백업, proc = 실제 사용. 분석 코드는 proc 만 읽는다.

| 파일 | 만드는 테이블 | 행 단위 |
|---|---|---|
| make_trading_calendar.py   | proc.calendar      | 한국 거래일 |
| make_index_membership.py   | proc.kospi200_membership | 거래일 × KOSPI200 편입 경험 종목 |
| make_intraday_bars.py      | proc.intraday_5m   | 거래일 × 국내 종목·상품·지수 × 5분 칸 |
| make_realized_volatility.py| proc.rv_daily      | 거래일 × 국내 종목·상품·지수 |
|                            | proc.rv_daily_us   | 미국 거래일 × 미국 종목 |
| make_market_controls.py    | proc.market_daily  | 한국 거래일 (국내 지수, 미국 지표는 전날 값) |
| make_stock_panel.py        | proc.stock_daily   | 거래일 × 주식 213종 |
| make_etf_panel.py          | proc.etf_daily     | 거래일 × ETF·ETN 392종 |
| make_standard_views.py     | 뷰 6개: futures_all, options_all, minute_kr, minute_idx, minute_us, etf_aum_all |

열 설명·점검 결과는 processing.md.

전부 다시 만들기:   py -m src.common.processing.run_all
하나만:             py -m src.common.processing.make_realized_volatility

공통 규칙
- `date` 는 DATE, `code` 는 접두사 없는 6자리 (분봉과 같음). FnGuide 원래 코드는 `fg_code`.
- 값을 0으로 채우지 않는다. 없는 값은 NULL.
- 테이블 주석에 만든 날짜·원천·코드 경로를 남긴다 (`SELECT comment FROM duckdb_tables()`).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import yaml

from src.common.db import ROOT, connect

PROC = "hyfin.proc"


def load_config(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))


def save(con: duckdb.DuckDBPyConnection, table: str, select_sql: str, src: str, code_file: str) -> int:
    """`select_sql` 결과로 proc.<table> 을 새로 만들고 주석을 단다. 행 수를 돌려준다."""
    full = f"{PROC}.{table}"
    con.sql(f"CREATE OR REPLACE TABLE {full} AS {select_sql}")
    code = Path(code_file).resolve().relative_to(ROOT).as_posix()
    note = f"A, {dt.date.today()}, src={src}, code={code}".replace("'", "''")
    con.sql(f"COMMENT ON TABLE {full} IS '{note}'")
    n = con.sql(f"SELECT count(*) FROM {full}").fetchone()[0]
    print(f"{full:28} {n:>12,} rows", flush=True)
    return n


def save_view(con: duckdb.DuckDBPyConnection, view: str, select_sql: str, src: str, code_file: str) -> None:
    """raw 를 형식만 맞춰 보여주는 뷰. 저장 공간을 쓰지 않고, 조회할 때마다 raw 를 읽는다."""
    full = f"{PROC}.{view}"
    con.sql(f"CREATE OR REPLACE VIEW {full} AS {select_sql}")
    code = Path(code_file).resolve().relative_to(ROOT).as_posix()
    note = f"VIEW, A, {dt.date.today()}, src={src}, code={code}".replace("'", "''")
    con.sql(f"COMMENT ON VIEW {full} IS '{note}'")
    print(f"{full:28} {'view':>12}", flush=True)


def main(build) -> None:
    """각 파일의 `if __name__ == "__main__": main(build)` 용."""
    build(connect())
