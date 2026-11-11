"""data/interim 의 정리본을 MotherDuck `hyfin` DB의 `raw` 스키마에 원본 그대로 올린다.

- F1~F17: parquet 파일 1개 = 테이블 1개 (열·값 변경 없음). 테이블 이름 = 파일 이름 소문자.
- 분봉: 509개 파일을 종류별 3개 테이블로 합침. 원래 열·값은 그대로 두고
  파일 이름에서 뽑은 열(file, code, venue, freq, price)만 앞에 붙인다.
- 부속 CSV(_catalog, _file_list, _market_gaps)도 테이블로.
- 올린 뒤 로컬 파일과 행 수를 대조한다.

사용:  py scripts/load_motherduck.py            (전체)
       py scripts/load_motherduck.py fnguide    (F1~F17만)
       py scripts/load_motherduck.py minute     (분봉만)
토큰은 .env 의 motherduck_token 에서 읽는다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
FN = ROOT / "data/interim/fnguide"
MIN = ROOT / "data/interim/minute"
DB = "hyfin"


def load_env():
    env = ROOT / ".env"
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    if not os.environ.get("motherduck_token"):
        sys.exit(".env 에 motherduck_token 이 없습니다")


def q(p: Path) -> str:
    return "'" + p.as_posix().replace("'", "''") + "'"


def qlist(ps) -> str:
    return "[" + ", ".join(q(p) for p in ps) + "]"


def local_count(src_sql: str) -> int:
    return duckdb.sql(f"SELECT count(*) FROM {src_sql}").fetchone()[0]


def put(con, table: str, select_sql: str, src_sql: str, report: list):
    con.sql(f"CREATE OR REPLACE TABLE raw.{table} AS {select_sql}")
    remote = con.sql(f"SELECT count(*) FROM raw.{table}").fetchone()[0]
    local = local_count(src_sql)
    ok = "OK" if remote == local else "MISMATCH"
    report.append((table, local, remote, ok))
    print(f"{ok:8} raw.{table:28} local={local:>11,} remote={remote:>11,}", flush=True)


def load_fnguide(con, report):
    for p in sorted(FN.glob("F*.parquet")):
        src = f"read_parquet({q(p)})"
        put(con, p.stem.lower(), f"SELECT * FROM {src}", src, report)
    src = f"read_csv({q(FN / '_catalog.csv')}, header=true)"
    put(con, "fnguide_catalog", f"SELECT * FROM {src}", src, report)


# 파일 이름 -> 열. 원래 열은 * 로 그대로 둔다.
KR_COLS = r"""
    f AS file,
    regexp_extract(f, '^([^_]+)_', 1)            AS code,
    regexp_extract(f, '^[^_]+_([A-Z]+)_', 1)     AS venue,
    regexp_extract(f, '_([0-9]+m)', 1)           AS freq,
    CASE WHEN f LIKE '%\_raw' ESCAPE '\' THEN 'raw' ELSE 'adj' END AS price"""
IDX_COLS = r"""
    f AS file,
    regexp_extract(f, '^IDX([^_]+)_', 1)         AS code,
    regexp_extract(f, '_([0-9]+m)$', 1)          AS freq"""
US_COLS = r"""
    f AS file,
    regexp_extract(f, '^US_(.+)_[0-9]+m$', 1)    AS ticker,
    regexp_extract(f, '_([0-9]+m)$', 1)          AS freq"""


def minute_select(files, cols, order):
    src = f"read_parquet({qlist(files)}, filename=true, union_by_name=true)"
    sql = f"""
      SELECT {cols}, * EXCLUDE (filename, f)
      FROM (SELECT *, regexp_extract(replace(filename, '\\', '/'), '([^/]+)\\.parquet$', 1) AS f
            FROM {src})
      ORDER BY {order}"""
    return sql, src


def load_minute(con, report):
    files = sorted(MIN.glob("*.parquet"))
    idx = [p for p in files if p.name.startswith("IDX")]
    us = [p for p in files if p.name.startswith("US_")]
    kr = [p for p in files if p not in idx and p not in us]
    print(f"분봉 파일: 국내 {len(kr)}, 지수 {len(idx)}, 미국 {len(us)}", flush=True)
    for table, fl, cols, order in [
        ("minute_idx", idx, IDX_COLS, "code, freq, ts"),
        ("minute_us", us, US_COLS, "ticker, freq, ts_et"),
        ("minute_kr", kr, KR_COLS, "code, venue, freq, price, ts"),
    ]:
        sql, src = minute_select(fl, cols, order)
        put(con, table, sql, src, report)
    for name, table in [("_file_list.csv", "minute_file_list"), ("_catalog.csv", "minute_catalog"),
                        ("_market_gaps.csv", "minute_market_gaps")]:
        src = f"read_csv({q(MIN / name)}, header=true, all_varchar=true)"
        put(con, table, f"SELECT * FROM {src}", src, report)


def main(what: str):
    load_env()
    con = duckdb.connect("md:")
    con.sql(f"CREATE DATABASE IF NOT EXISTS {DB}")
    con.sql(f"USE {DB}")
    con.sql("CREATE SCHEMA IF NOT EXISTS raw")
    con.sql("CREATE SCHEMA IF NOT EXISTS proc")
    report = []
    if what in ("all", "fnguide"):
        load_fnguide(con, report)
    if what in ("all", "minute"):
        load_minute(con, report)
    bad = [r for r in report if r[3] != "OK"]
    print(f"\n완료: 테이블 {len(report)}개, 행 수 불일치 {len(bad)}개")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
