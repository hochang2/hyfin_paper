"""MotherDuck `hyfin` DB 연결.

연구자 A(소유자)와 B(공유 읽기 전용) 모두 같은 코드로 쓴다.
연결 후에는 누구든 `hyfin.raw.*`, `hyfin.proc.*` 로 테이블을 부른다.

    from src.common.db import connect, sql

    con = connect()
    con.sql("SELECT * FROM hyfin.raw.f2_universe LIMIT 5").show()

    df = sql("SELECT * FROM hyfin.raw.f11_index WHERE code = ?", ["I.001"])

토큰은 저장소 루트 `.env` 의 `motherduck_token` 에서 읽는다 (docs/database_guide.md 2절).
사용자 폴더 경로에 한글이 있으면 `.env` 에 `duckdb_home=<영문 경로>` 도 넣는다.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB = "hyfin"
# 연구자 B용 공유. URL은 팀 밖으로 내보내지 않는다 (docs/database_guide.md 7절).
SHARE_URL = "md:_share/hyfin_share/5bcd81c8-b423-4817-b821-42f561dc1a93"

_con: duckdb.DuckDBPyConnection | None = None


def _load_env() -> None:
    """`.env` 의 `키=값` 줄을 환경변수로 (이미 있는 값은 덮어쓰지 않음)."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    if not os.environ.get("motherduck_token"):
        raise RuntimeError(f"motherduck_token 이 없습니다. {env} 에 `motherduck_token=<토큰>` 한 줄을 넣으세요.")


def connect(fresh: bool = False) -> duckdb.DuckDBPyConnection:
    """MotherDuck 에 연결하고 `hyfin` 을 쓸 수 있는 상태로 돌려준다.

    - A: 자기 계정에 `hyfin` DB가 있으므로 그대로 쓴다.
    - B: `hyfin` 이 없으면 공유를 `hyfin` 이라는 이름으로 붙인다 (계정에 남으므로 처음 한 번만 실제로 붙음).

    연결은 한 번 만들어 재사용한다. `fresh=True` 면 새로 만든다.
    """
    global _con
    if _con is not None and not fresh:
        return _con

    if not duckdb.__version__.startswith("1."):
        raise RuntimeError(f"duckdb {duckdb.__version__} — MotherDuck 은 2.0 미지원. `pip install duckdb==1.5.6`")

    _load_env()
    # 사용자 폴더 경로에 한글 등 비ASCII 문자가 있으면 MotherDuck 확장을 ~/.duckdb 에 쓰지 못한다.
    # 그럴 때는 .env 에 `duckdb_home=C:/duckdb_ext` 처럼 영문 경로를 넣는다 (없으면 기본값).
    home = os.environ.get("duckdb_home")
    config = {"home_directory": home, "extension_directory": home} if home else {}
    con = duckdb.connect("md:", config=config)
    names = {r[0] for r in con.sql("SELECT database_name FROM duckdb_databases()").fetchall()}
    if DB not in names:
        con.sql(f"ATTACH '{SHARE_URL}' AS {DB}")
    _con = con
    return con


def sql(query: str, params: list | None = None) -> pd.DataFrame:
    """쿼리 결과를 DataFrame 으로. 큰 집계는 SQL 안에서 끝내고 결과만 받는다."""
    return connect().execute(query, params or []).df()


def close() -> None:
    global _con
    if _con is not None:
        _con.close()
        _con = None
