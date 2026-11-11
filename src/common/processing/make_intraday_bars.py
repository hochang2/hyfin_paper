"""분봉 → 5분 격자 장중 패널 → proc.intraday_5m (날짜 × 국내 종목·상품·지수 × 5분 칸)

proc.rv_daily 는 이 테이블을 날짜별로 합쳐서 만든다 (make_realized_volatility.py). 두 테이블의 수익률은 항상 같다.

- 가격: KRX 1분봉 수정주가 (configs/analysis.yaml rv.price_venue), 정규장만.
- 격자: 그날 코스피(IDX001) 정규장 1분봉을 5분으로 묶은 칸. 보통 76칸 (09:00~15:15 시작 칸).
  개장 지연일·시장 공백일은 그날 실제 시장 칸만 있다.
- close   : 칸 안 마지막 1분봉 종가. 거래 없는 칸은 직전 칸 종가 (그날 첫 거래 전 칸은 NULL).
- traded  : 칸 안에 실제 1분봉이 있었는지.
- ret     : ln(close / 직전 칸 close). 그날 첫 수익률은 첫 정규장 봉 시가 기준 (overnight 제외).
- volume_krx : 칸 안 KRX 1분봉 거래량 합 (정규장). 지수는 단위가 달라 NULL.
- 종가 단일가(15:30)는 칸에 넣지 않는다 → proc.rv_daily 의 p_close_auction, rv5_close.
"""
from __future__ import annotations

from src.common.processing import load_config, main, save


def build(con) -> None:
    rv = load_config("analysis")["rv"]
    iv = f"INTERVAL '{int(rv['bar_minutes'])} minutes'"
    venue = rv["price_venue"]

    sql = f"""
    WITH grid AS (
      SELECT date, b, row_number() OVER (PARTITION BY date ORDER BY b)::INTEGER AS k
      FROM (SELECT DISTINCT date, time_bucket({iv}, ts::TIMESTAMP) AS b
            FROM hyfin.raw.minute_idx WHERE code = '001' AND session = 'regular')
    ),
    kinds AS (
      SELECT DISTINCT kiwoom_code AS code, kind FROM hyfin.raw.f2_universe
      UNION SELECT kiwoom_code, kind FROM hyfin.raw.f8_etf_info
    ),
    bars AS (
      SELECT coalesce(k.kind, 'unknown') AS asset, m.code, m.date, m.ts, m.open, m.close, m.volume
      FROM hyfin.raw.minute_kr m LEFT JOIN kinds k USING (code)
      WHERE m.venue = '{venue}' AND m.freq = '1m' AND m.price = 'adj' AND m.session = 'regular'
      UNION ALL
      SELECT 'index', 'IDX' || code, date, ts, open, close, NULL
      FROM hyfin.raw.minute_idx WHERE session = 'regular'
    ),
    day AS (
      SELECT asset, code, date, arg_min(open, ts) AS p_open FROM bars GROUP BY ALL
    ),
    bk AS (
      SELECT code, date, time_bucket({iv}, ts::TIMESTAMP) AS b,
             arg_max(close, ts) AS c, sum(volume) AS volume_krx, count(*)::INTEGER AS n_bars
      FROM bars GROUP BY ALL
    ),
    filled AS (
      SELECT d.asset, d.code, d.date, g.k, g.b, d.p_open, bk.c, bk.volume_krx, bk.n_bars,
             last_value(bk.c IGNORE NULLS) OVER (PARTITION BY d.code, d.date ORDER BY g.b
                                                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cf
      FROM day d JOIN grid g USING (date) LEFT JOIN bk USING (code, date, b)
    )
    SELECT strptime(date, '%Y%m%d')::DATE AS date, code, asset, k, b AS bucket,
           (hour(b) * 100 + minute(b))::INTEGER AS hhmm,
           cf AS close,
           c IS NOT NULL AS traded,
           ln(cf / coalesce(lag(cf) OVER (PARTITION BY code, date ORDER BY b), p_open)) AS ret,
           volume_krx,
           coalesce(n_bars, 0) AS n_bars
    FROM filled
    ORDER BY code, date, k
    """
    save(con, "intraday_5m", sql, f"raw.minute_kr({venue} 1m adj regular), raw.minute_idx", __file__)


if __name__ == "__main__":
    main(build)
