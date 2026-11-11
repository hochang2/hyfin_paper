"""분봉 → 일별 실현변동성 → proc.rv_daily (국내), proc.rv_daily_us (미국)

국내 (configs/analysis.yaml 의 rv 설정). 5분 수익률은 proc.intraday_5m 을 그대로 합친다 → 먼저 make_intraday_bars 실행.
- 가격: KRX 1분봉 수정주가, 정규장(`session = 'regular'`)만.
- 격자: 그날 코스피 1분봉 정규장 시각을 5분 단위로 묶은 칸 (보통 76칸, 수능일·시장 공백일도 그날 실제 시장에 맞춰짐).
- 칸 종가 = 칸 안 마지막 1분봉 종가. 거래가 없는 칸은 직전 칸 종가로 채운다 (0으로 채우지 않음).
- 첫 수익률은 그날 첫 정규장 봉의 시가 기준 → 전날 종가→시가(overnight)는 RV에 들어가지 않는다.
- rv5       = Σ (5분 로그수익률)²            (정규장만)
- rv5_close = rv5 + (ln 종가단일가 / 정규장 마지막 칸 종가)²   (15:30 종가 단일가 포함, 미확정 결정 → 둘 다 둠)
- rv_pk     = Parkinson = (ln 고가/저가)² / (4 ln 2)  (정규장 고저가)
- 거래량: volume_krx = KRX 1분봉 전 세션 합, volume_all = ALL(KRX+NXT) 전 세션 합 (1분봉 우선, 없으면 5분봉).
- asset: 주식 / ETF / ETN / index (F2·F8 의 kind).
- 지수(코드 `IDX001` KOSPI, `IDX201` KOSPI200, `IDX101` KOSDAQ, `IDX013` 전기·전자)도 같은 방식. 지수 거래량은 단위가 달라 넣지 않는다.

미국: 정규장(09:30~15:59 ET), 격자는 SPY 정규장 기준(보통 78칸), date = 미국 영업일(`bus_dt`). 종가 단일가 없음.
한국 날짜에 붙일 때는 전날(한국 기준) 값을 써야 한다 → proc.market_daily 참고.
"""
from __future__ import annotations

from src.common.processing import load_config, main, save


def build(con) -> None:
    rv = load_config("analysis")["rv"]
    bar = int(rv["bar_minutes"])
    venue = rv["price_venue"]
    iv = f"INTERVAL '{bar} minutes'"

    kr = f"""
    WITH bars AS (
      SELECT code, date, ts, session, open, high, low, close
      FROM hyfin.raw.minute_kr WHERE venue = '{venue}' AND freq = '1m' AND price = 'adj'
      UNION ALL
      SELECT 'IDX' || code, date, ts, session, open, high, low, close
      FROM hyfin.raw.minute_idx
    ),
    day AS (
      SELECT code, date,
             arg_min(open, ts) FILTER (session = 'regular') AS p_open,
             max(high) FILTER (session = 'regular') AS hi,
             min(low)  FILTER (session = 'regular') AS lo,
             count(*)  FILTER (session = 'regular') AS n_bars,
             arg_max(close, ts) FILTER (session = 'close') AS p_close_auction
      FROM bars GROUP BY ALL
      HAVING count(*) FILTER (session = 'regular') > 0
    ),
    agg AS (   -- 5분 수익률은 proc.intraday_5m 에서 (make_intraday_bars.py)
      SELECT code, date, any_value(asset) AS asset,
             sum(ret * ret) AS rv5, count(*) AS n_buckets, count(*) FILTER (traded) AS n_buckets_traded,
             arg_max(close, k) AS p_close_reg
      FROM hyfin.proc.intraday_5m GROUP BY ALL
    ),
    vk AS (
      SELECT code, date, sum(volume) AS volume_krx
      FROM hyfin.raw.minute_kr WHERE venue = 'KRX' AND freq = '1m' AND price = 'adj' GROUP BY ALL
    ),
    va AS (
      SELECT code, date, arg_min(v, freq) AS volume_all
      FROM (SELECT code, date, freq, sum(volume) AS v FROM hyfin.raw.minute_kr
            WHERE venue = 'ALL' AND price = 'adj' GROUP BY ALL)
      GROUP BY ALL
    ),
    out AS (
      SELECT a.date, d.code, a.asset,
             a.rv5,
             a.rv5 + coalesce(ln(d.p_close_auction / a.p_close_reg) ^ 2, 0) AS rv5_close,
             CASE WHEN d.lo > 0 THEN ln(d.hi / d.lo) ^ 2 / (4 * ln(2)) END AS rv_pk,
             d.p_open, a.p_close_reg, d.p_close_auction,
             coalesce(d.p_close_auction, a.p_close_reg) AS p_close,
             d.hi AS p_high, d.lo AS p_low,
             d.n_bars::INTEGER AS n_bars, a.n_buckets::INTEGER AS n_buckets,
             a.n_buckets_traded::INTEGER AS n_buckets_traded,
             vk.volume_krx, va.volume_all
      FROM day d JOIN agg a ON a.code = d.code AND a.date = strptime(d.date, '%Y%m%d')::DATE
      LEFT JOIN vk ON vk.code = d.code AND vk.date = d.date LEFT JOIN va ON va.code = d.code AND va.date = d.date
    )
    SELECT *,
           ln(p_close / p_open) AS ret_oc,
           ln(p_open / lag(p_close) OVER (PARTITION BY code ORDER BY date)) AS ret_overnight,
           ln(p_close / lag(p_close) OVER (PARTITION BY code ORDER BY date)) AS ret_cc
    FROM out
    ORDER BY code, date
    """
    save(con, "rv_daily", kr, f"proc.intraday_5m, raw.minute_kr({venue} 1m adj), raw.minute_idx", __file__)

    us = f"""
    WITH grid AS (
      SELECT DISTINCT bus_dt, time_bucket({iv}, ts_et::TIMESTAMP) AS b
      FROM hyfin.raw.minute_us WHERE ticker = 'SPY' AND session = 'regular'
    ),
    bars AS (SELECT * FROM hyfin.raw.minute_us WHERE session = 'regular'),
    day AS (
      SELECT ticker, bus_dt, arg_min(open, ts_et) AS p_open, max(high) AS hi, min(low) AS lo,
             count(*) AS n_bars, sum(volume) AS volume
      FROM bars GROUP BY ALL
    ),
    bk AS (
      SELECT ticker, bus_dt, time_bucket({iv}, ts_et::TIMESTAMP) AS b, arg_max(close, ts_et) AS c
      FROM bars GROUP BY ALL
    ),
    filled AS (
      SELECT d.ticker, d.bus_dt, g.b, bk.c, d.p_open,
             last_value(bk.c IGNORE NULLS) OVER (PARTITION BY d.ticker, d.bus_dt ORDER BY g.b
                                                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cf
      FROM day d JOIN grid g USING (bus_dt) LEFT JOIN bk USING (ticker, bus_dt, b)
    ),
    r AS (
      SELECT ticker, bus_dt, b, c, cf,
             ln(cf / coalesce(lag(cf) OVER (PARTITION BY ticker, bus_dt ORDER BY b), p_open)) AS r
      FROM filled
    ),
    agg AS (
      SELECT ticker, bus_dt, sum(r * r) AS rv5, count(*) AS n_buckets, count(c) AS n_buckets_traded,
             arg_max(cf, b) AS p_close
      FROM r GROUP BY ALL
    )
    SELECT strptime(d.bus_dt, '%Y%m%d')::DATE AS date, d.ticker,
           a.rv5,
           CASE WHEN d.lo > 0 THEN ln(d.hi / d.lo) ^ 2 / (4 * ln(2)) END AS rv_pk,
           d.p_open, a.p_close, d.hi AS p_high, d.lo AS p_low,
           d.n_bars::INTEGER AS n_bars, a.n_buckets::INTEGER AS n_buckets,
           a.n_buckets_traded::INTEGER AS n_buckets_traded, d.volume,
           ln(a.p_close / d.p_open) AS ret_oc,
           ln(a.p_close / lag(a.p_close) OVER (PARTITION BY d.ticker ORDER BY d.bus_dt)) AS ret_cc
    FROM day d JOIN agg a USING (ticker, bus_dt)
    ORDER BY ticker, date
    """
    save(con, "rv_daily_us", us, "raw.minute_us(1m regular)", __file__)


if __name__ == "__main__":
    main(build)
