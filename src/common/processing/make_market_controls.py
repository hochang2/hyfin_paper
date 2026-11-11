"""시장 통제변수 → proc.market_daily (한국 거래일 1행)

국내 지수 (F11, 같은 날 종가)
  kospi, k200, kosdaq, kq150, k200_it, krx_semi : 종가 + `r_` 로그수익률 (전 거래일 대비)
  vkospi                                         : 수준

미국 (F14) — 한국 장 마감 시점에 알 수 있는 값만 쓴다
  한국 날짜 d 에 '미국 날짜 < d' 인 가장 최근 값을 붙인다 (ASOF, 등호 없음).
  sox(SOXX 지수), ndx, spx, vix, nvda, tsm : 종가(NVDA·TSM 은 수정종가) + `r_` 로그수익률 (미국 전 거래일 대비)
  us_asof_date : S&P500 기준으로 붙은 미국 날짜 (보통 한국 날짜 - 1일, 월요일이면 금요일)

거시 (F16)
  FRED(미국채10년, 원달러환율 NY 정오) : 위와 같이 전날 값 → ust10, usdkrw
  국내(고객예탁금, 신용융자, 국고채)      : 같은 날 값. 억원 → 원 (×1e8)
    deposit_won, credit_won, credit_kospi_won, credit_kosdaq_won, ktb3, ktb10
    ※ 같은 날 장 마감 이후 집계되는 값이므로, 다음 날 RV 를 예측하는 모델에서는 그대로 써도 되지만
       같은 날 RV 를 설명하는 모델에서는 하루 늦춰서 쓸 것.
"""
from __future__ import annotations

from src.common.processing import main, save

KR = {"I.001": "kospi", "I.101": "k200", "I.201": "kosdaq", "I.253": "kq150", "I.136": "k200_it", "I.163": "krx_semi"}
US = {"I.SOXX": "sox", "I.NDX": "ndx", "I.GSPC": "spx", "I.VIX": "vix", "FJ.NVDA": "nvda", "FJ.TSM": "tsm"}
FRED = {"미국채10년": "ust10", "원달러환율": "usdkrw"}
KR_MACRO = {  # 변수: (열 이름, 배수)
    "고객예탁금": ("deposit_won", 1e8),
    "신용융자_합계": ("credit_won", 1e8),
    "신용융자_거래소": ("credit_kospi_won", 1e8),
    "신용융자_코스닥": ("credit_kosdaq_won", 1e8),
    "국고채3년": ("ktb3", 1),
    "국고채10년": ("ktb10", 1),
}


def _q(xs) -> str:
    return ", ".join(f"'{x}'" for x in xs)


def build(con) -> None:
    kr_cols = ",\n      ".join(
        f"max(close) FILTER (code = '{c}') AS {n}, max(r) FILTER (code = '{c}') AS r_{n}" for c, n in KR.items()
    )
    us_cols = ",\n      ".join(
        f"max(px) FILTER (code = '{c}') AS {n}, max(r) FILTER (code = '{c}') AS r_{n}" for c, n in US.items()
    )
    fred_cols = ",\n      ".join(f"max(v) FILTER (code = '{c}') AS {n}" for c, n in FRED.items())
    krm_cols = ",\n      ".join(
        f"max(값 * {m}) FILTER (변수 = '{c}') AS {n}" for c, (n, m) in KR_MACRO.items()
    )

    sql = f"""
    WITH cal AS (SELECT date FROM hyfin.proc.calendar),
    kr AS (
      SELECT date::DATE AS date, code, close,
             ln(close / lag(close) OVER (PARTITION BY code ORDER BY date)) AS r
      FROM hyfin.raw.f11_index WHERE close IS NOT NULL
    ),
    kr_w AS (
      SELECT date,
      {kr_cols},
      max(close) FILTER (code = 'I.110') AS vkospi
      FROM kr GROUP BY date
    ),
    us AS (
      SELECT date::DATE AS d, code, coalesce(adj_close, close) AS px,
             ln(coalesce(adj_close, close) / lag(coalesce(adj_close, close)) OVER (PARTITION BY code ORDER BY date)) AS r
      FROM hyfin.raw.f14_global WHERE coalesce(adj_close, close) IS NOT NULL
    ),
    fred AS (
      SELECT date::DATE AS d, 변수 AS code, 값 AS v, NULL::DOUBLE AS r
      FROM hyfin.raw.f16_macro WHERE 변수 IN ({_q(FRED)}) AND 값 IS NOT NULL
    ),
    lagged AS (   -- 한국 날짜 × 미국 계열, 미국 날짜 < 한국 날짜 중 최신
      SELECT c.date, c.code, u.d, u.px, u.r
      FROM (SELECT date, code FROM cal, (SELECT DISTINCT code FROM us)) c
      ASOF LEFT JOIN us u ON c.code = u.code AND c.date > u.d
      UNION ALL
      SELECT c.date, c.code, f.d, f.v, f.r
      FROM (SELECT date, code FROM cal, (SELECT DISTINCT code FROM fred)) c
      ASOF LEFT JOIN fred f ON c.code = f.code AND c.date > f.d
    ),
    us_w AS (
      SELECT date,
      {us_cols},
      max(d) FILTER (code = 'I.GSPC') AS us_asof_date
      FROM lagged GROUP BY date
    ),
    fred_w AS (
      SELECT date, {fred_cols}
      FROM (SELECT date, code, px AS v FROM lagged) GROUP BY date
    ),
    krm_w AS (
      SELECT date::DATE AS date,
      {krm_cols}
      FROM hyfin.raw.f16_macro GROUP BY 1
    )
    SELECT * FROM cal
    LEFT JOIN kr_w USING (date)
    LEFT JOIN us_w USING (date)
    LEFT JOIN fred_w USING (date)
    LEFT JOIN krm_w USING (date)
    ORDER BY date
    """
    save(con, "market_daily", sql, "proc.calendar, raw.f11_index, raw.f14_global, raw.f16_macro", __file__)


if __name__ == "__main__":
    main(build)
