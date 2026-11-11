"""ETF·ETN 일별 패널 → proc.etf_daily (거래일 × 392종)

기본: F7 (종가·NAV/IV·좌수·거래량·괴리율)
  traded       = 종가가 있는 날 (상장 전·상장폐지 후·무거래일은 FALSE, 값은 NULL 그대로)
  ret_log      = ln(종가 / 직전 행 종가)    — 직전 행이 무거래면 NULL
  nav_ret_log  = ln(NAV·IV / 직전 행 NAV·IV)
  flow_won     = (좌수 - 직전 좌수) × 오늘 NAV·IV   (설정 +, 환매 -)
상품 정보 (F8): kind(ETF/ETN), leverage(2/-1/…), replication, grp, grp_desc, index_name, manager,
                fee_pct, total_cost_pct, lp, list_date, delist_date
  ※ F8 grp 의 S1~S3 은 '1배 비교 상품' 그룹이라 F2 의 S 그룹(주식)과 뜻이 다르다.
F9 투자자별 순매수 (_qty, _amt), F10 순자산 aum_won (= 백만원 × 1e6)
"""
from __future__ import annotations

from src.common.processing import main, save


def build(con) -> None:
    sql = """
    WITH base AS (
      SELECT date::DATE AS date, substr(code, 2) AS code, code AS fg_code, name,
             close, nav, iv, nav_or_iv, premium_pct, shares, volume, value, aum_calc,
             close IS NOT NULL AS traded,
             ln(close / lag(close) OVER w) AS ret_log,
             ln(nav_or_iv / lag(nav_or_iv) OVER w) AS nav_ret_log,
             (shares - lag(shares) OVER w) * nav_or_iv AS flow_won
      FROM hyfin.raw.f7_etf_daily
      WINDOW w AS (PARTITION BY code ORDER BY date)
    )
    SELECT b.*,
           e.kind, e.leverage::INTEGER AS leverage, e.replication, e."group" AS grp, e.group_desc AS grp_desc,
           e.index_name, e.manager, e.fee_pct, e.total_cost_pct, e.lp,
           e.list_date::DATE AS list_date, e.delist_date::DATE AS delist_date,
           i.indiv_qty, i.indiv_amt, i.inst_qty, i.inst_amt, i.fin_inv_qty, i.fin_inv_amt,
           i.foreign_qty, i.foreign_amt,
           a.aum_mil * 1e6 AS aum_won
    FROM base b
    LEFT JOIN hyfin.raw.f8_etf_info e ON e.code = b.fg_code
    LEFT JOIN hyfin.raw.f9_etf_investor i ON i.code = b.fg_code AND i.date::DATE = b.date
    LEFT JOIN hyfin.raw.f10_etf_aum a ON a.code = b.fg_code AND a.date::DATE = b.date
    ORDER BY code, date
    """
    save(con, "etf_daily", sql, "raw.f7_etf_daily, raw.f8_etf_info, raw.f9_etf_investor, raw.f10_etf_aum", __file__)


if __name__ == "__main__":
    main(build)
