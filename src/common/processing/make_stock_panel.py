"""주식 일별 패널 → proc.stock_daily (거래일 × 주식 213종)

기본: F1 (가격·거래량·시총·유동비율·외국인지분율)
  ret_log    = ln(1 + ret_pct/100)  (FnGuide 수익률 = 권리락·분할 반영)
  halted     = 거래량 0 인 날 (거래정지 추정)
  first_date = 이 종목이 F1 에 처음 나온 날 (신규상장 확인용)
  grp        = F2 그룹 (S1_기초자산 / S2_KOSPI200 / S3_KOSPI200편출)
붙이는 것 (같은 날)
  F3 투자자별 순매수 (_qty, _amt 원 그대로), F4 신용·공매도, F6 PBR·PER, F5 FY2026 영업이익 컨센서스
미래정보 차단 (연간 자료)
  F5 영업이익(실적) : 잠정실적 공시일(없으면 사업보고서 제출일) 다음 날부터 → op_income_last, op_income_fy
  F6 자산·매출·부채비율 : 사업보고서 제출일(F5 report_date, 없으면 다음 해 3/31) 다음 날부터 → *_last, fin_fy
  F1 베타 : 전년도 값 → beta_prev
삼성전자·SK하이닉스만 (나머지는 NULL)
  F12 근월물 선물: fut_close, fut_settle, fut_spot, fut_volume(전 월물 합), fut_oi(전 월물 합), fut_roll_day,
                   fut_basis_pct = (선물종가 / 현물 - 1) × 100
  F13 ATM 옵션:   opt_iv_atm, opt_iv_call, opt_iv_put, opt_dte, opt_volume, opt_oi
"""
from __future__ import annotations

from src.common.processing import main, save

INVESTORS = ["indiv", "foreign", "foreign_other", "inst", "fin_inv", "insur", "trust", "bank", "pension",
             "corp_other", "pef"]
FUT_MAP = "CASE underlying WHEN '삼성전자' THEN 'A005930' WHEN 'SK하이닉스' THEN 'A000660' END"


def build(con) -> None:
    inv = ", ".join(f"i.{p}_qty, i.{p}_amt" for p in INVESTORS)
    sql = f"""
    WITH base AS (
      SELECT date::DATE AS date, substr(code, 2) AS code, code AS fg_code, name,
             open, high, low, close, volume, value, mktcap, shares, ret_pct, float_pct, foreign_pct,
             ln(1 + ret_pct / 100) AS ret_log,
             volume = 0 AS halted,
             min(date::DATE) OVER (PARTITION BY code) AS first_date
      FROM hyfin.raw.f1_stock_daily
    ),
    earn AS (
      SELECT code AS fg_code, fiscal_year, op_income,
             coalesce(prelim_date, report_date)::DATE AS avail
      FROM hyfin.raw.f5_earnings_annual WHERE coalesce(prelim_date, report_date) IS NOT NULL
    ),
    fin AS (
      SELECT f.code AS fg_code, f.fiscal_year, f.assets, f.sales, f.debt_ratio_pct,
             coalesce(e.report_date::DATE, make_date(f.fiscal_year::INTEGER + 1, 3, 31)) AS avail
      FROM hyfin.raw.f6_fin_annual f
      LEFT JOIN hyfin.raw.f5_earnings_annual e USING (code, fiscal_year)
    ),
    s1 AS (
      SELECT b.*, e.op_income AS op_income_last, e.fiscal_year::INTEGER AS op_income_fy
      FROM base b ASOF LEFT JOIN earn e ON b.fg_code = e.fg_code AND b.date > e.avail
    ),
    s2 AS (
      SELECT s1.*, f.assets AS assets_last, f.sales AS sales_last, f.debt_ratio_pct AS debt_ratio_pct_last,
             f.fiscal_year::INTEGER AS fin_fy
      FROM s1 ASOF LEFT JOIN fin f ON s1.fg_code = f.fg_code AND s1.date > f.avail
    ),
    fut AS (
      SELECT date::DATE AS date, {FUT_MAP} AS fg_code,
             close AS fut_close, settle AS fut_settle, spot AS fut_spot,
             volume_all AS fut_volume, oi_all AS fut_oi, roll_day AS fut_roll_day,
             (close / spot - 1) * 100 AS fut_basis_pct
      FROM hyfin.raw.f12_futures_front
    ),
    opt AS (
      SELECT date::DATE AS date, {FUT_MAP} AS fg_code,
             iv_atm AS opt_iv_atm, iv_call AS opt_iv_call, iv_put AS opt_iv_put, dte::INTEGER AS opt_dte,
             opt_volume, opt_oi
      FROM hyfin.raw.f13_option_atm_iv
    )
    SELECT s2.*, u."group" AS grp,
           {inv},
           c.credit_qty, c.credit_amt, c.short_qty, c.short_amt, c.short_bal_qty, c.short_bal_amt, c.lend_bal_qty,
           v.pbr, v.per,
           k.op_income_cons_fy2026,
           bt.beta AS beta_prev,
           fut.* EXCLUDE (date, fg_code),
           opt.* EXCLUDE (date, fg_code)
    FROM s2
    LEFT JOIN hyfin.raw.f2_universe u ON u.code = s2.fg_code
    LEFT JOIN hyfin.raw.f3_stock_investor i ON i.code = s2.fg_code AND i.date::DATE = s2.date
    LEFT JOIN hyfin.raw.f4_stock_credit_short c ON c.code = s2.fg_code AND c.date::DATE = s2.date
    LEFT JOIN hyfin.raw.f6_valuation_daily v ON v.code = s2.fg_code AND v.date::DATE = s2.date
    LEFT JOIN hyfin.raw.f5_consensus_daily k ON k.code = s2.fg_code AND k.date::DATE = s2.date
    LEFT JOIN hyfin.raw.f1_beta_annual bt ON bt.code = s2.fg_code AND bt.year = year(s2.date) - 1
    LEFT JOIN fut ON fut.fg_code = s2.fg_code AND fut.date = s2.date
    LEFT JOIN opt ON opt.fg_code = s2.fg_code AND opt.date = s2.date
    ORDER BY code, date
    """
    save(con, "stock_daily", sql,
         "raw.f1~f6, raw.f12_futures_front, raw.f13_option_atm_iv", __file__)


if __name__ == "__main__":
    main(build)
