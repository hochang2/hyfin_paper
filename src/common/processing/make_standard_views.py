"""raw 를 형식만 맞춘 뷰 → proc.futures_all, options_all, minute_kr, minute_idx, minute_us, etf_aum_all

가공할 것은 없지만 분석 코드가 raw 를 직접 읽지 않도록 proc 이름으로 열어 둔다.
뷰는 저장 공간을 쓰지 않고 조회할 때마다 raw 를 읽는다 → raw 를 지우면 뷰도 깨진다.

바꾸는 것 (값은 그대로)
- date: DATE (분봉 'YYYYMMDD', F 의 TIMESTAMP → DATE). 분봉 시각 열(ts, ts_et)은 그대로.
- code: 접두사 없는 6자리. FnGuide 원래 코드는 fg_code. 선물·옵션은 기초자산 code(005930/000660) 추가.
- 단위: F10 백만원 → aum_won (원). 나머지는 raw 와 같다.
- raw 의 `file` 열(적재 파일 이름)은 뺀다.
"""
from __future__ import annotations

from src.common.processing import main, save_view

UNDERLYING = "CASE underlying WHEN '삼성전자' THEN '005930' WHEN 'SK하이닉스' THEN '000660' END"


def build(con) -> None:
    views = {
        "futures_all": (f"""
            SELECT date::DATE AS date, {UNDERLYING} AS code, underlying, expiry_ym, contract, contract_name,
                   open, high, low, close, change, settle, spot, volume, value, oi
            FROM hyfin.raw.f12_futures_all""", "raw.f12_futures_all"),
        "options_all": (f"""
            SELECT date::DATE AS date, {UNDERLYING} AS code, underlying, expiry_ym, cp, strike, contract, contract_name,
                   close, iv, volume, value, oi, next_base_price
            FROM hyfin.raw.f13_option_all""", "raw.f13_option_all"),
        "minute_kr": ("""
            SELECT strptime(date, '%Y%m%d')::DATE AS date, code, venue, freq, price, session, ts, hhmm,
                   open, high, low, close, volume, acc_volume
            FROM hyfin.raw.minute_kr""", "raw.minute_kr"),
        "minute_idx": ("""
            SELECT strptime(date, '%Y%m%d')::DATE AS date, 'IDX' || code AS code, freq, session, ts, hhmm,
                   open, high, low, close, volume, acc_volume
            FROM hyfin.raw.minute_idx""", "raw.minute_idx"),
        "minute_us": ("""
            SELECT strptime(bus_dt, '%Y%m%d')::DATE AS date, ticker, freq, session, ts_et, hhmm_raw,
                   open, high, low, close, volume, adj_flag
            FROM hyfin.raw.minute_us""", "raw.minute_us"),
        "etf_aum_all": ("""
            SELECT date::DATE AS date, substr(code, 2) AS code, code AS fg_code, name,
                   aum_mil * 1e6 AS aum_won, shares
            FROM hyfin.raw.f10_etf_aum""", "raw.f10_etf_aum"),
    }
    for name, (sql, src) in views.items():
        save_view(con, name, sql, src, __file__)


if __name__ == "__main__":
    main(build)
