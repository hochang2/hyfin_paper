"""한국 거래일 달력 + 사건 더미 → proc.calendar

- 거래일: F11 코스피(`I.001`)가 있는 날 (2015-01-02 ~).
- 분봉 기간(2025-10-01 ~)에는 코스피 1분봉으로 정규장 봉 수, 개장 시각, 시장 공백을 붙인다.
- 사건일·제도 기간은 configs/events.yaml. 날짜가 null 인 사건(규제 시행일)은 열이 전부 NULL.
- rel_<사건>: 사건일(휴장이면 그다음 거래일)을 0으로 한 거래일 차이. 이벤트 스터디용.
"""
from __future__ import annotations

from src.common.processing import load_config, main, save


def build(con) -> None:
    cfg = load_config("events")
    ev = {e["id"]: e["date"] for e in cfg["events"]}

    def post(name: str) -> str:
        d = ev.get(name)
        return f"date >= DATE '{d}'" if d else "NULL::BOOLEAN"

    def rel(name: str) -> str:
        d = ev.get(name)
        if not d:
            return "NULL::INTEGER"
        return f"(tday - (SELECT min(tday) FROM days WHERE date >= DATE '{d}'))::INTEGER"

    regimes = ",\n      ".join(
        f"date BETWEEN DATE '{r['start']}' AND DATE '{r['end']}' AS {r['id']}" for r in cfg.get("regimes", [])
    )

    period_cases = []
    if ev.get("regulation_announce"):
        period_cases.append(f"WHEN date >= DATE '{ev['regulation_announce']}' THEN 'post_announce'")
    if ev.get("listing"):
        period_cases.append(f"WHEN date >= DATE '{ev['listing']}' THEN 'listing_to_announce'")
    period = "CASE " + " ".join(period_cases) + " ELSE 'pre_listing' END"

    sql = f"""
    WITH days AS (
      SELECT date, row_number() OVER (ORDER BY date) AS tday
      FROM (SELECT DISTINCT date::DATE AS date FROM hyfin.raw.f11_index WHERE code = 'I.001')
    ),
    m AS (
      SELECT strptime(date, '%Y%m%d')::DATE AS date,
             count(*) FILTER (session = 'regular')::INTEGER AS reg_bars,
             min(hhmm) FILTER (session = 'regular')::INTEGER AS open_hhmm,
             max(hhmm) FILTER (session = 'regular')::INTEGER AS last_hhmm
      FROM hyfin.raw.minute_idx WHERE code = '001'
      GROUP BY 1
    ),
    g AS (
      SELECT strptime(date, '%Y%m%d')::DATE AS date, CAST(missing_min AS INTEGER) AS gap_min, gaps AS gap_range
      FROM hyfin.raw.minute_market_gaps
    )
    SELECT
      date,
      year(date)::INTEGER AS year,
      m.date IS NOT NULL AS has_minute,
      m.reg_bars, m.open_hhmm, m.last_hhmm,
      CASE WHEN m.date IS NOT NULL THEN m.open_hhmm > 900 END AS late_open,
      CASE WHEN m.date IS NOT NULL THEN coalesce(g.gap_min, 0) END AS gap_min,
      g.gap_range,
      {post('listing')} AS post_listing,
      {post('regulation_announce')} AS post_announce,
      {post('regulation_effective')} AS post_effective,
      {period} AS period,
      {rel('listing')} AS rel_listing,
      {rel('regulation_announce')} AS rel_announce,
      {rel('regulation_effective')} AS rel_effective,
      {regimes}
    FROM days
    LEFT JOIN m USING (date)
    LEFT JOIN g USING (date)
    ORDER BY date
    """
    save(con, "calendar", sql, "raw.f11_index, raw.minute_idx, raw.minute_market_gaps, configs/events.yaml", __file__)


if __name__ == "__main__":
    main(build)
