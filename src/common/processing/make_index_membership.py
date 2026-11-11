"""KOSPI200 구성종목 스냅샷 → 일별 편입 여부 → proc.kospi200_membership (거래일 × 종목)

원천 F17
- members: 기준일 5개 (2025-06-13, 2025-10-01, 2025-12-12, 2026-06-12, 2026-10-30)
- changes: 기준일 사이의 편입·편출 27건

방법: 거래일 d 의 구성종목 = 기준일 <= d 인 가장 최근 스냅샷 (2025-06-13 이전은 행 없음).
- 정기변경 적용일(2025-12-12, 2026-06-12)이 스냅샷 날짜와 같아서 정기변경은 정확하다.
- **수시변경은 다음 스냅샷 날부터 반영된다.** 예) 한화머시너리앤서비스홀딩스(0220W0)는 2026-06-12 ~ 10-30 사이에 편입됐는데,
  실제 편입일을 몰라 2026-10-30 부터 TRUE 다. 이 종목을 쓸 때는 실제 편입일을 확인할 것.
- 대상 종목: 한 번이라도 스냅샷에 나온 종목. 그 기간 전체 거래일에 행을 만든다 (편입 전·편출 후는 FALSE).

열: date, code, name, in_k200, snapshot_date (판단에 쓴 스냅샷), mktcap_snapshot_won (그 스냅샷의 상장시총, 백만원 → 원),
    change (그 스냅샷에서 새로 편입 '편입' / 빠짐 '편출' 이면 해당 거래일 구간 전체에 표시, 아니면 NULL)
"""
from __future__ import annotations

from src.common.processing import main, save


def build(con) -> None:
    sql = """
    WITH snap AS (
      SELECT DISTINCT 기준일::DATE AS snapshot_date FROM hyfin.raw.f17_kospi200_members
    ),
    days AS (
      SELECT c.date, s.snapshot_date
      FROM (SELECT date FROM hyfin.proc.calendar WHERE date >= (SELECT min(snapshot_date) FROM snap)) c
      ASOF JOIN snap s ON c.date >= s.snapshot_date
    ),
    codes AS (
      SELECT 종목코드 AS code, arg_max(종목명, 기준일) AS name FROM hyfin.raw.f17_kospi200_members GROUP BY 1
      UNION
      SELECT 종목코드, arg_max(종목명, 이후기준일) FROM hyfin.raw.f17_kospi200_changes
      WHERE 종목코드 NOT IN (SELECT 종목코드 FROM hyfin.raw.f17_kospi200_members) GROUP BY 1
    ),
    m AS (
      SELECT 기준일::DATE AS snapshot_date, 종목코드 AS code, 상장시가총액 * 1e6 AS mktcap_won
      FROM hyfin.raw.f17_kospi200_members
    ),
    ch AS (
      SELECT 이후기준일::DATE AS snapshot_date, 종목코드 AS code, 구분 AS change
      FROM hyfin.raw.f17_kospi200_changes
    )
    SELECT d.date, c.code, c.name,
           m.code IS NOT NULL AS in_k200,
           d.snapshot_date,
           m.mktcap_won AS mktcap_snapshot_won,
           ch.change
    FROM days d CROSS JOIN codes c
    LEFT JOIN m ON m.snapshot_date = d.snapshot_date AND m.code = c.code
    LEFT JOIN ch ON ch.snapshot_date = d.snapshot_date AND ch.code = c.code
    ORDER BY c.code, d.date
    """
    save(con, "kospi200_membership", sql, "raw.f17_kospi200_members, raw.f17_kospi200_changes, proc.calendar", __file__)


if __name__ == "__main__":
    main(build)
