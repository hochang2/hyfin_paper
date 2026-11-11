# data/

로컬 보관용 폴더. 내용은 라이선스(키움 OpenAPI, FnGuide DataGuide 이용약관) 문제로 **Git에 올리지 않는다** (이 README와 `metadata/`만 커밋).
분석은 MotherDuck `hyfin` DB에 올린 사본으로 한다 → [../docs/database_guide.md](../docs/database_guide.md)

| 경로 | 내용 |
|---|---|
| `F1~F6.zip` | DataGuide 원본 xlsx (불변) |
| `F7_F17.zip` | F7~F17 정리본 원본 (불변) |
| `minute_2610.zip` | 키움 분봉 509개 (불변, **유일본 — 별도 백업 필수**) |
| `interim/fnguide/` | F1~F17 정리본 (csv + parquet) |
| `interim/minute/` | 분봉 zip을 푼 parquet |
| `metadata/` | 스키마, 카탈로그, 검증 리포트 (비민감) |

- 원본 zip은 읽기만 한다. 정리본을 다시 만드는 방법은 [../docs/data_spec.md](../docs/data_spec.md) 0절.
- 정리본을 바꾸면 `py scripts/load_motherduck.py`로 DB에 다시 올린다.
