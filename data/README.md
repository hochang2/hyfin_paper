# data/

원자료는 라이선스(키움 OpenAPI, FnGuide DataGuide 이용약관) 문제로 **Git에 올리지 않는다**.
각 연구자는 아래 상대경로 규칙에 맞춰 로컬에 배치한다.

| 경로 | 내용 | Git |
|---|---|---|
| `raw/kiwoom/` | 키움 API 응답 원본 (불변) | 제외 |
| `raw/fnguide/` | DataGuide 추출 원본 (불변) | 제외 |
| `external/` | KRX·금융위·SEIBro 등 외부 원자료 | 제외 |
| `interim/` | 정제·병합 중간산출물 | 제외 |
| `processed/` | 분석용 패널 | 제외 |
| `metadata/` | 스키마, 카탈로그, 검증 리포트 (비민감) | 포함 |

규칙
- raw 는 덮어쓰지 않는다. `source`, `retrieved_at`, `asof`, 파일 해시를 함께 기록.
- 보유 현황은 [../docs/data_inventory.md](../docs/data_inventory.md) 에서 관리.
