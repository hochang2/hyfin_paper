# RQ4 1차 연구 (main_study)

결과와 해석: [results/REPORT.md](results/REPORT.md). 설정은 `../../config.yaml`, 공용 코드는 `../../core/`.

## 파일

| 단계 | 모듈 | 결과 |
|---|---|---|
| 공통 | `core/common.py` (설정·로그·표 저장), `core/data_pipeline.py` (DB→캐시), `core/panel.py` (주 패널), `core/estim.py` (FE·SE), `core/collapsed.py` (접힌 격차·위약) | – |
| 1 데이터 | `product_universe.py` | t01* |
| 2 기술통계 | `descriptive.py` | t02*, f01, f02 |
| 3 주 분석 | `did.py`, `event_study.py`, `ddd.py` | t03*, t04*, t05, f03 |
| 4 진단 | `diagnostics.py` | t06, f04 |
| 5 강건성 | `placebo.py`, `robustness.py` | t07*, t08*, f05, f06 |
| 6 민감도 | `sensitivity.py` (HonestDiD·Oster·Cinelli–Hazlett·오염·LODO·결합) | t09*, f07, f08 |
| 7 메커니즘 | `mechanism.py` | t10*, f09 |
| 8 종합 | `synthesis.py` (요약표·로그 병합·RQ2 연결) | t11, experiment_log.{csv,md} |

- 단계별 로그 조각: `results/log_parts/`. `synthesis.py` 가 `experiment_log.{csv,md}` 로 합친다.
