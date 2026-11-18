# RQ3 1차 연구 (main_study)

질문: **HAR-RV 와 당일 수익률 충격을 고려한 뒤에도, 단일종목 레버리지 ETF 의 이론적 리밸런싱 압력이 다음 날 실현변동성 예측을 개선하는가?**
사전 지정 주 검정은 C(HAR + 상장 더미 + 수익률 충격 + 리밸런싱 압력) vs B(같은 모형에서 리밸런싱 압력 제외), Clark-West + Holm 이다 (`../../config.yaml`).

결과와 해석: [results/REPORT.md](results/REPORT.md)

## 원칙

- 데이터 로드·캐시·실험 로그는 `src.rq2.core` 를 수정 없이 쓴다.
- 원점 t 의 학습은 다음 날 값이 이미 실현된 행(index < t)만 쓴다.
- 결과는 동결한다. 다시 돌리면 결과 폴더를 새 커밋으로 남긴다.
- 사전 지정 밖 분석은 실험 로그에 탐색적으로 표시하고, 주 결론을 바꾸지 않는다.

## 파일

| 파일 | 내용 |
|---|---|
| `run_all.py` | 단계 실행 (`--stages data,primary,judgment,robustness,exploratory`) |
| `../../core/data.py` | 상품 18개 합산, 규모 비율 S, 설계 행렬(HAR·D·R·X), 보조 ETF 변수 |
| `../../core/forecast.py` | 확장/이동 창 표본외 예측, Clark-West, DM, QLIKE, 블록 셔플, NW OLS, VIF, Holm |
| `exploratory.py` | ML (Lasso, ENet, RF, XGB, LGBM) 표본외 예측 |
| `plots.py` | 그림 |
| `results/` | `REPORT.md`, `tables/`, `figures/`, `experiment_log.{csv,md}`, `environment.json` |

테스트: `.venv/Scripts/python -m pytest src/rq3/tests` (데이터 점검 11개, 예측·검정 모의자료 7개).
