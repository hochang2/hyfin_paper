# RQ3

- **담당:** 연구자 B
- **질문:** ETF 거래·자금흐름·AUM·리밸런싱 정보가 HAR-RV 대비 미래 RV 예측력을 높이는가?
- **기준모형 (baseline):** HAR-RV vs HAR-RV+ETF (rolling/expanding OOS)
- **확장 실험 (extension):** Lasso/Elastic Net, RF, XGBoost, LightGBM
- **주의:** ETF 는 2026-05 이후만 존재, 예측시점 가용 변수만 사용, QLIKE 등 손실함수

## 상태

1차 연구 완료 (2026-11-18). 사전 지정 주 검정 결과 두 종목 모두 '리밸런싱 압력의 추가 예측력 증거 없음'. 수익률 충격(+14%)이 주된 예측 정보이고, 개인 순매수는 탐색적 신호로 확증 평가 2차 가설에 등록했다.

- 설정·사전 지정: [config.yaml](config.yaml) (커밋 `rq3 사전 지정`)
- 진행 기록: [PROGRESS.md](PROGRESS.md)
- 보고서: [main_study/results/REPORT.md](main_study/results/REPORT.md)
- 남은 일: 11월 이후 자료로 확증 평가, PROJECT_CONTEXT 7절 결정 기록은 A 와 조율

## 파일

- 분석 코드: `main_study/*.py` (원자료 로더는 `src/rq2/core/` 재사용)
- 결과 그림·표: `main_study/results/`
- 테스트: `tests/test_*.py`
