# RQ3

- **담당:** 연구자 B
- **질문:** ETF 거래·자금흐름·AUM·리밸런싱 정보가 HAR-RV 대비 미래 RV 예측력을 높이는가?
- **기준모형 (baseline):** HAR-RV vs HAR-RV+ETF (rolling/expanding OOS)
- **확장 실험 (extension):** Lasso/Elastic Net, RF, XGBoost, LightGBM
- **주의:** ETF 는 2026-05 이후만 존재, 예측시점 가용 변수만 사용, QLIKE 등 손실함수

## 상태

미구현. 변수·표본기간이 확정되면 [PROJECT_CONTEXT.md](../../PROJECT_CONTEXT.md) 7절 결정 기록에 남기고 구현한다.

## 파일

- 분석 코드: 이 폴더의 `*.py` (공통 함수는 `src/common/`)
- 결과 그림·표: `results/`
- 테스트: `test_*.py`
