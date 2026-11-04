# RQ3

- **담당:** 연구자 B
- **질문:** ETF 거래·자금흐름·AUM·리밸런싱 정보가 HAR-RV 대비 미래 RV 예측력을 높이는가?
- **기준모형 (baseline):** HAR-RV vs HAR-RV+ETF (rolling/expanding OOS)
- **확장 실험 (extension):** Lasso/Elastic Net, RF, XGBoost, LightGBM
- **주의:** ETF 는 2026-05 이후만 존재, 예측시점 가용 변수만 사용, QLIKE 등 손실함수

## 상태

미구현. 변수·표본기간 확정 후 [docs/decisions.md](../../../docs/decisions.md) 에 기록하고 구현한다.
