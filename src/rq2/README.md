# RQ2

- **담당:** 연구자 A
- **질문:** 규제 이후 실제 RV 가 규제가 없었을 때의 반사실적 RV 보다 낮았는가?
- **기준모형 (baseline):** BSTS / CausalImpact
- **확장 실험 (extension):** Synthetic Control, XGBoost 반사실 예측, TimesFM
- **주의:** placebo, 사전기간 적합도, 대조군 민감도, 발표일 vs 시행일 구분. 사후 실제값 학습 누출 금지

## 상태

미구현. 변수·표본기간이 확정되면 [PROJECT_CONTEXT.md](../../PROJECT_CONTEXT.md) 7절 결정 기록에 남기고 구현한다.

## 파일

- 분석 코드: 이 폴더의 `*.py` (공통 함수는 `src/common/`)
- 결과 그림·표: `results/`
- 테스트: `test_*.py`
