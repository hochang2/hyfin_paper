# RQ2

- **담당:** 연구자 A
- **질문:** 규제 이후 실제 RV 가 규제가 없었을 때의 반사실적 RV 보다 낮았는가?
- **기준모형 (baseline):** BSTS / CausalImpact
- **확장 실험 (extension):** Synthetic Control, XGBoost 반사실 예측, TimesFM
- **주의:** placebo, 사전기간 적합도, 대조군 민감도, 발표일 vs 시행일 구분. 사후 실제값 학습 누출 금지

## 상태

분석 완료 (2026-11-16). 결론과 근거는 [REPORT.md](REPORT.md) 에 있다.
요약: 시행 후 61일 동안 두 종목의 변동성이 반사실보다 낮아졌다는 증거가 없다. 다만 SK하이닉스는 검정력이 부족하다.

## 실행

```
py -m src.rq2.run_all                          # 전 단계 (설정: configs/rq2.yaml)
py -m src.rq2.run_all --stages primary,figures # 일부 단계만
py -m src.rq2.synthesis                        # 종합 표 (재추정 없음)
py -m pytest src/rq2/test_bsts.py              # BSTS 구현 검증
```

Foundation model(Chronos-2, TimesFM 2.5)은 별도 가상환경 `.venv-fm` 에서 돌린다. 순서는 `foundation.py` 머리말에 있다.

## 파일

| 파일 | 내용 |
|---|---|
| `data.py` | 패널·통제변수 로드 (`hyfin.proc`, 캐시 `data/interim/rq_cache/`) |
| `bsts.py` | CausalImpact 방식 BSTS (Gibbs) |
| `models.py` | OLS·ridge·enet·SARIMAX·UC·HAR-X·LGBM·XGB·SCM 반사실 |
| `diagnostics.py` | 회귀 가정·정상성·Oster/Cinelli-Hazlett 진단 |
| `foundation.py` | Chronos-2 / TimesFM 반사실 |
| `pipeline.py`, `run_all.py` | 단계별 실행, 실험 로그 기록 |
| `synthesis.py` | 보고서용 종합 표 |
| `plots.py` | 그림 |
| `results/` | 표·그림·`experiment_log.{csv,md}`·`environment.json` |
