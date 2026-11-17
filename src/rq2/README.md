# RQ2

- **담당:** 연구자 A
- **질문:** 규제 이후 실제 RV 가 규제가 없었을 때의 반사실적 RV 보다 낮았는가?
- **기준모형 (baseline):** BSTS / CausalImpact
- **확장 실험 (extension):** Synthetic Control, ML 반사실 예측(LGBM·XGB·RF·ENet), Chronos-2·TimesFM
- **주의:** placebo, 사전기간 적합도, 대조군 민감도, 발표일 vs 시행일 구분. 사후 실제값 학습 누출 금지

## 상태

- **Phase 1 (2026-10 자료, 종료 2026-11-17)**: 종합 보고서는 [phase1/REPORT.md](phase1/REPORT.md).
  - 여러 반사실 상황(S1~S13)을 함께 읽었다.
  - 결론: 61일 동안 유의한 변동성 감소는 없다. 이 판단은 모든 상황과 현실적인 가정 위반에서 유지된다.
  - 다만 −10~−20% 감소는 배제되지 않는다. 부호는 시장 냉각을 반사실에 얼마나 반영하느냐에 따라 갈린다.

  | 하위 연구 | 내용 |
  |---|---|
  | `phase1/main_study` | 사전 지정 주 분석 (BSTS, 같은 기업 차분, placebo, 강건성) |
  | `phase1/ml_extension` | ML·FM 반사실 이득 검증. 이득 없음 |
  | `phase1/ml_followup` | 정확도·추론 개선 요인. 추론 방식(n_eff)이 핵심 |
  | `phase1/ensemble` | 선택적 앙상블. 정확도 소폭 개선, 추론 이득 없음 |
  | `phase1/sensitivity` | bias·오염·교란·창·결합 민감도 |

- **Phase 2 (2027-02 예정)**: 사후 표본을 연장해 같은 반사실 상황 세트를 다시 돌린다. ML 효과 검증은 하지 않는다.
  지시사항은 [phase2/INSTRUCTIONS.md](phase2/INSTRUCTIONS.md) 에 있다. 자료를 보기 전에 사전 지정한 것이다.

## 구조

```
src/rq2/
  core/        공용 라이브러리 (Phase 1·2 공통, 기존 함수 수정 금지)
    data.py      패널·통제변수 로드 (hyfin.proc, 캐시 data/interim/rq_cache/)
    bsts.py      CausalImpact 방식 BSTS (Gibbs)
    models.py    OLS·ridge·enet·SARIMAX·UC·HAR-X·LGBM·XGB·SCM 반사실
    pipeline.py  설정(configs/rq2.yaml)·사양 실행·Holm
    explog.py    실험 로그 (csv + md)
  tests/       BSTS 구현 검증 (모의자료)
  phase1/      Phase 1. 결과 동결, 읽기만 한다
    REPORT.md    종합 보고서
    main_study/  ml_extension/  ml_followup/  ensemble/  sensitivity/
      (각 폴더) README.md, run_all.py, *.py, results/{REPORT.md, tables/, figures/, experiment_log.*}
  phase2/      Phase 2 (phase1 을 import 하지 않고 core 만 쓴다)
    INSTRUCTIONS.md
```

## 실행 (Phase 1 재현)

```
py -m src.rq2.phase1.main_study.run_all                          # 주 분석 전 단계 (설정: configs/rq2.yaml)
py -m src.rq2.phase1.main_study.run_all --stages primary,figures # 일부 단계만
py -m src.rq2.phase1.main_study.synthesis                        # 종합 표 (재추정 없음)
py -m src.rq2.phase1.ml_extension.run_all                        # ML 확장
py -3.13 -m src.rq2.phase1.ml_followup.run_all
py -3.13 -m src.rq2.phase1.ensemble.run_all
py -3.13 -m src.rq2.phase1.sensitivity.run_all
py -m pytest src/rq2/tests                                       # BSTS 구현 검증
```

- Foundation model(Chronos-2, TimesFM 2.5)은 별도 가상환경 `.venv-fm` 에서 돌린다.
- 순서는 각 파일 머리말에 있다: `phase1/main_study/foundation.py`, `phase1/ml_extension/foundation_bridge.py`
- 캐시 폴더 이름(`data/interim/rq_cache/{ml_ext, ml_followup, ensemble, sensitivity}`)은 폴더 이동과 무관하게 그대로다.
