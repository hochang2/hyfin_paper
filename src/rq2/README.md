# RQ2

- **담당:** 연구자 A
- **질문:** 규제 이후 실제 RV 가 규제가 없었을 때의 반사실적 RV 보다 낮았는가?
- **기준모형 (baseline):** BSTS / CausalImpact
- **확장 실험 (extension):** Synthetic Control, ML 반사실 예측(LGBM·XGB·RF·ENet), Chronos-2·TimesFM
- **주의:** placebo, 사전기간 적합도, 대조군 민감도, 발표일 vs 시행일 구분. 사후 실제값 학습 누출 금지

## 상태

- **1차 연구 (main_study)**: 분석 완료 (2026-11-16). 결론과 근거는 [main_study/results/REPORT.md](main_study/results/REPORT.md).
  요약: 시행 후 61일 동안 두 종목의 변동성이 반사실보다 낮아졌다는 증거가 없다. 다만 SK하이닉스는 검정력이 부족하다.
- **ML 확장 하위실험 (ml_extension)**: "ML 을 넣어서 실제로 얻는 이익이 있었나"를 따로 검증한다.
  1차 연구 결과는 건드리지 않는다. 보고서는 [ml_extension/results/REPORT.md](ml_extension/results/REPORT.md).

## 구조

```
src/rq2/
  core/           공용 라이브러리 (두 연구가 같이 쓴다)
    data.py         패널·통제변수 로드 (hyfin.proc, 캐시 data/interim/rq_cache/)
    bsts.py         CausalImpact 방식 BSTS (Gibbs)
    models.py       OLS·ridge·enet·SARIMAX·UC·HAR-X·LGBM·XGB·SCM 반사실
    pipeline.py     설정(configs/rq2.yaml)·사양 실행·Holm
    explog.py       실험 로그 (csv + md)
  main_study/     1차 연구 (주 분석·진단·강건성·FM) — 결과 동결
  ml_extension/   ML 확장 하위실험 (실험 1~5 + 진단 1~12)
    (두 폴더 공통) README.md, run_all.py, *.py, results/{REPORT.md, tables/, figures/, experiment_log.*, environment.json}
  tests/          BSTS 구현 검증 (모의자료)
```

## 실행

```
py -m src.rq2.main_study.run_all                          # 1차 연구 전 단계 (설정: configs/rq2.yaml)
py -m src.rq2.main_study.run_all --stages primary,figures # 일부 단계만
py -m src.rq2.main_study.synthesis                        # 종합 표 (재추정 없음)
py -m src.rq2.ml_extension.run_all                        # ML 확장 전 단계 (설정: ml_extension/config.yaml)
py -m pytest src/rq2/tests                                # BSTS 구현 검증
```

Foundation model(Chronos-2, TimesFM 2.5)은 별도 가상환경 `.venv-fm` 에서 돌린다.
1차 연구는 `main_study/foundation.py`, 확장 실험은 `ml_extension/foundation_bridge.py` 머리말에 순서가 있다
(`ml_extension/run_all.py` 의 fm 단계는 이를 자동으로 호출한다).

## 데이터를 늘릴 때 (예: 2027-01 까지 재수집)

1. `configs/rq2.yaml` 의 `sample.end` 를 늘리고 1차 연구를 다시 돌린다 (결과 폴더는 새 커밋으로 남긴다).
2. `ml_extension/config.yaml` 은 위치(index) 기반 값(`last_index_exclusive`, `listing_index`)을 확인한다
   (표본 시작이 같으면 그대로). 캐시 `data/interim/rq_cache/ml_ext/` 를 지우고 `run_all` 을 다시 돌린다.
3. 사전 지정 선정 규칙(`selection_rule`)은 바꾸지 않는다 — 바꾸면 새 실험으로 기록한다.
