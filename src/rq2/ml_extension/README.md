# RQ2 ML 확장 하위실험

질문: **1차 연구의 반사실 모형에 ML(ENet·RF·LGBM·XGB)과 foundation model 을 넣어서 실제로 얻는 이익이 있었나?**
이 연구는 "ML 로 유의한 규제효과를 만들어내는 것"이 목적이 아니다.
예측 정확도(실험 1), 알려진 효과의 회복·검정력·위양성(실험 2), 튜닝(실험 3), 결합(실험 4)을 먼저 평가한다.
실제 효과(실험 5)는 가짜 개입일 결과만으로 **사전에 고른** 모형으로 추정한다.

결과와 해석: [results/REPORT.md](results/REPORT.md)

## 원칙

- 1차 연구 코드는 `src.rq2.core` 를 import 해서 그대로 쓴다. `main_study/results` 는 읽기만 한다.
- 5분봉 rv5 결과와 일별 GK 결과는 섞지 않는다. 표의 `dataset` 열로 구분한다.
- 통제변수 PCA·표준화, 튜닝, 앙상블 가중치는 모두 그 과제의 학습구간(가짜 개입일 전) 자료로만 정한다.
- 사후 y 는 어떤 모형의 입력에도 들어가지 않는다. 진단 1 의 불변성 검정으로 확인한다.
- seed·하이퍼파라미터·환경은 `config.yaml`, `results/experiment_log.*`, `results/environment.json` 에 남긴다.
- 실패한 적합도 지우지 않고 남긴다. 캐시는 `errors.csv`, 로그에는 `RQ2-ML-FAIL-*` 로 기록한다.

## 파일

| 파일 | 내용 |
|---|---|
| `config.yaml` | 데이터셋·가짜 개입일·창·모형 목록·주입효과·튜닝·앙상블·사전 지정 선정 규칙 |
| `data_pipeline.py` | 과제 `(dataset, code, i0)` → `(y, X, 학습, 사후)`, 국면 라벨, 주입효과 경로 |
| `models.py` | 모형 레지스트리 (core 재사용 + naive_mean·did_gap·RF·LGBM+HAR) |
| `benchmark.py` | 실험 1: rolling-origin 적합·캐시·지표·DM 검정·불변성 검정 |
| `foundation_bridge.py` | Chronos-2·TimesFM 을 `.venv-fm` 에서 같은 과제로 실행 |
| `synthetic_effect.py` | 실험 2: 효과 주입, native / time / space 추론, 검정력·MDE, 번짐·시차누출 |
| `tuning.py` | 실험 3: Optuna 중첩 시계열 CV (목적 A 예측, B 효과) |
| `ensemble.py` | 실험 4: 동일·역MSE·stacking·hybrid 앙상블, 오차 상관 |
| `real_effect.py` | 실험 5: 사전 지정 모형 선택 → 실제 효과 재추정 |
| `diagnostics.py` | 진단 1~12 |
| `plots.py` | 그림 |
| `run_all.py` | 단계 실행 (`--stages bench,fm,tune,ensemble,spill,leaky,ablation,diag_fit,tables,figures`) |

캐시: `data/interim/rq_cache/ml_ext/<dataset>/<model>_{daily,win}.parquet` (Git 제외).
완료된 과제는 다시 적합하지 않으므로, 중간에 멈춰도 같은 명령으로 이어서 돌릴 수 있다.

## 실행

```
py -m src.rq2.ml_extension.run_all                                   # 전부 (12코어 기준 수 시간)
py -m src.rq2.ml_extension.run_all --stages tables,figures           # 캐시에서 표·그림·로그만 다시
```

## 표본을 늘릴 때

`config.yaml` 에서 다음을 확인한다.
- `datasets.rv5.last_index_exclusive`(발표일 위치)와 `listing_index`: 표본 시작이 그대로면 바뀌지 않는다.
- `windows.post_effective`: 사후 길이가 61일보다 길어지면 끝값과 `max_h` 를 늘린다.
- `selection_rule` 은 그대로 둔다. 바꾸면 새 실험이다.
캐시 폴더를 지우고 다시 실행한다.
