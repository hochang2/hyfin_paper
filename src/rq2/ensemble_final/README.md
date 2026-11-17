# ensemble_final — RQ2 최종 추가 실험 (앙상블 최적화 · 표본 밖 검증)

기존 캐시 예측(ml_extension, ml_followup)만 재사용한다. **모형 재학습·Optuna·FM fine-tuning·global learning 없음.**
결과와 해석은 [results/REPORT.md](results/REPORT.md)에 있다.

## 실행

```bash
cd /f/hyfin_paper
PYTHONPATH=. py -3.13 -m src.rq2.ensemble_final.run_all          # 캐시가 있으면 재사용 (약 5분)
PYTHONPATH=. py -3.13 -m src.rq2.ensemble_final.run_all --force  # 온라인 앙상블 캐시 재계산
```

주의: `run_all`(또는 `exp4_oos`)을 다시 실행하면 `results/selection_frozen.json`의 `frozen_at`이 새로 기록된다.
선정 규칙은 결정적이므로 선택 결과는 바뀌지 않는다.

## 모듈

| 파일 | 역할 |
|---|---|
| `config.yaml` | 사전 등록 설정 (모형 pool, 창, 온라인 규칙, OOS 구간, 감사 설정). 결과를 보기 전에 고정 |
| `loader.py` | 기존 캐시 → 창평균 넓은 표 (`wide(ds, window)`), 사전 관측 국면 변수 (`regime_frame`) |
| `strategies.py` | 25개 결합 전략 (A 성능, B 다양성, C 제약 stacking, D 혼합) 및 온라인 학습 (`online`, 범위 global/code/hier/glocal/regime) |
| `online_run.py` | 자료별 온라인 앙상블 예측 생성 및 캐시 |
| `evaluate.py` | 지표, i0 블록 bootstrap, DM-HAC, White reality check, 가중치 안정성 |
| `exp1_diversity.py` | Exp1 개별 성능 · 오차 다양성 |
| `exp2_3_tables.py` | Exp2 선택적 앙상블 · Exp3 종목/국면 적응 (선택 편향 미보정 서술표) |
| `exp4_oos.py` | Exp4 선정/최종 구간 분리, 전이 검증, reality check |
| `exp5_inference.py` | Exp5 시간 placebo FPR, 검정력, MDE (선택 단계 포함, cross-fitted 대 in-sample 귀무) |
| `exp6_real.py` | Exp6 실제 규제일 효과 (탐색적 강건성) |
| `figures.py` | 공통 팔레트와 그림 헬퍼 |
| `run_all.py` | 전체 실행 및 실험 로그 (`results/experiment_log.csv`, `.md`) 작성 |

## 캐시

- 입력 (읽기 전용): `data/interim/rq_cache/ml_ext/`, `data/interim/rq_cache/ml_followup/`
- 출력: `data/interim/rq_cache/ensemble_final/`
  - `online_<ds>.parquet`, `weights_<ds>.parquet`, `metapicks_<ds>.parquet`
  - `audit_gkd`, `audit_gk`

기존 `main_study`, `ml_extension`, `ml_followup` 결과는 수정하지 않는다.
