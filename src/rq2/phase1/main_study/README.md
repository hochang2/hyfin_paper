# RQ2 1차 연구 (main_study)

질문: **규제 이후 실제 RV 가 규제가 없었을 때의 반사실적 RV 보다 낮았는가?**
사전 지정 주 분석은 BSTS(CausalImpact 방식)이고, 대안 모형·진단·placebo·강건성은 그 결론을 점검한다.

결과와 해석: [results/REPORT.md](results/REPORT.md)

## 원칙

- 공용 코드는 `src.rq2.core` 에 있다. 이 폴더는 실행·진단·그림만 담는다.
- 결과는 동결한다. 다시 돌리면 결과 폴더를 새 커밋으로 남긴다.
- 사후 실제값은 어떤 모형의 학습에도 들어가지 않는다.
- 설정은 `configs/rq2.yaml`, 실험 기록은 `results/experiment_log.*`, 환경은 `results/environment.json` 에 남긴다.

## 파일

| 파일 | 내용 |
|---|---|
| `run_all.py` | 단계 실행 (`--stages primary,models,withinfirm,diagnostics,placebo_time,placebo_space,longhist,robustness,figures`) |
| `synthesis.py` | 결과 csv 를 모아 보고서용 종합 표 (재추정 없음) |
| `diagnostics.py` | 사전구간 회귀 진단 + 정상성 + 전체표본 ITS |
| `foundation.py` | Chronos-2·TimesFM 2.5 반사실 (`.venv-fm` 에서 실행, 머리말에 순서) |
| `plots.py` | 그림 (결과 표만 읽는다) |
| `results/` | `REPORT.md`, `tables/`, `figures/`, `experiment_log.{csv,md}`, `environment.json` |

캐시: `data/interim/rq_cache/` (Git 제외).

## 실행

```
py -m src.rq2.phase1.main_study.run_all                          # 전 단계
py -m src.rq2.phase1.main_study.run_all --stages primary,figures # 일부 단계만
py -m src.rq2.phase1.main_study.synthesis                        # 종합 표
```
