# RQ2 최종 민감도 분석 (sensitivity)

RQ2 의 마지막 실험이다. 기존 결론("시행 후 61일 동안 두 종목 변동성이 반사실보다 낮아졌다는 증거가 없다")이
어떤 가정에 기대고 있는지, 각 가정이 얼마나 틀려야 결론이 바뀌는지를 수치로 보인다.
유의한 결과를 찾는 실험이 아니다. 결론은 [results/REPORT.md](results/REPORT.md).

- 기존 RQ2 결과·코드(main_study, ml_extension, ml_followup, ensemble)는 읽기만 한다.
- 새 예측모형, ML 재학습, Optuna, FM 미세조정은 하지 않는다.
- 새로 적합하는 것은 주 분석 BSTS·통계모형을 같은 설정과 seed 로 다시 돌리는 섭동 검증뿐이다.
  - 통제 오염 주입, 통제 하나씩 빼기(LOCO), 구조 주입이 여기에 해당한다.
  - 재적합 결과는 `data/interim/rq_cache/sensitivity/fits` 에 캐시한다.

## 실험

| ID | 파일 | 내용 |
|---|---|---|
| RQ2-S1-* | `bias_sensitivity.py` | b ∈ [−0.2, 0.2] 격자, 경험적 bias 원천 8종, 2022+ 별도 기간 보정 검증, 부호·−5/−10/−20/−30%·유의성 tipping point, 음성통제(005935) 보정 |
| RQ2-S2-* | `contamination.py` | 통제 오염 s = 0~100%. 시나리오는 A(donor 평균만), A2(donor 전체+PCA 재투영), B(KOSDAQ), C(결합), D(상관 상위 30%), E(factor β), F(005935 오염, 차분). 6개 모형의 반응계수 A와 실제 주입 재적합 검증 |
| RQ2-S3-* | `confounding.py` | 같은 기업 차분의 차별 충격 δ tipping point(역사 분포·사전추세·시장민감도와 비교). BSTS 숨은 교란 4가지 모양, LOCO 기준 |
| RQ2-S4-* | `event_window.py` | 사건일 7개, 시행 후 5/10/20/30일, 전·후반, 07-28 제외, donut, 대안 시행일, 사건 사이 구간 MDE. Holm·BH·max-T 보정. **사전 등록 아님** |
| RQ2-S5-* | `joint_sensitivity.py` | b × s × δ 제한 격자, 가산식과 BSTS 구조 주입 비교, frontier |
| RQ2-S6-DIAG | `diagnostics.py` | 검정력·MDE, donor 교환가능성, 실행 간 차이, 근사 검증 |

## 실행

```
py -3.13 -m src.rq2.phase1.sensitivity.run_all                # 전체 (캐시 있으면 1분 내외, 없으면 BSTS 재적합 ~150회)
py -3.13 -m src.rq2.phase1.sensitivity.run_all --only s3,s5    # 일부
```

순서: s1 → s2 → s3 → s4 → s5(s1·s2 결과표 사용) → s6(s2·s5 결과표 사용).
설정은 [config.yaml](config.yaml) 에 있다. 격자는 결과를 보기 전에 정했고, 값을 바꾸면 새 실험으로 기록한다.

## 결과 위치

- `results/REPORT.md`: 최종 보고서 (10절)
- `results/tables/s{1..6}_*.csv`: 결과표
- `results/figures/s{1..5}_*.png`: 그림
- `results/experiment_log.{csv,md}`: 실험 로그 (ID 25개)
