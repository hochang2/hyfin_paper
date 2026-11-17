# RQ2 ML 후속 실험 (ml_followup)

RQ2 주 분석(`src/rq2/phase1/main_study`)과 1차 ML 확장(`src/rq2/phase1/ml_extension`)에 이은 별도 후속 연구. 앞의 두 결과는 읽기만 한다.
결과 요약은 [results/REPORT.md](results/REPORT.md).

## 구성
| 파일 | 내용 |
|---|---|
| `config.yaml` | 모든 설정·seed (값 변경 = 새 실험) |
| `panel.py`, `common.py` | 자료 패널, 창·지표·추론(neff/strict, cluster CI) 공통 함수, 실행 로그 |
| `exp5_audit.py` | Exp5 추론 감사 (우선순위 1) |
| `exp1_global.py` | Exp1 Global/Pooled 학습 |
| `exp4_sample.py` | Exp4 표본 효율 (Q1~Q5) |
| `exp6_dgp.py` | Exp6 DGP 시뮬레이션·효과 복원 |
| `exp2_transfer.py` | Exp2 GK→rv5 전이 |
| `exp7_donor.py` | Exp7 donor pool 민감도 |
| `exp3_fm.py` | Exp3 Foundation model (Chronos-2, TimesFM 2.5) |
| `figures.py` | 결과 표 → `results/figures/*.png` (적합 불필요) |
| `run_all.py` | 단계 실행기 |

## 실행
```
py -3.13 -m src.rq2.phase1.ml_followup.run_all                 # 전체 (캐시가 있으면 적합 건너뜀)
py -3.13 -m src.rq2.phase1.ml_followup.run_all exp4_gk analysis  # 단계 지정
.venv-fm/Scripts/python.exe -m src.rq2.phase1.ml_followup.exp3_fm forecast zs ft   # FM (별도 환경)
py -3.13 -m src.rq2.phase1.ml_followup.exp3_fm analysis
py -3.13 -m src.rq2.phase1.ml_followup.run_all figures         # 그림만 다시
```
- 캐시: `data/interim/rq_cache/ml_followup/{exp1..exp7}/` (Git 제외)
- 결과 표: `results/tables/*.csv`, 실행 기록: `results/experiment_log/`, 환경: `results/environment.json`
- 대략 소요: 12코어 CPU 기준 전체 ≈ 15~20시간 (Exp5 gk_dense, Exp6, Exp4 gk, Exp3 fine-tune 이 대부분)
