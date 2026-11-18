# RQ1

- **담당:** 연구자 B
- **질문:** 상품 도입 후 삼성전자·SK하이닉스 변동성이 시장·반도체 공통요인 통제 후에도 변화했는가?
- **기준모형 (baseline):** GARCH 계열, 구조변화·이벤트 전후 회귀
- **확장 실험 (extension):** 대조군 기반 효과 비교, 비선형 변동성 모형
- **주의:** 도입~규제 전 구간이 짧음, 공통 시장충격 교란

## 상태

- **1차 연구 (main_study)**: 분석 완료 (2026-11-17). 결론과 근거는 [phase1/main_study/results/REPORT.md](phase1/main_study/results/REPORT.md).
  요약: 상장 후 두 종목의 변동성이 올랐지만 상품 없는 삼성전자우·반도체 종목도 같은 크기로 올라, 상장 효과의 증거는 없다. 검정력은 낮다.
- 진행 기록과 결정은 [PROGRESS.md](PROGRESS.md), 설정과 사전 지정은 [config.yaml](config.yaml).

## 구조 (RQ2 틀을 따름)

```
src/rq1/
  config.yaml   사건일·통제·표본·사전 지정(주 검정, 판정 규칙, 미리 알던 사실)
  PROGRESS.md   진행 기록·결정·A 와 조율할 것
  core/         공용 라이브러리: data.py (패널·역할), did.py (DiD·순열·Holm·MDE)
  tests/        core 검증 (패널 점검, DiD 모의자료)
  phase1/       Phase 1
    main_study/   README.md, run_all.py, *.py, results/{REPORT.md, tables/, figures/, experiment_log.*, environment.json}
  phase2/       Phase 2 (예정)
```

- 반사실 모형·데이터 로드·실험 로그는 `src/rq2/core/` 를 수정하지 않고 import 한다.
- 결과는 동결한다. 다시 돌리면 새 커밋으로 남긴다.
