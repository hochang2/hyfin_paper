# RQ1

- **담당:** 연구자 B
- **질문:** 상품 도입 후 삼성전자·SK하이닉스 변동성이 시장·반도체 공통요인 통제 후에도 변화했는가?
- **기준모형 (baseline):** GARCH 계열, 구조변화·이벤트 전후 회귀
- **확장 실험 (extension):** 대조군 기반 효과 비교, 비선형 변동성 모형
- **주의:** 도입~규제 전 구간이 짧음, 공통 시장충격 교란

## 상태

사전 지정 초안 작성 (2026-11-17). 진행 상황과 결정은 [PROGRESS.md](PROGRESS.md), 설정과 주 검정은 [config.yaml](config.yaml).

## 구조 (RQ2 틀을 따름)

```
src/rq1/
  config.yaml     사건일·통제·표본·사전 지정(주 검정, 판정 규칙, 미리 알던 사실)
  PROGRESS.md     진행 기록·결정·A 와 조율할 것
  main_study/     주 분석 (예정): run_all.py, *.py
    results/      REPORT.md, tables/, figures/, experiment_log.{csv,md}, environment.json
```

- 반사실 모형·데이터 로드·실험 로그는 `src/rq2/core/` 를 수정하지 않고 import 한다.
- 결과는 동결한다. 다시 돌리면 새 커밋으로 남긴다.
