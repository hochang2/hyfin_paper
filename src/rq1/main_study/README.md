# RQ1 1차 연구 (main_study)

질문: **상품 상장(2026-05-27) 후 삼성전자·SK하이닉스의 실현변동성이 상품 없는 비교 종목 대비 높아졌는가?**
사전 지정 주 분석은 종목별 패널 DiD + donor placebo-in-space 순열 p + Holm 이다 (`../config.yaml`).

결과와 해석: [results/REPORT.md](results/REPORT.md)

## 원칙

- 데이터 로드·BSTS·실험 로그는 `src.rq2.core` 를 수정 없이 쓴다.
- 결과는 동결한다. 다시 돌리면 결과 폴더를 새 커밋으로 남긴다.
- 표본은 07-15(규제 발표 전날)까지. 표준화·매칭은 상장 전 자료로만.

## 파일

| 파일 | 내용 |
|---|---|
| `run_all.py` | 단계 실행 (`--stages data,primary,judgment,robustness,exploratory`) |
| `data.py` | 종목 × 일 패널, 역할(처리·donor·음성통제), GK 패널, 매칭 변수, 만기일 |
| `did.py` | TWFE DiD, 이벤트 스터디, donor 순열, 포트폴리오 귀무, Holm, MDE |
| `robustness.py` | 측정치·표본·매칭·통합·통제 회귀·같은 기업·BSTS |
| `exploratory.py` | GJR-GARCH-X (직접 MLE), 평균 이동 sup-F |
| `plots.py` | 그림 |
| `results/` | `REPORT.md`, `tables/`, `figures/`, `experiment_log.{csv,md}`, `environment.json` |

테스트: `py -m pytest src/rq1/tests` (패널 점검 6개, DiD 모의자료 6개).
