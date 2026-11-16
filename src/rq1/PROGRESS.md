# RQ1 진행 기록 (연구자 B)

맥락이 끊겨도 이어서 작업할 수 있게 단계별로 기록한다. 최신 항목이 위에 온다.
작업 방식은 RQ2(연구자 A)의 틀을 따른다: 설정 파일, 실험 로그 자동 기록, REPORT.md, environment.json, 주 검정 사전 지정 + Holm, 결과 동결.

## 현재 상태

| 단계 | 상태 |
|---|---|
| 0. 사전 지정 (`config.yaml` pre_registration) | 초안 작성 (2026-11-17), 동결 커밋 전 |
| 1. 데이터·기초 그림 | 미착수 |
| 2. 주 분석 (P1, P2) | 미착수 |
| 3. 진단·placebo·MDE | 미착수 |
| 4. 탐색·강건성 | 미착수 |
| 5. REPORT.md | 미착수 |

## 위치

- 설정: `src/rq1/config.yaml` (사건일·통제·사전 지정)
- 코드: `src/rq1/main_study/` (예정). 반사실 모형·데이터 로드·실험 로그는 `src/rq2/core/` 를 수정하지 않고 import 한다
- 결과: `src/rq1/main_study/results/{REPORT.md, tables/, figures/, experiment_log.{csv,md}, environment.json}`
- 캐시: `data/interim/rq_cache/` (Git 제외)

## 내린 결정

1. (2026-11-17) 사건일은 A 가 확인한 `configs/rq2.yaml` timeline 을 쓴다. 공통 `configs/events.yaml` 갱신은 A 와 따로 진행한다.
2. (2026-11-17) 통제변수 `C_main`, donor pool(main 175), 음성통제 005935 는 RQ2 와 같은 정의를 쓴다.
   KOSPI·KOSPI200·전기전자 지수와 VKOSPI 는 처치 종목 비중이 커서 주 통제에서 뺀다.
3. (2026-11-17) 처치 구간은 상장~발표 전날(05-27~07-15, 35거래일). 학습은 상장 전날(05-26)까지.
4. (2026-11-17) 주 분석은 패널 DiD (처리 종목별, donor main 175 대조). 주 검정은 P1(005930), P2(000660) 2개, Holm 보정.
   처리 종목이 1개라 판정은 donor placebo-in-space 순열 p 로 한다. 상장 전 평행추세·사전 반응(05-22)은 이벤트 스터디로 확인한다.
   통합 효과와 통제변수 회귀는 보조, 같은 기업 차분과 RQ2 BSTS 는 강건성이다.
   005930 의 DiD 와 같은 기업 차분이 다르면 `config.yaml` disagreement_rule_005930 대로 해석한다.
5. (2026-11-17) 판정 조건 (b) 음성통제는 종목별로 다르다. 005930 은 같은 기업 우선주 005935, 000660 은 같은 기업 우선주가 없어
   반도체 관련주 7종목(삼성·SK 계열 제외, 상품 없음)을 업종 음성통제로 쓴다. 000660 은 기업 고유 충격을 배제하지 못하는 것을 한계로 적는다.
6. (2026-11-17) 커밋은 `src/rq1/` 파일만. 공통 파일(events.yaml, PROJECT_CONTEXT 7절) 변경은 A 와 합의 후 별도로.

## A 와 합의·조율할 것

- [ ] `configs/events.yaml` 에 rq2.yaml timeline 반영 (+ `run_all` 재실행으로 `proc.calendar` 갱신)
- [ ] PROJECT_CONTEXT 7절에 위 결정 기록
- [ ] `src/rq2/core/explog.py` 등 공용 부분을 `src/common/` 으로 옮길지
- [ ] 장 마감 30분 리밸런싱 분석을 RQ1 과 RQ2 중 어디서 할지
- [ ] 상품 상장 승인·공시일 확인 (05-22 사전 반응 가능성)
