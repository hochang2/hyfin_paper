# RQ4

- **담당:** 연구자 A
- **질문:** 단일종목 레버리지 규제 후 비규제 국내 레버리지 ETF 의 거래활동이 같은 기초 계열 1X 대비 상대적으로 증가했는가(H1)? 지수형 vs 반도체형의 차이(H2)?
- **설계:** 상품×일 패널 DID (상품 FE + 계열×일 FE), 주간 event study, DDD (2X/1X × 반도체/지수 × 사후/기준)
- **주 추론:** 접힌 격차 시계열의 역사적 위약(2016~2026-05 비중첩 창). CR1·wild·DK·2-way·무작위화는 보조
- **표현 원칙:** '수요 이동과 부합하는 상대적 거래활동 변화'. 개인 식별 자료가 없으므로 '동일 투자자 이동'은 쓰지 않는다

## 상태

완료 (2026-11-17). 결론과 전체 수치는 [phase1/main_study/results/REPORT.md](phase1/main_study/results/REPORT.md).

- H1 지지 안 됨 (β = −0.10, 위약 p 0.92)
- H2 차이 +1.54 (위약 p 0.087, Holm 0.174). 반도체 사전추세·1X 대조 붕괴에 취약
- 규제 대상 거래 −94%. 국내 2X 로 옮겨간 노출은 사라진 노출의 1.5~3.5%
- 계획했던 Double ML·Causal Forest 는 하지 않았다. 처치 단위가 11개(반도체 3)인 공통 시행일 설계라 이질 효과 학습이 의미가 없다 (REPORT 11절)

## 구조 (RQ2 틀을 따름)

```
src/rq4/
  config.yaml   시점, 상품군, 결과변수, 가설, 추론 (사전 설계)
  core/         공용 라이브러리: common.py, data_pipeline.py, panel.py, estim.py, collapsed.py
  tests/        테스트 (아직 없음)
  phase1/       Phase 1
    main_study/   README.md, run_all.py, *.py, results/{REPORT.md, tables/, figures/, log_parts/, experiment_log.*}
  phase2/       Phase 2 (예정)
```

## 실행

```
PYTHONIOENCODING=utf-8 py -3.13 -m src.rq4.phase1.main_study.run_all    # 전체 (약 15분, 캐시 있으면 DB 접속 없음)
PYTHONIOENCODING=utf-8 py -3.13 -m src.rq4.phase1.main_study.<모듈>      # 단계별
```

- DB 는 `hyfin.proc` 만 읽는다. 캐시는 `data/interim/rq_cache/rq4/` (Git 제외). 지우면 MotherDuck 에서 다시 받는다.
- 원자료·`src/common`·`configs/events.yaml` 은 바꾸지 않는다.
- 단계별 파일과 결과 표는 [phase1/main_study/README.md](phase1/main_study/README.md) 에 있다.
