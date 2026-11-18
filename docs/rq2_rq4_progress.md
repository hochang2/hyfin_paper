# RQ2·RQ4 진행 기록 (연구자 A)

맥락이 끊겨도 이어서 작업할 수 있게 단계별로 기록한다. 최신 항목이 위에 온다.
사용자 지시(2026-11-15): **RQ2 를 먼저 끝내고 충분히 검토한 뒤 RQ4 로 넘어간다.**

## 현재 상태

| 단계 | RQ2 | RQ4 |
|---|---|---|
| 1. 검토·설계 | 완료 | 완료 (src/rq4/config.yaml 사전 설계) |
| 2. Baseline | 완료 (BSTS 주 분석, ITS, within-firm) | 완료 (DID·event study·DDD, 역사적 위약 주 추론) |
| 3. 확장 모형 | 완료 (11개 모형 + Chronos-2·TimesFM + GARCH) | 완료 (매칭·IPW/DR·SDID) |
| 4. 진단 | 완료 (D01~D05, 통제 오염 점검 포함) | 완료 (G01~G09) |
| 5. 강건성 | 완료 (BSTS 24·OLS 20 사양, placebo 4종, 미국 변수 반사실) | 완료 (R00~R33, 위약 P1~P5) |
| 6. 종합 보고서 | 완료 (src/rq2/phase1/main_study/results/REPORT.md, 2026-11-16) | 완료 (src/rq4/results/REPORT.md, 2026-11-17) |
| 7. ML 확장 하위실험 | 완료 (src/rq2/phase1/ml_extension/results/REPORT.md, 2026-11-16) | - |

## 실행 방법

```
py -m src.rq2.phase1.main_study.run_all                       # 전 단계
py -m src.rq2.phase1.main_study.run_all --stages primary,figures
py -m pytest src/rq2/tests                  # BSTS 구현 검증 (모의자료)
```

- 설정: `configs/rq2.yaml`
- 결과: `src/rq2/phase1/main_study/results/{tables,figures}`, `experiment_log.{csv,md}`, `environment.json`
- DB 캐시: `data/interim/rq_cache/*.parquet` (Git 제외). 지우면 MotherDuck 에서 다시 받는다.

## 내린 결정 (공통 설정 변경 제안 포함)

1. **사건일**: 시행일은 2026-07-31 (7/24 금융위 발표로 앞당김; 원래 8/5경), 발표는 2026-07-16. 근거는 `configs/rq2.yaml` timeline 주석.
   `configs/events.yaml` 는 공통 설정이라 수정하지 않았다 → **연구자 B와 합의 후 반영 제안**.
2. **추정대상**: 학습구간 = 2025-10-01~2026-07-15 (상장기간 포함). 반사실은 "상품은 계속 있고 규제만 없음"이다.
   상장 전만으로 학습하면 추정대상이 "상품 자체가 없음"으로 바뀌므로 강건성 사양으로만 쓴다.
3. **통제변수**: KOSPI·KOSPI200·전기전자 지수와 VKOSPI 에는 삼성전자·SK하이닉스 비중이 커서 주 통제에서 뺐다.
   오염 확인용 사양(C_contaminated)으로만 쓴다.
4. **donor pool**: 반도체 관련 종목, 삼성·SK 계열, 늦게 상장한 종목을 뺐다 (main 175종목).
5. **음성통제**: 삼성전자우(005935)는 같은 기업이지만 단일종목 레버리지 상품이 없다. 모든 RQ2 분석을 005935 에도 똑같이 돌린다.
6. **TimesFM/Chronos**: 사용자 요청으로 별도 가상환경 `.venv-fm` 에 설치해 실행했다 (RQ2-E-FM-*).
   단변량 버전은 음성통제에서도 큰 음(−)의 효과가 나와 공통충격을 효과로 오인한다. 공변량 버전은 다른 모형과 비슷하다.
7. **주 검정 family**: {005930, 000660} × {transition 07-16~07-30, post_effective 07-31~10-30}. Holm 보정.
   그 밖의 결과는 탐색적 분석으로 취급한다.

## RQ2 결과 메모 (최종 해석은 src/rq2/phase1/main_study/results/REPORT.md)

- BSTS 주 분석: post_effective 효과는 005930 +0.05 [−0.18, 0.28], 000660 −0.05 [−0.27, 0.15]. 둘 다 0 근처다.
  transition 은 두 종목 모두 양(+)인데, 000660 의 비보정 p=0.038 은 Holm 보정 후 0.15 이다.
- 음성통제 005935 도 같은 크기의 양(+) 이탈을 보인다 → 기업·업종 충격일 가능성이 크다.
- 같은 기업 차분(005930−005935)은 상장기간 대비 transition +0.02, post +0.02 로 0 에 가깝다.
- 진단:
  - 000660 은 상장일 부근에서 회귀관계가 구조적으로 변했다 (Chow p=.001, sup-F p=.02).
  - 005930 잔차는 1차 자기상관 0.39 다 → HAC / block bootstrap / local level 로 대응했다.
- 국내 통제(donor 평균, KOSDAQ)가 상장기간 대비 크게 떨어졌다 (D05). 그래서 미국 변수만 쓴 반사실을 2016~ 긴 표본으로 보정했다 (R-USONLY).
  - post 61일에 donor 평균을 포함한 모든 계열이 미국 기준 반사실보다 높거나 같았다.
  - 국내 통제의 하락은 7월 급등의 되돌림으로 해석된다. 시장 전체 냉각의 증거는 없다.
- 같은 기업 BSTS 의 21~61일 −0.25 (p=0.023) 는 Holm 0.18, 긴 표본 보정 p 0.85 다. 근거로 쓰지 않는다 (R-WITHIN-CALIB).
- 다음: RQ2 결론을 사용자에게 보고한 뒤 RQ4 에 착수한다. events.yaml 사건일 갱신은 연구자 B 에게 제안할 것.

## RQ2 ML 확장 메모 (src/rq2/phase1/ml_extension/results/REPORT.md)

- 질문: 반사실 모형에 ML·FM 을 넣어 얻는 순이익이 있었나 → 없다 (예측·검출력·튜닝·앙상블 모두).
- rv5 30일 효과오차: BSTS 0.180, OLS 0.187, ML 0.217~0.267. gk 는 ens_stack 0.314 가 1위이고, 단독 ML 은 BSTS 보다 유의하게 나쁘다.
- MDE(검정력 0.8): rv5 10일 −45%(OLS·BSTS), ML −50~−54%. −20% 는 어떤 모형도 검출 못 한다.
- time 보정은 비겹침 유효 창 수(n_eff)로 한다. rv5 30일 이상은 검정 불가라서 사전 규칙의 fallback(10일)을 썼다 → harx 선정, 실제 효과 모두 비유의 (Holm p 1.0).
- 다음: 2027-01 까지 표본 연장 시 `src/rq2/phase1/ml_extension/config.yaml` 날짜만 바꾸고 재실행. 해외 ETF 는 가장 나중.

## RQ4 결과 메모 (src/rq4/results/REPORT.md, 2026-11-17)

- H1 지지 안 됨: 비규제 2X(B∪C) vs 1X β = −0.10, 역사적 위약 p 0.92. 모든 SE·사양에서 비유의.
- H2: β_C − β_B = +1.54 (위약 p 0.087, Holm 0.174; CR1 0.002). 반도체 사전 V자 추세·반도체 1X 대조 −73% 붕괴 때문. 기준선을 상장 직전 35일로 옮기면 0.65 (p 0.35). HonestDiD 붕괴점 M̄ < 0.25.
- 규제 대상 거래 −94%. 국내 2X 가 흡수한 삼성·하이닉스 노출은 사라진 노출의 1.5~3.5%. 교차상관은 양(+) → 대체 아님.
- 해외 상장 상품 경로는 자료 없음 → 다음 우선 과제 (SEIBro 해외주식 결제).
- RQ2 연결: '국내 대체상품으로 압력 유지' 설명을 약화시키는 정황 증거. 인과 검증 아님.

## RQ4 설계 메모 (착수 전, 기록용)

- 단위: ETF 상품.
  - 처치(대체재 후보): 규제 밖 국내 레버리지 2X — KOSPI200, KOSDAQ150, 반도체, 나스닥, SOX.
  - 통제: 같은 지수의 1X ETF.
- 1단계 확인: 규제 대상 P1 상품의 거래·개인 순매수가 실제로 줄었는가.
- 모형: TWFE DID, 주간 event study, DDD (2X vs 1X × 반도체 vs 지수), SDID, 매칭 DID.
- 추론: wild cluster bootstrap, Driscoll-Kraay, 무작위화 추론.
- 해외 대체: 국내 투자자의 해외 상품 거래 자료(F15/SEIBro)가 수집되지 않았다 → 기술통계로만 다루고 한계로 명시한다.
