# 전처리 (raw → proc)

**역할 분담: `raw` = 백업, `proc` = 실제 사용.** 분석 코드(`src/rq*/`)는 `hyfin.proc` 만 읽는다.
`hyfin.raw` 는 이 폴더의 전처리 코드만 읽고, 아무도 고치거나 지우지 않는다.
raw 에 있는 분석용 데이터는 모두 proc 에 테이블이나 뷰로 들어가 있다. 목록 정보(`*_catalog`, `minute_file_list`)만 raw 에 남는다.

proc 은 언제든 아래 명령으로 다시 만들 수 있다 (A만 실행, 쓰기 권한 필요). 전체 실행은 15초 안팎이다.

```bash
py -m src.common.processing.run_all                     # 전부 (순서대로)
py -m src.common.processing.make_realized_volatility    # 하나만
```

| 파일 | 만드는 테이블 | 행 단위 | 키 | 행 수 (2026-11-12) |
|---|---|---|---|---|
| `make_trading_calendar.py` | `proc.calendar` | 한국 거래일 2015-01-02 ~ 2026-10-30 | date | 2,902 |
| `make_market_controls.py` | `proc.market_daily` | 한국 거래일 | date | 2,902 |
| `make_index_membership.py` | `proc.kospi200_membership` | 거래일 × KOSPI200 편입 경험 종목 (2025-06-13 ~) | code, date | 71,994 |
| `make_intraday_bars.py` | `proc.intraday_5m` | 분봉 거래일 × 국내 249개 × 5분 칸 | code, date, k | 4,681,077 |
| `make_realized_volatility.py` | `proc.rv_daily` | 분봉 거래일 × 국내 종목·상품·지수 249개 | code, date | 61,790 |
| | `proc.rv_daily_us` | 미국 거래일 × 미국 8종 | ticker, date | 1,184 |
| `make_stock_panel.py` | `proc.stock_daily` | 거래일 × 주식 213종 | code, date | 554,758 |
| `make_etf_panel.py` | `proc.etf_daily` | 거래일 × ETF·ETN 392종 | code, date | 406,984 |
| `make_standard_views.py` | 뷰 6개 (아래) | raw 와 같음 | | raw 와 같음 |

`run_all.py` 는 calendar → market → membership → intraday → RV → stock → ETF → 뷰 순서로 돈다.
`market_daily` 와 `kospi200_membership` 은 `calendar` 를, `rv_daily` 는 `intraday_5m` 을 쓰기 때문이다.
공통 함수(`save`, `save_view`, `load_config`)는 `__init__.py` 에 있다. `save` 는 테이블을 새로 만들고(`CREATE OR REPLACE`) 주석을 단다.
주석에는 만든 사람, 날짜, 원천, 코드 경로가 들어간다.

## 공통 규칙

- `date` 는 DATE, `code` 는 접두사 없는 6자리로, 분봉 코드와 같다. FnGuide 원래 코드(`A005930`, `Q500057`)는 `fg_code` 에 둔다.
- 없는 값은 NULL 이다. 0으로 채우지 않는다.
- 금액은 원 단위다. F16 의 억원은 ×1e8, F10 의 백만원은 ×1e6 으로 바꿨다. F3·F9 투자자 금액은 FnGuide 단위를 그대로 둔다.
- 수익률은 모두 로그수익률이다.
- **미래정보 차단**
  - 연간 실적·재무는 공시일 다음 날부터 붙인다.
  - 베타는 전년도 값을 쓴다.
  - 미국 값과 FRED 값은 한국 날짜보다 앞선 날짜(`<`) 중 가장 최근 것을 붙인다.

## 테이블별 내용

### proc.calendar

| 열 | 뜻 |
|---|---|
| `tday` | 거래일 순번 (1부터) |
| `has_minute` | 분봉이 있는 날 (2025-10-01 ~). 이 기간에만 아래 4개 열이 있다 |
| `reg_bars`, `open_hhmm`, `last_hhmm` | 코스피 1분봉 정규장 봉 수, 첫 봉 시각, 마지막 봉 시각 |
| `late_open` | 개장 지연일 (2025-11-13 수능 10:00, 2026-01-02 10:00) |
| `gap_min`, `gap_range` | 장중 시장 공백 분 수와 시간대. 공백일은 10일이고 각 29분 |
| `post_listing`, `post_announce`, `post_effective` | 사건일 이후 여부. 규제 시행일은 미정이라 NULL |
| `period` | `pre_listing` / `listing_to_announce` / `post_announce` |
| `rel_listing`, `rel_announce`, `rel_effective` | 사건일을 0으로 둔 거래일 차이. 이벤트 스터디 창에 쓴다 |
| `short_sale_ban` | 공매도 전면 금지 기간 (2023-11-06 ~ 2025-03-28) |

사건일과 제도 기간은 `configs/events.yaml` 에서 읽는다. 날짜를 바꾸면 `run_all` 을 다시 돌린다.

### proc.rv_daily (국내 실현변동성, 2025-10-01 ~)

방법 (`configs/analysis.yaml` → `rv`):
- 가격은 KRX 1분봉 수정주가를 쓰고, 정규장만 본다.
- 칸은 그날 코스피 정규장 시각을 5분으로 나눈 격자다. 보통 76칸이고, 개장 지연일·공백일은 그날 시장에 맞춰진다.
- 칸 종가는 칸 안 마지막 1분봉 종가다. 거래가 없는 칸은 직전 칸 종가를 이어 쓰므로 수익률이 0이 된다.
- 첫 수익률은 첫 정규장 봉의 시가를 기준으로 한다. 따라서 전날 종가에서 시가까지의 변동(overnight)은 빠진다.

| 열 | 뜻 |
|---|---|
| `asset` | 주식 / ETF / ETN / index |
| `rv5` | Σ(5분 로그수익률)². 정규장만. **일간 분산**이므로 연율 변동성은 `sqrt(rv5*252)` |
| `rv5_close` | `rv5` + (종가 단일가 / 정규장 마지막 칸)². 종가 단일가가 없는 날은 `rv5` 와 같다 |
| `rv_pk` | Parkinson 추정치 (정규장 고가·저가) |
| `p_open`, `p_close_reg`, `p_close_auction`, `p_close`, `p_high`, `p_low` | 시가, 정규장 마지막 가격, 종가 단일가, 종가(단일가가 있으면 그것), 고가, 저가 |
| `ret_oc`, `ret_overnight`, `ret_cc` | 시가→종가, 전날 종가→시가, 종가→종가 |
| `n_bars`, `n_buckets`, `n_buckets_traded` | 정규장 1분봉 수, 격자 칸 수, 실제 거래가 있던 칸 수 |
| `volume_krx` | KRX 1분봉 전 세션 거래량 합 |
| `volume_all` | KRX+NXT 전 세션 거래량. ALL 1분봉이 있으면 그것, 없으면 ALL 5분봉. ALL 이 없는 상품은 NULL |

지수 코드는 `IDX001` KOSPI, `IDX201` KOSPI200, `IDX101` KOSDAQ, `IDX013` 전기·전자다.
분봉 코드는 FnGuide 와 반대이니 주의한다 (FnGuide `I.101` = KOSPI200, `I.201` = KOSDAQ). 지수 거래량은 단위가 달라서 넣지 않았다.

### proc.intraday_5m (5분 장중 패널)

`rv_daily` 의 5분 수익률이 이 테이블에서 나온다. `rv5` = 그날 `ret²` 의 합이다.

| 열 | 뜻 |
|---|---|
| `k`, `bucket`, `hhmm` | 그날 몇 번째 칸인지, 칸 시작 시각 (TIMESTAMP), 칸 시작 HHMM. 개장 지연일은 k=1 이 10:00 이다 |
| `close` | 칸 마지막 종가. 거래 없는 칸은 직전 칸 값이고, 그날 첫 거래 전 칸은 NULL |
| `traded`, `n_bars` | 칸 안에 실제 1분봉이 있었는지, 그 개수 |
| `ret` | 직전 칸 대비 로그수익률. k=1 은 그날 첫 정규장 봉의 시가 대비 |
| `volume_krx` | 칸 안 KRX 거래량. 지수는 NULL |

종가 단일가(15:30)는 칸에 없다. 필요하면 `rv_daily.p_close_auction` 을 쓴다.

### proc.kospi200_membership (KOSPI200 편입 여부)

- 거래일마다 `기준일 <= 그날` 인 가장 최근 F17 스냅샷으로 판단한다.
  - 스냅샷은 5개이고, 정기변경 적용일(2025-12-12, 2026-06-12)과 날짜가 같아서 정기변경은 정확하다.
  - **수시변경은 다음 스냅샷 날부터 반영된다.** 한화머시너리앤서비스홀딩스(0220W0)는 실제로 2026-06~10 사이에 편입됐지만 2026-10-30 부터 TRUE 다.
- 열: `in_k200`, `snapshot_date`, `mktcap_snapshot_won` (스냅샷 상장시총), `change` (그 스냅샷 구간에서 편입·편출된 종목이면 '편입'/'편출').
- 대상은 한 번이라도 스냅샷에 나온 종목이다. 편출 후에도 행이 있고 `in_k200 = FALSE` 다. OCI 처럼 빠졌다 다시 들어온 종목도 그대로 보인다.

### proc.rv_daily_us (미국, 2026-04-01 ~)

- 대상: SPY, SOXX, SOXL, TQQQ, NVDL, TSLL, MU, TSM.
- 정규장만 보고, 격자는 SPY 기준(보통 78칸)이다.
- `date` 는 미국 영업일이다. 한국 날짜에 붙일 때는 `date < 한국 날짜` 중 가장 최근 값으로 ASOF join 한다.
- 열은 `rv5`, `rv_pk`, 가격, `ret_oc`, `ret_cc`, 칸 수, `volume` 이다.

### proc.market_daily (시장 통제변수)

| 묶음 | 열 | 시점 |
|---|---|---|
| 국내 지수 (F11) | `kospi`, `k200`, `kosdaq`, `kq150`, `k200_it`, `krx_semi` 와 각 `r_*`, `vkospi` | 같은 날 종가 |
| 미국 (F14) | `sox`, `ndx`, `spx`, `vix`, `nvda`, `tsm` 와 각 `r_*`, `us_asof_date` | **전날 미국 종가** (월요일이면 금요일 값) |
| FRED (F16) | `ust10` (미국채 10년), `usdkrw` (뉴욕 정오 환율) | 전날 미국 값 |
| 국내 거시 (F16) | `deposit_won`, `credit_won`, `credit_kospi_won`, `credit_kosdaq_won` (원), `ktb3`, `ktb10` (%) | 같은 날 |

국내 거시 변수는 장 마감 후 집계된다. 같은 날 RV 를 설명하는 모형에서는 하루 늦춰서 써야 한다.

### proc.stock_daily (주식 패널)

| 묶음 | 열 |
|---|---|
| F1 | `open`~`close`, `volume`, `value`, `mktcap`, `shares`, `ret_pct`, `float_pct`, `foreign_pct`, `ret_log` (= ln(1+ret_pct/100), 분할 반영), `halted` (거래량 0), `first_date` |
| F2 | `grp` (S1_기초자산 / S2_KOSPI200 / S3_KOSPI200편출) |
| F3 | `<투자자>_qty`, `<투자자>_amt`. 투자자: indiv, foreign, foreign_other, inst, fin_inv, insur, trust, bank, pension, corp_other, pef |
| F4 | `credit_*`, `short_*`, `short_bal_*`, `lend_bal_qty` |
| F5·F6 일별 | `pbr`, `per`, `op_income_cons_fy2026` |
| 연간 (공시 후) | `op_income_last`, `op_income_fy`: 잠정실적 공시일(없으면 사업보고서일) **다음 날**부터 붙는다 |
| | `assets_last`, `sales_last`, `debt_ratio_pct_last`, `fin_fy`: 사업보고서 제출일(없으면 다음 해 3/31) **다음 날**부터 붙는다 |
| | `beta_prev`: 전년도 베타 |
| 삼성전자·SK하이닉스만 | `fut_close`, `fut_settle`, `fut_spot`, `fut_volume`, `fut_oi` (전 월물 합), `fut_roll_day`, `fut_basis_pct` |
| | `opt_iv_atm`, `opt_iv_call`, `opt_iv_put`, `opt_dte`, `opt_volume`, `opt_oi` (옵션은 2025-01 ~ 444일) |

예를 들어 삼성전자 2025년 잠정실적은 2026-01-08 에 공시됐으므로 01-09 부터 붙는다. 같은 해 사업보고서는 2026-02-12 에 제출됐으므로 02-13 부터 붙는다.

### proc.etf_daily (ETF·ETN 패널)

| 묶음 | 열 |
|---|---|
| F7 | `close`, `nav`, `iv`, `nav_or_iv`, `premium_pct`, `shares`, `volume`, `value`, `aum_calc` |
| 계산 | `traded` (종가 있음), `ret_log`, `nav_ret_log` (직전 행 대비), `flow_won` = (좌수 − 직전 좌수) × 오늘 NAV·IV |
| F8 | `kind`, `leverage`, `replication`, `grp`, `grp_desc`, `index_name`, `manager`, `fee_pct`, `total_cost_pct`, `lp`, `list_date`, `delist_date` |
| F9 | `indiv_*`, `inst_*`, `fin_inv_*`, `foreign_*`. 33종만 있다 |
| F10 | `aum_won`. 2025-01 ~ 만 있다 |

F8 의 `grp` S1~S3 은 '1배 비교 상품' 그룹이다. F2 의 S 그룹(주식)과 이름은 같지만 뜻이 다르다.

### 뷰 (make_standard_views.py)

가공 없이 형식만 맞춘 뷰다. 저장 공간을 쓰지 않고, 조회할 때마다 raw 를 읽는다.
바꾼 것은 `date` → DATE, `code` → 6자리(원래 코드는 `fg_code`), F10 금액 → 원, raw 의 `file` 열 제거뿐이다.

| 뷰 | 원본 | 내용 |
|---|---|---|
| `proc.futures_all` | F12 전 월물 | 선물 전 월물. `code` = 기초자산 (005930/000660) |
| `proc.options_all` | F13 전 종목 | 옵션 행사가별. `code` = 기초자산 |
| `proc.minute_kr` | 국내 분봉 | 전 세션(pre/regular/close/after), KRX·NXT·ALL, 1분·5분, 수정·원주가 |
| `proc.minute_idx` | 지수 분봉 | `code` = IDX001 등 (`intraday_5m` 과 같은 표기) |
| `proc.minute_us` | 미국 분봉 | 전 세션 (pre/regular/post/overnight). `date` = 미국 영업일 |
| `proc.etf_aum_all` | F10 | 시장 전체 ETF 1,186종 순자산 (`aum_won`) |

분봉 뷰는 크다(국내 3,500만 행). 집계는 SQL 로 하고 결과만 pandas 로 받는다.

## 점검 결과 (2026-11-12)

| 점검 | 결과 |
|---|---|
| 키 중복 (모든 테이블) | 0 |
| `rv_daily` 를 `intraday_5m` 기반으로 바꾼 전후 | `rv5`, `rv5_close`, 정규장 종가 모두 같음 (차이 < 1e-16) |
| 뷰 행 수 vs raw | 6개 모두 같음, `date` NULL 0개 |
| KOSPI200 편입 종목 수 | 스냅샷 구간마다 하루 200개 (2026-10-30 은 201개) |
| `rv_daily` 종가 vs F1 종가 (주식 55,046행) | 0.1% 넘게 차이 나는 행 0개 (최대 0.03%) |
| 코스피 격자 칸 수 | 76칸 250일, 71칸 10일(공백일), 64칸 1일(2026-01-02) |
| 연율 RV 중앙값 | 지수 28%, 주식 45%, ETF 62%, ETN 105% |
| `ret_log` vs 종가 비율 (주식) | 분할 반영 후 모두 일치 |
| 미국 값 미래정보 | `us_asof_date >= date` 인 행 0개 |
| 연간 자료 시점 | 삼성전자 2025 실적·재무가 각각 공시 다음 날부터 붙는 것 확인 |

## 알아둘 점

- **ETF·ETN `ret_log` 는 가격 그대로다.**
  - |수익률|>40% 인 행이 96개 있다. 대부분 ETN 괴리 해소, 해외 기초자산 시차, 2026-03-09 원유 급변 같은 실제 현상이다. 좌수 분할 흔적은 거의 없다.
  - 기초자산 수익률이 필요하면 `nav_ret_log` 를 쓴다.
- **`flow_won` 이상치가 1개 있다.** TIGER 삼성그룹(138520) 2025-06-17 은 순자산의 7배가 넘는 환매로 나온다. 좌수 데이터 오류일 수 있다.
- **ETF·ETN 15종은 종가 단일가 봉이 없는 날이 있다** (`p_close_auction` NULL 187행). 이날은 `rv5_close = rv5` 다.
- `rv_daily` 에는 `name` 이 없다. 이름은 `stock_daily` 나 `etf_daily` 와 `code` 로 붙인다.

## RV 사용 규칙 (2026-11-12 결정)

- 주분석은 `rv5` (정규장만), 강건성 검정은 `rv5_close` (종가 단일가 포함).
- 종속변수: RQ1·RQ2·RQ4 는 `ln(rv5)`, RQ3 는 `rv5` 와 `ln(rv5)` 둘 다 (HAR 관례). 변환은 분석 코드에서 한다.
- 규제 시행일은 미정이다. 정해지면 `configs/events.yaml` 에 넣고 다시 실행한다.
