# 공유 DB 이용 가이드 (MotherDuck `hyfin`)

작성일: 2026-11-11 · 대상: 연구자 A(DB 관리자)·연구자 B · 데이터 내용은 [data_spec.md](data_spec.md) 참고

## 1. 한눈에 보기

| 항목 | 내용 |
|---|---|
| 서비스 | MotherDuck (클라우드 DuckDB), 무료 Lite 요금제 |
| DB | `hyfin` — 소유자 연구자 A. **쓰기는 A만**, B는 공유(share)로 읽기 전용 |
| 리전 | `aws-ap-northeast-1` (도쿄). 공유는 같은 리전 안에서만 됨 |
| 스키마 | `raw` = 받은 원본 그대로, **백업용** (수정·삭제 금지) / `proc` = **분석에 쓰는 것 전부** (전처리 테이블 + 형식만 맞춘 뷰) |
| 접속 | Python `duckdb` **1.5.6** (MotherDuck은 2.0 미지원), 또는 웹 UI app.motherduck.com |
| 무료 한도 | 저장 10GB (현재 약 0.9GB), 계산 월 10시간, 스냅샷 1일 보관 |

## 2. 처음 설정

### 연구자 B
1. A가 보낸 **조직 초대 메일**의 링크로 가입·참여 (조직에 들어가면 리전이 자동으로 같아짐).
2. 웹 UI → 왼쪽 위 조직 이름 → **Settings → Integrations → Access Tokens → Create token**. 토큰 복사.
3. 저장소 루트에 `.env` 파일을 만들고 한 줄 넣기. **`.env` 는 커밋하지 않는다** (gitignore 됨).
   ```
   motherduck_token=<내 토큰>
   ```
4. `py -m pip install "duckdb==1.5.6"`
5. 공유 DB 붙이기 (한 번만 하면 계정에 남음):
   ```python
   import duckdb, os
   # .env 를 읽어 os.environ["motherduck_token"] 에 넣은 뒤
   con = duckdb.connect("md:")
   con.sql("ATTACH 'md:_share/hyfin_share/5bcd81c8-b423-4817-b821-42f561dc1a93' AS hyfin")
   con.sql("SELECT * FROM hyfin.raw.f2_universe LIMIT 5").show()
   ```
   웹 UI에서는 왼쪽 Shared with me 목록에서 붙이거나 위 `ATTACH` 문을 그대로 실행한다.

   **저장소 코드에서는** `src/common/db.py` 를 쓰면 `.env` 읽기와 공유 붙이기를 알아서 한다 (A·B 공통):
   ```python
   from src.common.db import connect, sql
   con = connect()                                   # 분석은 hyfin.proc.* 만
   df = sql("SELECT date, code, rv5 FROM hyfin.proc.rv_daily WHERE code = ?", ["005930"])
   ```

### 연구자 A (관리자, 이미 완료)
- DB 생성·업로드: `py scripts/load_motherduck.py` (토큰은 `.env`)
- 공유 생성 (완료):
  ```sql
  CREATE SHARE hyfin_share FROM hyfin (ACCESS RESTRICTED, VISIBILITY HIDDEN, UPDATE AUTOMATIC);
  ```
- B 초대: 웹 UI → Settings → **Members → Invite** (B 이메일)
- B 가입 후 권한 부여 (사용자 이름은 Settings → Members 에서 확인, `@` 가 있으면 큰따옴표):
  ```sql
  GRANT READ ON SHARE hyfin_share TO USER "<B의 사용자 이름>";
  ```
- 공유 URL: `md:_share/hyfin_share/5bcd81c8-b423-4817-b821-42f561dc1a93` (다시 보기: `LIST SHARES;`)

## 3. 테이블 규칙

- **분석 코드는 `proc.*` 만 읽는다.** `raw.*` 는 백업이고, 읽는 것은 `src/common/processing/` 의 전처리 코드뿐이다.
- **`raw.*` 는 원본 그대로**. 열 이름·값·단위를 바꾸지 않았다. 고칠 일이 있으면 `raw` 를 건드리지 말고 전처리 코드를 고쳐 `proc` 을 다시 만든다. `proc` 의 뷰가 `raw` 를 읽으므로 `raw` 를 지우면 안 된다.
- **`proc.*` 이름 규칙**: `<내용>_<단위>` 소문자 + 밑줄. 현재 테이블 9개: `calendar`, `market_daily`, `kospi200_membership`, `intraday_5m`, `rv_daily`, `rv_daily_us`, `stock_daily`, `etf_daily`.
  뷰 6개: `futures_all`, `options_all`, `minute_kr`, `minute_idx`, `minute_us`, `etf_aum_all`. 열 설명은 [processing.md](../src/common/processing/processing.md).
  - 만든 사람·원천·코드를 테이블 주석으로 남긴다:
    ```sql
    COMMENT ON TABLE proc.rv_daily IS 'A, 2026-11-20, src=raw.minute_kr(KRX 1m adj), code=src/common/processing/make_realized_volatility.py';
    ```
  - 만드는 코드는 저장소 `src/` 에 커밋한다. DB의 표는 언제든 코드로 다시 만들 수 있어야 한다.
- **B가 만든 결과**: 공유 DB는 B가 쓸 수 없다. B는 자기 DB(`hyfin_b`)를 만들어 쓰고, 공유를 만들어 A에게 URL을 보낸다 (2절과 같은 방법, 방향만 반대). 둘 다 쓸 결과는 A가 `hyfin.proc` 로 옮긴다.

## 4. 테이블 목록

### 4.1 FnGuide·KRX 등 (F1~F17) — 파일 1개 = 테이블 1개

열·키·단위·주의사항은 [data_spec.md](data_spec.md) 2~5절에 그대로 적용된다. 테이블 이름 = 파일 이름 소문자.

| 테이블 | 내용 | 키 |
|---|---|---|
| `raw.f1_stock_daily` | 주식 일별 시세·시총·지분율 (213종) | date, code |
| `raw.f1_beta_annual` | 베타 (연 1값) | code, year |
| `raw.f2_universe` | 주식 213 + 상품 288 목록·그룹 | code |
| `raw.f3_stock_investor` | 투자자별 순매수 | date, code |
| `raw.f4_stock_credit_short` | 신용·공매도·대차 | date, code |
| `raw.f5_earnings_annual` | 잠정치발표일·영업이익 | code, fiscal_year |
| `raw.f5_consensus_daily` | FY2026 영업이익 컨센서스 | date, code |
| `raw.f6_fin_annual` | 자산·매출·부채비율 | code, fiscal_year |
| `raw.f6_valuation_daily` | PBR·PER | date, code |
| `raw.f7_etf_daily` | ETF·ETN 일별 시세 | date, code |
| `raw.f8_etf_info` | ETF·ETN 기본정보·분류 | code |
| `raw.f9_etf_investor` | ETF·ETN 투자자별 순매수 | date, code |
| `raw.f10_etf_aum` | 전체 ETF 순자산 (2025-01~) | date, code |
| `raw.f11_index` | 국내 지수 7개 | date, code |
| `raw.f12_futures_front` / `raw.f12_futures_all` | 주식선물 최근월물 / 전 월물 | date, underlying / contract |
| `raw.f13_option_atm_iv` / `raw.f13_option_all` | 주식옵션 ATM IV / 전 종목 (2025-01~) | date, underlying / contract |
| `raw.f14_global` | 해외 지수·종목 | date, code |
| `raw.f16_macro` | 거시 8개 변수 (열 이름 한글: `date, 변수, 값, 단위, 출처`) | date, 변수 |
| `raw.f17_kospi200_members` / `raw.f17_kospi200_changes` | KOSPI200 구성종목 / 편입·편출 (열 이름 한글) | 기준일, 종목코드 |
| `raw.fnguide_catalog` | 위 테이블별 행 수·기간 요약 | — |

### 4.2 분봉 (키움, M1~M7) — 종류별로 합친 3개 테이블

파일 509개를 종류별로 합쳤다. **원래 열·값은 그대로**, 파일 이름에서 뽑은 열만 앞에 붙였다. 원래 열 설명은 [data_spec.md](data_spec.md) 6절.

| 테이블 | 원래 파일 | 붙인 열 | 원래 열 |
|---|---|---|---|
| `raw.minute_kr` | 국내 주식·ETF·ETN `<code>_<venue>_<freq>[_raw]` 497개 (32,900,786행) | `file, code, venue(KRX/NXT/ALL), freq(1m/5m), price(adj/raw)` | `ts, date, hhmm, open, high, low, close, volume, acc_volume, session` |
| `raw.minute_idx` | 지수 `IDX<code>_1m` 4개 (397,731행) | `file, code(001/013/101/201), freq` | 위와 같음 (가격 포인트, 거래량 천주) |
| `raw.minute_us` | 미국 `US_<ticker>_1m` 8개 (1,504,367행) | `file, ticker, freq` | `bus_dt, ts_et, hhmm_raw, open, high, low, close, volume, adj_flag, session` |
| `raw.minute_file_list` | `_file_list.csv` | — | 파일별 종목명·그룹(M1~M7)·기간·행 수 |
| `raw.minute_catalog` | `_catalog.csv` | — | 파일별 품질 지표 |
| `raw.minute_market_gaps` | `_market_gaps.csv` | — | 시장 전체 공백 10일 |

부속 CSV 3개는 모든 열을 문자열(VARCHAR)로 올렸다. 숫자로 쓸 때 `CAST(rows AS BIGINT)` 처럼 바꾼다.

**한 파일만 고를 때는 반드시 `code, venue, freq, price` 를 모두 지정한다.** 안 그러면 KRX·ALL·NXT 또는 수정·원주가가 섞여 같은 시각이 중복된다.

## 5. 쿼리 예

```sql
-- 분봉: 삼성전자 KRX 1분 수정주가 정규장
SELECT ts, close, volume
FROM hyfin.raw.minute_kr
WHERE code = '005930' AND venue = 'KRX' AND freq = '1m' AND price = 'adj'
  AND session = 'regular' AND date BETWEEN '20260501' AND '20260630'
ORDER BY ts;

-- 분봉 → 일별 마지막 종가 (분봉 date 는 'YYYYMMDD' 문자열, F1 date 는 DATE)
WITH d AS (
  SELECT strptime(date, '%Y%m%d')::DATE AS date, code, arg_max(close, ts) AS close_min
  FROM hyfin.raw.minute_kr
  WHERE venue = 'KRX' AND freq = '1m' AND price = 'adj' AND code IN ('005930', '000660')
    AND session IN ('regular', 'close')              -- 시간외(after) 빼야 종가와 맞음
  GROUP BY ALL
)
SELECT d.*, f.close AS close_f1
FROM d JOIN hyfin.raw.f1_stock_daily f
  ON f.date::DATE = d.date AND f.code = 'A' || d.code   -- F 테이블 date는 TIMESTAMP, 코드는 'A'+6자리
ORDER BY d.code, d.date;

-- 대조군 그룹(M5) 종목 목록
SELECT DISTINCT code, name FROM hyfin.raw.minute_file_list WHERE "group" LIKE 'M5%';
```

> 이 쿼리로 확인한 결과: 2종목 × 261일 모두 분봉 마지막 종가 = F1 종가.
> F1~F17 의 `date` 는 parquet 그대로 `TIMESTAMP_NS` 이다. 분봉 `date` 는 `'YYYYMMDD'` 문자열. 합칠 때 둘 다 `DATE` 로 맞춘다.
> `close` 같은 열 이름을 별칭으로 쓸 때는 `AS` 를 꼭 붙인다 (`close` 는 SQL 예약어).

```python
# Python: 결과를 pandas 로
df = con.sql("SELECT * FROM hyfin.raw.f11_index WHERE code = 'I.001'").df()

# proc 테이블 만들기 (A만, hyfin 에 직접)
con.sql("CREATE OR REPLACE TABLE hyfin.proc.rv_daily AS SELECT ...")
# 또는 pandas DataFrame 을 그대로
con.sql("CREATE OR REPLACE TABLE hyfin.proc.my_table AS SELECT * FROM df")
```

계산 시간 절약: 분봉 전체(약 3,500만 행)를 매번 pandas 로 받지 말고, 집계는 SQL로 하고 결과만 받는다. 자주 쓰는 중간 결과는 `proc` 에 저장한다.

## 6. 데이터 갱신

1. 로컬 `data/interim/` 의 파일을 새로 만든다 (data_spec.md 0절, 분봉은 6.8절).
2. A가 `py scripts/load_motherduck.py` (또는 `fnguide` / `minute` 만) 실행. `raw` 테이블을 통째로 다시 만들고 로컬 파일과 행 수를 대조한다. `MISMATCH` 가 나오면 종료 코드 1.
3. 공유는 `UPDATE AUTOMATIC` 이라 B에게 **5분 안에** 반영된다. 바로 보려면 B가 `REFRESH DATABASES;`.
4. 공유 URL을 유지하려면 공유를 `CREATE OR REPLACE SHARE` 로 다시 만들지 않는다 (URL이 바뀜). 설정 변경은 `ALTER SHARE`.

## 7. 주의

- **라이선스**: FnGuide DataGuide 데이터는 계약자 내부 연구용. 공유는 연구팀 안으로만 하고, 공유 설정을 `UNRESTRICTED` 로 바꾸거나 URL을 밖에 올리지 않는다.
- **토큰** 은 비밀번호와 같다. 채팅·코드·커밋에 넣지 않는다. 노출되면 Settings → Access Tokens 에서 지우고 새로 만든다.
- **백업은 따로**: MotherDuck 스냅샷은 1일만 보관된다. 원본 zip(특히 **분봉 `minute_2610.zip` 은 다시 받을 수 없음**)은 외장 디스크·드라이브에 별도 보관한다. DB는 사본이다.
- **계산 한도** 월 10시간: 큰 쿼리를 반복하지 말고 결과를 `proc` 에 저장해 재사용한다. 사용량은 웹 UI Settings → Usage.
- `raw` 를 실수로 덮어쓰지 않도록, A도 분석 코드에서는 `raw` 에 쓰기 쿼리를 쓰지 않는다 (`load_motherduck.py` 만 씀).
