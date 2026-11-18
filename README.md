# 단일종목 레버리지 상품 도입·규제 연구

2026년 한국 단일종목 레버리지 ETF·ETN의 도입과 개인투자자 접근 규제가
**삼성전자(005930)·SK하이닉스(000660)** 의 변동성과 레버리지 투자 수요에 미친 영향을 분석한다.

| RQ | 질문 | 담당 | 기준모형 |
|---|---|---|---|
| [RQ1](src/rq1/README.md) | 상품 도입 후 기초자산 변동성이 달라졌는가? | 연구자 B | GARCH 계열, 이벤트 전후 회귀 |
| [RQ2](src/rq2/README.md) | 규제 후 변동성이 반사실적 경로보다 낮아졌는가? | 연구자 A | BSTS / CausalImpact |
| [RQ3](src/rq3/README.md) | ETF 정보가 미래 변동성 예측력을 높이는가? | 연구자 B | HAR-RV vs HAR-RV+ETF |
| [RQ4](src/rq4/README.md) | 규제 후 비규제 레버리지 상품으로 거래활동이 이동했는가? | 연구자 A | DiD + 이벤트 스터디 |

주요 사건일: 2026-05-27 상장 / 2026-07-16 규제 발표 / 시행일 **미확정** (→ [configs/events.yaml](configs/events.yaml))

## 현재 상태 (2026-11-11)

- **데이터 수집 완료, 분석 코드는 아직 없음.**
- FnGuide·KRX 일별 F1~F17 (F15 제외, 2015-01 ~ 2026-10-30)과 키움 분봉 M1~M7 (국내 2025-10-01 ~ 2026-10-30)을 정리해 MotherDuck `hyfin` DB의 `raw` 스키마에 올렸다.
- 데이터 내용과 주의사항은 [docs/data_spec.md](docs/data_spec.md), DB 접속과 사용 규칙은 [docs/database_guide.md](docs/database_guide.md)를 본다.

## 구조

```text
configs/        사건일·universe·분석 설정 (YAML)
data/           로컬 보관용 원본 zip + 정리본 (Git 제외). 분석은 DuckDB로 함
docs/           data_spec.md (데이터 명세), database_guide.md (DB 사용법)
scripts/        임시·일회성 코드 (변환, DB 적재 등)
src/
  common/       RQ 공통 코드: db.py (DB 연결), processing/ (raw → proc 전처리)
  rq1/ ~ rq4/   RQ별 README, core/ (공용 코드), phase1/·phase2/ (연구별 코드·results), tests/
```

- 분석 데이터는 **MotherDuck `hyfin` DB가 기준**이다. `raw` = 원본 백업, `proc` = 분석에 쓰는 것 전부. **분석 코드는 `proc` 만 읽는다.** `proc` 테이블은 언제든 `src/` 코드로 다시 만들 수 있어야 한다.
- import는 저장소 루트 기준으로 `from src.common import ...` 처럼 쓴다.

## 설치

```bash
py -m venv .venv
.venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

저장소 루트에 `.env` 파일을 만들고 MotherDuck 토큰을 넣는다 (토큰 받는 법은 [docs/database_guide.md](docs/database_guide.md) 2절).

```
motherduck_token=<내 토큰>
```

Windows 사용자 폴더 경로에 한글이 있으면(예: `C:\Users\홍길동`) 접속이 실패한다. 영문 폴더를 하나 만들고 `.env`에 한 줄 더 넣는다.

```
duckdb_home=C:/duckdb_ext
```

## 데이터 정책

- 키움·FnGuide 원자료, 토큰, 유료 데이터는 커밋하지 않는다. FnGuide 데이터는 팀 내부 연구용이다.
- **`data/minute_2610.zip`은 다시 받을 수 없는 유일본**이다 (키움 서버는 최근 12개월만 보관). 외장 디스크 등에 따로 백업한다.

## 협업

- 연구자 A: `src/rq2/`, `src/rq4/` · 연구자 B: `src/rq1/`, `src/rq3/`
- `src/common/`은 둘이 합의한 뒤 수정한다.
- DB 쓰기는 A만 한다. B의 결과는 B의 DB(`hyfin_b`)에서 공유한다 ([database_guide.md](docs/database_guide.md) 3절).
- 브랜치 예: `feat/rv-pipeline`, `feat/rq2-bsts`, `feat/rq4-did`
- 실험은 `baseline` / `extension`으로 나누고, 시계열은 날짜순으로만 분할한다.
