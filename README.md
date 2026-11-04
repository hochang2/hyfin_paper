# 단일종목 레버리지 상품 도입·규제 연구

2026년 한국 단일종목 레버리지 ETF·ETN의 도입과 개인투자자 접근 규제가
**삼성전자(005930)·SK하이닉스(000660)** 의 변동성과 레버리지 투자 수요에 미친 영향을 분석한다.

| RQ | 질문 | 담당 | 기준모형 |
|---|---|---|---|
| RQ1 | 상품 도입 후 기초자산 변동성이 달라졌는가? | 연구자 B | GARCH 계열, 이벤트 전후 회귀 |
| RQ2 | 규제 후 변동성이 반사실적 경로보다 낮아졌는가? | 연구자 A | BSTS / CausalImpact |
| RQ3 | ETF 정보가 미래 변동성 예측력을 높이는가? | 연구자 B | HAR-RV vs HAR-RV+ETF |
| RQ4 | 규제 후 비규제 레버리지 상품으로 거래활동이 이동했는가? | 연구자 A | DiD + 이벤트 스터디 |

주요 사건일: 2026-05-27 상장 / 2026-07-16 규제 발표 / 시행일 **미확정** (→ [configs/events.yaml](configs/events.yaml))

## 현재 상태 (2026-11-04)

- 저장소 스켈레톤만 존재. **분석 코드·데이터 파이프라인은 아직 구현되지 않음.**
- 키움 분봉(2025-10~2026-10)은 로컬 보유로 보고됐으나 범위·스키마 미검증 → [docs/data_inventory.md](docs/data_inventory.md)
- FnGuide(DataGuide)는 조회 가능 여부만 확인, 다운로드 전.

## 문서

- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) — 프로젝트 근거 문서 (연구 설계·데이터 원칙·단계별 계획)
- [docs/source/DATA_SPEC.md](docs/source/DATA_SPEC.md) — 2026-10-23 데이터 명세 원문 (수정하지 않음)
- [docs/decisions.md](docs/decisions.md) — 의사결정 기록
- [docs/data_inventory.md](docs/data_inventory.md) — 실제 보유 데이터 현황

## 구조

```text
configs/              사건일·universe·분석 설정 (YAML)
docs/                 설계·데이터 문서, source/ 에 원문 보존
data/                 로컬 데이터 (raw/interim/processed/external 은 Git 제외)
src/leverage_study/   공통 패키지: ingestion → processing → features → evaluation, rq1~rq4
scripts/              CLI 진입점
notebooks/            탐색용 (정식 로직은 src 에)
results/              표·그림 (대용량 제외)
tests/
```

## 설치

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"
cp .env.example .env          # API 키 입력
```

## 데이터 정책

키움·FnGuide 원자료, API 키, 유료 데이터는 커밋하지 않는다. 자세한 내용은 [data/README.md](data/README.md).

## 협업

- 연구자 A: `rq2/`, `rq4/` · 연구자 B: `rq1/`, `rq3/`
- 공통 `ingestion/`, `processing/`, `features/` 는 합의 후 수정
- 브랜치 예: `feat/rv-pipeline`, `feat/rq2-bsts`, `feat/rq4-did`
- 실험은 `baseline` / `extension` 으로 구분하고, 시계열은 날짜순 분할만 사용
