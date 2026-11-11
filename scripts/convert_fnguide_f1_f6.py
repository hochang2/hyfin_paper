"""FnGuide DataGuide F1~F6 xlsx(wide) -> long CSV + parquet.

F7~F17(`F7_F17.zip`)과 같은 규칙으로 맞춘다.
  - long 형식: 일간 = (date, code), 결산 = (code, fiscal_year)
  - CSV는 UTF-8 BOM, 같은 이름의 parquet 동시 저장
  - 값이 없으면 NaN (0으로 채우지 않음), 값이 전부 빈 행은 버림
  - 금액은 모두 원으로 환산 (원본 단위는 docs/data_spec.md 참고)

사용:
  py -I scripts/convert_fnguide_f1_f6.py data/F1~F6.zip data/interim/fnguide
원본 zip은 읽기만 한다.
"""
from __future__ import annotations

import datetime as dt
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

F12 = "F1~F6/F1, F2.xlsx"

# 아이템코드 -> (열 이름, 원 환산 배수)
ITEMS = {
    # F1 일간
    "S410000650": ("open", 1), "S410000660": ("high", 1), "S410000670": ("low", 1),
    "S410000700": ("close", 1), "S41000080F": ("volume", 1), "S410000900": ("value", 1),
    "S410001250": ("mktcap", 1e6), "S420003800": ("shares", 1), "S41000170F": ("ret_pct", 1),
    "S420005150": ("float_pct", 1), "S410001300": ("foreign_pct", 1),
    # F1 결산
    "6009911035": ("beta", 1),
    # F3 (수량 주, 대금 만원 -> 원)
    **{f"CI20{k}3010": (f"{n}_qty", 1) for k, n in [
        ("08", "indiv"), ("11", "foreign"), ("10", "foreign_other"), ("00", "inst"),
        ("01", "fin_inv"), ("02", "insur"), ("03", "trust"), ("04", "bank"),
        ("06", "pension"), ("07", "corp_other"), ("31", "pef")]},
    **{f"CI20{k}3020": (f"{n}_amt", 1e4) for k, n in [
        ("08", "indiv"), ("11", "foreign"), ("10", "foreign_other"), ("00", "inst"),
        ("01", "fin_inv"), ("02", "insur"), ("03", "trust"), ("04", "bank"),
        ("06", "pension"), ("07", "corp_other"), ("31", "pef")]},
    # F4
    "S410011000": ("credit_qty", 1), "S410011400": ("credit_amt", 1e6),
    "S410000910": ("short_qty", 1), "S410000920": ("short_amt", 1),
    "S410000992": ("short_bal_qty", 1), "S410000994": ("short_bal_amt", 1e6),
    "S410100400": ("lend_bal_qty", 1),
    # F5
    "FP56000500": ("prelim_date", None), "FP56000900": ("report_date", None),
    "FP56000250": ("op_income", 1e3), "FM30011250": ("op_income_cons_fy2026", 1e8),
    # F6
    "6000901001": ("assets", 1e3), "6000904001": ("sales", 1e3),
    "6000906001": ("op_income", 1e3), "6001130000": ("debt_ratio_pct", 1),
    "S420006202": ("pbr", 1), "S420006002": ("per", 1),
}

# 출력 파일: (원본 xlsx, 시트, 주기, 열 목록)
OUTPUTS = {
    "F1_stock_daily": (F12, "주식_213", "daily",
                       ["open", "high", "low", "close", "volume", "value", "mktcap", "shares",
                        "ret_pct", "float_pct", "foreign_pct"]),
    "F1_beta_annual": (F12, "주식_213", "annual", ["beta"]),
    "F3_stock_investor": ("F1~F6/F3.xlsx", None, "daily",
                          [f"{n}_{t}" for n in ["indiv", "foreign", "foreign_other", "inst",
                                                "fin_inv", "insur", "trust", "bank", "pension",
                                                "corp_other", "pef"] for t in ["qty", "amt"]]),
    "F4_stock_credit_short": ("F1~F6/F4.xlsx", None, "daily",
                              ["credit_qty", "credit_amt", "short_qty", "short_amt",
                               "short_bal_qty", "short_bal_amt", "lend_bal_qty"]),
    "F5_earnings_annual": ("F1~F6/F5.xlsx", None, "annual",
                           ["prelim_date", "report_date", "op_income"]),
    "F5_consensus_daily": ("F1~F6/F5.xlsx", None, "daily", ["op_income_cons_fy2026"]),
    "F6_fin_annual": ("F1~F6/F6.xlsx", None, "annual",
                      ["assets", "sales", "op_income", "debt_ratio_pct"]),
    "F6_valuation_daily": ("F1~F6/F6.xlsx", None, "daily", ["pbr", "per"]),
}


def read_wide(z: zipfile.ZipFile, member: str, sheet: str | None):
    """DataGuide 시트 -> {열이름: DataFrame(code x date)}, {code: name}."""
    wb = openpyxl.load_workbook(io.BytesIO(z.read(member)), read_only=True)
    ws = wb[sheet] if sheet else wb.worksheets[0]
    rows = ws.iter_rows(values_only=True)
    for _ in range(8):
        next(rows)
    hdr = next(rows)
    ndates = sum(isinstance(d, dt.datetime) for d in hdr[6:])
    dates = pd.DatetimeIndex(hdr[6:6 + ndates])
    names, buf = {}, {}
    for r in rows:
        if not r[0]:
            continue
        code, item = r[0], str(r[3])  # 숫자형 아이템코드(6009911035 등)는 int로 읽힘
        if item not in ITEMS:
            raise KeyError(f"모르는 아이템 {item} ({r[4]}) in {member}")
        names[code] = r[1]
        col, mult = ITEMS[item]
        vals = [np.nan if v in (None, "") else v for v in r[6:6 + ndates]]
        if mult is None:  # YYYYMMDD 날짜
            vals = pd.to_datetime(pd.Series(vals, dtype="float64").astype("Int64").astype("string"),
                                  format="%Y%m%d", errors="raise")
        else:
            vals = pd.Series(vals, dtype="float64") * mult
        buf.setdefault(col, {})[code] = vals.to_numpy()
    wb.close()
    wide = {c: pd.DataFrame(d, index=dates).T for c, d in buf.items()}
    return wide, names


def to_daily(wide, names, cols):
    parts = []
    for c in cols:
        s = wide[c].stack(future_stack=True)
        s.name = c
        parts.append(s)
    df = pd.concat(parts, axis=1)
    df.index.names = ["code", "date"]
    df = df.dropna(how="all").reset_index()
    df.insert(2, "name", df["code"].map(names))
    return df.sort_values(["code", "date"])[["date", "code", "name", *cols]].reset_index(drop=True)


def to_annual(wide, names, cols, log):
    """결산 아이템: 연도 Y 구간에 찍힌 값을 FY Y 한 행으로. 연도 안에서 값이 바뀌면 경고."""
    parts = []
    for c in cols:
        w = wide[c]
        g = w.T.groupby(w.columns.year)
        first = g.first().T  # code x year (첫 non-NaN)
        nuniq = g.nunique().T
        bad = (nuniq > 1).stack()
        if bad.any():
            log.append(f"  [경고] {c}: 연도 안에서 값이 바뀐 (종목, 연도) {int(bad.sum())}건 "
                       f"예: {list(bad[bad].index[:5])}")
        s = first.stack()
        s.name = c
        parts.append(s)
    df = pd.concat(parts, axis=1)
    df.index.names = ["code", "fiscal_year"]
    df = df.dropna(how="all").reset_index()
    df.insert(1, "name", df["code"].map(names))
    df["fiscal_year"] = df["fiscal_year"].astype("int64")
    return df.sort_values(["code", "fiscal_year"]).reset_index(drop=True)


def universe(z: zipfile.ZipFile) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(z.read(F12)), sheet_name="전체_501", dtype=str)
    df.columns = ["code", "name", "kiwoom_code", "kind", "group", "list_date"]
    df["list_date"] = pd.to_datetime(df["list_date"], format="%Y%m%d")
    df.insert(4, "core33", df["group"].isin(["P1_단일종목상품", "P2_대체상품핵심"]))
    return df


def save(df: pd.DataFrame, out: Path, name: str, catalog: list):
    df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    df.to_parquet(out / f"{name}.parquet", index=False)
    fmt = {"date": lambda v: v.strftime("%Y-%m-%d"), "fiscal_year": lambda v: f"FY{v}", "year": str}
    tcol = next((c for c in fmt if c in df), None)
    catalog.append({
        "file": name, "rows": len(df), "cols": df.shape[1], "keys": float(df["code"].nunique()),
        "start": "" if tcol is None else fmt[tcol](df[tcol].min()),
        "end": "" if tcol is None else fmt[tcol](df[tcol].max()),
    })


def main(zip_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(zip_path)
    log, catalog, cache = [], [], {}
    save(universe(z), out, "F2_universe", catalog)
    for name, (member, sheet, freq, cols) in OUTPUTS.items():
        key = (member, sheet)
        if key not in cache:
            cache.clear()
            cache[key] = read_wide(z, member, sheet)
        wide, names = cache[key]
        log.append(name)
        df = to_daily(wide, names, cols) if freq == "daily" else to_annual(wide, names, cols, log)
        if name == "F1_beta_annual":
            # 베타는 2026 구간에도 값이 있어 '회계연도'가 아니라 '값이 찍힌 연도'로만 둔다
            df = df.rename(columns={"fiscal_year": "year"})
        save(df, out, name, catalog)
        log.append(f"  rows={len(df):,} codes={df['code'].nunique()}")
    cat = pd.DataFrame(catalog)
    cpath = out / "_catalog.csv"
    if cpath.exists():  # F7~F17 카탈로그와 합침
        old = pd.read_csv(cpath, encoding="utf-8-sig")
        old = old[~old["file"].str.match(r"F[1-6]_")]
        cat = pd.concat([cat, old], ignore_index=True)
    cat.to_csv(cpath, index=False, encoding="utf-8-sig")
    (out / "_convert_F1_F6.log").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
