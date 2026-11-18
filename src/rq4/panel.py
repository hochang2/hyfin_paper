"""분석 패널 구성 (product_universe 결과를 받아 모든 모듈이 같은 패널을 쓴다).

build_panel(...) → 상품×일 DataFrame
  열: date, code, name, grp4(A/B/C/D/E/SP), family(idx/semi/None), lev2(1=2X 처치, 0=1X 대조),
      value, volume, log_value, asinh_value, log_volume, turnover, flow_aum, period, post, trans, base, week
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.rq4.common import CFG, TL
from src.rq4.data_pipeline import load_etf

G = CFG["groups"]
P = {k: [pd.Timestamp(v[0]), pd.Timestamp(v[1])] if isinstance(v, list) else pd.Timestamp(v)
     for k, v in CFG["periods"].items()}


def code_group() -> dict:
    out = {}
    for g, codes in G.items():
        for c in codes:
            out.setdefault(str(c), g)
    return out


def period_of(d: pd.Series) -> pd.Series:
    s = pd.Series("other", index=d.index)
    s[(d >= P["long_pre"][0]) & (d <= P["long_pre"][1])] = "long_pre"
    s[(d >= P["listing_base"][0]) & (d <= P["listing_base"][1])] = "base"
    s[(d >= P["transition"][0]) & (d <= P["transition"][1])] = "trans"
    s[(d >= P["post"][0]) & (d <= P["post"][1])] = "post"
    s[d < P["long_pre"][0]] = "early"
    return s


def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["code", "date"]).copy()
    v = df["value"].astype(float)
    df["log_value"] = np.where(v > 0, np.log(v.where(v > 0)), np.nan)
    df["asinh_value"] = np.arcsinh(v / 1e6)          # 백만원 단위 asinh
    vol = df["volume"].astype(float)
    df["log_volume"] = np.where(vol > 0, np.log(vol.where(vol > 0)), np.nan)
    aum = df["aum_won"].where(df["aum_won"] > 0, df["aum_calc"])
    lag_aum = aum.groupby(df["code"]).shift(1)
    df["turnover"] = np.log(v.where(v > 0) / lag_aum.where(lag_aum > 0))
    df["flow_aum"] = df["flow_won"] / lag_aum.where(lag_aum > 0)
    df["indiv_net_intensity"] = df["indiv_amt"] / v.where(v > 0)
    return df


def build_panel(start="2025-10-01", end="2026-10-30", groups=("B_index_2x", "C_semi_2x", "E_index_1x", "E_semi_1x"),
                balanced=True, min_value=None, extra_codes=(), family_override=None) -> pd.DataFrame:
    e = load_etf()
    cg = code_group()
    codes = sorted({c for g in groups for c in G[g]} | set(extra_codes))
    d = e[e["code"].isin(codes)].copy()
    d = add_outcomes(d)
    d = d[(d["date"] >= pd.Timestamp(start)) & (d["date"] <= pd.Timestamp(end))]
    d["grp4"] = d["code"].map(cg).str[0]
    d["sub"] = d["code"].map(cg)
    fam = {}
    for f, gl in CFG["family"].items():
        for g in gl:
            for c in G[g]:
                fam[str(c)] = f
    if family_override:
        fam.update(family_override)
    d["family"] = d["code"].map(fam)
    d["lev2"] = d["sub"].str[0].isin(["B", "C", "D", "A"]).astype(int)
    if balanced:
        days = d["date"].nunique()
        ok = d[d["value"] > 0].groupby("code")["date"].nunique()
        d = d[d["code"].isin(ok[ok == days].index)]
    if min_value is not None:
        pre = d[(d["date"] >= P["long_pre"][0]) & (d["date"] <= P["long_pre"][1])]
        med = pre.groupby("code")["value"].median()
        d = d[d["code"].isin(med[med >= min_value].index)]
    d["period"] = period_of(d["date"])
    d["post"] = (d["period"] == "post").astype(int)
    d["trans"] = (d["period"] == "trans").astype(int)
    d["base"] = (d["period"] == "base").astype(int)
    d["fd"] = d["family"].astype(str) + "_" + d["date"].dt.strftime("%Y%m%d")
    return d.reset_index(drop=True)


def main_panel(**kw) -> pd.DataFrame:
    """주 분석 패널: listing_base ~ post (05-27 ~ 10-30), 고정 코호트, 활동 하한."""
    kw.setdefault("min_value", CFG["cohort"]["min_pre_median_value_won"])
    full = build_panel(start=CFG["cohort"]["balanced_window"][0], end=CFG["cohort"]["balanced_window"][1], **kw)
    return full


def window(df, start=None, end=None):
    start = start or P["listing_base"][0]
    end = end or P["post"][1]
    return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].copy()
