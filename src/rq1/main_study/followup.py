"""RQ1 두번째 연구 (사후 탐색, 2026-11-18). 1차 주 결론('상장 효과의 증거 없음')을 바꾸지 않는다.

(1) 마감창 − 장중 DiD: y = ln RV(14:50~15:20) − ln RV(09:30~14:50). 같은 종목 안의 차이라 종목 전체 변동성 충격이
    상쇄되고, 리밸런싱이 몰리는 마감 시간대의 상대 변동성만 남는다. 추론은 1차와 같은 donor 순열.
(2) 일중 모멘텀 기울기: r_last(14:50→정규장 종가) = a + b·r_rest(전일 종가→14:50) + c·r_rest·Post + d·Post.
    레버리지 ETF 는 당일 수익률 방향으로 마감에 매매하므로 상장 후 c > 0 을 예상. donor 대비 순열.
(3) SK스퀘어(402340) 음성통제: SK하이닉스에 없는 같은 기업 대조의 근사 (지주사라 불완전).
(4) 상장 전 가짜 처치 창(35거래일)으로 결과변수별 최소 검출효과(MDE) 비교.

`proc.intraday_5m` 칸 시각은 시작 시각이고 ret 는 직전 칸 종가 대비 log 수익률이다. 마지막 칸은 15:15~15:20.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.rq1.main_study import data as D
from src.rq1.main_study import did as M
from src.rq2.core import data as rq2data

SKSQ = "402340"
CLOSE_WIN, MID_WIN = (1450, 1515), (930, 1445)


def window_raw() -> pd.DataFrame:
    """종목·일별 마감창 RV, 장중창 RV, 14:45 칸 종가(=14:50 가격), 정규장 종가, 종가(단일가 포함)."""
    df = rq2data._cached("rq1_window_raw", f"""
        WITH w AS (
          SELECT date, code,
                 sum(CASE WHEN hhmm BETWEEN {CLOSE_WIN[0]} AND {CLOSE_WIN[1]} THEN ret*ret END) AS rvc,
                 sum(CASE WHEN hhmm BETWEEN {MID_WIN[0]} AND {MID_WIN[1]} THEN ret*ret END) AS rvi,
                 max(CASE WHEN hhmm = 1445 THEN close END) AS c1445
          FROM hyfin.proc.intraday_5m GROUP BY date, code)
        SELECT w.*, r.p_close_reg, r.p_close
        FROM w LEFT JOIN hyfin.proc.rv_daily r USING (date, code)""")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def roles() -> dict[str, str]:
    r = D.role_of()
    r[SKSQ] = "neg_sksq"
    return r


def _base() -> pd.DataFrame:
    """날짜별 post, rel, bin (처리 종목 행에서 가져옴). 표본은 1차와 같다."""
    p = D.panel_long("rv5")
    return p[p.code == "005930"][["date", "post", "rel", "bin"]]


def panel(kind: str = "diff") -> pd.DataFrame:
    """1차 패널과 같은 모양 (date, code, y, role, post, rel, bin).
    kind: diff = ln rvc − ln rvi, rvc = ln rvc, rvi = ln rvi, rv5 = 1차 결과변수 (+ SK스퀘어)."""
    rl = roles()
    if kind == "rv5":
        lv = rq2data.load_panel("rv5")
        y = lv[[c for c in rl if c in lv.columns]].stack().rename("y").reset_index()
        y.columns = ["date", "code", "y"]
    else:
        w = window_raw()
        w = w[w.code.isin(rl)].copy()
        rvc, rvi = w.rvc.where(w.rvc > 0), w.rvi.where(w.rvi > 0)
        w["y"] = {"diff": np.log(rvc) - np.log(rvi), "rvc": np.log(rvc), "rvi": np.log(rvi)}[kind]
        y = w[["date", "code", "y"]]
    out = _base().merge(y.dropna(), on="date", how="inner")
    out["role"] = out.code.map(rl)
    return out.sort_values(["code", "date"]).reset_index(drop=True)


def did_family(pn: pd.DataFrame, donors: list[str], seed: int, n_pf: int = 1000) -> tuple[pd.DataFrame, pd.Series]:
    """처리 2종목 + 음성통제(005935, 반도체 7 포트폴리오, SK스퀘어)의 DiD β 와 donor 순열 p."""
    null = M.placebo_space(pn, donors)["D"]
    rows = []
    for c in D.CFG["treated"] + [D.CFG["negative_control"], SKSQ]:
        b = M.did_beta(pn, c, donors)
        rows.append({"unit": c, "beta": b, "perm_p": M.perm_p(b, null), "null": f"donor 개별 {len(null)}"})
    semis = list(D.CFG["semis_negative_control"])
    pp = M.portfolio_panel(pn, semis, "SEMIS7")
    b = M.did_beta(pp, "SEMIS7", donors)
    null_pf = M.placebo_portfolios(pn, donors, k=len(semis), n_draw=n_pf, seed=seed)
    rows.append({"unit": "SEMIS7", "beta": b, "perm_p": M.perm_p(b, null_pf), "null": f"donor 무작위 7종목 {n_pf}"})
    out = pd.DataFrame(rows)
    out["mde_log"] = M.mde(null)
    out["placebo_sd"] = float(null.std(ddof=1))
    return out, null


def pseudo_mde(pn: pd.DataFrame, donors: list[str], length: int = 35, starts=range(60, 121, 10)) -> pd.DataFrame:
    """상장 전 자료만으로 가짜 처치 창을 두고 donor placebo sd → MDE. 결과변수별 검정력 비교용."""
    pre = pn[pn.post == 0]
    days = np.sort(pre.date.unique())
    rows = []
    for s in starts:
        if s + length > len(days):
            continue
        win = days[s:s + length]
        sub = pre[pre.date <= win[-1]].copy()
        sub["post"] = sub.date.isin(win).astype(int)
        null = M.placebo_space(sub, donors)["D"]
        rows.append({"fake_start": pd.Timestamp(win[0]).date(), "fake_end": pd.Timestamp(win[-1]).date(),
                     "n_days": len(days[:s + length]), "placebo_sd": float(null.std(ddof=1)), "mde_log": M.mde(null)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 일중 모멘텀
def momentum_frame(auction: bool = False) -> pd.DataFrame:
    """date, code, r_rest(전일 종가→14:50), r_last(14:50→정규장 종가, auction=True 면 단일가 종가), post."""
    w = window_raw()
    w = w[w.code.isin(roles())].copy()
    w["prev_close"] = w.groupby("code").p_close.shift(1)
    end = w.p_close if auction else w.p_close_reg
    w["r_rest"] = np.log(w.c1445 / w.prev_close)
    w["r_last"] = np.log(end / w.c1445)
    out = _base().merge(w[["date", "code", "r_rest", "r_last"]].dropna(), on="date")
    out["role"] = out.code.map(roles())
    return out


def slope_change(g: pd.DataFrame, hac_lags: int = 5) -> dict:
    X = np.column_stack([g.r_rest, g.r_rest * g.post, g.post])
    r = sm.OLS(g.r_last.to_numpy(), sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return {"b_pre": r.params[1], "c_change": r.params[2], "se_c": r.bse[2], "p_hac": r.pvalues[2], "n": len(g)}


def momentum_test(mf: pd.DataFrame, donors: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """통계량 = c_i − donor 평균 c (i 자신 제외). donor 각각의 같은 통계량을 귀무로 순열 p (양측)."""
    c = {code: slope_change(g) for code, g in mf.groupby("code")}
    cd = pd.Series({k: c[k]["c_change"] for k in donors if k in c})
    null = cd - (cd.sum() - cd) / (len(cd) - 1)
    rows = []
    for u in D.CFG["treated"] + [D.CFG["negative_control"], SKSQ] + list(D.CFG["semis_negative_control"]):
        stat = c[u]["c_change"] - cd.mean()
        rows.append({"code": u, "role": roles()[u], **c[u], "stat_vs_donor_mean": stat, "perm_p": M.perm_p(stat, null)})
    out = pd.DataFrame(rows)
    out["donor_mean_c"] = cd.mean()
    return out, null


def pre_corr() -> pd.DataFrame:
    """상장 전 ln rv5 상관: 같은 기업 대조 후보의 적합도."""
    lv = rq2data.load_panel("rv5")
    pre = lv.loc[D.CFG["sample"]["start"]:D.CFG["sample"]["pre_end"]]
    return pd.DataFrame([{"pair": "005930–005935", "corr": pre["005930"].corr(pre["005935"])},
                         {"pair": "000660–402340", "corr": pre["000660"].corr(pre[SKSQ])},
                         {"pair": "000660–005930", "corr": pre["000660"].corr(pre["005930"])}])
