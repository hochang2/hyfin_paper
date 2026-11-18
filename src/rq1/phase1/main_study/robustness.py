"""RQ1 4단계: 보조·강건성 분석 (config.yaml pre_registration.auxiliary / robustness).

결론은 주 검정으로만 내린다. 여기 결과로 주 결론을 뒤집지 않고, 차이와 원인을 함께 보고한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.rq1.core import data as D
from src.rq1.core import did as M

TREATED = D.CFG["treated"]


def did_with_placebo(panel: pd.DataFrame, donors: list[str], codes: list[str] = TREATED) -> list[dict]:
    """주 설계와 같은 DiD + donor 순열 p 를 다른 패널(측정치·표본 변경)에 적용."""
    null = M.placebo_space(panel, donors)["D"]
    out = []
    for c in codes:
        b = M.did_beta(panel, c, donors)
        out.append({"code": c, "beta": b, "p": M.perm_p(b, null), "p_type": f"donor 순열 ({len(null)})",
                    "mde_log": M.mde(null)})
    return out


def matched_donors(code: str, donors: list[str], panel: pd.DataFrame, k: int = 20) -> list[str]:
    """상장 전 자료로만: 상장 전 평균 log RV, log 시가총액, 전년도 베타의 표준화 거리 최근접 k."""
    mv = D.match_vars().set_index("code")
    pre = panel[panel.post == 0].groupby("code").y.mean().rename("lrv_pre")
    X = pd.concat([pre, np.log(mv["mktcap"]).rename("lmcap"), mv["beta_prev"]], axis=1).dropna()
    Z = (X - X.loc[donors].mean()) / X.loc[donors].std()
    pool = [d for d in donors if d in Z.index]
    dist = np.sqrt(((Z.loc[pool] - Z.loc[code]) ** 2).sum(axis=1))
    return list(dist.nsmallest(k).index)


def matched_did(panel: pd.DataFrame, donors: list[str], k: int = 20) -> list[dict]:
    """매칭 대조군 k 종목. 순열은 매칭된 k 종목 안에서 (각자를 가짜 처리로, 나머지 k−1 을 대조로)."""
    out = []
    for c in TREATED:
        mt = matched_donors(c, donors, panel, k)
        b = M.did_beta(panel, c, mt)
        null = M.placebo_space(panel, mt)["D"]
        out.append({"code": c, "beta": b, "p": M.perm_p(b, null), "p_type": f"매칭 {k} 안 순열", "note": ",".join(mt)})
    return out


def pooled_did(panel: pd.DataFrame, donors: list[str], n_draw: int, seed: int) -> dict:
    """두 처리 종목을 함께 처리로 둔 DiD. 귀무: donor 무작위 2종목을 처리로 둔 같은 DiD."""
    def beta2(pn, pair, ctrl):
        d = pn[pn.code.isin(list(pair) + list(ctrl))].copy()
        d["D"] = d.code.isin(pair).astype(float) * d.post
        return float(M.twfe(d, ["D"])["D"])
    b = beta2(panel, TREATED, donors)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_draw):
        pair = list(rng.choice(donors, 2, replace=False))
        null.append(beta2(panel, pair, [c for c in donors if c not in pair]))
    return {"code": "005930+000660", "beta": b, "p": M.perm_p(b, np.array(null)), "p_type": f"donor 무작위 2종목 {n_draw}"}


def panel_controls(panel: pd.DataFrame, donors: list[str]) -> pd.DataFrame:
    """종목마다 값이 다른 통제변수: 자기 log RV 시차(1일, 직전 5일 평균), 전년도 베타 × 전날 미국 log VIX.
    시장 공통 변수는 날짜 FE 에 흡수되므로 넣지 않는다. 실적 발표일 더미는 분기 발표일 자료가 없어 뺐다."""
    from src.rq2.core.data import raw_market
    p = panel.sort_values(["code", "date"]).copy()
    full = D.panel_long("rv5")   # 표본 첫날 시차를 위해 같은 패널 사용 (표본 이전 값 없음 → 첫 5일 제외)
    p["lag1"] = p.groupby("code").y.shift(1)
    p["lag5"] = p.groupby("code").y.transform(lambda s: s.shift(1).rolling(5).mean())
    mv = D.match_vars().set_index("code")["beta_prev"]
    lvix = np.log(raw_market()["vix"])
    p["beta_x_vix"] = p.code.map(mv) * p.date.map(lvix)
    del full
    return p.dropna(subset=["lag1", "lag5", "beta_x_vix"])


def did_controls(panel: pd.DataFrame, donors: list[str]) -> list[dict]:
    p = panel_controls(panel, donors)
    regs = ["lag1", "lag5", "beta_x_vix"]
    out, nulls = [], []
    for j in donors:
        d, _ = M.design(p, j, [c for c in donors if c != j])
        nulls.append(float(M.twfe(d, ["D"] + regs)["D"]))
    for c in TREATED:
        d, _ = M.design(p, c, donors)
        b = float(M.twfe(d, ["D"] + regs)["D"])
        out.append({"code": c, "beta": b, "p": M.perm_p(b, np.array(nulls)), "p_type": "donor 순열 (175)",
                    "note": "통제: 자기 RV 시차 1·5일, 베타×전날 VIX"})
    return out


def ts_event_regression(cfg2: dict) -> list[dict]:
    """시계열 이벤트 회귀: ln RV_T ~ post + C_main + HAR(1,5,22). HAC(5). 모형 p 이므로 탐색·보조로만."""
    from src.rq2.core.data import build_controls, load_panel
    s = D.CFG["sample"]
    lv = load_panel("rv5").loc[s["start"]:s["treat_end"]]
    X = build_controls(cfg2, "rv5", "main", train_end=s["pre_end"])[cfg2["controls"]["C_main"]]
    out = []
    for c in TREATED:
        y = lv[c]
        df = pd.DataFrame({"y": y, "post": (y.index >= s["treat_start"]).astype(float),
                           "har1": y.shift(1), "har5": y.shift(1).rolling(5).mean(), "har22": y.shift(1).rolling(22).mean()})
        df = df.join(X).dropna()
        r = sm.OLS(df.y, sm.add_constant(df.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        # HAR 시차가 있으면 post 계수는 단기 효과다. 장기 효과 = β / (1 − Σ HAR 계수)
        rho = r.params[["har1", "har5", "har22"]].sum()
        out.append({"code": c, "beta": float(r.params["post"]), "p": float(r.pvalues["post"]), "p_type": "HAC 모형 p (보고만)",
                    "note": f"장기효과 β/(1−ρ)={r.params['post'] / (1 - rho):+.3f}, ρ={rho:.2f}, n={int(r.nobs)}"})
    return out


def within_firm(seed: int) -> list[dict]:
    """같은 기업 차분 d_t = lrv(005930) − lrv(005935), β=1 고정.
    (1) rv5: 처치 평균 − 상장 전 평균, HAC p (보고만)
    (2) 일별 GK: 같은 정의의 효과를 2016~2025-09 의 가짜 상장일(10일 간격) 분포로 보정한 p."""
    s = D.CFG["sample"]
    from src.rq2.core.data import load_panel
    lv = load_panel("rv5").loc[s["start"]:s["treat_end"]]
    d = lv["005930"] - lv["005935"]
    post = (d.index >= s["treat_start"]).astype(float)
    r = sm.OLS(d.values, sm.add_constant(post)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    out = [{"code": "005930−005935", "beta": float(r.params[1]), "p": float(r.pvalues[1]), "p_type": "rv5, HAC 모형 p (보고만)"}]

    g = D.long_gk_wide(["005930", "005935"])
    dg = (g["005930"] - g["005935"]).dropna()
    n_pre, n_post = 155, 35

    def eff(series, pos):
        return series.iloc[pos:pos + n_post].mean() - series.iloc[pos - n_pre:pos].mean()

    pos_real = int(np.searchsorted(dg.index.values, s["treat_start"].to_datetime64()))
    real = eff(dg, pos_real)
    lo = int(np.searchsorted(dg.index.values, pd.Timestamp("2016-06-01").to_datetime64()))
    hi = int(np.searchsorted(dg.index.values, s["start"].to_datetime64())) - n_post   # 가짜 처치 구간이 분봉 표본 전에 끝나게
    null = np.array([eff(dg, p) for p in range(max(lo, n_pre), hi, 10)])
    out.append({"code": "005930−005935", "beta": float(real), "p": M.perm_p(real, null),
                "p_type": f"일별 GK, 긴 표본 가짜 상장일 {len(null)}개 보정", "note": f"null sd={null.std(ddof=1):.3f}, MDE={2.8 * null.std(ddof=1):.3f}"})
    return out


def bsts_rq2(cfg2: dict) -> list[dict]:
    """RQ2 의 BSTS (C_main) 를 상장 전(05-26)까지만 학습해 05-27~07-15 반사실을 만든다. 모형 구간 p (보고만)."""
    from src.rq2.core.pipeline import run_spec
    s = D.CFG["sample"]
    out = []
    for c in TREATED + [D.CFG["negative_control"]]:
        _, _, rows = run_spec({"code": c, "model": "bsts", "train_end": s["pre_end"], "sample_end": s["treat_end"],
                               "seed": D.CFG["seed"]})
        r = next(x for x in rows if x.get("window") == "listing")
        out.append({"code": c, "beta": r["effect_log"], "p": r["p_two_sided"], "p_type": "BSTS 모형 p (보고만)",
                    "note": f"95% [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}], 상장 전 RMSE {r.get('pre_rmse', float('nan')):.3f}"})
    return out


def did_lag_controls_with_negatives(panel: pd.DataFrame, donors: list[str]) -> list[dict]:
    """종목별 통제(자기 log RV 시차 1일·직전 5일 평균)만 넣은 패널 DiD 를 처리 종목과 음성통제에 똑같이 적용.
    005935 는 베타 자료가 없어 베타×VIX 항은 뺐다. 반도체 7종목은 평균을 가상 종목으로 둔다."""
    semis = D.CFG["semis_negative_control"]
    pp = M.portfolio_panel(panel, semis, "SEMIS7")
    p = pp.sort_values(["code", "date"]).copy()
    p["lag1"] = p.groupby("code").y.shift(1)
    p["lag5"] = p.groupby("code").y.transform(lambda s: s.shift(1).rolling(5).mean())
    p = p.dropna(subset=["lag1", "lag5"])
    regs = ["lag1", "lag5"]
    null = []
    for j in donors:
        d, _ = M.design(p, j, [c for c in donors if c != j])
        null.append(float(M.twfe(d, ["D"] + regs)["D"]))
    out = []
    for c in TREATED + [D.CFG["negative_control"], "SEMIS7"]:
        d, _ = M.design(p, c, donors)
        b = float(M.twfe(d, ["D"] + regs)["D"])
        out.append({"code": c, "beta": b, "p": M.perm_p(b, np.array(null)), "p_type": "donor 순열 (175, 개별)",
                    "note": "통제: 자기 RV 시차 1·5일. SEMIS7 은 포트폴리오라 개별 귀무 대비 p 는 보수적이지 않음(참고)"})
    return out


def pooled_negative(panel: pd.DataFrame, donors: list[str], n_draw: int, seed: int) -> dict:
    """통합 효과의 음성통제판: 005935 + 반도체 7종목 평균을 함께 처리로 둔 DiD (같은 귀무)."""
    pp = M.portfolio_panel(panel, D.CFG["semis_negative_control"], "SEMIS7")
    d = pp[pp.code.isin(["005935", "SEMIS7"] + list(donors))].copy()
    d["D"] = d.code.isin(["005935", "SEMIS7"]).astype(float) * d.post
    b = float(M.twfe(d, ["D"])["D"])
    ref = pooled_did(panel, donors, n_draw=n_draw, seed=seed)   # 같은 seed → 같은 귀무
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_draw):
        pair = list(rng.choice(donors, 2, replace=False))
        dd = panel[panel.code.isin(donors)].copy()
        dd["D"] = dd.code.isin(pair).astype(float) * dd.post
        null.append(float(M.twfe(dd, ["D"])["D"]))
    return {"code": "005935+SEMIS7", "beta": b, "p": M.perm_p(b, np.array(null)), "p_type": f"donor 무작위 2종목 {n_draw}",
            "note": f"처리 통합 β={ref['beta']:+.3f} 와 비교"}
