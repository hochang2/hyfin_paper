"""RQ2 진단. 대상: 사전구간 회귀 (y ~ X), 그리고 전체표본 ITS 회귀 (y ~ X + 사건 더미).

모형별로 맞는 검정만 쓴다:
- 회귀계 (OLS/ITS/BSTS 회귀부): 잔차 자기상관(Ljung-Box, Breusch-Godfrey), 이분산(BP, White), 정규성(JB),
  다중공선성(VIF, 조건수), 함수형(RESET), 이상치·영향점(Cook's D), 구조변화(Chow@상장일, sup-F wild bootstrap, CUSUM)
- 시계열 자체: 정상성 (ADF, KPSS) — y, 각 통제변수, 잔차
- 예측오차 패턴: placebo-in-time 오차의 평균(편향)·자기상관 → run_all 의 placebo 단계에서
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import (acorr_breusch_godfrey, acorr_ljungbox, breaks_cusumolsresid,
                                          het_breuschpagan, het_white, linear_reset)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity(s: pd.Series) -> dict:
    s = s.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf = adfuller(s, autolag="AIC")
        kp = kpss(s, regression="c", nlags="auto")
    return {"adf_stat": adf[0], "adf_p": adf[1], "kpss_stat": kp[0], "kpss_p": kp[1],
            "verdict": ("stationary" if adf[1] < .05 and kp[1] > .05 else
                        "unit_root" if adf[1] >= .05 and kp[1] <= .05 else "ambiguous")}


def chow(y, X, split_mask) -> dict:
    Xc = sm.add_constant(X, has_constant="add")
    r = sm.OLS(y, Xc).fit()
    r1 = sm.OLS(y[~split_mask], Xc[~split_mask]).fit()
    r2 = sm.OLS(y[split_mask], Xc[split_mask]).fit()
    k = Xc.shape[1]
    n = len(y)
    ssr_u = r1.ssr + r2.ssr
    F = ((r.ssr - ssr_u) / k) / (ssr_u / (n - 2 * k))
    return {"chow_F": F, "chow_p": 1 - stats.f.cdf(F, k, n - 2 * k), "n_after": int(split_mask.sum())}


def sup_f(y, X, trim=0.15, nboot=199, seed=0) -> dict:
    """Andrews sup-F (전 계수 구조변화), p 는 Hansen(2000) 고정설계 wild bootstrap."""
    Xc = sm.add_constant(X, has_constant="add").values
    yv = y.values
    n, k = Xc.shape
    lo, hi = int(n * trim), int(n * (1 - trim))

    def supf(yy):
        b = np.linalg.lstsq(Xc, yy, rcond=None)[0]
        ssr0 = np.sum((yy - Xc @ b) ** 2)
        best, arg = -np.inf, None
        for t in range(lo, hi):
            s = 0.0
            for A, B in ((Xc[:t], yy[:t]), (Xc[t:], yy[t:])):
                bb = np.linalg.lstsq(A, B, rcond=None)[0]
                s += np.sum((B - A @ bb) ** 2)
            F = ((ssr0 - s) / k) / (s / (n - 2 * k))
            if F > best:
                best, arg = F, t
        return best, arg

    F, t = supf(yv)
    b = np.linalg.lstsq(Xc, yv, rcond=None)[0]
    e = yv - Xc @ b
    rng = np.random.default_rng(seed)
    Fb = np.array([supf(e * rng.standard_normal(n))[0] for _ in range(nboot)])
    return {"supF": F, "supF_date": str(y.index[t].date()), "supF_p_boot": float((Fb >= F).mean())}


def regression_diagnostics(y: pd.Series, X: pd.DataFrame, listing_date, hac_lags=5, seed=0, nboot=199) -> dict:
    Xc = sm.add_constant(X, has_constant="add")
    m = sm.OLS(y, Xc).fit()
    e = m.resid
    out = {"n": int(m.nobs), "r2": m.rsquared, "adj_r2": m.rsquared_adj}
    lb = acorr_ljungbox(e, lags=[5, 10])
    out.update({"lb5_p": float(lb.lb_pvalue.iloc[0]), "lb10_p": float(lb.lb_pvalue.iloc[1])})
    out["bg5_p"] = float(acorr_breusch_godfrey(m, nlags=5)[1])
    out["resid_acf1"] = float(e.autocorr(1))
    out["bp_p"] = float(het_breuschpagan(e, Xc)[1])
    try:
        out["white_p"] = float(het_white(e, Xc)[1])
    except Exception:  # noqa: BLE001  (더미 때문에 특이행렬일 수 있음)
        out["white_p"] = np.nan
    jb = stats.jarque_bera(e)
    out.update({"jb_p": float(jb.pvalue), "skew": float(stats.skew(e)), "kurt": float(stats.kurtosis(e) + 3)})
    vif = {c: variance_inflation_factor(Xc.values, i) for i, c in enumerate(Xc.columns) if c != "const"}
    out["vif_max"] = float(max(vif.values())) if vif else np.nan
    out["vif"] = {k: round(v, 2) for k, v in vif.items()}
    out["cond_number"] = float(np.linalg.cond(((X - X.mean()) / X.std()).values)) if X.shape[1] else np.nan
    try:
        out["reset_p"] = float(linear_reset(m, power=3, use_f=True).pvalue)
    except Exception:  # noqa: BLE001
        out["reset_p"] = np.nan
    infl = m.get_influence()
    cd = pd.Series(infl.cooks_distance[0], index=y.index)
    out["n_cooks_gt_4n"] = int((cd > 4 / len(y)).sum())
    out["top_influential"] = [str(d.date()) for d in cd.nlargest(5).index]
    stud = pd.Series(infl.resid_studentized_external, index=y.index)
    out["n_abs_stud_gt3"] = int((stud.abs() > 3).sum())
    out.update(chow(y, X, pd.Series(y.index >= listing_date, index=y.index).values))
    out["cusum_p"] = float(breaks_cusumolsresid(e.values, ddof=Xc.shape[1])[1])
    out.update(sup_f(y, X, nboot=nboot, seed=seed))
    # HAC vs 일반 SE 비율 (자기상관 보정이 SE 를 얼마나 바꾸는지)
    mh = sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    out["se_ratio_hac_over_ols"] = {c: round(float(mh.bse[c] / m.bse[c]), 2) for c in Xc.columns}
    return out


# ---------------------------------------------------------------- ITS 회귀 + 민감도 (Oster, Cinelli-Hazlett)
def its_regression(y, X, dummies: pd.DataFrame, hac_lags=5):
    Z = sm.add_constant(pd.concat([dummies, X], axis=1), has_constant="add")
    return sm.OLS(y, Z).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags}), sm.OLS(y, Z).fit()


def oster_delta(b_short, r2_short, b_long, r2_long, rmax_mult=1.3) -> dict:
    """Oster (2019) 근사식. delta = 효과를 0으로 만드는 비관측 선택 강도 (관측 대비). |delta|>1 이면 강건 기준 통과."""
    rmax = min(rmax_mult * r2_long, 1.0)
    num = b_long * (r2_long - r2_short)
    den = (b_short - b_long) * (rmax - r2_long)
    delta = num / den if den != 0 else np.inf
    beta_star = b_long - (b_short - b_long) * (rmax - r2_long) / (r2_long - r2_short) if r2_long != r2_short else np.nan
    return {"rmax": rmax, "delta_for_zero": delta, "beta_star_delta1": beta_star}


def robustness_value(t: float, dof: int, q: float = 1.0, alpha: float = 0.05) -> dict:
    """Cinelli & Hazlett (2020). RV_q: 처치·결과의 잔차 분산을 각각 이만큼 설명하는 교란이면 효과가 q 만큼 줄어듦."""
    f = abs(t) / np.sqrt(dof)
    fq = q * f
    rv = 0.5 * (np.sqrt(fq ** 4 + 4 * fq ** 2) - fq ** 2)
    tcrit = stats.t.ppf(1 - alpha / 2, dof - 1)
    fqa = fq - tcrit / np.sqrt(dof - 1)
    rva = 0.0 if fqa <= 0 else 0.5 * (np.sqrt(fqa ** 4 + 4 * fqa ** 2) - fqa ** 2)
    return {"partial_r2_treat": t ** 2 / (t ** 2 + dof), "RV_q1": rv, "RV_q1_alpha05": rva}
