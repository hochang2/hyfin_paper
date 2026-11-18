"""RQ1 패널 DiD (TWFE), 이벤트 스터디, donor placebo-in-space 순열 추론.

처리 종목이 1개인 DiD 이므로 군집 표준오차는 쓰지 않는다 (보고만 HAC). 판정은 순열 p 로 한다
(config.yaml pre_registration.primary_family).

TWFE 는 종목·날짜 이중 평균 제거(alternating projections)로 푼다. 결측 1칸이 있어 완전 균형이 아니므로
단순 평균 차이 공식 대신 회귀로 계산한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.rq1.core.data import REF_BIN_MAX


def _twfe_demean(M: np.ndarray, unit: np.ndarray, time: np.ndarray, tol: float = 1e-10, max_iter: int = 500) -> np.ndarray:
    """열마다 종목·날짜 고정효과를 제거한다 (불균형 패널용 반복 평균 제거)."""
    M = M.astype(float).copy()
    nu, nt = unit.max() + 1, time.max() + 1
    cu, ct = np.bincount(unit, minlength=nu), np.bincount(time, minlength=nt)
    for _ in range(max_iter):
        prev = M.copy()
        for j in range(M.shape[1]):
            M[:, j] -= (np.bincount(unit, M[:, j], nu) / cu)[unit]
            M[:, j] -= (np.bincount(time, M[:, j], nt) / ct)[time]
        if np.max(np.abs(M - prev)) < tol:
            break
    return M


def twfe(df: pd.DataFrame, regressors: list[str]) -> pd.Series:
    """y ~ 종목 FE + 날짜 FE + regressors. 점추정만 반환."""
    unit = pd.factorize(df["code"])[0]
    time = pd.factorize(df["date"])[0]
    M = _twfe_demean(df[["y"] + regressors].to_numpy(), unit, time)
    beta, *_ = np.linalg.lstsq(M[:, 1:], M[:, 0], rcond=None)
    return pd.Series(beta, index=regressors)


def design(panel: pd.DataFrame, treated: str, controls: list[str], event_study: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """처리 종목 1개 + 대조 종목들로 회귀용 표를 만든다."""
    d = panel[panel.code.isin([treated] + list(controls))].copy()
    tr = (d.code == treated).astype(float)
    if not event_study:
        d["D"] = tr * d["post"]
        return d, ["D"]
    cols = []
    for b in sorted(d.loc[d.bin > REF_BIN_MAX, "bin"].unique()):
        c = f"b{b:+d}"
        d[c] = tr * (d["bin"] == b)
        cols.append(c)
    return d, cols


def did_beta(panel: pd.DataFrame, treated: str, controls: list[str]) -> float:
    d, cols = design(panel, treated, controls)
    return float(twfe(d, cols)["D"])


def did_hac(panel: pd.DataFrame, treated: str, controls: list[str], lags: int = 5) -> dict:
    """보고용 모형 SE: d_t = y_T − 대조 평균 을 post 에 회귀 (HAC). 균형 패널에서 계수는 DiD β 와 같다."""
    w = panel[panel.code.isin([treated] + list(controls))].pivot(index="date", columns="code", values="y")
    dt = w[treated] - w[list(controls)].mean(axis=1)
    post = panel.drop_duplicates("date").set_index("date")["post"].reindex(dt.index)
    r = sm.OLS(dt.values, sm.add_constant(post.values.astype(float))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"beta_hac": float(r.params[1]), "se_hac": float(r.bse[1]), "p_hac": float(r.pvalues[1])}


def placebo_space(panel: pd.DataFrame, donors: list[str], event_study: bool = False) -> pd.DataFrame:
    """donor 각 종목을 가짜 처리 종목으로 두고 나머지 donor 를 대조로 같은 DiD 를 돌린다."""
    rows = []
    for j in donors:
        others = [c for c in donors if c != j]
        d, cols = design(panel, j, others, event_study=event_study)
        b = twfe(d, cols)
        rows.append({"code": j, **b.to_dict()})
    return pd.DataFrame(rows).set_index("code")


def perm_p(stat: float, null: np.ndarray) -> float:
    """양측 순열 p = (1 + #{|null| >= |stat|}) / (1 + N)."""
    null = np.asarray(null, float)
    return float((1 + np.sum(np.abs(null) >= abs(stat) - 1e-12)) / (1 + len(null)))


def mde(null: np.ndarray, power_mult: float = 2.8) -> float:
    """80% 검정력·양측 5% 근사: 2.8 × placebo sd (RQ2 와 같은 정의)."""
    return float(power_mult * np.std(np.asarray(null, float), ddof=1))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = sorted(pvals, key=pvals.get)
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        out[k] = run
    return out


# ---------------------------------------------------------------- 3단계 판정용
def portfolio_panel(panel: pd.DataFrame, codes: list[str], name: str) -> pd.DataFrame:
    """여러 종목의 날짜별 평균 log RV 를 하나의 가상 종목(name)으로 패널에 추가한다."""
    p = panel[panel.code.isin(codes)].groupby("date").agg(y=("y", "mean"), post=("post", "first"),
                                                          rel=("rel", "first"), bin=("bin", "first")).reset_index()
    p["code"], p["role"] = name, "portfolio"
    return pd.concat([panel, p[panel.columns]], ignore_index=True)


def placebo_portfolios(panel: pd.DataFrame, donors: list[str], k: int, n_draw: int, seed: int) -> np.ndarray:
    """donor 에서 k 종목을 무작위로 골라 평균을 가짜 처리 종목으로 두고 나머지 donor 대비 DiD β.
    포트폴리오 평균은 개별 종목보다 덜 흔들리므로 같은 크기(k)의 포트폴리오로 귀무분포를 만든다."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_draw):
        pick = list(rng.choice(donors, size=k, replace=False))
        rest = [c for c in donors if c not in pick]
        pp = portfolio_panel(panel[panel.code.isin(donors)], pick, "__PF__")
        out.append(did_beta(pp, "__PF__", rest))
    return np.array(out)


def pretrend_stat(coefs: pd.Series, pre_bins: list[str]) -> float:
    """상장 전 구간 계수가 함께 0 인지: 제곱합."""
    return float(np.sum(np.square(coefs[pre_bins].to_numpy(float))))
