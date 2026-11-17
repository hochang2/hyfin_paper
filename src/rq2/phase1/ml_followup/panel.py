"""후속 실험 공용 패널: 종목 전체 rv5 / GK wide 패널, leave-one-out 통제, 단위(종목) 특성, donor pool.

원칙
  - 통제 PCA·표준화·단위 특성은 모두 해당 가짜 개입일의 학습구간(< i0)으로만 추정한다.
  - donor u 를 표적으로 쓸 때 통제에서 u 자신을 뺀다 (don_mean_{-u}, PC_{-u}).
    PC_{-u} 는 학습구간 고유벡터를 그대로 두고 u 의 기여(z_u·v_u)만 뺀 근사다 (PCA 재추정 아님, REPORT 에 명시).
  - 처치 3 종목은 donor 가 아니므로 전체 pool 통제 = 1차 연구 build_controls 와 같은 값 (verify_against_core 로 확인).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.rq2.core.data import donor_codes, load_panel, raw_calendar, raw_f1, raw_market, garman_klass
from src.rq2.core.pipeline import CFG

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RES = HERE / "results"
FCFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
CACHE = ROOT / FCFG["cache_dir"]
EXT = ROOT / FCFG["ext_cache_dir"]
WIN = {k: tuple(v) for k, v in FCFG["windows"].items()}
HZ = FCFG["horizons"]
TREATED = ["005930", "000660"]
UNITS3 = ["005930", "000660", "005935"]
FEATS = FCFG["rv5"]["features"]
SEED = FCFG["seed"]


# ---------------------------------------------------------------- rv5 wide
@lru_cache(maxsize=1)
def rv5_wide() -> pd.DataFrame:
    return load_panel("rv5").loc[CFG["sample"]["start"]:CFG["sample"]["end"]]


@lru_cache(maxsize=1)
def market_frame() -> pd.DataFrame:
    """종목과 무관한 시장 통제 (rv5 표본 날짜)."""
    lv = rv5_wide()
    X = pd.DataFrame(index=lv.index)
    X["kosdaq_lrv"] = lv["IDX101"]
    m = raw_market().reindex(X.index)
    X["us_lvix"] = np.log(m["vix"])
    X["us_abs_sox"] = m["r_sox"].abs() * 100
    X["kosdaq_ret_abs"] = m["r_kosdaq"].abs() * 100
    cal = raw_calendar().reindex(X.index)
    X["gap_day"] = (cal["gap_min"].fillna(0) > 0).astype(float)
    return X


def pool(name: str) -> list[str]:
    return donor_codes(CFG, name)


def _pca(D: pd.DataFrame, n_tr: int, k: int = 3):
    """core._pca_scores 와 같은 계산. 점수와 함께 표준화 행렬 S, 고유벡터 vt 를 돌려준다 (LOO 용)."""
    Dt = D.iloc[:n_tr]
    mu, sd = Dt.mean(), Dt.std(ddof=0).replace(0, 1)
    S = ((D - mu) / sd).fillna(0.0)
    St = S.values[:n_tr]
    _, _, vt = np.linalg.svd(St - St.mean(0), full_matrices=False)
    vt = vt[:k].copy()
    for i in range(k):
        if vt[i].sum() < 0:
            vt[i] = -vt[i]
    return S, vt


@lru_cache(maxsize=256)
def rv5_controls_all(i0: int, pool_name: str = "main", members: tuple | None = None) -> dict:
    """가짜 개입일 i0 의 통제. 반환: {'full': DataFrame(date × FEATS), 'loo': {code: DataFrame}}.
    members 를 주면 그 종목 목록을 pool 로 쓴다 (실험 7 의 유사도 pool)."""
    lv = rv5_wide()
    donors = list(members) if members else pool(pool_name)
    D = lv[donors]
    S, vt = _pca(D, i0)
    sc = S.values @ vt.T
    M = market_frame()
    full = pd.DataFrame(index=lv.index)
    full["don_mean"] = D.mean(axis=1)
    for j in range(3):
        full[f"don_pc{j+1}"] = sc[:, j]
    full = full.join(M)
    Dv = D.values
    cnt = np.isfinite(Dv).sum(1)
    tot = np.nansum(Dv, 1)
    loo = {}
    for j, u in enumerate(donors):
        x = full.copy()
        f = np.isfinite(Dv[:, j])
        x["don_mean"] = np.where(f, (tot - np.where(f, Dv[:, j], 0)) / np.maximum(cnt - f, 1), tot / cnt)
        sub = S.values[:, j:j + 1] * vt[:, j][None, :]
        for k in range(3):
            x[f"don_pc{k+1}"] = sc[:, k] - sub[:, k]
        loo[u] = x
    return {"full": full, "loo": loo, "donors": donors}


def unit_X(i0: int, code: str, pool_name: str = "main", members: tuple | None = None) -> pd.DataFrame:
    c = rv5_controls_all(i0, pool_name, members)
    return c["loo"].get(code, c["full"])


def rv5_task(code: str, i0: int, H: int | None = None, pool_name: str = "main", members: tuple | None = None,
             feats: list[str] | None = None, train_len: int | None = None):
    """(y, X, n_tr): 행 = [학습구간 | 사후 H 일]. 결측 행 제거. train_len 이면 최근 train_len 일만."""
    lv = rv5_wide()
    feats = feats or FEATS
    X = unit_X(i0, code, pool_name, members)[feats]
    H = H if H is not None else min(FCFG["rv5"]["max_h"], len(lv) - i0)
    df = pd.concat([lv[code].rename("y"), X], axis=1).iloc[:i0 + H]
    tr = np.arange(len(df)) < i0
    if train_len:
        keep = np.arange(len(df)) >= i0 - train_len
        df, tr = df[keep], tr[keep]
    ok = df.notna().all(1).values
    df, tr = df[ok], tr[ok]
    return df["y"], df.drop(columns="y"), int(tr.sum())


def rv5_horizon_max(i0: int) -> int:
    return min(FCFG["rv5"]["max_h"], FCFG["rv5"]["last_index_exclusive"] - i0)


def rv5_cutoffs() -> list[int]:
    r = FCFG["rv5"]
    return list(range(r["first_cutoff"], r["last_index_exclusive"] - 5 + 1, r["cutoff_step"]))


def real_index() -> int:
    return rv5_wide().index.get_loc(CFG["timeline"]["announce"])


# ---------------------------------------------------------------- 단위 특성 (학습구간만)
@lru_cache(maxsize=1)
def mktcap_wide() -> pd.DataFrame:
    m = pd.read_parquet(ROOT / FCFG["mktcap_cache"])
    m["date"] = pd.to_datetime(m["date"])
    return m.pivot(index="date", columns="code", values="mktcap").sort_index()


def unit_features(y: pd.Series, X: pd.DataFrame, n_tr: int, code: str) -> dict:
    yt, xt = y.values[:n_tr], X["don_mean"].values[:n_tr]
    mu, sd = yt.mean(), yt.std(ddof=1)
    c = np.cov(yt, xt)
    beta = c[0, 1] / c[1, 1] if c[1, 1] > 0 else 0.0
    r2 = c[0, 1] ** 2 / (c[0, 0] * c[1, 1]) if c[0, 0] * c[1, 1] > 0 else 0.0
    mc = mktcap_wide()
    d_end = y.index[n_tr - 1]
    s = mc[code].loc[:d_end].tail(60) if code in mc.columns else pd.Series(dtype=float)
    lmcap = float(np.log(s.mean())) if len(s.dropna()) else np.nan
    return {"u_mu": mu, "u_sd": sd, "u_beta": beta, "u_r2": r2, "u_lmcap": lmcap}


# ---------------------------------------------------------------- GK wide (2015~)
@lru_cache(maxsize=1)
def gk_frame() -> pd.DataFrame:
    """core.load_long_daily 와 같은 정의의 종목별 log GK wide + 공통 통제(LOO 전 don_mean 합계 포함)."""
    f1 = raw_f1()
    f1 = f1[(f1.volume > 0) & (f1.high > f1.low)].copy()
    f1["gk"] = garman_klass(f1.open, f1.high, f1.low, f1.close)
    f1 = f1[f1.gk > 0]
    f1["lgk"] = np.log(f1.gk)
    w = f1.pivot(index="date", columns="code", values="lgk").sort_index().loc["2015-06-01":]
    return w


@lru_cache(maxsize=1)
def gk_long() -> pd.DataFrame:
    from src.rq2.core.data import load_long_daily
    return load_long_daily(CFG).loc["2015-06-01":]


@lru_cache(maxsize=1)
def gk_units() -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """(donors_long, y_wide[treated3 + donors_long], X_common(date × [us_lvix, us_abs_sox, kosdaq_ret_abs]))."""
    L = gk_long()
    dl = list(L.attrs["donors_long"])
    w = gk_frame().reindex(L.index)
    Y = w[UNITS3 + dl]
    Xc = L[["us_lvix", "us_abs_sox", "kosdaq_ret_abs"]]
    return dl, Y, Xc


def gk_X(code: str) -> pd.DataFrame:
    """GK 통제: don_mean (donor 표적이면 자기 제외) + 공통 3개. core gk 데이터셋과 같은 열 순서."""
    dl, Y, Xc = gk_units()
    D = Y[dl]
    tot, cnt = D.sum(1, min_count=1), D.notna().sum(1)
    if code in dl:
        f = Y[code].notna()
        dm = (tot - Y[code].fillna(0)) / (cnt - f.astype(int))
    else:
        dm = tot / cnt
    X = pd.DataFrame({"don_mean": dm}, index=Y.index).join(Xc)
    return X


def gk_task(code: str, i0: int, n_tr: int, H: int):
    X = gk_X(code)
    _, Y, _ = gk_units()
    df = pd.concat([Y[code].rename("y"), X], axis=1).iloc[max(i0 - n_tr, 0):i0 + H]
    tr = np.arange(len(df)) < (i0 - max(i0 - n_tr, 0))
    ok = df.notna().all(1).values
    df, tr = df[ok], tr[ok]
    return df["y"], df.drop(columns="y"), int(tr.sum())


def verify_against_core(n: int = 3) -> pd.DataFrame:
    """처치 종목 통제 = core.build_controls 인지, 그리고 OLS 창평균이 ml_ext 캐시와 같은지 확인."""
    import statsmodels.api as sm
    from src.rq2.phase1.ml_extension import data_pipeline as DP
    W = pd.read_parquet(EXT / "rv5" / "ols_win.parquet")
    rows = []
    for i0 in rv5_cutoffs()[:: max(1, len(rv5_cutoffs()) // n)][:n]:
        for code in UNITS3:
            y0, X0, tr0, po0 = DP.task_data("rv5", code, i0)
            y, X, ntr = rv5_task(code, i0, rv5_horizon_max(i0))
            dx = float(np.nanmax(np.abs(X0.values - X.loc[X0.index, X0.columns].values)))
            Xc = sm.add_constant(X.values)
            b = np.linalg.lstsq(Xc[:ntr], y.values[:ntr], rcond=None)[0]
            pr = Xc[ntr:] @ b
            w = W[(W.code == code) & (W.i0 == i0) & (W.window == "h10")].iloc[0]
            rows.append({"i0": i0, "code": code, "max_abs_dX": dx,
                         "predbar_h10_here": pr[:10].mean(), "predbar_h10_ext": w.predbar})
    return pd.DataFrame(rows)
