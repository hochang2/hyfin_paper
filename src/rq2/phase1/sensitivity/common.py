"""공통: 경로·설정·로그·기준 추정치·null 분포·섭동 재적합.

원칙
· 기존 결과표와 캐시는 읽기만 한다 (main_study, ml_extension, ml_followup, ensemble).
· 효과 θ = mean(y − 반사실) (log RV), bias b = (실제 무처치 결과 − 반사실) 의 창평균. 보정 효과 = θ − b.
· 오염·교란은 사후구간에만 들어가므로 학습 적합은 변하지 않는다. 같은 seed 로 다시 돌리면 Gibbs 표본도 같고,
  사후 예측만 X_post 변화만큼 바뀐다 → 섭동 재적합 = '고정된 예측함수에 바뀐 X 를 넣은 것'과 같다 (diagnostics 에서 확인).
"""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.rq2.core import models as M
from src.rq2.core.data import build_controls, donor_codes, load_panel
from src.rq2.core.explog import ExpLog
from src.rq2.core.pipeline import CFG as RQ2, MODEL_FUN, effect_windows

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RES = HERE / "results"
TAB = RES / "tables"
FIG = RES / "figures"
CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
CACHE = ROOT / CFG["cache_dir"]
SEED = CFG["seed"]
MAIN = ROOT / "src/rq2/phase1/main_study/results/tables"
ENS = ROOT / "src/rq2/phase1/ensemble/results/tables"
EXT = ROOT / "data/interim/rq_cache/ml_ext"
FU = ROOT / "data/interim/rq_cache/ml_followup"
ENSC = ROOT / "data/interim/rq_cache/ensemble"
TREAT = RQ2["treated"]
NEG = RQ2["within_firm_control"]
TRAIN_END = pd.Timestamp(RQ2["sample"]["train_end"])
EFFECTIVE = pd.Timestamp(RQ2["timeline"]["effective"])
BSTS_MAIN = dict(niter=RQ2["bsts"]["niter"], burn=RQ2["bsts"]["burn"],
                 expected_model_size=RQ2["bsts"]["expected_model_size"],
                 prior_level_sd=RQ2["bsts"]["prior_level_sd"], prior_level_df=RQ2["bsts"]["prior_level_df"])
POST_L = 61          # post_effective 창 길이 (거래일)
SIMPLEX_MEMBERS = ["bsts", "ols", "ridge", "uc", "sarimax", "harx"]

LOG = ExpLog(RES / "experiment_log.csv")


def pct(x):
    return 100 * np.expm1(x)


def lg(p):
    """% 변화 → log."""
    return np.log1p(np.asarray(p, float) / 100)


def save_table(df: pd.DataFrame, name: str) -> Path:
    TAB.mkdir(parents=True, exist_ok=True)
    p = TAB / f"{name}.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return p


def n_eff(n_unique_i0: int, step: int, L: int = POST_L) -> float:
    """비겹침 창 수 근사 (ensemble.strategies.n_eff 와 같은 규칙)."""
    return n_unique_i0 * min(1.0, step / max(L, 1))


def block_boot_se(x: np.ndarray, block: int, B: int = 2000, rng=None) -> float:
    """이동 블록 부트스트랩 평균 SE (겹치는 창 의존성 반영). 반복 수 B 는 표본 수가 아니다."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2 * block or n < 4:
        return np.nan
    rng = rng or np.random.default_rng(SEED)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(B, k))
    idx = (starts[..., None] + np.arange(block)).reshape(B, -1)[:, :n]
    return float(x[idx].mean(1).std(ddof=1))


# ------------------------------------------------------------------ 기준 추정치와 null
@functools.lru_cache(None)
def base_estimates() -> pd.DataFrame:
    """주 분석·앙상블 실험의 61일 post_effective (및 transition) 추정치. 기존 표에서 읽기만 한다."""
    pb = pd.read_csv(MAIN / "rq2_primary_bsts.csv", dtype={"code": str})
    wf = pd.read_csv(MAIN / "rq2_withinfirm.csv", dtype={"code": str})
    ex = pd.read_csv(ENS / "exp6_real_effect.csv", dtype={"code": str})
    sy = pd.read_csv(MAIN / "rq2_synthesis_main.csv", dtype={"code": str})
    rows = []
    for w in ["transition", "post_effective"]:
        for c in TREAT:
            r = pb[(pb.code == c) & (pb.window == w)].iloc[0]
            p = sy[(sy.code == c) & (sy.window == w) & sy.design.str.startswith("BSTS")].calibrated_p.iat[0]
            rows.append(dict(design=f"BSTS_{c}", code=c, kind="bsts", window=w, theta=r.effect_log, p_reported=p))
            s = ex[(ex.strategy == "stat_simplex") & (ex.code == c) & (ex.window == w)].iloc[0]
            rows.append(dict(design=f"SIMPLEX_{c}", code=c, kind="simplex", window=w, theta=s.effect_log,
                             p_reported=s.p_space))
        d = wf[(wf.model == "within_DID_HAC_vs_listing") & (wf.window == w)].iloc[0]
        p = sy[(sy.window == w) & sy.design.str.startswith("같은 기업 차분")].calibrated_p.iat[0]
        rows.append(dict(design="DIFF_005930", code="005930", kind="diff", window=w, theta=d.effect_log, p_reported=p))
    B = pd.DataFrame(rows)
    B["theta_pct"] = pct(B.theta)
    return B


def theta(design: str, window: str = "post_effective") -> float:
    B = base_estimates()
    return float(B[(B.design == design) & (B.window == window)].theta.iat[0])


@functools.lru_cache(None)
def null_dist(kind: str, window: str = "post_effective") -> np.ndarray:
    """설계별 효과 null 분포 (참효과 0 일 때의 효과 추정치 분포).
    bsts    : 주 분석 BSTS placebo-in-space (donor 175)
    simplex : ensemble Exp6 stat_simplex donor 175
    diff    : 주 분석 긴 표본 GK 같은 기업 차분 placebo (237)"""
    if kind == "bsts":
        s = pd.read_csv(MAIN / "rq2_placebo_space_bsts.csv", dtype={"code": str})
        return s.loc[~s.treated, window].to_numpy(float)
    if kind == "simplex":
        d = pd.read_csv(ENS / "exp6_space_donor_effects.csv", dtype={"code": str})
        return d[(d.window == window) & (d.strategy == "stat_simplex")].effect.to_numpy(float)
    if kind == "diff":
        l = pd.read_csv(MAIN / "rq2_longhist_placebo.csv")
        return l[(l.series == "005930-005935") & (~l.actual)][window].to_numpy(float)
    raise KeyError(kind)


def null_space_model(model: str, window: str = "post_effective") -> np.ndarray:
    """ml_ext space 캐시의 donor 효과 (모형별 null). 처치 2종목·005935 제외."""
    d = pd.read_parquet(EXT / "space" / f"{model}_win.parquet")
    d = d[(d.window == window) & ~d.code.isin(TREAT + [NEG])]
    return (d.ybar - d.predbar).to_numpy(float)


def crit95(null: np.ndarray) -> float:
    """중심화한 |null| 의 95% 분위 (주 분석 calibrated p 와 같은 중심화 규칙)."""
    z = np.asarray(null, float)
    return float(np.quantile(np.abs(z - z.mean()), 0.95))


def emp_p(est: float, null: np.ndarray) -> float:
    z = np.asarray(null, float) - np.mean(null)
    return float((1 + np.sum(np.abs(z) >= abs(est))) / (1 + len(z)))


# ------------------------------------------------------------------ 섭동 재적합 (주 분석 설정 그대로)
@functools.lru_cache(None)
def donor_panel() -> pd.DataFrame:
    lv = load_panel("rv5").loc[RQ2["sample"]["start"]:RQ2["sample"]["end"]]
    return lv[donor_codes(RQ2, "main")]


@functools.lru_cache(None)
def base_controls() -> pd.DataFrame:
    return build_controls(RQ2, "rv5", "main", RQ2["sample"]["train_end"])


def pca_loadings():
    """build_controls 의 PCA 와 같은 표준화·고유벡터 (학습구간만). 사후구간 섭동의 재투영용."""
    D = donor_panel()
    tm = D.index <= TRAIN_END
    Zt = D[tm]
    mu, sd = Zt.mean(), Zt.std(ddof=0).replace(0, 1)
    S = ((D - mu) / sd).fillna(0.0)
    St = S[tm]
    _, _, vt = np.linalg.svd(St.values - St.values.mean(0), full_matrices=False)
    vt = vt[:3]
    for i in range(3):
        if vt[i].sum() < 0:
            vt[i] = -vt[i]
    return mu, sd, vt


def controls_shifted(donor_shift: pd.Series | float = 0.0, kosdaq_shift: float = 0.0,
                     mean_only: bool = False, dates: pd.Index | None = None) -> pd.DataFrame:
    """사후 날짜(dates, 기본 post_effective)에 donor log RV 를 donor_shift 만큼, KOSDAQ log RV 를 kosdaq_shift 만큼 옮긴 통제변수.
    mean_only=True : don_mean 만 바꾸고 PCA 점수는 그대로 (점수만 바꾸는 단순 시나리오)
    mean_only=False: donor 패널 자체에 주입 → don_mean 재계산 + 학습구간 고유벡터로 PCA 재투영"""
    X = base_controls().copy()
    dates = dates if dates is not None else X.index[X.index >= EFFECTIVE]
    D = donor_panel()
    sh = pd.Series(donor_shift, index=D.columns) if np.isscalar(donor_shift) else donor_shift.reindex(D.columns).fillna(0)
    m = X.index.isin(dates)
    if mean_only:
        X.loc[m, "don_mean"] += float(sh.mean())
    else:
        D2 = D.copy()
        D2.loc[D2.index.isin(dates)] += sh.values
        X["don_mean"] = D2.mean(axis=1)
        X["don_median"] = D2.median(axis=1)
        mu, sd, vt = pca_loadings()
        S = ((D2 - mu) / sd).fillna(0.0)
        sc = S.values @ vt.T
        for i in range(3):
            X[f"don_pc{i+1}"] = sc[:, i]
    X.loc[m, "kosdaq_lrv"] += kosdaq_shift
    return X


def _key(*a) -> str:
    return hashlib.md5(json.dumps(a, default=str, sort_keys=True).encode()).hexdigest()[:16]


def fit_effect(code: str, model: str, X: pd.DataFrame | None = None, controls: list[str] | None = None,
               y_shift: pd.Series | None = None, tag: str = "", cache: bool = True) -> dict:
    """주 분석 run_spec 과 같은 표본·학습구간·seed 로 반사실을 적합하고 창별 효과와 일별 예측을 돌려준다.
    X 를 주면 그 통제변수 표를, y_shift 를 주면 y 에 더한 값을 쓴다 (주입 시뮬레이션). 결과는 캐시."""
    controls = controls or RQ2["controls"]["C_main"]
    X = base_controls() if X is None else X
    lv = load_panel("rv5")
    y = lv[code].loc[RQ2["sample"]["start"]:RQ2["sample"]["end"]].rename("y")
    if y_shift is not None:
        y = y.add(y_shift.reindex(y.index).fillna(0.0)).rename("y")
    df = pd.concat([y, X[controls]], axis=1).dropna()
    y, Xd = df["y"], df.drop(columns="y")
    k = _key(code, model, controls, tag, np.round(Xd.values, 10).sum(), np.round(Xd.values ** 2, 10).sum(),
             np.round(y.values, 10).sum())
    f = CACHE / "fits" / f"{code}_{model}_{k}.parquet"
    if cache and f.exists():
        P = pd.read_parquet(f)
    else:
        tr = pd.Series(y.index <= TRAIN_END, index=y.index)
        po = ~tr
        kw = BSTS_MAIN if model == "bsts" else {}
        cf = MODEL_FUN[model](y, Xd, tr, po, seed=RQ2["seed"], **kw)
        P = pd.DataFrame({"date": y.index[po], "y": y[po].values, "pred": np.asarray(cf.pred, float)})
        if model == "bsts":
            P["pred_mean"] = cf.draws.mean(0)
        f.parent.mkdir(parents=True, exist_ok=True)
        P.to_parquet(f)
    P = P.set_index("date")
    win = effect_windows(P.index)
    eff = {w: float((P.y - P.pred)[m].mean()) for w, m in win.items() if m.any()}
    return {"effects": eff, "daily": P}


def stat_simplex_weights(window: str = "post_effective") -> dict:
    """ensemble Exp6 와 같은 규칙의 실제 stat_simplex 가중치 (rv5 가짜 창, i0=190 이전 완전 실현분)."""
    from src.rq2.phase1.ensemble import exp6_real as E
    T = E._task(window)
    w, c, ms = E._predictor(T, "stat_simplex", np.arange(len(T.y)), None, None, None)
    return {"weights": dict(zip(ms, map(float, w))), "intercept": float(c)}
