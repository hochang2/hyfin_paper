"""ML 확장 실험의 데이터와 '과제(task)' 정의. 1차 연구 코드(src.rq2.core)를 그대로 재사용한다.

과제 = (dataset, code, i0). i0 는 가짜(또는 실제) 개입일의 표본 내 위치이고, 학습은 i0 직전까지만 쓴다.
  - rv5   : 분봉 log RV. 학습은 표본 시작부터의 확장창. 통제 PCA·표준화는 학습구간으로만 다시 추정한다.
  - gk    : F1 일별 log Garman-Klass. 학습은 직전 250일 고정창.
  - space : 실제 개입일(07-16)에 donor 175 종목 + 처치 3 종목. donor 는 자기 자신을 뺀 donor 평균을 통제로 쓴다.

정보 유출 차단:
  - 사후 y 는 어떤 모형 입력에도 들어가지 않는다 (benchmark 의 불변성 검정으로 확인).
  - 통제 X 는 그날 장 마감에 관측 가능한 값이다. 사후 X 에는 실제값을 넣는다 (1차 연구와 같은 조건).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.rq2.core.data import build_controls, donor_codes, load_long_daily, load_panel
from src.rq2.core.pipeline import CFG

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RES = HERE / "results"


def load_ecfg() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


ECFG = load_ecfg()
CACHE = ROOT / ECFG["cache_dir"]
WIN = {k: tuple(v) for k, v in ECFG["windows"].items()}


# ---------------------------------------------------------------- 공통 패널
@lru_cache(maxsize=1)
def rv5_panel() -> pd.DataFrame:
    return load_panel("rv5").loc[CFG["sample"]["start"]:CFG["sample"]["end"]]


@lru_cache(maxsize=64)
def rv5_controls(train_end: pd.Timestamp) -> pd.DataFrame:
    """train_end 까지로만 PCA·표준화를 추정한 통제변수 (rv5, main donor)."""
    return build_controls(CFG, "rv5", "main", train_end)


@lru_cache(maxsize=1)
def gk_panel() -> pd.DataFrame:
    return load_long_daily(CFG).loc["2015-06-01":]


def ctrl_names(ds: str) -> list[str]:
    c = ECFG["datasets"][ds]["controls"]
    return CFG["controls"][c] if isinstance(c, str) else list(c)


# ---------------------------------------------------------------- 가짜 개입일
def cutoffs(ds: str) -> list[int]:
    d = ECFG["datasets"][ds]
    if ds == "rv5":
        return list(range(d["first_cutoff"], d["last_index_exclusive"] - 5 + 1, d["cutoff_step"]))
    if ds == "gk":
        idx = gk_panel().index
        lst = idx.searchsorted(CFG["timeline"]["listing"])
        a = max(idx.searchsorted(pd.Timestamp(d["first_cutoff_date"])), d["train_len"])   # 학습창 250일 확보
        return list(range(a, lst - d["max_h"] + 1, d["cutoff_step"]))
    raise ValueError(ds)


def horizon_max(ds: str, i0: int) -> int:
    d = ECFG["datasets"][ds]
    if ds == "rv5":
        return min(d["max_h"], d["last_index_exclusive"] - i0)
    return d["max_h"]


def actual_index(ds: str) -> int:
    if ds in ("rv5", "space"):
        return rv5_panel().index.get_loc(CFG["timeline"]["announce"])
    return gk_panel().index.get_loc(CFG["timeline"]["announce"])


# ---------------------------------------------------------------- 과제 → (y, X, tr, po)
def task_data(ds: str, code: str, i0: int, H: int | None = None):
    """y: Series, X: DataFrame, tr/po: bool Series. po 는 i0 부터 H 일 (결측 행 제거 후 위치 기준)."""
    if ds == "rv5":
        lv = rv5_panel()
        idx = lv.index
        te = idx[i0 - 1]
        X = rv5_controls(te)[ctrl_names("rv5")]
        df = pd.concat([lv[code].rename("y"), X], axis=1).dropna()
        H = H or horizon_max(ds, i0)
        df = df[df.index <= idx[min(i0 + H - 1, len(idx) - 1)]]
    elif ds == "space":
        lv = rv5_panel()
        idx = lv.index
        te = idx[i0 - 1]
        names = ctrl_names("space")
        X = rv5_controls(te)[names].copy()
        donors = donor_codes(CFG, "main")
        if code in donors:   # 자기 자신을 뺀 donor 평균 (1차 연구 P02 와 같음)
            S = lv[donors].sum(axis=1)
            X["don_mean"] = (S - lv[code]) / (len(donors) - 1)
        df = pd.concat([lv[code].rename("y"), X], axis=1).dropna()
    elif ds == "gk":
        L = gk_panel()
        idx = L.index
        te = idx[i0 - 1]
        names = ctrl_names("gk")
        n_tr = ECFG["datasets"]["gk"]["train_len"]
        H = H or horizon_max(ds, i0)
        df = pd.concat([L[code].rename("y"), L[names]], axis=1).iloc[i0 - n_tr:i0 + H].dropna()
    else:
        raise ValueError(ds)
    y = df["y"]
    X = df.drop(columns="y")
    tr = pd.Series(df.index <= te, index=df.index)
    return y, X, tr, ~tr


def state_z(ds: str, code: str, i0: int) -> pd.Series:
    """시나리오 D/E 용 시장 상태 z_t: donor 평균 변동성을 학습구간 평균·sd 로 표준화 (사전 관측 가능 통계만)."""
    y, X, tr, po = task_data(ds, code, i0)
    v = X["don_mean"]
    mu, sd = v[tr].mean(), v[tr].std()
    return ((v - mu) / sd)[po]


def regime_labels(ds: str, cut_list: list[int]) -> pd.DataFrame:
    """가짜 개입일 시점에 관측 가능한 시장 상태. 경계(삼분위)는 평가 묶음용이며 예측에는 쓰지 않는다."""
    rows = []
    for i0 in cut_list:
        if ds == "rv5":
            X = rv5_controls(rv5_panel().index[i0 - 1])
            idx = rv5_panel().index
            vol = X["don_mean"].iloc[i0 - 20:i0].mean()
            jump = (X["don_mean"].iloc[i0 - 5:i0].max() - X["don_mean"].iloc[i0 - 25:i0 - 5].mean())
        else:
            L = gk_panel()
            idx = L.index
            vol = L["don_mean"].iloc[i0 - 20:i0].mean()
            jump = (L["don_mean"].iloc[i0 - 5:i0].max() - L["don_mean"].iloc[i0 - 25:i0 - 5].mean())
        rows.append({"cutoff": idx[i0], "i0": i0, "pre_vol20": vol, "pre_jump5": jump})
    df = pd.DataFrame(rows)
    df["vol_regime"] = pd.qcut(df.pre_vol20, 3, labels=["low", "mid", "high"]).astype(str)
    df["jump_regime"] = np.where(df.pre_jump5 >= df.pre_jump5.quantile(2 / 3), "spike", "calm")
    return df


def space_units() -> list[str]:
    return donor_codes(CFG, "main") + ["005930", "000660", "005935"]


def tau_path(scn: dict, pct: float, H: int, z: np.ndarray | None = None) -> np.ndarray:
    """사후 일별 주입 효과 (log 척도). pct = RV 척도 % 변화 → tau = ln(1+pct/100)."""
    tau = np.log1p(pct / 100.0)
    t = np.arange(1, H + 1)
    k = scn["kind"]
    if k == "constant":
        return np.full(H, tau)
    if k == "ramp":
        return tau * np.minimum(1.0, t / scn["ramp_days"])
    if k == "decay":
        return tau * 0.5 ** ((t - 1) / scn["half_life"])
    if k == "state":
        return tau * np.clip(1 + scn["slope"] * z[:H], 0, 2)
    if k == "highvol":
        return 2 * tau * (z[:H] > 0)
    raise ValueError(k)
