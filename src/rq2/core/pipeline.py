"""RQ2 실험 한 건을 돌리는 공통 함수.

run_spec(spec) → (CF, y, summary rows). spec 은 dict:
  code, model, controls(이름), measure, pool, train_start, train_end, post_start, drop_dates, seed, model_kw
학습 = [train_start, train_end], 반사실 예측 = train_end 다음 날부터 표본 끝까지.
효과 창(window)은 effect_windows() 로 정의하고 모든 실험에서 같은 정의를 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.rq2.core import models as M
from src.rq2.core.data import build_controls, build_design, donor_codes, load_cfg, load_panel

CFG = load_cfg()
TL = CFG["timeline"]

MODEL_FUN = {
    "bsts": M.cf_bsts, "ols": M.cf_ols, "ridge": M.cf_ridge,
    "enet": lambda *a, **k: M.cf_ridge(*a, kind="enet", **k),
    "lgbm": lambda *a, **k: M.cf_gbm(*a, lib="lgbm", **k),
    "xgb": lambda *a, **k: M.cf_gbm(*a, lib="xgb", **k),
    "uc": M.cf_uc, "sarimax": M.cf_sarimax, "harx": M.cf_harx,
}


def base_spec(**kw) -> dict:
    s = dict(code="005930", model="bsts", controls="C_main", measure="rv5", pool="main",
             train_start=CFG["sample"]["start"], train_end=CFG["sample"]["train_end"],
             post_start=None, drop_dates=None, seed=CFG["seed"], model_kw={})
    s.update(kw)
    return s


def effect_windows(idx: pd.DatetimeIndex, post_start: pd.Timestamp | None = None) -> dict[str, pd.Series]:
    """주 효과 창. post_start 가 주어지면 'from_event' (사건일 이후 전체) 창을 추가."""
    s = lambda m: pd.Series(m, index=idx)  # noqa: E731
    eff = idx >= TL["effective"]
    k = np.cumsum(eff)
    w = {
        "listing": s((idx >= TL["listing"]) & (idx < TL["announce"])),
        "transition": s((idx >= TL["announce"]) & (idx < TL["effective"])),
        "post_effective": s(eff),
        "post_eff_20d": s(eff & (k <= 20)),
        "post_eff_21_61d": s(eff & (k > 20)),
    }
    if post_start is not None:
        w["from_event"] = s(idx >= post_start)
    return w


def run_spec(spec: dict):
    s = base_spec(**spec)
    controls = CFG["controls"][s["controls"]] if isinstance(s["controls"], str) else list(s["controls"])
    if s["model"] == "scm":
        return _run_scm(s)
    y, X = build_design(CFG, s["code"], controls, s["measure"], s["pool"], s["train_end"])
    if s.get("extra_X") is not None:
        X = X.join(s["extra_X"], how="left").fillna(0.0)
    if s.get("transform"):
        y, X = s["transform"](y, X, s)
    keep = y.index >= s["train_start"]
    if s.get("sample_end") is not None:
        keep &= y.index <= s["sample_end"]
    y, X = y[keep], X[keep]
    tr = pd.Series(y.index <= s["train_end"], index=y.index)
    po = ~tr
    cf = MODEL_FUN[s["model"]](y, X, tr, po, seed=s["seed"], **s["model_kw"])
    y_post = y[po]
    windows = effect_windows(y_post.index, s["post_start"])
    if s["drop_dates"] is not None:
        a, b = s["drop_dates"]
        for k in windows:
            windows[k] = windows[k] & ~((windows[k].index >= a) & (windows[k].index <= b))
    rows = M.summarize(y_post, cf, windows)
    fit = M.pre_fit_metrics(y[tr], cf.fitted_pre)
    for r in rows:
        r.update(fit)
    return cf, y, rows


def _run_scm(s):
    lv = load_panel(s["measure"]).loc[CFG["sample"]["start"]:CFG["sample"]["end"]]
    donors = [d for d in donor_codes(CFG, s["pool"]) if d != s["code"]]
    W = lv[[s["code"]] + donors].dropna(axis=1)
    W = W[W.index >= s["train_start"]]
    if s.get("sample_end") is not None:
        W = W[W.index <= s["sample_end"]]
    donors = [d for d in donors if d in W.columns]
    tr = pd.Series(W.index <= s["train_end"], index=W.index)
    po = ~tr
    cf = M.cf_scm(W, s["code"], donors, tr, po, seed=s["seed"], **s["model_kw"])
    y = W[s["code"]]
    windows = effect_windows(y[po].index, s["post_start"])
    rows = M.summarize(y[po], cf, windows)
    for r in rows:
        r.update(M.pre_fit_metrics(y[tr], cf.fitted_pre))
    return cf, y, rows


def rows_df(rows: list[dict], **tags) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for k, v in tags.items():
        df[k] = v
    return df


def holm(p: pd.Series) -> pd.Series:
    """Holm-Bonferroni 보정 p."""
    order = np.argsort(p.values)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p.values[i])
        adj[i] = min(run, 1.0)
    return pd.Series(adj, index=p.index)


# ---------------------------------------------------------------- 보조 변환 (강건성용)
def winsorize_transform(y, X, s, q=(0.01, 0.99)):
    tr = y.index <= s["train_end"]
    lo, hi = y[tr].quantile(q[0]), y[tr].quantile(q[1])
    y = y.clip(lo, hi)
    Xc = X.copy()
    for c in X.columns:
        if X[c].nunique() > 2:
            a, b = X.loc[tr, c].quantile(q[0]), X.loc[tr, c].quantile(q[1])
            Xc[c] = X[c].clip(a, b)
    return y, Xc


def listing_dummy() -> pd.DataFrame:
    idx = load_panel("rv5").index
    return pd.DataFrame({"listing_regime": (idx >= TL["listing"]).astype(float)}, index=idx)


def within_firm_control_X(code_ctrl: str = "005935", measure: str = "rv5") -> pd.DataFrame:
    lv = load_panel(measure)
    return pd.DataFrame({f"lrv_{code_ctrl}": lv[code_ctrl]})


__all__ = ["CFG", "TL", "run_spec", "base_spec", "effect_windows", "rows_df", "holm",
           "winsorize_transform", "listing_dummy", "within_firm_control_X", "build_controls"]
