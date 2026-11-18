"""접힌(collapsed) 격차 시계열과 역사적 시간 위약(historical time placebo) — 주 추론.

식별 문제: 처치·대조 상품 수가 적고(클러스터 11 vs 34) 날짜 간 공통충격이 강해 클러스터 SE 가 믿기 어렵다.
해결: 상품을 계열×집단으로 접어 일별 격차 g_f,t = mean log v(2X) − mean log v(1X) 를 만들고,
      실제 효과 Δ = mean_post g − mean_base g 를, 상장 이전(2016~2026-05) 같은 길이 가짜 창들의 Δ 분포와 비교.
      가짜 창은 겹치지 않게 잡는다 (RQ2 에서 겹치는 창은 n_eff 를 부풀려 위양성을 키웠다).
      코호트 규칙(창 안 매일 거래, 기준구간 중앙값 ≥ 1억)은 실제·가짜 창에 똑같이 적용.
균형 패널에서 Δ_pooled = Σ w_f Δ_f / Σ w_f, w_f = nT·nC/(nT+nC) 는 상품 FE + 계열×일 FE 회귀의 β 와 같다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.rq4.common import CFG, TL
from src.rq4.data_pipeline import load_etf
from src.rq4.panel import P

G = CFG["groups"]
FLOOR = CFG["cohort"]["min_pre_median_value_won"]


def default_spec():
    return {"idx": (G["B_index_2x"], G["E_index_1x"]), "semi": (G["C_semi_2x"], G["E_semi_1x"])}


_WIDE = {}


def wide(outcome="log_value"):
    if outcome in _WIDE:
        return _WIDE[outcome]
    from src.rq4.panel import add_outcomes
    e = load_etf()
    allc = sorted({c for v in G.values() for c in v})
    d = add_outcomes(e[e["code"].isin(allc)])
    W = d.pivot(index="date", columns="code", values=outcome).sort_index()
    V = d.pivot(index="date", columns="code", values="value").sort_index()
    _WIDE[outcome] = (W, V)
    return W, V


def window_lengths(post_end=None, base=None):
    W, _ = wide()
    days = W.index
    b0, b1 = base or P["listing_base"]
    nb = int(((days >= b0) & (days <= b1)).sum())
    nt = int(((days >= P["transition"][0]) & (days <= P["transition"][1])).sum())
    pe = pd.Timestamp(post_end) if post_end else P["post"][1]
    npost = int(((days >= P["post"][0]) & (days <= pe)).sum())
    return nb, nt, npost


def window_effect(idx_base, idx_post, spec=None, outcome="log_value", floor=FLOOR, min_units=(1, 1),
                  require_window=None):
    """주어진 기준·사후 날짜 위치(정수 배열)에서 계열별 Δ_f 와 가중치."""
    W, V = wide(outcome)
    spec = spec or default_spec()
    allpos = np.concatenate([idx_base, idx_post]) if require_window is None else require_window
    out = {}
    for f, (tr, co) in spec.items():
        res = []
        for codes in (tr, co):
            cs = [c for c in codes if c in W.columns]
            sub = W.iloc[allpos][cs]
            vv = V.iloc[allpos][cs]
            ok = sub.notna().all() & (vv > 0).all()
            med = V.iloc[idx_base][cs].median()
            ok &= med >= floor if floor else True
            res.append([c for c in cs if ok[c]])
        tr_ok, co_ok = res
        if len(tr_ok) < min_units[0] or len(co_ok) < min_units[1]:
            out[f] = dict(delta=np.nan, w=0.0, nT=len(tr_ok), nC=len(co_ok))
            continue
        g = W[tr_ok].mean(axis=1) - W[co_ok].mean(axis=1)
        delta = g.iloc[idx_post].mean() - g.iloc[idx_base].mean()
        out[f] = dict(delta=delta, w=len(tr_ok) * len(co_ok) / (len(tr_ok) + len(co_ok)),
                      nT=len(tr_ok), nC=len(co_ok))
    ws = sum(v["w"] for v in out.values() if np.isfinite(v["delta"]))
    pooled = sum(v["w"] * v["delta"] for v in out.values() if np.isfinite(v["delta"])) / ws if ws else np.nan
    res = {"pooled": pooled}
    for f, v in out.items():
        res[f] = v["delta"]
        res[f"nT_{f}"], res[f"nC_{f}"] = v["nT"], v["nC"]
    if "semi" in out and "idx" in out:
        res["diff"] = out["semi"]["delta"] - out["idx"]["delta"]
    return res


def actual_positions(post_end=None, base=None, post_start=None):
    W, _ = wide()
    days = W.index
    b0, b1 = base or P["listing_base"]
    ib = np.where((days >= b0) & (days <= b1))[0]
    ps = pd.Timestamp(post_start) if post_start else P["post"][0]
    pe = pd.Timestamp(post_end) if post_end else P["post"][1]
    ip = np.where((days >= ps) & (days <= pe))[0]
    return ib, ip


def placebo_windows(nb, nt, npost, step=None, start=None, end=None):
    """기준 nb 일, 공백 nt 일, 사후 npost 일. 겹치지 않게 (step = 전체 길이)."""
    W, _ = wide()
    days = W.index
    s0 = pd.Timestamp(start or CFG["inference"]["placebo_range"][0])
    s1 = pd.Timestamp(end or CFG["inference"]["placebo_range"][1])
    pos = np.where((days >= s0) & (days <= s1))[0]
    L = nb + nt + npost
    step = step or L
    wins = []
    i = pos[0]
    while i + L - 1 <= pos[-1]:
        wins.append((np.arange(i, i + nb), np.arange(i + nb + nt, i + L)))
        i += step
    return wins


def placebo_dist(spec=None, outcome="log_value", step=None, post_end=None, base=None, floor=FLOOR, start=None,
                 end=None):
    nb, nt, npost = window_lengths(post_end, base)
    if base is not None:   # 다른 기준선이면 기준 끝~시행 사이 공백
        W, _ = wide()
        days = W.index
        nt = int(((days > pd.Timestamp(base[1])) & (days < P["post"][0])).sum())
    rows = []
    for ib, ip in placebo_windows(nb, nt, npost, step, start, end):
        r = window_effect(ib, ip, spec, outcome, floor)
        W, _ = wide(outcome)
        r["start"] = W.index[ib[0]]
        rows.append(r)
    return pd.DataFrame(rows)


def placebo_p(actual: float, dist: pd.Series) -> dict:
    x = dist.dropna().values
    n = len(x)
    if n == 0 or not np.isfinite(actual):
        return dict(p=np.nan, n=n, sd=np.nan, z=np.nan, p_norm=np.nan, ci_low=np.nan, ci_high=np.nan)
    from scipy import stats
    p = (1 + np.sum(np.abs(x - np.median(x)) >= abs(actual - np.median(x)))) / (n + 1)
    sd = x.std(ddof=1)
    z = (actual - x.mean()) / sd
    p_norm = 2 * stats.t.sf(abs(z), n - 1)
    q = stats.t.ppf(0.975, n - 1)
    # 가짜 창 분포를 영가설 하 오차분포로 보고, 효과 = actual − 분포 평균 의 CI
    return dict(p=p, n=n, sd=sd, z=z, p_norm=p_norm, mean_null=x.mean(),
                ci_low=actual - x.mean() - q * sd * np.sqrt(1 + 1 / n),
                ci_high=actual - x.mean() + q * sd * np.sqrt(1 + 1 / n))
