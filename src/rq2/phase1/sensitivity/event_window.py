"""Exp4. 사건일·추정창 민감도 (RQ2-S4-*). 사전 등록 분석이 아니라 민감도 분석이다.

- 효과는 기존 placebo-in-space 캐시(ml_ext/space, 개입 2026-07-16 기준 h=1..71 일별 예측)에서 창별로 다시 평균한다.
  새 적합 없음. 주 분석 BSTS(0.052)와 space 캐시 BSTS(0.060)는 실행이 달라 값이 조금 다르다 → 창 비교는 space 캐시로 통일.
- SIMPLEX = 실제 stat_simplex 가중치(post_effective 규칙)로 bsts·ols·uc 일별 예측을 결합 (창마다 가중치 재학습 안 함).
- DIFF = rv5 d=005930−005935 의 창 평균 − 상장기간 평균 (단순 차이; 주 분석 HAC 회귀의 gap_day 통제는 생략).
  null = GK 2015~ d 의 같은 상대 위치 창 (가짜 개입일 1일 간격, n_eff 보고).
- p = 같은 창 donor null 대비 중심화 경험 p. 다중검정: 설계별 Holm·BH, donor 벡터 max-|z| (창 간 상관 반영).
- 식별 가능성: 사건 사이 구간별 MDE(80% 검정력, 양측 5%) ≈ 2.8 × null sd. MDE 가 20% 를 넘으면 그 구간을 따로 식별할 수 없다고 본다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from src.rq2.core.data import load_long_daily, load_panel
from src.rq2.core.pipeline import effect_windows
from src.rq2.phase1.sensitivity import common as C
from src.rq2.phase1.sensitivity import figs as F

EV = {k: pd.Timestamp(v) for k, v in C.CFG["events"].items()}
END = pd.Timestamp(C.RQ2["sample"]["end"])


def windows(dates: pd.DatetimeIndex) -> dict[str, tuple[str, np.ndarray]]:
    """이름 → (분류, 날짜 마스크). dates = 07-16 이후 거래일."""
    d = dates
    W = {}
    eff = d >= EV["effective"]
    pos = np.cumsum(eff) - 1          # 시행일 이후 거래일 번호 (0부터)
    W["transition (07-16~07-30)"] = ("기준", d < EV["effective"])
    W["post_effective 61일 (주 분석)"] = ("기준", eff)
    for k in C.CFG["windows_after_effective"]:
        if k < C.POST_L:   # 61일 = 주 분석 창 (중복 검정 제외)
            W[f"시행 후 {k}일" + (" (= 전반 1~30일)" if k == 30 else "")] = ("창 길이", eff & (pos < k))
    W["시행 후 31~61일 (후반)"] = ("전·후반", eff & (pos >= 30))
    W["transition − 07-28 제외"] = ("서킷브레이커", (d < EV["effective"]) & (d != EV["circuit_breaker"]))
    W["announce~effective 전체 − 07-28 제외"] = ("서킷브레이커", d != EV["circuit_breaker"])
    lo, hi = (pd.Timestamp(x) for x in C.CFG["donut"])
    W["donut: 61일 − (07-28~08-04) (= 원래 시행일 08-05 이후)"] = ("donut", eff & ~((d >= lo) & (d <= hi)))
    W["프리미엄 규정 08-19 이후"] = ("대안 시행일", d >= EV["premium_rule"])
    W["조기시행 공지 07-24 이후"] = ("대안 시행일", d >= EV["early_effective_notice"])
    # 사건 사이 구간
    cuts = [EV[k] for k in ["announce", "early_effective_notice", "effective", "additional_measures",
                            "original_effective", "premium_rule"]] + [END + pd.Timedelta(days=1)]
    names = ["발표~조기공지", "조기공지~시행", "시행~추가조치", "추가조치~원시행일", "원시행일~프리미엄", "프리미엄 이후"]
    for nm, a, b in zip(names, cuts[:-1], cuts[1:]):
        W[f"구간: {nm}"] = ("사건 구간", (d >= a) & (d < b))
    return W


def _space_gap(models_w: dict[str, float]) -> pd.DataFrame:
    """code × date 일별 (y − pred). models_w = {model: weight}."""
    out = None
    for m, w in models_w.items():
        D = pd.read_parquet(C.EXT / "space" / f"{m}_daily.parquet")
        g = D.pivot(index="date", columns="code", values="pred") * w
        out = g if out is None else out + g
        y = D.pivot(index="date", columns="code", values="y")
    return y - out


def space_effects() -> dict[str, pd.DataFrame]:
    sw = C.stat_simplex_weights("post_effective")["weights"]
    sw = {m: w for m, w in sw.items() if w > 1e-9}
    gaps = {"bsts": _space_gap({"bsts": 1.0}), "simplex": _space_gap(sw)}
    res = {}
    for kind, G in gaps.items():
        W = windows(G.index)
        res[kind] = pd.DataFrame({nm: G[m].mean() for nm, (_, m) in W.items()})   # code × window
    return res, gaps, sw


def diff_effects() -> tuple[pd.Series, pd.DataFrame]:
    lv = load_panel("rv5").loc[C.RQ2["sample"]["start"]:C.RQ2["sample"]["end"]]
    d = (lv["005930"] - lv[C.NEG]).dropna()
    base = d[effect_windows(d.index)["listing"]].mean()
    post = d[d.index >= EV["announce"]]
    W = windows(post.index)
    real = pd.Series({nm: post[m].mean() - base for nm, (_, m) in W.items()})
    # null: GK 긴 표본, 같은 상대 위치
    LD = load_long_daily(C.RQ2).loc[:"2026-07-15"]
    g = (LD["005930"] - LD[C.NEG]).dropna().to_numpy()
    nb, H = C.CFG["confounding"]["diff_listing_days"], len(post)
    rows = []
    for t0 in range(nb, len(g) - H + 1):
        seg = g[t0:t0 + H]
        b0 = g[t0 - nb:t0].mean()
        rows.append({nm: seg[m].mean() - b0 for nm, (_, m) in W.items()})
    return real, pd.DataFrame(rows)


def analyse(real: pd.Series, null: pd.DataFrame, design: str, n_eff: float) -> pd.DataFrame:
    W = windows(pd.DatetimeIndex([]))  # 이름·분류만
    mu, sd = null.mean(), null.std()
    Z = (null - mu) / sd
    zr = (real - mu) / sd
    maxz = Z.abs().max(axis=1).to_numpy()
    rows = []
    for nm in real.index:
        nl = null[nm].dropna().to_numpy()
        rows.append(dict(design=design, window=nm, group=W[nm][0], theta=real[nm], pct=float(C.pct(real[nm])),
                         null_mean=mu[nm], null_sd=sd[nm], p=C.emp_p(real[nm], nl),
                         p_maxT=float((1 + np.sum(maxz >= abs(zr[nm]))) / (1 + len(maxz))),
                         mde_pct=float(C.pct(-2.8 * sd[nm])), n_null=len(nl), n_eff=n_eff))
    T = pd.DataFrame(rows)
    test = T.group != "기준"
    T.loc[test, "p_holm"] = multipletests(T.loc[test, "p"], method="holm")[1]
    T.loc[test, "p_bh"] = multipletests(T.loc[test, "p"], method="fdr_bh")[1]
    T["identifiable_20pct"] = T.mde_pct > -20
    return T


def fig_forest(T: pd.DataFrame):
    import matplotlib.pyplot as plt
    des = T.design.unique()
    wins = T.window.unique()[::-1]
    fig, ax = plt.subplots(figsize=(11, 9), layout="constrained")
    for j, d in enumerate(des):
        g = T[T.design == d].set_index("window").reindex(wins)
        y = np.arange(len(wins)) + (j - (len(des) - 1) / 2) * 0.15
        lo = C.pct(g.theta - 1.96 * g.null_sd)
        hi = C.pct(g.theta + 1.96 * g.null_sd)
        ax.errorbar(g.pct * 100, y, xerr=[(g.pct - lo) * 100, (hi - g.pct) * 100], fmt="o", ms=3.5, lw=0.9,
                    color=F.COL.get(d, F.CAT[j]), label=d, capsize=0)
    ax.axvline(0, color=F.INK2, lw=0.8)
    for v in (-10, -20):
        ax.axvline(v, color=F.CAT[7], lw=0.6, ls=":")
    ax.set_yticks(np.arange(len(wins)), wins, fontsize=7.5)
    ax.set_xlabel("효과 % (점: 추정치, 막대: ±1.96 × donor/역사 null sd) — 민감도 분석, 사전 등록 아님")
    ax.set_title("창·사건일별 효과", loc="left")
    ax.legend(fontsize=7, loc="lower right")
    F.save(fig, "s4_window_forest")


def fig_event_time(gaps):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    for ax, kind in zip(axs, ["bsts", "simplex"]):
        G = gaps[kind]
        don = G.drop(columns=C.TREAT + [C.NEG], errors="ignore")
        cum = lambda s: s.expanding().mean()  # noqa: E731
        Dc = don.apply(lambda s: s[s.index >= EV["effective"]].expanding().mean())
        ax.fill_between(Dc.index, Dc.quantile(.05, axis=1), Dc.quantile(.95, axis=1), color=F.GRID, alpha=0.8,
                        label="donor 5~95% (시행일부터 누적 평균)")
        for c in C.TREAT:
            s = G[c]
            ax.plot(s.index, s, color=F.COL[c], lw=0.6, alpha=0.35)
            ax.plot(Dc.index, cum(s[s.index >= EV["effective"]]), color=F.COL[c], lw=1.8, label=f"{c} 누적 평균 gap")
        for k, t in EV.items():
            ax.axvline(t, color=F.INK2, lw=0.6, ls="--")
            ax.text(t, ax.get_ylim()[1], k, rotation=90, fontsize=6, va="top", ha="right", color=F.INK2)
        ax.axhline(0, color=F.INK, lw=0.6)
        ax.set_ylabel(f"y − cf (log)  [{kind}]")
        ax.legend(fontsize=7, loc="lower left")
    axs[0].set_title("사건 시간축: 일별 gap(옅은 선)과 시행일 이후 누적 평균", loc="left")
    F.save(fig, "s4_event_time")


def run():
    res, gaps, sw = space_effects()
    out = []
    for kind, R in res.items():
        donors = R.drop(index=C.TREAT + [C.NEG], errors="ignore")
        for code in C.TREAT:
            out.append(analyse(R.loc[code], donors, f"{kind.upper()}_{code}", n_eff=len(donors)))
    real, null = diff_effects()
    out.append(analyse(real, null, "DIFF_005930", n_eff=C.n_eff(len(null), 1)))
    T = pd.concat(out, ignore_index=True)
    # 같은 창 재추정이 주 분석과 얼마나 다른지 (space 캐시 vs 주 분석)
    ref = {f"BSTS_{c}": C.theta(f"BSTS_{c}") for c in C.TREAT} | {f"SIMPLEX_{c}": C.theta(f"SIMPLEX_{c}") for c in C.TREAT} \
        | {"DIFF_005930": C.theta("DIFF_005930")}
    T["main_theta_post_effective"] = T.design.map(ref)
    C.save_table(T, "s4_event_window")
    S = (T[T.group != "기준"].groupby("design")
         .agg(n_windows=("window", "size"), min_pct=("pct", "min"), max_pct=("pct", "max"),
              n_negative=("theta", lambda x: int((x < 0).sum())), min_p=("p", "min"), min_p_holm=("p_holm", "min"),
              min_p_bh=("p_bh", "min"), min_p_maxT=("p_maxT", "min")).reset_index())
    C.save_table(S, "s4_event_window_summary")
    seg = T[T.group == "사건 구간"][["design", "window", "pct", "p", "mde_pct", "identifiable_20pct"]]
    C.save_table(seg, "s4_event_segments_identifiability")
    fig_forest(T)
    fig_event_time(gaps)
    for d, g in T.groupby("design"):
        C.LOG.add(id=f"RQ2-S4-{d}", rq="RQ2", kind="robustness", purpose="사건일·창 민감도 (사전 등록 아님)",
                  data="ml_ext space 일별 캐시 / rv5·GK 차분", model=d,
                  settings={"windows": C.CFG["windows_after_effective"], "donut": C.CFG["donut"],
                            "simplex_weights": {k: round(v, 3) for k, v in sw.items()}},
                  result=g[["window", "pct", "p", "p_holm", "p_maxT"]].round(3).to_dict("records"),
                  diagnostics={"mde_pct": g[["window", "mde_pct"]].round(3).to_dict("records")}, seed=C.SEED)
    return dict(T=T, S=S, seg=seg)


if __name__ == "__main__":
    out = run()
    C.LOG.save()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    for k, v in out.items():
        print("==", k)
        print(v.round(3).to_string())
