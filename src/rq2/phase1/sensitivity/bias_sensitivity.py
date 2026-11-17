"""Exp1. 반사실 예측 bias 민감도 (RQ2-S1-*).

정의: bias b = (실제 무처치 결과 − 반사실 예측) 의 창평균. 보정 효과 = θ̂ − b.
  b > 0 : 모형이 무처치 변동성을 과소예측 → θ̂ 가 실제 효과보다 큼 → 실제 효과는 θ̂ − b (더 음수)
이것은 '시나리오' 이며 보정된 인과효과가 아니다. 경험적 bias 는 가짜 개입일(참효과 0)의 효과 추정치 = bias 실현값.

경험적 bias 원천 (실제 발표일 이전에 완전히 관측된 창만):
  gk_all       ml_ext GK BSTS (3종목 × ~232, 10일 간격, 2016-06~2026-01)        → BSTS 설계
  gk_recent    같은 자료, 2024-01 이후 가짜 개입일
  gk_regime    실제 발표일과 같은 국면 (donor 변동성 높음 · KOSPI 20일 급락) 의 가짜 개입일
  rv5_recent   ml_ext rv5 BSTS 의 post_effective 가짜 창 (i0+71 <= 190, 종목당 10개 — n_eff < 1)
  space_all    실제 기간 donor 175 의 효과 (같은 시기 단면 bias, 처치 종목과 교환가능하지 않음)
  space_similar 사전 적합(pre_rmse)이 처치 종목과 가장 비슷한 donor 25
  stat_simplex 는 같은 원천에서 구성원 예측을 실제 가중치로 결합 (GK 는 ensemble online_gk 캐시)
  DIFF 는 GK 긴 표본 차분 placebo (주 분석, 237) · GK 단순 차분 (상장기간 대응 34일 기준창) · rv5 단순 차분
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.rq2.core.data import load_long_daily, load_panel
from src.rq2.phase1.ensemble import loader as L
from src.rq2.phase1.sensitivity import common as C
from src.rq2.phase1.sensitivity import figs as F

BC = C.CFG["bias"]
DESIGNS = list(C.CFG["designs"])
REAL_REG = None


def _regime_gk() -> pd.DataFrame:
    r = L.regime_frame("gk")[["i0", "cutoff", "vol_hi", "trend"]]
    return r


def _real_regime():
    s = L.regime_frame("space").iloc[0]
    return bool(s.vol_hi), s.trend


# ------------------------------------------------------------------ 원천별 bias 실현값 (long: source, design, code, cutoff, pos, e)
def _gk_long(model_col: str, simplex: bool = False) -> pd.DataFrame:
    if simplex:
        o = pd.read_parquet(C.ENSC / "online_gk.parquet")
        o = o[(o.window == "post_effective") & (o.strategy == "stat_simplex")]
        d = o[["code", "i0", "cutoff", "err"]].rename(columns={"err": "e"})
    else:
        d = pd.read_parquet(C.EXT / "gk" / f"{model_col}_win.parquet")
        d = d[d.window == "post_effective"]
        d = d.assign(e=d.ybar - d.predbar)[["code", "i0", "cutoff", "e"]]
    return d


def diff_simple(series_a: pd.Series, series_b: pd.Series, base_len: int, step: int, last_end: int | None = None):
    """같은 기업 차분 가짜 효과: mean(d[t0+10, t0+71)) − mean(d[t0−base_len, t0)). d 결측일은 제외."""
    d = (series_a - series_b).dropna()
    v, idx = d.values, d.index
    rows = []
    hi = (last_end if last_end is not None else len(v)) - 71
    for t0 in range(base_len, hi + 1, step):
        rows.append({"pos": t0, "cutoff": idx[t0], "e": v[t0 + 10:t0 + 71].mean() - v[t0 - base_len:t0].mean(),
                     "transition_e": v[t0:t0 + 10].mean() - v[t0 - base_len:t0].mean()})
    return pd.DataFrame(rows)


def bias_draws() -> pd.DataFrame:
    """모든 원천의 bias 실현값을 한 표로."""
    out = []
    gidx = L.gk_index()
    reg = _regime_gk().set_index("i0")
    vh, tr = _real_regime()
    for kind, simplex in [("bsts", False), ("simplex", True)]:
        g = _gk_long("bsts", simplex)
        g["pos"] = g.i0
        g["same_regime"] = g.i0.map(lambda i: bool(reg.vol_hi.get(i)) == vh and reg.trend.get(i) == tr)
        for c in C.TREAT + [C.NEG]:
            gc = g[g.code == c]
            des = f"{'BSTS' if kind == 'bsts' else 'SIMPLEX'}_{c}"
            out.append(gc.assign(source="gk_all", design=des, step=10))
            out.append(gc[gc.cutoff >= BC["recent_from"]].assign(source="gk_recent", design=des, step=10))
            out.append(gc[gc.same_regime].assign(source="gk_regime", design=des, step=10))
    # rv5 최근 (발표 전 완전 실현 post_effective 가짜 창)
    W = L.wide("rv5", "post_effective")
    w = C.stat_simplex_weights()["weights"]
    ms = [m for m in w if m in W.columns]
    for c in C.TREAT + [C.NEG]:
        Wc = W[W.code == c]
        out.append(pd.DataFrame({"code": c, "i0": Wc.i0, "pos": Wc.i0, "cutoff": Wc.cutoff, "e": Wc.ybar - Wc.bsts,
                                 "source": "rv5_recent", "design": f"BSTS_{c}", "step": 3}))
        out.append(pd.DataFrame({"code": c, "i0": Wc.i0, "pos": Wc.i0, "cutoff": Wc.cutoff,
                                 "e": Wc.ybar - Wc[ms].values @ np.array([w[m] for m in ms]),
                                 "source": "rv5_recent", "design": f"SIMPLEX_{c}", "step": 3}))
    # 공간 (같은 시기 donor)
    sp = pd.read_parquet(C.EXT / "space" / "bsts_win.parquet")
    sp = sp[sp.window == "post_effective"].set_index("code")
    dn = pd.read_csv(C.ENS / "exp6_space_donor_effects.csv", dtype={"code": str})
    dn = dn[(dn.window == "post_effective") & (dn.strategy == "stat_simplex")].set_index("code").effect
    bs = pd.read_csv(C.MAIN / "rq2_placebo_space_bsts.csv", dtype={"code": str}).set_index("code")
    donors = bs.index[~bs.treated]
    for c in C.TREAT + [C.NEG]:
        pr = sp.pre_rmse.reindex(donors).dropna()
        near = (pr - sp.pre_rmse[c]).abs().nsmallest(BC["similar_prefit_k"]).index
        for des, ser in [(f"BSTS_{c}", bs.post_effective), (f"SIMPLEX_{c}", dn)]:
            out.append(pd.DataFrame({"code": donors, "e": ser.reindex(donors).values, "source": "space_all",
                                     "design": des, "step": np.nan}))
            out.append(pd.DataFrame({"code": near, "e": ser.reindex(near).values, "source": "space_similar",
                                     "design": des, "step": np.nan}))
    # DIFF
    lh = pd.read_csv(C.MAIN / "rq2_longhist_placebo.csv")
    lh = lh[(lh.series == "005930-005935") & ~lh.actual]
    lh["cutoff"] = pd.to_datetime(lh.date)
    lh["pos"] = gidx.searchsorted(lh.cutoff)
    out.append(pd.DataFrame({"code": "005930", "cutoff": lh.cutoff, "pos": lh.pos, "e": lh.post_effective,
                             "source": "gk_diff_long", "design": "DIFF_005930", "step": 10}))
    LD = load_long_daily(C.RQ2)
    LD = LD.loc[:"2026-07-15"]
    gs = diff_simple(LD["005930"], LD["005935"], C.CFG["confounding"]["diff_listing_days"], 5)
    gs = gs[gs.cutoff + pd.Timedelta(days=110) < pd.Timestamp("2026-07-16")]
    out.append(gs.assign(code="005930", source="gk_diff_simple", design="DIFF_005930", step=5)
               .drop(columns="transition_e"))
    lv = load_panel("rv5").loc[C.RQ2["sample"]["start"]:"2026-07-15"]
    rs = diff_simple(lv["005930"], lv["005935"], C.CFG["confounding"]["diff_listing_days"], 3)
    out.append(rs.assign(code="005930", source="rv5_diff_simple", design="DIFF_005930", step=3)
               .drop(columns="transition_e"))
    D = pd.concat(out, ignore_index=True)
    return D[np.isfinite(D.e)]


def summarize_sources(D: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(C.SEED)
    for (des, src), g in D.groupby(["design", "source"], sort=False):
        e = g.sort_values("cutoff").e.to_numpy() if g.cutoff.notna().all() else g.e.to_numpy()
        n = len(e)
        step = g.step.iat[0]
        if np.isfinite(step):
            ne = C.n_eff(g.pos.nunique() if g.pos.notna().any() else n, int(step))
            blk = int(np.ceil(C.POST_L / step))
            se_blk = C.block_boot_se(e, blk, rng=rng)
        else:   # 공간 원천: 같은 기간 donor 단면 → 서로 다른 종목 (시간 겹침 없음, 단 공통 충격으로 상관)
            ne, se_blk = float(n), np.nan
        sd = e.std(ddof=1) if n > 1 else np.nan
        rows.append(dict(design=des, source=src, n=n, n_eff=ne, mean=e.mean(), sd=sd,
                         se_naive=sd / np.sqrt(n), se_neff=sd / np.sqrt(ne) if ne >= 3 else np.nan,
                         se_block=se_blk, q05=np.quantile(e, .05), q25=np.quantile(e, .25),
                         q50=np.median(e), q75=np.quantile(e, .75), q95=np.quantile(e, .95),
                         se_usable=bool(ne >= 3)))
    S = pd.DataFrame(rows)
    # 이동 2년 평균의 범위 (GK, 시간 의존 bias)
    roll = []
    for des, g in D[D.source.isin(["gk_all", "gk_diff_long"])].groupby("design"):
        g = g.sort_values("cutoff").set_index("cutoff").e
        r = g.rolling(f"{365 * BC['rolling_years']}D", min_periods=20).mean().dropna()
        roll.append(dict(design=des, source="gk_rolling2y", n=len(r), n_eff=np.nan, mean=r.mean(), sd=r.std(),
                         q05=r.min(), q95=r.max(), q50=r.median(), se_usable=False))
    return pd.concat([S, pd.DataFrame(roll)], ignore_index=True)


# ------------------------------------------------------------------ 별도 기간 검증 (온라인 bias 보정)
def validate(D: pd.DataFrame) -> pd.DataFrame:
    """검증 구간(2022-01~) 가짜 개입일마다, 그 전에 완전히 관측된 창(pos' + 71 <= pos)만으로 bias 를 추정하고
    보정 후 오차·구간 coverage·위양성률을 잰다. 참효과 0 이므로 coverage = 1 − FPR."""
    V = BC["validation"]
    reg = _regime_gk().set_index("i0")
    rows, per = [], []
    srcs = [("gk_all", d) for d in DESIGNS + ["BSTS_005935"] if not d.startswith("DIFF")] + [("gk_diff_long", "DIFF_005930")]
    for src, des in srcs:
        g = D[(D.source == src) & (D.design == des)].sort_values("pos").reset_index(drop=True)
        step = int(g.step.iat[0])
        pos, e, cut = g.pos.to_numpy(), g.e.to_numpy(), g.cutoff.to_numpy()
        tgt = np.where(cut >= np.datetime64(V["start"]))[0]
        for j in tgt:
            past = np.where(pos + 71 <= pos[j])[0]
            if len(past) < 10:
                continue
            ep = e[past]
            ne = C.n_eff(len(past), step)
            sd = ep.std(ddof=1)
            se = sd / np.sqrt(ne)
            recent = past[cut[past] >= cut[j] - np.timedelta64(365 * BC["rolling_years"], "D")]
            est = {"none": (0.0, 0.0),
                   "expanding": (ep.mean(), se),
                   "rolling2y": (e[recent].mean() if len(recent) >= 5 else ep.mean(),
                                 (e[recent].std(ddof=1) / np.sqrt(max(C.n_eff(len(recent), step), 1)))
                                 if len(recent) >= 5 else se),
                   "shrink": (ep.mean() * ne / (ne + V["shrink_kappa"]), se * ne / (ne + V["shrink_kappa"]))}
            if src == "gk_all":
                ip = g.i0.to_numpy()
                same = [k for k in past if reg.vol_hi.get(ip[k]) == reg.vol_hi.get(ip[j])
                        and reg.trend.get(ip[k]) == reg.trend.get(ip[j])]
                if len(same) >= 10:
                    es = e[same]
                    est["regime"] = (es.mean(), es.std(ddof=1) / np.sqrt(max(C.n_eff(len(same), step), 1)))
                else:
                    est["regime"] = est["expanding"]
            for k, (b, s) in est.items():
                per.append(dict(design=des, source=src, method=k, cutoff=cut[j], e=e[j], b_hat=b, se_b=s, sd_past=sd))
    P = pd.DataFrame(per)
    for (des, k), g in P.groupby(["design", "method"], sort=False):
        r = g.e - g.b_hat
        crit0 = 1.96 * g.sd_past
        crit1 = 1.96 * np.sqrt(g.sd_past ** 2 + g.se_b ** 2)
        step = 10
        rows.append(dict(design=des, method=k, n_target=len(g), n_eff_target=C.n_eff(len(g), step),
                         mean_err=r.mean(), rmse=np.sqrt(np.mean(r ** 2)), mae=np.mean(np.abs(r)),
                         mean_b_hat=g.b_hat.mean(), mean_se_b=g.se_b.mean(),
                         coverage95_no_se=np.mean(np.abs(r) <= crit0), coverage95_with_se=np.mean(np.abs(r) <= crit1),
                         fpr_with_se=np.mean(np.abs(r) > crit1)))
    R = pd.DataFrame(rows)
    base = R[R.method == "none"].set_index("design").rmse
    R["rmse_ratio_vs_none"] = R.rmse / R.design.map(base)
    return R, P


# ------------------------------------------------------------------ 격자 · 임계점
def grid_table(S: pd.DataFrame) -> pd.DataFrame:
    B = C.base_estimates()
    rows = []
    for des, spec in C.CFG["designs"].items():
        th = C.theta(des)
        null = C.null_dist(spec["kind"])
        c95 = C.crit95(null)
        sd0 = null.std(ddof=1)
        se_b = S[(S.design == des) & (S.source == ("gk_all" if spec["kind"] != "diff" else "gk_diff_long"))].se_neff.iat[0]
        for b in BC["grid"]:
            adj = th - b
            crit = np.sqrt(c95 ** 2 + (1.96 * se_b) ** 2)
            rows.append(dict(design=des, theta=th, b=b, adjusted_log=adj, adjusted_pct=C.pct(adj),
                             sign_changed=bool(np.sign(adj) != np.sign(th) and adj != 0),
                             p_emp=C.emp_p(adj, null), crit95=c95, se_b=se_b, crit95_with_se=crit,
                             sig_no_se=bool(abs(adj) > c95), sig_with_se=bool(abs(adj) > crit),
                             z_normal=adj / np.sqrt(sd0 ** 2 + se_b ** 2)))
    return pd.DataFrame(rows)


def tipping(S: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for des, spec in C.CFG["designs"].items():
        th = C.theta(des)
        null = C.null_dist(spec["kind"])
        c95 = C.crit95(null)
        main_src = "gk_all" if spec["kind"] != "diff" else "gk_diff_long"
        se_b = S[(S.design == des) & (S.source == main_src)].se_neff.iat[0]
        crit = np.sqrt(c95 ** 2 + (1.96 * se_b) ** 2)
        tp = {"sign_flip": th}
        for t in BC["targets_pct"]:
            tp[f"effect_{t}pct"] = th - float(C.lg(t))
        tp["sig_negative_no_se"] = th + c95
        tp["sig_negative_with_se"] = th + crit
        tp["sig_positive_with_se"] = th - crit
        for k, b in tp.items():
            r = dict(design=des, theta=th, theta_pct=C.pct(th), tipping=k, b_needed=b)
            for src, g in S[S.design == des].set_index("source").iterrows():
                inside = g.q05 <= b <= g.q95
                r[f"{src}__in_q05_q95"] = bool(inside)
            rows.append(r)
    T = pd.DataFrame(rows)
    # 경험적 분포에서 b >= b_needed (또는 b <= for 양의 방향) 일 확률
    return T


def tail_prob(D: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, t in T.iterrows():
        for src, g in D[D.design == t.design].groupby("source"):
            e = g.e.to_numpy()
            pos_dir = t.b_needed >= 0 if t.tipping != "sig_positive_with_se" else False
            p = np.mean(e >= t.b_needed) if t.tipping != "sig_positive_with_se" else np.mean(e <= t.b_needed)
            rows.append(dict(design=t.design, tipping=t.tipping, b_needed=t.b_needed, source=src, n=len(e),
                             share_bias_beyond=p, mean_bias=e.mean(), sd_bias=e.std(ddof=1),
                             mean_beyond=bool(e.mean() >= t.b_needed) if t.tipping != "sig_positive_with_se"
                             else bool(e.mean() <= t.b_needed), direction="b>=" if pos_dir or
                             t.tipping != "sig_positive_with_se" else "b<="))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 그림
def fig_curve(G, S):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), layout="constrained", sharey=False)
    bb = np.linspace(-0.3, 0.8, 221)
    for ax, group in zip(axs, [["BSTS_005930", "BSTS_000660", "SIMPLEX_005930", "SIMPLEX_000660"], ["DIFF_005930"]]):
        for des in group:
            th = C.theta(des)
            ax.plot(bb, C.pct(th - bb), color=F.COL[des], lw=1.8, label=f"{des} (θ̂={th:+.3f})")
            g = G[G.design == des]
            crit = g.crit95_with_se.iat[0]
            ax.axvspan(th + crit, bb.max(), color=F.COL[des], alpha=0.06, lw=0)
            ax.axvline(th + crit, color=F.COL[des], lw=0.8, ls=":")
        for t in BC["targets_pct"]:
            ax.axhline(t, color=F.GRID, lw=1, ls="--")
            ax.text(bb.min(), t, f"{t}%", fontsize=7, color=F.INK2, va="bottom")
        ax.axhline(0, color=F.INK2, lw=1)
        # 경험적 bias 범위 (q05~q95, 평균 점)
        srcs = S[S.design.isin(group) & S.source.isin(["gk_all", "gk_recent", "gk_regime", "rv5_recent", "space_similar",
                                                        "gk_diff_long", "gk_diff_simple", "rv5_diff_simple"])]
        y0 = -75
        for k, (_, s) in enumerate(srcs.iterrows()):
            yy = y0 - 4 * k
            ax.plot([s.q05, s.q95], [yy, yy], color=F.COL[s.design], lw=2, alpha=0.6)
            ax.plot(s["mean"], yy, "o", color=F.COL[s.design], ms=4)
            ax.text(s.q95 + 0.01, yy, f"{s.design[:6]}·{s.source}", fontsize=5.5, va="center", color=F.INK2)
        ax.set_xlabel("가정한 bias b (log, = 실제 무처치 − 반사실)")
        ax.set_ylabel("보정 효과 θ̂ − b (%)")
        ax.set_xlim(bb.min(), bb.max())
        ax.legend(fontsize=7, loc="upper right")
    axs[0].set_title("Bias 민감도 곡선 — 점선 오른쪽(음영) = 통계적으로 유의한 '감소' 판단 (bias 불확실성 포함)\n"
                     "아래 막대 = 경험적 bias q05~q95 (점=평균)", fontsize=8.5, loc="left")
    axs[1].set_title("같은 기업 차분 005930−005935", fontsize=8.5, loc="left")
    F.save(fig, "s1_bias_sensitivity_curve")


def fig_tipping(T, S):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6.2), layout="constrained")
    keys = ["sign_flip", "effect_-10pct", "effect_-20pct", "effect_-30pct", "sig_negative_with_se"]
    mk = {"sign_flip": "o", "effect_-10pct": "s", "effect_-20pct": "D", "effect_-30pct": "^", "sig_negative_with_se": "X"}
    lab = {"sign_flip": "부호 전환", "effect_-10pct": "−10%", "effect_-20pct": "−20%", "effect_-30pct": "−30%",
           "sig_negative_with_se": "유의한 감소 (bias SE 포함)"}
    for i, des in enumerate(C.CFG["designs"]):
        y = i
        main = ["gk_all", "gk_regime", "rv5_recent", "space_similar"] if not des.startswith("DIFF") else \
            ["gk_diff_long", "gk_diff_simple", "rv5_diff_simple"]
        for j, src in enumerate(main):
            s = S[(S.design == des) & (S.source == src)]
            if len(s):
                s = s.iloc[0]
                yy = y + 0.12 * (j - 1.5)
                ax.plot([s.q05, s.q95], [yy, yy], color=F.CAT[j + 3 if j < 2 else j + 4], lw=3, alpha=0.45,
                        solid_capstyle="butt", label=src if i in (0, 4) else None)
        for k in keys:
            b = T[(T.design == des) & (T.tipping == k)].b_needed.iat[0]
            ax.plot(b, y, mk[k], color=F.COL[des], ms=8, mec=F.SURF, label=lab[k] if i == 0 else None)
    ax.set_yticks(range(len(C.CFG["designs"])), list(C.CFG["designs"]))
    ax.invert_yaxis()
    ax.axvline(0, color=F.INK2, lw=1)
    ax.set_xlabel("결론을 바꾸는 데 필요한 bias b (log)")
    ax.set_title("Tipping point — 기호: 그 판단에 필요한 b / 막대: 경험적 bias q05~q95 (원천별)", fontsize=9, loc="left")
    ax.legend(fontsize=7, ncol=3, loc="lower right")
    F.save(fig, "s1_tipping_points")


def negative_control(S: pd.DataFrame) -> pd.DataFrame:
    """음성통제 005935 (규제 비대상) 에 같은 bias 보정을 적용: 보정 후 효과가 0 근처로 가면 그 원천의 bias 보정이 타당하다는 근거."""
    pb = pd.read_csv(C.MAIN / "rq2_primary_bsts.csv", dtype={"code": str})
    ex = pd.read_csv(C.ENS / "exp6_real_effect.csv", dtype={"code": str})
    th = {"BSTS": pb[(pb.code == C.NEG) & (pb.window == "post_effective")].effect_log.iat[0],
          "SIMPLEX": ex[(ex.code == C.NEG) & (ex.window == "post_effective") & (ex.strategy == "stat_simplex")].effect_log.iat[0]}
    rows = []
    for fam in ["BSTS", "SIMPLEX"]:
        for src in ["gk_all", "gk_recent", "gk_regime", "rv5_recent", "space_all", "space_similar"]:
            r = dict(family=fam, source=src)
            for c in C.TREAT + [C.NEG]:
                s = S[(S.design == f"{fam}_{c}") & (S.source == src)]
                b = s["mean"].iat[0] if len(s) else np.nan
                t0 = th[fam] if c == C.NEG else C.theta(f"{fam}_{c}")
                r[f"b_{c}"] = b
                r[f"adj_{c}"] = t0 - b
                r[f"adj_pct_{c}"] = C.pct(t0 - b)
            rows.append(r)
    return pd.DataFrame(rows)


def run():
    D = bias_draws()
    S = summarize_sources(D)
    R, P = validate(D)
    G = grid_table(S)
    T = tipping(S)
    TP = tail_prob(D, T)
    NC = negative_control(S)
    C.save_table(NC, "s1_negative_control_correction")
    C.save_table(S, "s1_bias_sources")
    C.save_table(R, "s1_bias_validation")
    C.save_table(G, "s1_bias_grid")
    C.save_table(T, "s1_tipping_points")
    C.save_table(TP, "s1_tipping_vs_empirical")
    fig_curve(G, S)
    fig_tipping(T, S)
    # 로그
    for des in C.CFG["designs"]:
        s = S[S.design == des]
        t = T[T.design == des].set_index("tipping").b_needed
        C.LOG.add(id=f"RQ2-S1-{des}", rq="RQ2", kind="robustness",
                  purpose="반사실 bias 민감도: 보정 효과 θ̂−b, 부호·크기·유의성 임계점, 경험적 bias 범위와 비교",
                  data="주 분석 θ̂ + 기존 placebo 캐시 (GK 238, rv5 32, space 175, GK 차분 237)",
                  model=des, settings={"grid": BC["grid"], "targets_pct": BC["targets_pct"]},
                  diagnostics={r.source: round(r["mean"], 3) for _, r in s.iterrows()},
                  result={k: round(v, 3) for k, v in t.items()},
                  interpretation="시나리오 분석 (보정된 인과효과 아님)", seed=C.SEED)
    C.LOG.add(id="RQ2-S1-VAL", rq="RQ2", kind="diagnostic", purpose="온라인 bias 보정의 별도 기간(2022~) 검증",
              data="GK 가짜 개입일", model="none/expanding/rolling2y/shrink/regime",
              settings=BC["validation"],
              result=R[["design", "method", "rmse_ratio_vs_none", "coverage95_with_se"]].round(3).to_dict("records"),
              seed=C.SEED)
    return dict(D=D, S=S, R=R, G=G, T=T, TP=TP, NC=NC)


if __name__ == "__main__":
    out = run()
    C.LOG.save()
    pd.set_option("display.width", 250)
    print(out["S"].round(3).to_string())
    print(out["R"].round(3).to_string())
    print(out["NC"].round(3).to_string())
    print(out["TP"][out["TP"].tipping.isin(["effect_-10pct", "effect_-20pct", "sig_negative_with_se"])].round(3).to_string())
