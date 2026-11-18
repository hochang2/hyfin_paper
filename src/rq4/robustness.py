"""5단계: 강건성. 모든 명세를 같은 '창 함수' 로 계산하고, 같은 함수를 역사적 비중첩 가짜 창에 적용해 위약 p 를 얻는다.
(명세마다 귀무분포를 새로 만들어 명세 탐색으로 인한 유리한 선택을 막음.)

R-O 결과변수: log 거래량, 회전율(log 거래대금/전일 순자산), 순유입/순자산, 2X 거래대금 점유율, 합계 log 격차
R-C 표본·통제: 하한 없음, 불균형 패널(FE, CR1), 공변량(변동성×2X)
R-W 창: 사후 10/20/30/60일, 발표일 기준(과도기 포함), 도넛(사후 첫 5일 제외), 기준 마지막 10/20일,
        장기 기준선(long_pre), 상장 직전 35일 기준
R-K 구성: 계열별 최대 2X·1X 제외, 243880 제외, 306950 추가, 처치·대조 하나씩 빼기(LOO 범위)
R-M 모형: 매칭 DID(기준 수준·기울기 최근접 3), IPW, DR, SDID(기준=base, 기준=long_pre+base)
다중검정: H1(풀링) 명세들, H2(차이) 명세들 각각 BH. (주 결론은 사전 지정 주 명세의 Holm)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize

from src.rq4 import collapsed as C
from src.rq4.collapsed import FLOOR, actual_positions, default_spec, placebo_p, placebo_windows, wide
from src.rq4.common import CFG, FIG, PAL, INK2, ExpLog, plt_setup, save_table
from src.rq4.did import fit
from src.rq4.estim import bh
from src.rq4.panel import P, build_panel, main_panel, window

LOG = ExpLog("08_robustness")
G = CFG["groups"]
KEYS = ["pooled", "idx", "semi", "diff"]


# ---------- 창 함수들 ----------
def cohort(ib, ip, codes, outcome="log_value", floor=FLOOR):
    W, V = wide(outcome)
    cs = [c for c in codes if c in W.columns]
    allpos = np.concatenate([ib, ip])
    ok = W.iloc[allpos][cs].notna().all() & (V.iloc[allpos][cs] > 0).all()
    if floor:
        ok &= V.iloc[ib][cs].median() >= floor
    return [c for c in cs if ok[c]]


def combine(per):
    """per: {f: (delta, nT, nC)} → pooled/idx/semi/diff"""
    out, ws, num = {}, 0.0, 0.0
    for f, (dl, nT, nC) in per.items():
        out[f] = dl
        if np.isfinite(dl) and nT and nC:
            w = nT * nC / (nT + nC)
            ws += w
            num += w * dl
    out["pooled"] = num / ws if ws else np.nan
    out["diff"] = out.get("semi", np.nan) - out.get("idx", np.nan)
    return out


def fn_gap(spec=None, outcome="log_value", floor=FLOOR, agg=None):
    """agg=None: 동일가중 평균 격차, 'logsum': log Σ2X − log Σ1X, 'share': Σ2X/(Σ2X+Σ1X)"""
    spec = spec or default_spec()

    def f(ib, ip):
        W, V = wide(outcome)
        per = {}
        for fam, (tr, co) in spec.items():
            t, c = cohort(ib, ip, tr, outcome, floor), cohort(ib, ip, co, outcome, floor)
            if not t or not c:
                per[fam] = (np.nan, 0, 0)
                continue
            if agg is None:
                g = W[t].mean(axis=1) - W[c].mean(axis=1)
            elif agg == "logsum":
                g = np.log(V[t].sum(axis=1)) - np.log(V[c].sum(axis=1))
            else:
                g = V[t].sum(axis=1) / (V[t].sum(axis=1) + V[c].sum(axis=1))
            per[fam] = (g.iloc[ip].mean() - g.iloc[ib].mean(), len(t), len(c))
        return combine(per)
    return f


def fn_match(k=3):
    spec = default_spec()

    def f(ib, ip):
        W, _ = wide()
        per = {}
        for fam, (tr, co) in spec.items():
            t, c = cohort(ib, ip, tr), cohort(ib, ip, co)
            if not t or not c:
                per[fam] = (np.nan, 0, 0)
                continue
            Yb = W.iloc[ib]
            lvl = Yb.mean()
            tt = np.arange(len(ib))
            slope = Yb.apply(lambda s: np.polyfit(tt, s.values, 1)[0] if s.notna().all() else np.nan)
            Z = pd.DataFrame({"lvl": lvl, "slope": slope * len(ib)})
            Z = (Z - Z.loc[t + c].mean()) / Z.loc[t + c].std()
            dlt = W.iloc[ip].mean() - Yb.mean()
            eff = []
            for u in t:
                dist = ((Z.loc[c] - Z.loc[u]) ** 2).sum(axis=1).sort_values()
                eff.append(dlt[u] - dlt[dist.index[:k]].mean())
            per[fam] = (float(np.mean(eff)), len(t), min(k, len(c)) * 0 + len(c))
        return combine(per)
    return f


def fn_ipw(dr=False):
    spec = default_spec()

    def f(ib, ip):
        W, _ = wide()
        rows = []
        for fam, (tr, co) in spec.items():
            t, c = cohort(ib, ip, tr), cohort(ib, ip, co)
            Yb = W.iloc[ib]
            for u in t + c:
                rows.append(dict(code=u, fam=fam, D=int(u in t), lvl=Yb[u].mean(), sd=Yb[u].diff().std(),
                                 dlt=W.iloc[ip][u].mean() - Yb[u].mean()))
        df = pd.DataFrame(rows)
        res = {"flag": ""}
        per = {}
        for fam, g in df.groupby("fam"):
            X = sm.add_constant(g[["lvl", "sd"]])
            try:
                m = sm.Logit(g["D"], X).fit(disp=0, maxiter=200)
                ps = m.predict(X).clip(1e-6, 1 - 1e-6)
                if (ps > .99).any() or (ps < .01).any() or not m.mle_retvals.get("converged", True):
                    res["flag"] += f"{fam}:극단성향 "
            except Exception as ex:  # 완전분리
                res["flag"] += f"{fam}:{type(ex).__name__} "
                per[fam] = (np.nan, 0, 0)
                continue
            wc = ps / (1 - ps)
            T, Cm = g["D"] == 1, g["D"] == 0
            if dr:
                om = sm.WLS(g.loc[Cm, "dlt"], X[Cm], weights=wc[Cm]).fit()
                mu = om.predict(X)
                rT = (g.loc[T, "dlt"] - mu[T]).mean()
                rC = np.average(g.loc[Cm, "dlt"] - mu[Cm], weights=wc[Cm])
            else:
                rT = g.loc[T, "dlt"].mean()
                rC = np.average(g.loc[Cm, "dlt"], weights=wc[Cm])
            per[fam] = (rT - rC, int(T.sum()), int(Cm.sum()))
        out = combine(per)
        out["flag"] = res["flag"]
        return out
    return f


def _simplex_ls(A, b, reg=0.0, intercept=True):
    """min ||a0 + A w − b||² + reg ||w||², w ≥ 0, Σw = 1"""
    n = A.shape[1]

    def obj(x):
        a0, w = (x[0], x[1:]) if intercept else (0.0, x)
        r = a0 + A @ w - b
        return r @ r + reg * (w @ w)
    x0 = np.r_[0.0, np.ones(n) / n] if intercept else np.ones(n) / n
    k = 1 if intercept else 0
    cons = [{"type": "eq", "fun": lambda x: x[k:].sum() - 1}]
    bnds = ([(None, None)] if intercept else []) + [(0, 1)] * n
    r = optimize.minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 500})
    return r.x[k:]


def sdid_one(YT, YC, npre):
    """YT: (T,) 처치 평균 경로, YC: (T,Nc). 앞 npre 행이 사전. Arkhangelsky et al.(2021) 단순화 구현."""
    pre, post = slice(0, npre), slice(npre, None)
    Tpost = YT[post].shape[0]
    sig = np.std(np.diff(YC[pre], axis=0))
    zeta = (1 * Tpost) ** 0.25 * sig
    w = _simplex_ls(YC[pre], YT[pre], reg=zeta ** 2 * npre)
    lam = _simplex_ls(YC[pre].T, YC[post].mean(axis=0), reg=1e-6 * sig ** 2)
    tau = (YT[post].mean() - lam @ YT[pre]) - (w @ YC[post].mean(axis=0) - w @ (lam @ YC[pre]))
    return tau


def fn_sdid():
    spec = default_spec()

    def f(ib, ip):
        W, _ = wide()
        per = {}
        for fam, (tr, co) in spec.items():
            t, c = cohort(ib, ip, tr), cohort(ib, ip, co)
            if not t or not c:
                per[fam] = (np.nan, 0, 0)
                continue
            pos = np.concatenate([ib, ip])
            YT = W.iloc[pos][t].mean(axis=1).values
            YC = W.iloc[pos][c].values
            per[fam] = (sdid_one(YT, YC, len(ib)), len(t), len(c))
        return combine(per)
    return f


# ---------- 실행 틀 ----------
def run_spec(fn, nb=None, ngap=None, npost=None, ib=None, ip=None):
    W, _ = wide()
    if ib is None:
        ib, ip = actual_positions()
    nb = nb or len(ib)
    npost = npost or len(ip)
    ngap = ngap if ngap is not None else ip[0] - ib[-1] - 1
    act = fn(ib, ip)
    dist = pd.DataFrame([fn(b, p) for b, p in placebo_windows(nb, ngap, npost)])
    res = {}
    for k in KEYS:
        res[k] = act[k]
        pr = placebo_p(act[k], dist[k]) if k in dist else dict(p=np.nan, n=0, ci_low=np.nan, ci_high=np.nan)
        res[f"p_{k}"], res[f"n_{k}"] = pr["p"], pr["n"]
        res[f"lo_{k}"], res[f"hi_{k}"] = pr["ci_low"], pr["ci_high"]
    res["flag"] = act.get("flag", "")
    return res


def pos_between(a, b):
    days = wide()[0].index
    return np.where((days >= pd.Timestamp(a)) & (days <= pd.Timestamp(b)))[0]


def main():
    t0 = time.time()
    plt = plt_setup()
    W, V = wide()
    days = W.index
    ib, ip = actual_positions()
    specs = []   # (group, label, kwargs for run_spec)
    add = lambda g, lab, fn, **kw: specs.append((g, lab, fn, kw))

    add("주", "주 명세 (log 거래대금, base vs post)", fn_gap())
    # 결과변수
    for o, lab in [("log_volume", "log 거래량"), ("turnover", "회전율 log(거래대금/전일 순자산)"),
                   ("flow_aum", "순유입/전일 순자산")]:
        add("결과", lab, fn_gap(outcome=o))
    add("결과", "합계 log 격차 (거래대금 가중)", fn_gap(agg="logsum"))
    add("결과", "2X 점유율 Σ2X/(Σ2X+Σ1X) (수준, %p)", fn_gap(agg="share"))
    # 표본
    add("표본", "활동 하한 없음", fn_gap(floor=None))
    # 창
    for n in [10, 20, 30, 60]:
        add("창", f"사후 첫 {n}일", fn_gap(), ib=ib, ip=ip[:n])
    ann = pos_between(P["transition"][0], P["post"][1])
    add("창", "발표일 기준 (과도기 포함 사후)", fn_gap(), ib=ib, ip=ann)
    add("창", "도넛: 사후 첫 5일 제외", fn_gap(), ib=ib, ip=ip[5:])
    for n in [10, 20]:
        add("창", f"기준 = base 마지막 {n}일", fn_gap(), ib=ib[-n:], ip=ip)
    lp = pos_between(*P["long_pre"])
    add("창", "기준 = long_pre (상장 전 2025-10~05-26)", fn_gap(), ib=lp, ip=ip)
    add("창", "기준 = 상장 직전 35일", fn_gap(), ib=lp[-35:], ip=ip)
    add("창", "기준 = 2025-01~05-26 (장기)", fn_gap(), ib=pos_between("2025-01-02", P["long_pre"][1]), ip=ip)
    # 구성
    sp = default_spec()
    for drop, lab in [(["122630", "494310"], "계열별 최대 2X 제외 (122630, 494310)"),
                      (["069500", "091160"], "계열별 최대 1X 제외 (069500, 091160)"),
                      (G["C_semi_2x_sens_drop"], "243880 제외 (반도체 2X 정의 민감)")]:
        s2 = {f: ([c for c in t if c not in drop], [c for c in co if c not in drop]) for f, (t, co) in sp.items()}
        add("구성", lab, fn_gap(spec=s2))
    s3 = {"idx": (G["B_index_2x"] + G["B_index_2x_sens"], G["E_index_1x"]), "semi": sp["semi"]}
    add("구성", "306950 추가 (지수 2X 정의 민감)", fn_gap(spec=s3))
    s4 = {"idx": sp["idx"], "semi": (sp["semi"][0], G["E_semi_1x"] + G["SP_concentrated_1x"])}
    add("구성", "반도체 대조에 삼성·SK 집중 1X 추가 (오염 위험)", fn_gap(spec=s4))
    # 모형
    add("모형", "매칭 DID (기준 수준·기울기 최근접 3)", fn_match())
    add("모형", "IPW (성향: 기준 수준·변동)", fn_ipw())
    add("모형", "DR (IPW + 결과회귀)", fn_ipw(dr=True))
    add("모형", "SDID (사전 = base)", fn_sdid())
    lpb = pos_between(P["long_pre"][0], P["listing_base"][1])
    add("모형", "SDID (사전 = long_pre+base)", fn_sdid(), ib=lpb, ip=ip)

    rows = []
    for g, lab, fn, kw in specs:
        ts = time.time()
        r = run_spec(fn, **kw)
        r.update(group=g, spec=lab, runtime_s=time.time() - ts)
        rows.append(r)
        print(f"{lab[:30]:32s} pooled {r['pooled']:+.3f} (p {r['p_pooled']:.3f})  idx {r['idx']:+.3f} (p {r['p_idx']:.3f})"
              f"  semi {r['semi']:+.3f} (p {r['p_semi']:.3f})  diff {r['diff']:+.3f} (p {r['p_diff']:.3f}) {r['flag']}",
              flush=True)

    # LOO (점추정 범위만; 위약 없음)
    loo = []
    f0 = None
    for fam, (tr, co) in sp.items():
        for u in tr + co:
            s2 = {f: ([c for c in t if c != u], [c for c in cc if c != u]) for f, (t, cc) in sp.items()}
            r = fn_gap(spec=s2)(ib, ip)
            loo.append(dict(drop=u, family=fam, role="2X" if u in tr else "1X", **{k: r[k] for k in KEYS}))
    loo = pd.DataFrame(loo)
    save_table(loo, "t08b_leave_one_out")
    for k in KEYS:
        rows.append(dict(group="구성", spec=f"LOO 범위 ({k})", **{k: np.nan for k in KEYS},
                         flag=f"min {loo[k].min():+.3f} ({loo.loc[loo[k].idxmin(), 'drop']}), "
                              f"max {loo[k].max():+.3f} ({loo.loc[loo[k].idxmax(), 'drop']})"))

    # FE 기반 (CR1): 불균형 패널, 공변량
    from src.rq4.diagnostics import covariates
    for lab, d, extra in [("불균형 패널 (FE, CR1)", window(build_panel(balanced=False, min_value=FLOOR)), []),
                          ("공변량: 계열 실현변동성×2X, VKOSPI×2X (FE, CR1)", covariates(window(main_panel())),
                           ["x_rv", "x_vk"])]:
        dd = d.dropna(subset=extra) if extra else d
        ms, _ = fit(dd, extra=extra)
        mp, _ = fit(dd, split=False, extra=extra)
        s = ms.summary("cr1", cluster="code")
        sp_ = mp.summary("cr1", cluster="code")
        w = np.array([1.0 if n == "post_semi" else -1.0 if n == "post_idx" else 0 for n in ms.names])
        lc = ms.lincomb(w, "cr1", cluster="code") if hasattr(ms, "lincomb") else None
        rows.append(dict(group="표본" if "불균형" in lab else "통제", spec=lab, pooled=mp.b[0], p_pooled=sp_.loc["post_2x", "p"],
                         idx=s.loc["post_idx", "coef"], p_idx=s.loc["post_idx", "p"],
                         semi=s.loc["post_semi", "coef"], p_semi=s.loc["post_semi", "p"],
                         diff=w @ ms.b, p_diff=lc["p"] if lc is not None else np.nan,
                         flag=f"CR1 상품 (위약 아님), N상품={dd.code.nunique()}"))
    out = pd.DataFrame(rows)
    # 다중검정 BH (위약 p 있는 명세)
    for k in ["pooled", "diff"]:
        m = out[f"p_{k}"].notna() & (out["group"] != "주") & ~out["flag"].astype(str).str.startswith("CR1")
        adj = bh(dict(zip(out.index[m], out.loc[m, f"p_{k}"])))
        out[f"p_{k}_bh"] = out.index.map(adj)
    save_table(out, "t08_robustness")

    # forest plot
    show = out[out[["lo_diff"]].notna().all(axis=1)].reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.32 * len(show) + 1.4), sharey=True)
    for ax, k, lab in zip(axes, ["pooled", "idx", "semi"] if False else ["pooled", "semi", "diff"],
                          ["H1 풀링 β", "β_C 반도체", "H2 β_C-β_B"]):
        y = np.arange(len(show))[::-1]
        col = [PAL[1] if g == "주" else PAL[0] for g in show.group]
        ax.hlines(y, show[f"lo_{k}"] + 0 * show[k], show[f"hi_{k}"], color=col, lw=1.2)
        # ci 는 (actual − null mean) 중심 → 점도 같은 중심으로
        ctr = (show[f"lo_{k}"] + show[f"hi_{k}"]) / 2
        ax.scatter(ctr, y, color=col, s=16, zorder=3)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(lab, fontsize=9)
    axes[0].set_yticks(np.arange(len(show))[::-1])
    axes[0].set_yticklabels([s[:34] for s in show.spec], fontsize=7)
    fig.suptitle("그림 6. 강건성 forest plot (위약-평균 보정 효과, 역사적 위약 95% 예측구간)", fontsize=10)
    fig.savefig(FIG / "f06_forest.png")
    plt.close(fig)

    for i, r in out.iterrows():
        LOG.add(id=f"R{i:02d}", phase="5_robustness", kind="robustness", hypothesis="H1/H2", purpose=r["group"],
                treated="B∪C", control="E", period="base vs post (명세별)", outcome=r["spec"],
                model=r["spec"], estimand="pooled / β_B / β_C / diff",
                estimate=f"pooled {r['pooled']:+.3f}; idx {r['idx']:+.3f}; semi {r['semi']:+.3f}; diff {r['diff']:+.3f}"
                if np.isfinite(r["pooled"]) else "",
                p=r.get("p_diff"), p_adj=r.get("p_diff_bh"), correction="BH (H2 명세군)",
                inference="역사적 위약 비중첩" if not str(r["flag"]).startswith("CR1") else "CR1",
                robustness=f"p_pooled {r.get('p_pooled', np.nan):.3f} (BH {r.get('p_pooled_bh', np.nan):.3f}); "
                           f"p_idx {r.get('p_idx', np.nan):.3f}; p_semi {r.get('p_semi', np.nan):.3f}",
                failure_revision=str(r["flag"]), runtime_s=r.get("runtime_s"))
    LOG.save()
    print(f"총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
