"""3단계-1: 주 DID (H1, H2). 상품 FE + 계열×일 FE.

  log v_it = α_i + γ_{f(i),t} + β·(2X_i×Post_t) + δ·(2X_i×Trans_t) + ε_it
  분리형: β_B (지수), β_C (반도체). H2 = β_C − β_B.

식별 문제와 대응
· 시장 전체 거래 증감 → 같은 기초 계열 1X 를 대조로, 계열×일 FE 로 날짜별 공통충격 제거 (질문 2).
· 7/16~7/30 은 발표·조기시행 예고·서킷브레이커가 겹친 과도기 → 별도 더미, 기준선에 넣지 않음.
· 클러스터가 적다(처치 11) → 주 추론은 역사적 시간 위약(collapsed.py). 클러스터 SE 들은 비교용.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.rq4.collapsed import actual_positions, placebo_dist, placebo_p, window_effect
from src.rq4.common import CFG, SEED, ExpLog, save_table
from src.rq4.estim import FE, bh, holm
from src.rq4.panel import main_panel, window

LOG = ExpLog("03_did")


def design(d: pd.DataFrame, split=True) -> tuple[pd.DataFrame, list[str]]:
    d = d.copy()
    if split:
        cols = []
        for f in ["idx", "semi"]:
            m = (d["family"] == f).astype(int) * d["lev2"]
            d[f"post_{f}"] = m * d["post"]
            d[f"trans_{f}"] = m * d["trans"]
            cols += [f"post_{f}", f"trans_{f}"]
        return d, cols
    d["post_2x"] = d["lev2"] * d["post"]
    d["trans_2x"] = d["lev2"] * d["trans"]
    return d, ["post_2x", "trans_2x"]


def fit(d, y="log_value", split=True, extra=(), fe=("code", "fd")):
    d, X = design(d, split)
    return FE(d, y, X + list(extra), list(fe)), X


def all_se(m, names, wild_reps=None, seed=SEED, h2=True):
    """같은 적합에서 여러 SE. 반환: 행 = (term, method)."""
    rows = []
    specs = [("CR1 상품", "cr1", "code", None), ("CR1 날짜", "cr1", "date", None),
             ("2-way 상품·날짜", "twoway", "code", "date"), ("Driscoll–Kraay", "dk", "date", None),
             ("HC1(독립)", "hc1", "code", None)]
    for lab, k, c1, c2 in specs:
        s = m.summary(k, cluster=c1, cluster2=c2)
        for t in s.index:
            rows.append(dict(term=t, method=lab, **s.loc[t].to_dict()))
        if h2 and "post_semi" in m.names:
            w = np.array([1.0 if n == "post_semi" else -1.0 if n == "post_idx" else 0.0 for n in m.names])
            V = m.vcov(k, c1, c2)
            est, se = w @ m.b, float(np.sqrt(w @ V @ w))
            from scipy import stats
            dof = s["dof"].iloc[0]
            rows.append(dict(term="H2: post_semi − post_idx", method=lab, coef=est, se=se, t=est / se,
                             p=2 * stats.t.sf(abs(est / se), dof), ci_low=est - stats.t.ppf(.975, dof) * se,
                             ci_high=est + stats.t.ppf(.975, dof) * se, dof=dof))
    if wild_reps:
        for j, t in enumerate(m.names):
            if not t.startswith("post"):
                continue
            r = m.wild(j=j, reps=wild_reps, seed=seed)
            rows.append(dict(term=t, method="Wild cluster(Webb) 상품", coef=m.b[j], se=np.nan, t=r["t"], p=r["p"],
                             ci_low=r["ci_low"], ci_high=r["ci_high"], dof=r["G"]))
        if h2 and "post_semi" in m.names:
            w = np.array([1.0 if n == "post_semi" else -1.0 if n == "post_idx" else 0.0 for n in m.names])
            r = m.wild(w=w, reps=wild_reps, seed=seed)
            rows.append(dict(term="H2: post_semi − post_idx", method="Wild cluster(Webb) 상품", coef=w @ m.b,
                             se=np.nan, t=r["t"], p=r["p"], ci_low=r["ci_low"], ci_high=r["ci_high"], dof=r["G"]))
    return pd.DataFrame(rows)


def randomization(d, reps=5000, seed=SEED):
    """상품 무작위화: 계열 안에서 2X 라벨을 같은 개수로 재배정 (교환가능성 가정 — 2X 와 1X 는 성질이 달라
    이 가정은 약하다는 점을 보고서에 명시)."""
    rng = np.random.default_rng(seed)
    base, X = fit(d, split=True)
    obs = {"post_idx": base.b[0], "post_semi": base.b[2]}
    obs["pooled"] = fit(d, split=False)[0].b[0]
    obs["diff"] = obs["post_semi"] - obs["post_idx"]
    prods = d.groupby("code").agg(family=("family", "first"), lev2=("lev2", "first")).reset_index()
    draws = []
    # 빠른 계산: 접힌 격차로 (균형 패널이면 FE 회귀와 동일)
    piv = d.pivot(index="date", columns="code", values="log_value")
    post = d.groupby("date")["post"].first().reindex(piv.index).values == 1
    base_ = d.groupby("date")["base"].first().reindex(piv.index).values == 1
    delta_i = piv[post].mean() - piv[base_].mean()   # 상품별 사후−기준 변화
    for _ in range(reps):
        res = {}
        ws, num = 0, 0
        for f in ["idx", "semi"]:
            p = prods[prods.family == f]
            nT = int(p.lev2.sum())
            perm = rng.permutation(p.code.values)
            tr, co = perm[:nT], perm[nT:]
            dl = delta_i[tr].mean() - delta_i[co].mean()
            res[f] = dl
            w = nT * len(co) / (nT + len(co))
            ws += w
            num += w * dl
        res["pooled"] = num / ws
        res["diff"] = res["semi"] - res["idx"]
        draws.append(res)
    dr = pd.DataFrame(draws)
    out = {}
    for k, kk in [("pooled", "pooled"), ("post_idx", "idx"), ("post_semi", "semi"), ("diff", "diff")]:
        out[k] = dict(obs=obs[k], p=(1 + (dr[kk].abs() >= abs(obs[k])).sum()) / (reps + 1))
    return out


def primary(d=None, log=True):
    d = window(d if d is not None else main_panel())
    mp, _ = fit(d, split=False)
    ms, _ = fit(d, split=True)
    se_p = all_se(mp, mp.names, wild_reps=CFG["inference"]["wild_reps"], h2=False)
    se_s = all_se(ms, ms.names, wild_reps=CFG["inference"]["wild_reps"])
    se_all = pd.concat([se_p.assign(model="pooled"), se_s.assign(model="split")])
    # 주 추론: 역사적 시간 위약
    ib, ip = actual_positions()
    act = window_effect(ib, ip)
    dist = placebo_dist()
    dist_ovl = placebo_dist(step=20)
    dist.to_csv(save_table(dist, "t03b_placebo_windows_nonoverlap"), index=False, encoding="utf-8-sig")
    save_table(dist_ovl, "t03c_placebo_windows_overlap20")
    pl = {}
    for k in ["pooled", "idx", "semi", "diff"]:
        pl[k] = placebo_p(act[k], dist[k])
        pl[k + "_ovl"] = placebo_p(act[k], dist_ovl[k])
    ri = randomization(d, CFG["inference"]["ri_reps"])
    for k, lab in [("pooled", "pooled"), ("post_idx", "idx"), ("post_semi", "semi"), ("diff", "diff")]:
        se_all = pd.concat([se_all, pd.DataFrame([dict(term=k, method="무작위화(계열 내 라벨)", coef=ri[k]["obs"],
                                                        p=ri[k]["p"], model="split" if k != "pooled" else "pooled")])])
    for k in ["pooled", "idx", "semi", "diff"]:
        for suf, lab in [("", "역사적 위약(비중첩)"), ("_ovl", "역사적 위약(20일 간격 중첩, 참고)")]:
            r = pl[k + suf]
            se_all = pd.concat([se_all, pd.DataFrame([dict(term=k, method=lab, coef=act[k], se=r["sd"], p=r["p"],
                                                            ci_low=r["ci_low"], ci_high=r["ci_high"], dof=r["n"],
                                                            model="collapsed")])])
    # 다중검정: 주 가족 Holm (주 추론 p)
    prim = {"H1_pooled": pl["pooled"]["p"], "H2_diff": pl["diff"]["p"]}
    prim_adj = holm(prim)
    sec = {"beta_idx": pl["idx"]["p"], "beta_semi": pl["semi"]["p"]}
    se_all["p_holm_primary"] = np.nan
    save_table(se_all, "t03_primary_did_all_se")
    if log:
        LOG.add(id="M01", phase="3_main", kind="baseline", hypothesis="H1",
                purpose="비규제 국내 2X ETF 거래대금이 1X 대비 상대적으로 늘었는가 (풀링)",
                treated=f"B∪C {d[d.lev2 == 1].code.nunique()}종", control=f"E {d[d.lev2 == 0].code.nunique()}종",
                period="base 05-27~07-15 vs post 07-31~10-30 (trans 별도)", outcome="log 거래대금",
                model="상품 FE + 계열×일 FE", estimand="2X−1X 격차 변화 (동일가중)",
                estimate=mp.b[0], ci_low=pl["pooled"]["ci_low"], ci_high=pl["pooled"]["ci_high"],
                se=pl["pooled"]["sd"], p=pl["pooled"]["p"], p_adj=prim_adj["H1_pooled"], correction="Holm(H1,H2)",
                inference=f"역사적 시간 위약 비중첩 n={pl['pooled']['n']}",
                robustness="; ".join(f"{r.method}: p={r.p:.3f}" for r in se_all[(se_all.model == 'pooled') &
                                                                              (se_all.term.isin(['post_2x', 'pooled']))].itertuples()),
                interpretation=f"{100 * np.expm1(mp.b[0]):+.1f}%. 지지 안 됨 (유의하지 않고 부호도 음)" if mp.b[0] < 0 or prim_adj["H1_pooled"] > .05 else "지지")
        LOG.add(id="M02", phase="3_main", kind="baseline", hypothesis="H2",
                purpose="반도체형과 지수형의 변화 차이 (방향 무가정)", treated="C vs B", control="각 계열 1X",
                period="base vs post", outcome="log 거래대금", model="분리형 DID (=DDD)",
                estimand="β_C − β_B", estimate=act["diff"], ci_low=pl["diff"]["ci_low"], ci_high=pl["diff"]["ci_high"],
                se=pl["diff"]["sd"], p=pl["diff"]["p"], p_adj=prim_adj["H2_diff"], correction="Holm(H1,H2)",
                inference=f"역사적 위약 비중첩 n={pl['diff']['n']}",
                robustness="; ".join(f"{r.method}: p={r.p:.3f}" for r in se_all[se_all.term.astype(str).str.startswith(('H2', 'diff'))].itertuples()),
                interpretation="방향: 반도체 > 지수" if act["diff"] > 0 else "방향: 지수 > 반도체")
        for k, lab in [("idx", "β_B 지수 2X"), ("semi", "β_C 반도체 2X")]:
            LOG.add(id=f"M03_{k}", phase="3_main", kind="baseline", hypothesis="H1(계열별)", purpose=lab,
                    treated=k, outcome="log 거래대금", model="분리형 DID", estimand=lab, estimate=act[k],
                    ci_low=pl[k]["ci_low"], ci_high=pl[k]["ci_high"], se=pl[k]["sd"], p=pl[k]["p"],
                    p_adj=bh(sec)["beta_" + k], correction="BH(보조: 계열별 β)",
                    inference=f"역사적 위약 n={pl[k]['n']}; CR1 p={se_s[(se_s.term == 'post_' + k) & (se_s.method == 'CR1 상품')].p.iloc[0]:.3f}")
        LOG.save()
    return dict(mp=mp, ms=ms, se=se_all, act=act, pl=pl, ri=ri, holm=prim_adj, dist=dist)


def main():
    r = primary()
    s = r["se"]
    print(s[["model", "term", "method", "coef", "se", "p", "ci_low", "ci_high"]].round(4).to_string())
    print("Holm:", r["holm"])


if __name__ == "__main__":
    main()
