"""RQ3 1차 연구 실행기. RQ1·RQ2 와 같은 방식이다.

    .venv/Scripts/python -m src.rq3.main_study.run_all                  # 전 단계
    .venv/Scripts/python -m src.rq3.main_study.run_all --stages data    # 일부 단계만

- 설정: src/rq3/config.yaml (사전 지정, 커밋 "rq3 사전 지정")
- 결과: src/rq3/main_study/results/{tables,figures}, experiment_log.{csv,md}, environment.json
- 실험 로그는 단계별로 누적된다. 같은 id 는 새 결과로 바꾼다.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.rq2.core.explog import ExpLog
from src.rq3.main_study import data as D
from src.rq3.main_study import forecast as F
from src.rq3.main_study import plots as P

OUT = Path(__file__).resolve().parent / "results"
TAB, FIG = OUT / "tables", OUT / "figures"
LOG = ExpLog(OUT / "experiment_log.csv")
SEED = D.CFG["seed"]
CODES = list(D.CFG["targets"])


# ====================================================================== stages
def stage_data():
    """1단계: 상품 합산, 설명변수 구성, 점검표, 기초 그림 (예측·검정 없음)."""
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    # 상품 목록
    p = D.raw_products()
    post = p[p.date >= D.CFG["sample"]["listing"]]
    prod = post.groupby("code").agg(
        name=("name", "first"), underlying=("underlying", "first"), kind=("kind", "first"),
        leverage=("leverage", "first"), index_name=("index_name", "first"),
        n_days=("date", "nunique"), n_aum_calc=("aum_calc", "count"), n_aum_won=("aum_won", "count"),
        aum_calc_mean_bn=("aum_calc", lambda s: s.mean() / 1e9), value_mean_bn=("value", lambda s: s.mean() / 1e9))
    prod["L(L-1)"] = prod.leverage * (prod.leverage - 1)
    prod.sort_values(["underlying", "kind", "leverage"]).round(2).to_csv(
        TAB / "rq3_t01_products.csv", encoding="utf-8-sig")

    # 기초자산별 규모 비중 (ETN 몫: aum_won 강건성에서 빠지는 부분)
    share = post.assign(w=post.leverage * (post.leverage - 1) * post.aum_calc).groupby(
        ["underlying", "kind"]).w.sum().unstack()
    share = share.div(share.sum(axis=1), axis=0)

    # 설명변수 요약: 구간별
    designs = {c: D.design(c) for c in CODES + [D.CFG["negative_control"]]}
    designs[D.CFG["negative_control"]] = D.design(D.CFG["negative_control"], "005930")
    rows = []
    for c, d in designs.items():
        per = D.period_of(d.index).values
        for k, g in d.groupby(per):
            row = {"code": c, "period": k, "n": len(g)}
            for v in ["har_d", "R", "S", "X"]:
                row[f"{v}_mean"], row[f"{v}_sd"] = g[v].mean(), g[v].std()
            post_g = g[g.D == 1]
            row["corr_R_X"] = post_g.R.corr(post_g.X) if len(post_g) > 2 else np.nan
            rows.append(row)
    order = ["pre_listing", "listing_announce", "announce_effective", "post_effective"]
    summ = pd.DataFrame(rows)
    summ["period"] = pd.Categorical(summ.period, order, ordered=True)
    summ.sort_values(["code", "period"]).round(4).to_csv(
        TAB / "rq3_t02_variables_by_period.csv", index=False, encoding="utf-8-sig")

    # 표본 점검표
    fo = D.first_origin()
    chk = []
    for c, d in designs.items():
        est = d.dropna(subset=["y_next"])
        chk.append({"code": c, "rows": len(d), "first_row": d.index.min().date(), "last_row": d.index.max().date(),
                    "n_pre_listing": int((d.D == 0).sum()), "n_post_listing": int((d.D == 1).sum()),
                    "first_origin": fo.date(), "n_forecasts": int((est.index >= fo).sum()),
                    "n_zero_return_days": int((d.r == 0).sum()),
                    "corr_R_X_post": round(d[d.D == 1].R.corr(d[d.D == 1].X), 4)})
    chk = pd.DataFrame(chk)
    chk.to_csv(TAB / "rq3_t03_sample_check.csv", index=False, encoding="utf-8-sig")

    P.data_overview({c: designs[c] for c in CODES}, D.CFG["timeline"], fo, FIG / "rq3_f01_data_overview.png")

    LOG.add(id="RQ3-D00-DATA", rq="RQ3", kind="diagnostic",
            purpose="1단계: 상품 합산·설명변수 구성·데이터 점검 (예측·검정 아님)",
            data="proc.rv_daily rv5, proc.stock_daily ret_log·value, proc.etf_daily P1 18개, 2025-10-01~2026-10-30",
            model="-",
            settings={"size": "aum_calc", "denom": "value", "har": [1, D.HAR_W, D.HAR_M],
                      "first_origin": str(fo.date())},
            diagnostics=("; ".join(f"{r.code}: rows={r.rows}, post={r.n_post_listing}, forecasts={r.n_forecasts}, "
                                   f"corr(R,X|post)={r.corr_R_X_post}" for r in chk.itertuples())
                         + "; ETN share of ΣL(L−1)AUM: "
                         + ", ".join(f"{u}={share.loc[u, 'ETN']:.3f}" for u in share.index)),
            result="상품 18개(종목별 9), 상장 전 S=X=0, 상장 후 S>0. 결측 없음",
            interpretation="예측 단계로 넘어갈 수 있음",
            seed=SEED)


_OOS: dict = {}


def oos_models(code: str, underlying: str | None = None, **kw) -> dict[str, pd.DataFrame]:
    """A·B·C 표본외 예측 (같은 설정이면 다시 계산하지 않음)."""
    key = (code, underlying, tuple(sorted(kw.items())))
    if key not in _OOS:
        d = D.design(code, underlying, **{k: v for k, v in kw.items() if k in ("measure", "size", "denom")})
        okw = {k: v for k, v in kw.items() if k in ("window", "rolling", "start")}
        fo = kw.get("first_origin", D.first_origin())
        _OOS[key] = {m: F.oos(d, cols, fo, **okw) for m, cols in F.MODELS.items()}
    return _OOS[key]


def stage_primary():
    """2단계: 사전 지정 주 검정. C vs B, Clark-West (ln RV MSE, 한쪽), Newey-West, Holm."""
    rows, fc = [], []
    for c in CODES:
        f = oos_models(c)
        cw = F.clark_west(f["B"].y, f["B"].yhat, f["C"].yhat)
        rows.append({"id": {"005930": "P1", "000660": "P2"}[c], "code": c, "n": cw["n"], "nw_lag": cw["nw_lag"],
                     "mse_A": np.mean((f["A"].y - f["A"].yhat) ** 2), "mse_B": cw["mse_small"],
                     "mse_C": cw["mse_big"], "r2_oos_C_vs_B": cw["r2_oos"], "cw_mean": cw["cw_mean"],
                     "cw_t": cw["cw_t"], "p_one_sided": cw["p_one_sided"]})
        fc.append(pd.DataFrame({"code": c, "y": f["C"].y, **{f"yhat_{m}": f[m].yhat for m in f},
                                "period": D.period_of(f["C"].index).values}))
    t = pd.DataFrame(rows)
    hp = F.holm(dict(zip(t.code, t.p_one_sided)))
    t["p_holm"] = t.code.map(hp)
    t["decision"] = np.where(t.p_holm < D.CFG["pre_registration"]["alpha"], "예측 개선", "추가 예측력의 증거 없음")
    t.round(6).to_csv(TAB / "rq3_t04_primary.csv", index=False, encoding="utf-8-sig")
    fcs = pd.concat(fc).rename_axis("date").reset_index()
    fcs.to_csv(TAB / "rq3_t05_forecasts.csv", index=False, encoding="utf-8-sig")
    P.cum_loss_diff({c: oos_models(c) for c in CODES}, D.CFG["timeline"], FIG / "rq3_f02_cum_loss_diff.png")

    for r in t.itertuples():
        LOG.add(id=f"RQ3-{r.id}-{r.code}", rq="RQ3", kind="baseline",
                purpose=f"주 검정 {r.id}: {r.code} 다음 날 ln RV 예측에서 C(B+X) 가 B(HAR+D+R) 보다 나은가",
                data=f"design({r.code}), 2025-11-06~2026-10-30, 표본외 {r.n}개 (원점 {D.first_origin():%Y-%m-%d}~)",
                model="OLS 확장 창, 원점마다 재추정",
                settings={"cols_B": F.MODELS["B"], "cols_C": F.MODELS["C"], "loss": "MSE ln RV",
                          "test": "Clark-West one-sided", "nw_lag": r.nw_lag, "holm_family": CODES},
                result=(f"MSE A={r.mse_A:.4f}, B={r.mse_B:.4f}, C={r.mse_C:.4f}, R2oos(C|B)={r.r2_oos_C_vs_B:+.4f}, "
                        f"CW t={r.cw_t:.3f}, p={r.p_one_sided:.4f}, Holm p={r.p_holm:.4f}"),
                interpretation=r.decision, seed=SEED)


N_PLACEBO, BLOCK = 1000, 5
PERIODS = ["pre_listing", "listing_announce", "announce_effective", "post_effective"]


def _cw(f_small: pd.DataFrame, f_big: pd.DataFrame) -> dict:
    return F.clark_west(f_small.y, f_small.yhat, f_big.yhat)


def _qlike_dm(f_small: pd.DataFrame, f_big: pd.DataFrame) -> dict:
    """RV 수준 예측 = exp(ŷ + s²/2) 로 QLIKE, DM (한쪽)."""
    rv = np.exp(f_small.y)
    l1 = F.qlike(rv, np.exp(f_small.yhat + f_small.s2 / 2))
    l2 = F.qlike(rv, np.exp(f_big.yhat + f_big.s2 / 2))
    out = F.dm_test(l1, l2)
    out.update(qlike_small=l1.mean(), qlike_big=l2.mean())
    return out


def stage_judgment():
    """3단계: 해석 근거(X 부호·계수 경로·VIF, 구간 내 블록 위약)와 사전 지정 보조 분석."""
    rng = np.random.default_rng(SEED)
    fo = D.first_origin()
    prim = pd.read_csv(TAB / "rq3_t04_primary.csv", dtype={"code": str}).set_index("code")

    # --- 해석 근거 1: 부호, 계수 경로, VIF
    sign_rows, paths = [], []
    for c in CODES:
        d = D.design(c).dropna(subset=["y_next"])
        cols = F.MODELS["C"]
        b, se = F.ols_hac(d.y_next.to_numpy(), d[cols].to_numpy(float), F.nw_lag(len(d)))
        k = cols.index("X") + 1
        kr = cols.index("R") + 1
        post = d[d.D == 1]
        v_all = F.vif(d[[x for x in cols if x != "D"]])
        v_post = F.vif(post[[x for x in cols if x != "D"]])
        bp = oos_models(c)["C"]["b_X"]
        sign_rows.append({"code": c, "n": len(d), "b_X": b[k], "se_X_nw": se[k], "t_X": b[k] / se[k],
                          "b_R": b[kr], "se_R_nw": se[kr], "vif_X_all": v_all["X"], "vif_R_all": v_all["R"],
                          "vif_X_post": v_post["X"], "vif_R_post": v_post["R"],
                          "oos_b_X_min": bp.min(), "oos_b_X_max": bp.max(), "oos_b_X_share_pos": (bp > 0).mean()})
        paths.append(oos_models(c)["C"][["b_R", "b_X"]].assign(code=c))
    sign = pd.DataFrame(sign_rows)
    sign.round(4).to_csv(TAB / "rq3_t06_sign_vif.csv", index=False, encoding="utf-8-sig")
    pd.concat(paths).rename_axis("date").reset_index().to_csv(
        TAB / "rq3_t06b_coef_path.csv", index=False, encoding="utf-8-sig")

    # --- 해석 근거 2: 규제 구간 안에서 5일 블록으로 S 를 섞은 위약
    plc_rows, plc_null = [], {}
    for c in CODES:
        d = D.design(c)
        f = oos_models(c)
        t_real = _cw(f["B"], f["C"])["cw_t"]
        post = d.index[d.D == 1]
        grp = D.period_of(post).to_numpy()
        absr = d.loc[post, "r"].abs()
        null = np.empty(N_PLACEBO)
        for j in range(N_PLACEBO):
            s_star = F.block_shuffle(d.loc[post, "S"], grp, BLOCK, rng)
            dj = d.copy()
            dj.loc[post, "X"] = np.log1p(100 * absr * s_star)
            null[j] = _cw(f["B"], F.oos(dj, F.MODELS["C"], fo))["cw_t"]
        plc_null[c] = null
        plc_rows.append({"code": c, "cw_t_real": t_real, "null_mean": null.mean(), "null_sd": null.std(),
                         "null_q95": np.quantile(null, 0.95), "rank_p": (1 + (null >= t_real).sum()) / (1 + N_PLACEBO),
                         "n_placebo": N_PLACEBO, "block": BLOCK})
    plc = pd.DataFrame(plc_rows)
    plc.round(4).to_csv(TAB / "rq3_t07_placebo.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(plc_null).to_csv(TAB / "rq3_t07b_placebo_null.csv", index=False, encoding="utf-8-sig")

    # --- 사전 지정 보조 분석
    aux = []

    def add(label, code, und, small, big, f):
        cw = _cw(f[small], f[big])
        q = _qlike_dm(f[small], f[big])
        aux.append({"analysis": label, "target": code, "pressure_from": und, "compare": f"{big} vs {small}",
                    "n": cw["n"], "r2_oos": cw["r2_oos"], "cw_t": cw["cw_t"], "cw_p": cw["p_one_sided"],
                    "qlike_small": q["qlike_small"], "qlike_big": q["qlike_big"], "dm_t": q["dm_t"],
                    "dm_p": q["p_one_sided"]})
    for c in CODES:
        add("주 비교 (QLIKE 병기)", c, c, "B", "C", oos_models(c))
        add("A vs B: 수익률 충격 R", c, c, "A", "B", oos_models(c))
    nc = D.CFG["negative_control"]
    add("음성통제", nc, "005930", "B", "C", oos_models(nc, "005930"))
    add("교차", "000660", "005930", "B", "C", oos_models("000660", "005930"))
    add("교차", "005930", "000660", "B", "C", oos_models("005930", "000660"))
    aux = pd.DataFrame(aux)
    aux.round(4).to_csv(TAB / "rq3_t08_auxiliary.csv", index=False, encoding="utf-8-sig")

    # 구간별 평균 손실 차이 (기술 통계)
    per = []
    for c in CODES:
        f = oos_models(c)
        e = {m: (f[m].y - f[m].yhat) ** 2 for m in "ABC"}
        df = pd.DataFrame({"B_minus_C": e["B"] - e["C"], "A_minus_B": e["A"] - e["B"],
                           "period": D.period_of(f["C"].index).values})
        g = df.groupby("period").agg(n=("B_minus_C", "size"), mean_B_minus_C=("B_minus_C", "mean"),
                                     mean_A_minus_B=("A_minus_B", "mean"))
        per.append(g.reset_index().assign(code=c))
    per = pd.concat(per)[["code", "period", "n", "mean_B_minus_C", "mean_A_minus_B"]]
    per.round(5).to_csv(TAB / "rq3_t09_loss_by_period.csv", index=False, encoding="utf-8-sig")

    P.placebo_hist(plc_null, plc.set_index("code").cw_t_real.to_dict(), FIG / "rq3_f03_placebo.png")
    P.coef_path({c: oos_models(c)["C"] for c in CODES}, D.CFG["timeline"], FIG / "rq3_f04_coef_path.png")

    reading = D.CFG["pre_registration"]["interpretation_evidence"]["reading"]
    for r in sign.itertuples():
        p = plc.set_index("code").loc[r.code]
        improved = prim.loc[r.code, "decision"] == "예측 개선"
        if not improved:
            txt = reading["none"]
        elif r.b_X < 0:
            txt = reading["improve__sign_neg"]
        elif p.rank_p >= 0.05:
            txt = reading["improve__placebo_fail"]
        else:
            txt = reading["improve__sign_pos__placebo_pass"]
        LOG.add(id=f"RQ3-J-{r.code}", rq="RQ3", kind="diagnostic",
                purpose=f"해석 근거: {r.code} X 계수 부호·VIF·구간 내 블록 위약",
                data=f"design({r.code}) 전체 {r.n}행 / 표본외 76개",
                model="C 전체 표본 OLS (NW) / S 블록 셔플 후 C 재예측",
                settings={"n_placebo": N_PLACEBO, "block": BLOCK, "blocks_within": PERIODS[1:], "seed": SEED},
                result=(f"b_X={r.b_X:+.4f} (t={r.t_X:.2f}), b_R={r.b_R:+.4f}, VIF X={r.vif_X_all:.1f}/post {r.vif_X_post:.1f}; "
                        f"OOS b_X 양수 비율 {r.oos_b_X_share_pos:.2f}; 위약: 실제 CW t={p.cw_t_real:.3f}, "
                        f"위약 95%={p.null_q95:.3f}, rank p={p.rank_p:.3f}"),
                interpretation=f"주 검정 '{prim.loc[r.code, 'decision']}' → {txt}", seed=SEED)
    for r in aux.itertuples():
        LOG.add(id=f"RQ3-AUX-{zlib.crc32(f'{r.analysis}{r.target}{r.pressure_from}{r.compare}'.encode()):08x}",
                rq="RQ3", kind="robustness" if r.analysis in ("음성통제", "교차") else "diagnostic",
                purpose=f"보조: {r.analysis} — {r.target} (X 는 {r.pressure_from} 상품) {r.compare}",
                data=f"표본외 {r.n}개", model="OLS 확장 창",
                settings={"loss": ["MSE ln RV (CW)", "QLIKE level (DM)"]},
                result=(f"R2oos={r.r2_oos:+.4f}, CW t={r.cw_t:.3f} p={r.cw_p:.4f}; QLIKE {r.qlike_small:.4f}→{r.qlike_big:.4f}, "
                        f"DM t={r.dm_t:.3f} p={r.dm_p:.4f}"),
                interpretation="보조 분석 (결론 근거 아님)", seed=SEED)


def spec_cw(code: str, cols_small: list[str], cols_big: list[str], underlying: str | None = None,
            design_kw: dict | None = None, oos_kw: dict | None = None, first_origin=None) -> dict:
    """한 사양의 big vs small Clark-West."""
    d = D.design(code, underlying, **(design_kw or {}))
    okw = dict(oos_kw or {})
    fo = first_origin or D.first_origin()
    fs, fb = F.oos(d, cols_small, fo, **okw), F.oos(d, cols_big, fo, **okw)
    lag = max(F.nw_lag(len(fs)), okw.get("horizon", 1) - 1)
    cw = F.clark_west(fs.y, fs.yhat, fb.yhat, lag=lag)
    return {"n": cw["n"], "nw_lag": lag, "mse_small": cw["mse_small"], "mse_big": cw["mse_big"],
            "r2_oos": cw["r2_oos"], "cw_t": cw["cw_t"], "cw_p": cw["p_one_sided"],
            "b_last": fb[[c for c in fb.columns if c.startswith("b_")]].iloc[-1].to_dict()}


def stage_robustness():
    """4단계: 사전 지정 강건성 (C vs B) + 탐색적 교차 분석(다른 종목 |r| 통제)."""
    B, C = F.MODELS["B"], F.MODELS["C"]
    nod = lambda cols: [c for c in cols if c != "D"]  # noqa: E731
    days = D.trading_days()
    listing = D.CFG["sample"]["listing"]
    post = days[days >= listing]
    specs = [
        ("주 사양 (기준)", {}, {}, None, B, C),
        ("상장 후 106일만 (D 자동 제외)", {}, {"start": listing}, None, B, C),
        ("상장 더미 없음", {}, {}, None, nod(B), nod(C)),
        ("이동 창 60일", {}, {"window": "rolling", "rolling": 60}, None, B, C),
        ("첫 원점 20번째 거래일", {}, {}, post[19], B, C),
        ("첫 원점 40번째 거래일", {}, {}, post[39], B, C),
        ("측정치 rv5_close", {"measure": "rv5_close"}, {}, None, B, C),
        ("측정치 rv_pk", {"measure": "rv_pk"}, {}, None, B, C),
        ("대상 5일 평균 ln RV", {}, {"target": "y_next5", "horizon": 5}, None, B, C),
        ("규모 aum_won (ETN 제외)", {"size": "aum_won"}, {}, None, B, C),
        ("분모 시가총액", {"denom": "mktcap"}, {}, None, B, C),
    ]
    rows = []
    for c in CODES:
        for label, dkw, okw, fo, cs, cb in specs:
            r = spec_cw(c, cs, cb, design_kw=dkw, oos_kw=okw, first_origin=fo)
            rows.append({"spec": label, "code": c, **{k: v for k, v in r.items() if k != "b_last"},
                         "b_X_last": r["b_last"].get("b_X", np.nan)})
    rob = pd.DataFrame(rows)
    rob.round(5).to_csv(TAB / "rq3_t10_robustness.csv", index=False, encoding="utf-8-sig")

    # 탐색: 교차 X 의 개선이 다른 종목의 수익률 충격 때문인지. B+ = B + R_u (다른 종목 |r|), C+ = B+ + X
    ex = []
    for tgt, und in [("000660", "005930"), ("005930", "000660"), ("005935", "005930")]:
        bplus = B + ["R_u"]
        r1 = spec_cw(tgt, B, bplus, underlying=und)          # 다른 종목 |r| 만 추가
        r2 = spec_cw(tgt, bplus, bplus + ["X"], underlying=und)   # 그 위에 X 추가
        for lab, r in [("B+R_u vs B (다른 종목 |r| 효과)", r1), ("B+R_u+X vs B+R_u (X 효과)", r2)]:
            ex.append({"target": tgt, "pressure_from": und, "compare": lab,
                       **{k: v for k, v in r.items() if k != "b_last"}})
    ex = pd.DataFrame(ex)
    ex.round(5).to_csv(TAB / "rq3_t11_exploratory_cross.csv", index=False, encoding="utf-8-sig")
    P.robustness_forest(rob, FIG / "rq3_f05_robustness.png")

    for r in rob.itertuples():
        LOG.add(id=f"RQ3-R-{zlib.crc32(f'{r.spec}{r.code}'.encode()):08x}", rq="RQ3", kind="robustness",
                purpose=f"강건성: {r.spec} — {r.code} C vs B", data=f"표본외 {r.n}개", model="OLS",
                settings={"spec": r.spec, "nw_lag": r.nw_lag},
                result=f"R2oos={r.r2_oos:+.4f}, CW t={r.cw_t:.3f}, p={r.cw_p:.4f}, 마지막 b_X={r.b_X_last:+.3f}",
                interpretation="강건성 (결론 근거 아님)", seed=SEED)
    for r in ex.itertuples():
        LOG.add(id=f"RQ3-X-{zlib.crc32(f'{r.target}{r.pressure_from}{r.compare}'.encode()):08x}", rq="RQ3",
                kind="extension", purpose=f"탐색(사전 지정 밖): {r.target} ← {r.pressure_from} 상품, {r.compare}",
                data=f"표본외 {r.n}개", model="OLS 확장 창", settings={"nw_lag": r.nw_lag},
                result=f"R2oos={r.r2_oos:+.4f}, CW t={r.cw_t:.3f}, p={r.cw_p:.4f}",
                interpretation="탐색적 (3단계 교차 결과를 보고 추가함)", seed=SEED)


ML_KINDS = ["lasso", "enet", "rf", "xgb", "lgbm"]


def stage_exploratory():
    """5단계 탐색: 보조 ETF 변수 하나씩·전부, ML. 주 결론을 바꾸지 않는다."""
    from src.rq3.main_study import exploratory as E
    fo = D.first_origin()
    B, C = F.MODELS["B"], F.MODELS["C"]
    allv = C + D.AUX_VARS
    rows, ml_rows, fcs = [], [], []
    for c in CODES:
        d = D.design(c).join(D.aux_vars(c))
        d["RNEG"] = 100 * d.r.clip(upper=0)          # 하락 수익률 (레버리지 효과 점검용)
        fB, fC = oos_models(c)["B"], oos_models(c)["C"]
        # (1) 보조 변수 하나씩 B 에 추가, (2) C + 전부 (선형)
        # (1b) 개인 순매수(IND)가 하락 수익률(레버리지 효과)을 대신하는지: B+RNEG 기준으로 IND 추가
        specs = ([(f"B + {v}", B, B + [v]) for v in D.AUX_VARS] + [("C + 보조 4개 (OLS)", B, allv)]
                 + [("B + RNEG (하락 수익률)", B, B + ["RNEG"]),
                    ("B + RNEG + IND vs B + RNEG", B + ["RNEG"], B + ["RNEG", "IND"])])
        for lab, small, cols in specs:
            fs = fB if small == B else F.oos(d, small, fo)
            f = F.oos(d, cols, fo)
            cw = F.clark_west(fs.y, fs.yhat, f.yhat)
            rows.append({"code": c, "model": lab, "n": cw["n"], "r2_oos_vs_small": cw["r2_oos"],
                         "cw_t": cw["cw_t"], "cw_p": cw["p_one_sided"]})
        post = d[d.D == 1]
        rows.append({"code": c, "model": "참고: 상장 후 corr(IND, r)", "n": len(post),
                     "r2_oos_vs_small": np.nan, "cw_t": post.IND.corr(post.r), "cw_p": np.nan})
        # (3) ML: 특징 = C + 보조 4개. 기준 B·C 와 MSE 비교 (포함 관계 아님 → DM)
        for k in ML_KINDS:
            f = E.oos_ml(d, allv, fo, k, SEED)
            eB, eC, eM = (fB.y - fB.yhat) ** 2, (fC.y - fC.yhat) ** 2, (f.y - f.yhat) ** 2
            dmB, dmC = F.dm_test(eB, eM), F.dm_test(eC, eM)
            ml_rows.append({"code": c, "model": k, "n": len(f), "mse": eM.mean(), "mse_B": eB.mean(),
                            "r2_oos_vs_B": 1 - eM.mean() / eB.mean(), "dm_t_vs_B": dmB["dm_t"],
                            "dm_p_vs_B": dmB["p_one_sided"], "r2_oos_vs_C": 1 - eM.mean() / eC.mean(),
                            "dm_p_vs_C": dmC["p_one_sided"]})
            fcs.append(f.assign(code=c, model=k))
    aux = pd.DataFrame(rows)
    ml = pd.DataFrame(ml_rows)
    aux.round(5).to_csv(TAB / "rq3_t12_aux_vars.csv", index=False, encoding="utf-8-sig")
    ml.round(5).to_csv(TAB / "rq3_t13_ml.csv", index=False, encoding="utf-8-sig")
    pd.concat(fcs).rename_axis("date").reset_index().to_csv(TAB / "rq3_t13b_ml_forecasts.csv", index=False,
                                                           encoding="utf-8-sig")
    for r in aux[aux.cw_p.notna()].itertuples():
        LOG.add(id=f"RQ3-E-{zlib.crc32(f'{r.code}{r.model}'.encode()):08x}", rq="RQ3", kind="extension",
                purpose=f"탐색: 보조 ETF 변수 {r.model} ({r.code})", data=f"표본외 {r.n}개",
                model="OLS 확장 창", settings={"vars": D.AUX_VARS},
                result=f"R2oos={r.r2_oos_vs_small:+.4f}, CW t={r.cw_t:.3f}, p={r.cw_p:.4f}",
                interpretation="탐색적 (주 결론 불변)", seed=SEED)
    for r in ml.itertuples():
        LOG.add(id=f"RQ3-ML-{r.model}-{r.code}", rq="RQ3", kind="extension",
                purpose=f"탐색: ML {r.model} (특징 C + 보조 4개) vs B·C ({r.code})", data=f"표본외 {r.n}개",
                model=r.model, settings={"features": allv, "hyper": "RQ2 ML 확장 기본값 고정 / 선형은 원점별 시계열 CV"},
                result=(f"MSE={r.mse:.4f} (B {r.mse_B:.4f}), R2oos vs B={r.r2_oos_vs_B:+.4f} (DM p {r.dm_p_vs_B:.4f}), "
                        f"vs C={r.r2_oos_vs_C:+.4f} (DM p {r.dm_p_vs_C:.4f})"),
                interpretation="탐색적 (주 결론 불변)", seed=SEED)


SHOCK_WIN = ("2026-07-24", "2026-08-05")   # 조기시행 예고~시행 직후 급변기


def followup_design(c: str) -> pd.DataFrame:
    """두번째 연구용 설계 행렬: 기본 + 보조 ETF 변수 + 경쟁 변수 + RNEG + 마감창 RV."""
    d = D.design(c).join(D.aux_vars(c)).join(D.spot_vars(c))
    d["RNEG"] = 100 * d.r.clip(upper=0)
    w = D.window_rv()
    w = w[w.code == c].set_index("date").reindex(D.trading_days())
    for col, name in [("rvc", "c"), ("rvc_auc", "ca")]:
        v = w[col].where(w[col] > 0)
        d[f"lrv{name}"] = np.log(v).reindex(d.index)
        d[f"y{name}_next"] = np.log(v).shift(-1).reindex(d.index)
    return d


def _cw_row(d, small, big, fo, **okw):
    fs, fb = F.oos(d, small, fo, **okw), F.oos(d, big, fo, **okw)
    cw = F.clark_west(fs.y, fs.yhat, fb.yhat)
    return cw, fs, fb


def stage_followup():
    """두번째 연구 (사후 탐색, 2026-11-18 이후 사용자 요청). 주 결론·S1·S2 를 바꾸지 않는다.
    (1) IND 신호의 날짜 집중도 (2) IND 블록 위약 (3) 경쟁 변수 통제 (4) 예측 대상 = 마감창 RV."""
    rng = np.random.default_rng(SEED + 1)
    fo = D.first_origin()
    B = F.MODELS["B"]
    BI = B + ["IND"]
    s0, s1 = map(pd.Timestamp, SHOCK_WIN)
    conc, loo_tab, plc_rows, plc_null, comp, clo = [], [], [], {}, [], []
    for c in CODES:
        d = followup_design(c)
        # ---- (1) 날짜 집중도
        cw, fs, fb = _cw_row(d, B, BI, fo)
        contrib = (fs.y - fs.yhat) ** 2 - (fb.y - fb.yhat) ** 2        # 양수 = IND 모형이 그날 더 잘 맞힘
        mse_s = ((fs.y - fs.yhat) ** 2)
        tot = contrib.sum()
        conc.append({"code": c, "case": "기준 (B+IND vs B)", "n": cw["n"], "r2_oos": cw["r2_oos"],
                     "cw_p": cw["p_one_sided"]})
        m = (fs.index >= s0) & (fs.index <= s1)
        r2 = contrib[~m].sum() / mse_s[~m].sum()
        conc.append({"code": c, "case": f"평가에서 {SHOCK_WIN[0]}~{SHOCK_WIN[1]} 제외", "n": int((~m).sum()),
                     "r2_oos": r2, "cw_p": F.clark_west(fs.y[~m], fs.yhat[~m], fb.yhat[~m])["p_one_sided"]})
        dd = d[(d.index < s0) | (d.index > s1)]
        cw2, _, _ = _cw_row(dd, B, BI, fo)
        conc.append({"code": c, "case": "학습·평가 모두에서 급변기 제외", "n": cw2["n"], "r2_oos": cw2["r2_oos"],
                     "cw_p": cw2["p_one_sided"]})
        order = contrib.sort_values(ascending=False)
        for k in (1, 3, 5):
            keep = ~fs.index.isin(order.index[:k])
            conc.append({"code": c, "case": f"기여 상위 {k}일 제외", "n": int(keep.sum()),
                         "r2_oos": contrib[keep].sum() / mse_s[keep].sum(),
                         "cw_p": F.clark_west(fs.y[keep], fs.yhat[keep], fb.yhat[keep])["p_one_sided"]})
        loo = [(contrib.sum() - contrib[t]) / (mse_s.sum() - mse_s[t]) for t in fs.index]
        conc.append({"code": c, "case": "하루씩 제외 (76번) 최솟값", "n": cw["n"] - 1, "r2_oos": min(loo), "cw_p": np.nan})
        conc.append({"code": c, "case": "하루씩 제외 (76번) 최댓값", "n": cw["n"] - 1, "r2_oos": max(loo), "cw_p": np.nan})
        top = order.head(5)
        loo_tab.append(pd.DataFrame({"code": c, "date": top.index, "contrib": top.values,
                                     "share_of_total": top.values / tot, "IND": d.loc[top.index, "IND"].values,
                                     "r": d.loc[top.index, "r"].values}))
        # ---- (2) IND 블록 위약 (X 위약과 같은 방식: 규제 구간 안 5일 블록)
        post = d.index[d.D == 1]
        grp = D.period_of(post).to_numpy()
        null = np.empty(N_PLACEBO)
        for j in range(N_PLACEBO):
            dj = d.copy()
            dj.loc[post, "IND"] = F.block_shuffle(d.loc[post, "IND"], grp, BLOCK, rng).to_numpy()
            null[j] = F.clark_west(fs.y, fs.yhat, F.oos(dj, BI, fo).yhat)["cw_t"]
        plc_null[c] = null
        plc_rows.append({"code": c, "cw_t_real": cw["cw_t"], "null_mean": null.mean(), "null_q95": np.quantile(null, .95),
                         "rank_p": (1 + (null >= cw["cw_t"]).sum()) / (1 + N_PLACEBO)})
        # ---- (3) 경쟁 변수 통제
        K = B + ["LIV", "SIND", "SFOR"]
        for lab, small, big in [("B + 경쟁변수 vs B", B, K),
                                ("B + 경쟁변수 + IND vs B + 경쟁변수", K, K + ["IND"]),
                                ("B + 경쟁변수 + RNEG + IND vs B + 경쟁변수 + RNEG", K + ["RNEG"], K + ["RNEG", "IND"])]:
            cw3, _, _ = _cw_row(d, small, big, fo)
            comp.append({"code": c, "compare": lab, "n": cw3["n"], "r2_oos": cw3["r2_oos"], "cw_t": cw3["cw_t"],
                         "cw_p": cw3["p_one_sided"]})
        post_d = d[d.D == 1]
        comp.append({"code": c, "compare": "참고: 상장 후 corr(IND, SIND / SFOR / LIV)", "n": len(post_d),
                     "r2_oos": np.nan, "cw_t": np.nan, "cw_p": np.nan,
                     "corr": f"{post_d.IND.corr(post_d.SIND):.3f} / {post_d.IND.corr(post_d.SFOR):.3f} / "
                             f"{post_d.IND.corr(post_d.LIV):.3f}"})
        # ---- (4) 예측 대상 = 다음 날 마감창 ln RV
        for tgt, lag in [("yc_next", "lrvc"), ("yca_next", "lrvca")]:
            Bc = B + [lag]
            for lab, big in [("C (X)", Bc + ["X"]), ("IND", Bc + ["IND"])]:
                dc = d.dropna(subset=[lag])
                cw4, _, _ = _cw_row(dc, Bc, big, fo, target=tgt)
                clo.append({"code": c, "target": tgt, "compare": f"B_close + {lab} vs B_close", "n": cw4["n"],
                            "r2_oos": cw4["r2_oos"], "cw_t": cw4["cw_t"], "cw_p": cw4["p_one_sided"]})
    conc, comp, clo = pd.DataFrame(conc), pd.DataFrame(comp), pd.DataFrame(clo)
    plc = pd.DataFrame(plc_rows)
    conc.round(5).to_csv(TAB / "rq3_t14_ind_concentration.csv", index=False, encoding="utf-8-sig")
    pd.concat(loo_tab).round(5).to_csv(TAB / "rq3_t14b_ind_top_days.csv", index=False, encoding="utf-8-sig")
    plc.round(4).to_csv(TAB / "rq3_t15_ind_placebo.csv", index=False, encoding="utf-8-sig")
    comp.round(5).to_csv(TAB / "rq3_t16_competitors.csv", index=False, encoding="utf-8-sig")
    clo.round(5).to_csv(TAB / "rq3_t17_closing_target.csv", index=False, encoding="utf-8-sig")
    P.placebo_hist(plc_null, plc.set_index("code").cw_t_real.to_dict(), FIG / "rq3_f06_ind_placebo.png",
                   title="IND 위약: 개인 순매수를 규제 구간 안에서 5일 블록으로 섞은 1000번", xlabel="Clark-West t (B+IND vs B)")

    tag = "탐색적 (두번째 연구, 사후). 주 결론·S2 근거 아님"
    for r in conc.itertuples():
        LOG.add(id=f"RQ3-F1-{zlib.crc32(f'{r.code}{r.case}'.encode()):08x}", rq="RQ3", kind="diagnostic",
                purpose=f"IND 날짜 집중도: {r.case} ({r.code})", data=f"표본외 {r.n}개", model="OLS 확장 창",
                result=f"R2oos={r.r2_oos:+.4f}" + ("" if pd.isna(r.cw_p) else f", CW p={r.cw_p:.4f}"),
                interpretation=tag, seed=SEED)
    for r in plc.itertuples():
        LOG.add(id=f"RQ3-F2-{r.code}", rq="RQ3", kind="placebo", purpose=f"IND 블록 위약 ({r.code})",
                data="표본외 76개", model="IND 를 규제 구간 안 5일 블록으로 섞음",
                settings={"n": N_PLACEBO, "block": BLOCK, "seed": SEED + 1},
                result=f"실제 CW t={r.cw_t_real:.3f}, 위약 95%={r.null_q95:.3f}, rank p={r.rank_p:.4f}",
                interpretation=tag, seed=SEED + 1)
    for r in comp[comp.cw_p.notna()].itertuples():
        LOG.add(id=f"RQ3-F3-{zlib.crc32(f'{r.code}{r.compare}'.encode()):08x}", rq="RQ3", kind="extension",
                purpose=f"경쟁 변수 통제: {r.compare} ({r.code})", data=f"표본외 {r.n}개", model="OLS 확장 창",
                settings={"competitors": ["LIV=ln ATM IV", "SIND=현물 개인 순매수/거래대금", "SFOR=현물 외국인 순매수/거래대금"]},
                result=f"R2oos={r.r2_oos:+.4f}, CW t={r.cw_t:.3f}, p={r.cw_p:.4f}", interpretation=tag, seed=SEED)
    for r in clo.itertuples():
        LOG.add(id=f"RQ3-F4-{zlib.crc32(f'{r.code}{r.target}{r.compare}'.encode()):08x}", rq="RQ3", kind="extension",
                purpose=f"마감창 RV 예측: {r.compare} ({r.code}, {r.target})", data=f"표본외 {r.n}개",
                model="OLS 확장 창", settings={"close_window": "14:50~15:20", "auction": r.target == "yca_next"},
                result=f"R2oos={r.r2_oos:+.4f}, CW t={r.cw_t:.3f}, p={r.cw_p:.4f}", interpretation=tag, seed=SEED)


STAGES = {"data": stage_data, "primary": stage_primary, "judgment": stage_judgment,
          "robustness": stage_robustness, "exploratory": stage_exploratory, "followup": stage_followup}


def write_env():
    import importlib
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ["numpy", "pandas", "statsmodels", "scipy", "sklearn", "xgboost", "lightgbm", "matplotlib", "duckdb"]:
        try:
            v[m] = getattr(importlib.import_module(m), "__version__", "?")
        except ImportError:
            v[m] = "not installed"
    (OUT / "environment.json").write_text(json.dumps(v, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(STAGES))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    old = OUT / "experiment_log.csv"
    if old.exists():
        LOG.rows = pd.read_csv(old, dtype=str).fillna("").to_dict("records")
    write_env()
    for st in a.stages.split(","):
        t0 = time.time()
        print(f"[RQ3] {st} ...", flush=True)
        n_before = len(LOG.rows)
        STAGES[st]()
        new_ids = {r["id"] for r in LOG.rows[n_before:]}
        LOG.rows = [r for r in LOG.rows[:n_before] if r["id"] not in new_ids] + LOG.rows[n_before:]
        LOG.save()
        print(f"[RQ3] {st} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
