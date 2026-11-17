"""시계열 foundation model 반사실 (Chronos-2, TimesFM 2.5) — 확장 실험 RQ2-E-FM.

환경이 둘로 나뉜다 (기존 분석 환경의 패키지 버전을 바꾸지 않으려고):
  1) 분석 환경   py -m src.rq2.phase1.main_study.foundation export    → data/interim/rq_cache/fm_input.parquet
  2) FM 환경     .venv-fm/Scripts/python -m src.rq2.phase1.main_study.foundation forecast   → data/interim/rq_cache/fm_output.parquet
  3) 분석 환경   py -m src.rq2.phase1.main_study.foundation collect   → results/tables/rq2_foundation*.csv + 실험 로그

변형 (같은 절단일·같은 정보):
  chronos2_uni   y 만 (공통충격 통제 없음 → '통제 안 하면 어떻게 되는가' 대조용)
  chronos2_cov   Chronos-2 기본 공변량 입력 (과거·미래 공변량)
  timesfm_uni    y 만
  timesfm_xreg   TimesFM 의 "xreg + timesfm" 방식을 직접 구현: 학습구간 OLS(y~X) → 잔차를 TimesFM 으로 예측 → Xβ 더함
공변량 = don_mean, kosdaq_lrv, us_lvix, us_abs_sox, gap_day (PCA 성분은 절단일마다 다시 추정해야 해서 뺐다).
사전학습 데이터가 2026 이전이므로 사후구간 정보 유출은 없다. 미래 공변량은 실제값 (다른 모형과 같은 조건).

추론: FM 은 경로 표본을 주지 않아 창 평균 효과의 구간을 직접 만들 수 없다.
→ 상장 전 가짜 절단일(placebo-in-time)에서 나온 창 평균 예측오차 분포로 보정한 구간/경험 p 를 쓴다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "data" / "interim" / "rq_cache"
COV = ["don_mean", "kosdaq_lrv", "us_lvix", "us_abs_sox", "gap_day"]
CODES = ["005930", "000660", "005935"]


# ---------------------------------------------------------------- 1) export (분석 환경)
def export():
    from src.rq2.core.data import build_controls, load_panel
    from src.rq2.core.pipeline import CFG
    lv = load_panel("rv5").loc[CFG["sample"]["start"]:CFG["sample"]["end"]]
    X = build_controls(CFG)[COV]
    df = pd.concat([lv[CODES], X], axis=1).dropna()
    p = CFG["placebo"]
    idx = df.index
    cands = idx[(idx >= pd.Timestamp(p["in_time_start"])) & (idx <= pd.Timestamp(p["in_time_end"]))][::p["in_time_step"]]
    cut = []
    for d in cands:
        i = idx.get_loc(d)
        if idx[min(i + p["in_time_horizon"] - 1, len(idx) - 1)] < CFG["timeline"]["listing"]:
            cut.append((idx[i - 1], p["in_time_horizon"], "placebo"))
    te = CFG["sample"]["train_end"]
    cut.append((te, int((idx > te).sum()), "actual"))
    df.to_parquet(CACHE / "fm_input.parquet")
    pd.DataFrame(cut, columns=["cutoff", "horizon", "kind"]).to_parquet(CACHE / "fm_cutoffs.parquet")
    print(f"export: {len(df)} rows, {len(cut)} cutoffs")


# ---------------------------------------------------------------- 2) forecast (FM 환경)
def forecast():
    import torch
    torch.manual_seed(0)
    df = pd.read_parquet(CACHE / "fm_input.parquet")
    cuts = pd.read_parquet(CACHE / "fm_cutoffs.parquet")
    from chronos import Chronos2Pipeline
    import timesfm
    ch = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    tf = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    tf.compile(timesfm.ForecastConfig(max_context=512, max_horizon=128, normalize_inputs=True,
                                      use_continuous_quantile_head=True, fix_quantile_crossing=True))
    out = []
    for c in cuts.itertuples():
        ctx = df[df.index <= c.cutoff]
        fut = df[df.index > c.cutoff].iloc[:c.horizon]
        H = len(fut)
        for code in CODES:
            y = ctx[code].values.astype(np.float32)
            preds = {}
            # Chronos-2 univariate / covariates
            q, m = ch.predict_quantiles([y], prediction_length=H, quantile_levels=[0.5])
            preds["chronos2_uni"] = m[0].numpy().ravel()
            inp = [{"target": y, "past_covariates": {k: ctx[k].values.astype(np.float32) for k in COV},
                    "future_covariates": {k: fut[k].values.astype(np.float32) for k in COV}}]
            q, m = ch.predict_quantiles(inp, prediction_length=H, quantile_levels=[0.5])
            preds["chronos2_cov"] = m[0].numpy().ravel()
            # TimesFM univariate
            pt, _ = tf.forecast(horizon=H, inputs=[y])
            preds["timesfm_uni"] = np.asarray(pt)[0, :H]
            # TimesFM xreg + timesfm (직접 구현)
            Z = np.column_stack([np.ones(len(ctx)), ctx[COV].values])
            b = np.linalg.lstsq(Z, y, rcond=None)[0]
            r = (y - Z @ b).astype(np.float32)
            pr, _ = tf.forecast(horizon=H, inputs=[r])
            Zf = np.column_stack([np.ones(H), fut[COV].values])
            preds["timesfm_xreg"] = Zf @ b + np.asarray(pr)[0, :H]
            for name, v in preds.items():
                out.append(pd.DataFrame({"cutoff": c.cutoff, "kind": c.kind, "code": code, "model": name,
                                         "date": fut.index, "h": np.arange(1, H + 1), "pred": v, "y": fut[code].values}))
        print(c.cutoff, c.kind, flush=True)
    pd.concat(out).to_parquet(CACHE / "fm_output.parquet")


# ---------------------------------------------------------------- 3) collect (분석 환경)
def collect():
    from src.rq2.core.explog import ExpLog
    from src.rq2.core.pipeline import TL, effect_windows
    from src.rq2.phase1.main_study.run_all import LOG, OUT, TAB, save
    import json
    o = pd.read_parquet(CACHE / "fm_output.parquet")
    o["e"] = o.y - o.pred
    pl = o[o.kind == "placebo"]
    ps = []
    for (m, code, cut), g in pl.groupby(["model", "code", "cutoff"]):
        ps.append({"model": m, "code": code, "cutoff": cut, "h10": g.e.iloc[:10].mean(), "h30": g.e.mean(),
                   "rmse30": np.sqrt(np.mean(g.e ** 2))})
    ps = pd.DataFrame(ps)
    save(ps, "rq2_foundation_placebo")
    act = o[o.kind == "actual"]
    rows = []
    for (m, code), g in act.groupby(["model", "code"]):
        g = g.set_index("date")
        W = effect_windows(g.index)
        null = ps[(ps.model == m) & (ps.code == code)]
        for w in ["transition", "post_effective", "post_eff_20d"]:
            mk = W[w].values
            eff = g.e[mk].mean()
            ref = null.h10 if mk.sum() <= 15 else null.h30   # 창 길이에 가까운 placebo 창
            sd = ref.std()
            rows.append({"model": m, "code": code, "window": w, "n_days": int(mk.sum()), "effect_log": eff,
                         "placebo_mean": ref.mean(), "placebo_sd": sd,
                         "ci_lo": eff - 1.96 * sd, "ci_hi": eff + 1.96 * sd,
                         "emp_p": (np.sum(np.abs(ref) >= abs(eff)) + 1) / (len(ref) + 1), "n_placebo": len(ref)})
    res = pd.DataFrame(rows)
    save(res, "rq2_foundation")
    old = OUT / "experiment_log.csv"
    prev = pd.read_csv(old, dtype=str).fillna("")
    LOG.rows = [r for r in prev.to_dict("records") if not r["id"].startswith("RQ2-E-FM") and r["id"] != "RQ2-E-FOUNDATION"]
    summ = ps.groupby("model").agg(mean_h30=("h30", "mean"), sd_h30=("h30", "std"), rmse=("rmse30", "mean"))
    for m in res.model.unique():
        g = res[(res.model == m) & res.window.isin(["transition", "post_effective"])]
        LOG.add(id=f"RQ2-E-FM-{m}", rq="RQ2", kind="extension",
                purpose="시계열 foundation model 반사실 (예측력 ≠ 인과식별; 공변량 유무 대조)",
                data="rv5 005930/000660/005935, 절단일 2026-07-15 + 상장 전 가짜 절단일 17개",
                model={"chronos2_uni": "Chronos-2 (amazon/chronos-2) 단변량", "chronos2_cov": "Chronos-2 + 공변량",
                       "timesfm_uni": "TimesFM 2.5 200M 단변량",
                       "timesfm_xreg": "TimesFM 2.5 xreg+timesfm (OLS 잔차 예측)"}[m],
                settings={"covariates": COV, "venv": ".venv-fm", "inference": "placebo 오차 분포 보정"},
                diagnostics=f"placebo h30 평균오차 {summ.loc[m, 'mean_h30']:+.3f}, sd {summ.loc[m, 'sd_h30']:.3f}, RMSE {summ.loc[m, 'rmse']:.3f}",
                result="; ".join(f"{r.code}/{r.window}: {r.effect_log:+.3f} [{r.ci_lo:+.3f},{r.ci_hi:+.3f}] 경험p={r.emp_p:.3f}"
                                 for r in g.itertuples()))
    LOG.save()
    print(res.round(3).to_string())
    print(summ.round(3))


if __name__ == "__main__":
    {"export": export, "forecast": forecast, "collect": collect}[sys.argv[1]]()
