"""Foundation model(Chronos-2, TimesFM 2.5)을 실험 1·2 과제에 연결한다.

환경이 둘이다 (분석 환경 패키지를 바꾸지 않기 위해, 1차 연구와 같은 방식):
  1) 분석 환경  py -m src.rq2.phase1.ml_extension.foundation_bridge export [ds ...]
       → CACHE/fm/input_<ds>.parquet  (과제별 학습구간·사후구간 y 와 그 과제의 통제 X; PCA 는 과제 학습구간으로 추정된 것)
  2) FM 환경    .venv-fm/Scripts/python -m src.rq2.phase1.ml_extension.foundation_bridge forecast [ds ...]
       → CACHE/fm/output_<ds>.parquet (numpy·pandas·torch 만 import)
  3) 분석 환경  py -m src.rq2.phase1.ml_extension.foundation_bridge collect [ds ...]
       → benchmark 캐시 형식(<model>_daily / <model>_win) 으로 변환

모형:
  chronos2_uni  y 만         chronos2_cov  과거·미래 공변량
  timesfm_uni   y 만         timesfm_xreg  학습구간 OLS(y~X) + 잔차를 TimesFM 으로 예측
FM 은 경로 표본(draws)이 없어 창평균 원시 구간이 없다 → 실험 2 에서 보정(가짜 개입일 null) 추론만 쓴다.
일별 90% 구간은 Chronos-2 분위수(0.05/0.95)만 저장한다. TimesFM 은 결측.
사후 y 는 출력에 평가용으로만 붙고 예측 입력에는 들어가지 않는다 (context = 학습구간만).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FM_MODELS = ["chronos2_uni", "chronos2_cov", "timesfm_uni", "timesfm_xreg"]


def _fm_dir() -> Path:
    import yaml
    c = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    p = ROOT / c["cache_dir"] / "fm"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- 1) export
def export(ds: str):
    from src.rq2.phase1.ml_extension import benchmark as B
    from src.rq2.phase1.ml_extension import data_pipeline as DP
    rows = []
    for code, i0 in B.tasks_for(ds):
        y, X, tr, po = DP.task_data(ds, code, i0)
        d = X.copy()
        d.columns = [f"x_{c}" for c in d.columns]
        d.insert(0, "y", y.values)
        d.insert(0, "is_post", po.values)
        d.insert(0, "i0", i0)
        d.insert(0, "code", code)
        d.index.name = "date"
        rows.append(d.reset_index())
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(_fm_dir() / f"input_{ds}.parquet")
    print(f"export {ds}: {df.groupby(['code', 'i0']).ngroups} tasks, {len(df)} rows", flush=True)


# ---------------------------------------------------------------- 2) forecast (FM 환경)
def forecast(ds: str, batch: int = 64):
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(max(1, (torch.get_num_threads())))
    from chronos import Chronos2Pipeline
    import timesfm
    df = pd.read_parquet(_fm_dir() / f"input_{ds}.parquet")
    xcols = [c for c in df.columns if c.startswith("x_")]
    ch = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    tf = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    tf.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=128, normalize_inputs=True,
                                      use_continuous_quantile_head=True, fix_quantile_crossing=True))
    tasks = []
    for (code, i0), g in df.groupby(["code", "i0"], sort=False):
        ctx, fut = g[~g.is_post], g[g.is_post]
        tasks.append((code, i0, ctx, fut))
    by_h: dict[int, list] = {}
    for t in tasks:
        by_h.setdefault(len(t[3]), []).append(t)
    out = []
    t0 = time.time()
    done = 0
    for H, ts in by_h.items():
        for k in range(0, len(ts), batch):
            B_ = ts[k:k + batch]
            ys = [t[2].y.values.astype(np.float32) for t in B_]
            res = {m: [None] * len(B_) for m in FM_MODELS}
            lo = {m: [None] * len(B_) for m in FM_MODELS}
            hi = {m: [None] * len(B_) for m in FM_MODELS}
            q, mn = ch.predict_quantiles(ys, prediction_length=H, quantile_levels=[0.05, 0.5, 0.95])
            for j in range(len(B_)):
                qq = q[j].numpy()[0]
                res["chronos2_uni"][j], lo["chronos2_uni"][j], hi["chronos2_uni"][j] = qq[:, 1], qq[:, 0], qq[:, 2]
            inp = [{"target": ys[j],
                    "past_covariates": {c: B_[j][2][c].values.astype(np.float32) for c in xcols},
                    "future_covariates": {c: B_[j][3][c].values.astype(np.float32) for c in xcols}}
                   for j in range(len(B_))]
            q, mn = ch.predict_quantiles(inp, prediction_length=H, quantile_levels=[0.05, 0.5, 0.95])
            for j in range(len(B_)):
                qq = q[j].numpy()[0]
                res["chronos2_cov"][j], lo["chronos2_cov"][j], hi["chronos2_cov"][j] = qq[:, 1], qq[:, 0], qq[:, 2]
            pt, _ = tf.forecast(horizon=H, inputs=ys)
            pt = np.asarray(pt)
            rs, xb = [], []
            for j, t in enumerate(B_):
                res["timesfm_uni"][j] = pt[j, :H]
                Z = np.column_stack([np.ones(len(t[2])), t[2][xcols].values])
                b = np.linalg.lstsq(Z, ys[j], rcond=None)[0]
                rs.append((ys[j] - Z @ b).astype(np.float32))
                xb.append(np.column_stack([np.ones(H), t[3][xcols].values]) @ b)
            pr, _ = tf.forecast(horizon=H, inputs=rs)
            pr = np.asarray(pr)
            for j in range(len(B_)):
                res["timesfm_xreg"][j] = xb[j] + pr[j, :H]
            for j, (code, i0, ctx, fut) in enumerate(B_):
                for m in FM_MODELS:
                    out.append(pd.DataFrame({"code": code, "i0": i0, "model": m, "date": fut.date.values,
                                             "h": np.arange(1, H + 1), "y": fut.y.values, "pred": res[m][j],
                                             "lo90": lo[m][j] if lo[m][j] is not None else np.nan,
                                             "hi90": hi[m][j] if hi[m][j] is not None else np.nan}))
            done += len(B_)
            print(f"  {ds} H={H} {done}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    o = pd.concat(out, ignore_index=True)
    o["sec_per_task"] = (time.time() - t0) / len(tasks) / len(FM_MODELS)
    o.to_parquet(_fm_dir() / f"output_{ds}.parquet")


# ---------------------------------------------------------------- 3) collect (분석 환경)
def collect(ds: str):
    from src.rq2.phase1.ml_extension import benchmark as B
    from src.rq2.phase1.ml_extension import data_pipeline as DP
    o = pd.read_parquet(_fm_dir() / f"output_{ds}.parquet")
    for m, g in o.groupby("model"):
        dl, wl = [], []
        for (code, i0), t in g.groupby(["code", "i0"]):
            t = t.sort_values("h")
            yp, pred = t.y.values, t.pred.values
            H = len(t)
            d = t[["code", "i0", "h", "date", "y", "pred", "lo90", "hi90"]].copy()
            d["cutoff"] = t.date.iloc[0]
            d["model"] = m
            dl.append(d)
            for w, (a, b) in DP.WIN.items():
                if b > H:
                    continue
                wl.append({"code": code, "i0": i0, "cutoff": t.date.iloc[0], "window": w, "a": a, "b": b,
                           "ybar": yp[a:b].mean(), "predbar": pred[a:b].mean(), "cf_sd": np.nan,
                           "cf_q025": np.nan, "cf_q975": np.nan, "cf_q05": np.nan, "cf_q95": np.nan,
                           "pre_rmse": np.nan, "H": H, "fit_sec": float(t.sec_per_task.iloc[0]), "model": m})
        pd.concat(dl).to_parquet(B.cache_path(ds, m, "daily"))
        pd.DataFrame(wl).to_parquet(B.cache_path(ds, m, "win"))
    print(f"collect {ds}: {o.model.nunique()} models", flush=True)


if __name__ == "__main__":
    cmd, dss = sys.argv[1], sys.argv[2:] or ["rv5", "gk", "space"]
    for ds in dss:
        {"export": export, "forecast": forecast, "collect": collect}[cmd](ds)
