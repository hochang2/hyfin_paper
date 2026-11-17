"""실험 3: Time Series Foundation Model (Chronos-2, TimesFM 2.5).

1차 연구 foundation.py 와 같은 3 단계 구조 (분석 환경의 패키지 버전을 건드리지 않는다):
  1) py -3.13 -m src.rq2.phase1.ml_followup.exp3_fm export                → ml_followup/exp3/fm_*.parquet
  2) .venv-fm/Scripts/python -m src.rq2.phase1.ml_followup.exp3_fm forecast [zs|ft]
  3) py -3.13 -m src.rq2.phase1.ml_followup.exp3_fm analysis

버전별 지원 (설치본 확인, 2026-11-16)
  chronos-forecasting 2.3.2 : Chronos2Pipeline.fit(finetune_mode='full'|'lora') 지원. LoRA 는 peft 필요 → 미설치.
                              peft 설치는 환경 변경이므로 하지 않았고, 'lora' 를 주면 조용히 full 로 바뀌므로 LoRA 는 실행하지 않는다.
  timesfm 3.0.2 (2.5 200M)  : 공식 fine-tuning API 없음 → zero-shot 만 (공변량은 xreg 방식 직접 구현).

변형 (평가 = rv5 가짜 개입일 32 + 실제 발표일, 표적 = 처치 3 + 평가 donor 20)
  zero-shot : chronos2_uni, chronos2_cov, timesfm_uni, timesfm_xreg
  fine-tune (Chronos-2 full, 다종목, refit 시점 r ∈ {90,150} 이전 자료만, i0 ∈ [r, 다음 r) 에 사용, 100 step —
             CPU 비용 때문에 계획(4 시점·300 step)에서 축소, config 주석 참조)
    ft_rv5_{uni,cov} : rv5 donor 175 + 처치 3 의 r 이전 구간 (종목당 90~180 점 → 자료 부족 조건)
    ft_gk_{uni,cov}  : GK donors_long 131 + 처치 3 의 r 날짜 이전 2015~ 전체 (종목당 ~2000 점, 영역 이동 있음)
공변량 = [don_mean(자기 제외), kosdaq_ret_abs, us_lvix, us_abs_sox] — GK·rv5 양쪽에 있는 공통 통제 (ft_gk 와 같은 이름).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / "data" / "interim" / "rq_cache" / "ml_followup" / "exp3"
COV = ["don_mean", "kosdaq_ret_abs", "us_lvix", "us_abs_sox"]
FT_PRED = 30          # fine-tuning 예측길이 (r=90 에서도 창이 생기도록; 그보다 긴 지평은 모델 자체 외삽)
FT_MIN_PAST = 40


# ---------------------------------------------------------------- 1) export (분석 환경)
def export():
    from src.rq2.phase1.ml_followup import panel as P
    from src.rq2.phase1.ml_followup.exp1_global import eval_donors
    OUT.mkdir(parents=True, exist_ok=True)
    lv = P.rv5_wide()
    M = P.market_frame()
    c = P.rv5_controls_all(P.rv5_cutoffs()[0])               # don_mean LOO 는 i0 와 무관 (PC 만 i0 의존)
    rows = []
    for u in P.pool("main") + P.UNITS3:
        X = c["loo"].get(u, c["full"])
        rows.append(pd.DataFrame({"code": u, "t": np.arange(len(lv)), "date": lv.index, "y": lv[u].values,
                                  "don_mean": X["don_mean"].values, **{k: M[k].values for k in COV[1:]}}))
    R = pd.concat(rows)
    R.to_parquet(OUT / "fm_rv5.parquet")
    dl, Y, _ = P.gk_units()
    g = []
    for u in P.UNITS3 + dl:
        X = P.gk_X(u)
        g.append(pd.DataFrame({"code": u, "date": Y.index, "y": Y[u].values, **{k: X[k].values for k in COV}}))
    pd.concat(g).to_parquet(OUT / "fm_gk.parquet")
    cuts = [(i0, P.rv5_horizon_max(i0)) for i0 in P.rv5_cutoffs()] + [(P.real_index(), 71)]
    pd.DataFrame(cuts, columns=["i0", "H"]).to_parquet(OUT / "fm_cutoffs.parquet")
    pd.DataFrame({"code": P.UNITS3 + eval_donors()[:20]}).to_parquet(OUT / "fm_targets.parquet")
    print("export", R.shape, len(cuts))


# ---------------------------------------------------------------- 2) forecast (FM 환경)
def _series(R, code, i0, H):
    g = R[R.code == code].sort_values("t")
    ctx, fut = g[g.t < i0].dropna(), g[(g.t >= i0) & (g.t < i0 + H)]
    return ctx, fut


def _ch_inputs(R, codes, i0, H, cov):
    inp, futs = [], []
    for u in codes:
        ctx, fut = _series(R, u, i0, H)
        fut = fut.dropna()
        d = {"target": ctx.y.values.astype(np.float32)}
        if cov:
            d["past_covariates"] = {k: ctx[k].values.astype(np.float32) for k in COV}
            d["future_covariates"] = {k: fut[k].values.astype(np.float32) for k in COV}
        inp.append(d)
        futs.append(fut)
    return inp, futs


def _emit(out, futs, codes, preds, i0, model, extra=None):
    for u, fut, p in zip(codes, futs, preds):
        H = len(fut)
        out.append(pd.DataFrame({"code": u, "i0": i0, "model": model, "h": np.arange(H), "y": fut.y.values,
                                 "pred": np.asarray(p).ravel()[:H], **(extra or {})}))


def _ch_predict(pipe, inp, H):
    _, m = pipe.predict_quantiles(inp, prediction_length=H, quantile_levels=[0.5])
    return [x.numpy().ravel() for x in m]


def forecast_zs():
    import torch
    torch.manual_seed(0)
    from chronos import Chronos2Pipeline
    import timesfm
    R = pd.read_parquet(OUT / "fm_rv5.parquet")
    cuts = pd.read_parquet(OUT / "fm_cutoffs.parquet")
    codes = pd.read_parquet(OUT / "fm_targets.parquet").code.tolist()
    ch = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    tf = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    tf.compile(timesfm.ForecastConfig(max_context=512, max_horizon=128, normalize_inputs=True,
                                      use_continuous_quantile_head=True, fix_quantile_crossing=True))
    out = []
    for c in cuts.itertuples():
        for cov in (False, True):
            inp, futs = _ch_inputs(R, codes, c.i0, c.H, cov)
            H = max(len(f) for f in futs)
            _emit(out, futs, codes, _ch_predict(ch, inp, H), c.i0, "chronos2_cov" if cov else "chronos2_uni")
        inp, futs = _ch_inputs(R, codes, c.i0, c.H, False)
        H = max(len(f) for f in futs)
        pt, _ = tf.forecast(horizon=H, inputs=[d["target"] for d in inp])
        _emit(out, futs, codes, list(np.asarray(pt)), c.i0, "timesfm_uni")
        res, xb = [], []
        for u, fut in zip(codes, futs):
            ctx, _ = _series(R, u, c.i0, c.H)
            Z = np.column_stack([np.ones(len(ctx)), ctx[COV].values])
            b = np.linalg.lstsq(Z, ctx.y.values, rcond=None)[0]
            res.append((ctx.y.values - Z @ b).astype(np.float32))
            xb.append(np.column_stack([np.ones(len(fut)), fut[COV].values]) @ b)
        pr, _ = tf.forecast(horizon=H, inputs=res)
        _emit(out, futs, codes, [x + np.asarray(p)[:len(x)] for x, p in zip(xb, pr)], c.i0, "timesfm_xreg")
        print("zs", c.i0, flush=True)
    pd.concat(out).to_parquet(OUT / "fm_zs.parquet")


def _ft_train(pipe, series, cov, steps, batch, lr, ctx_len, tag):
    data = []
    for g in series:
        g = g.dropna()
        if len(g) < FT_MIN_PAST + FT_PRED:
            continue
        d = {"target": g.y.values.astype(np.float32)}
        if cov:
            d["past_covariates"] = {k: g[k].values.astype(np.float32) for k in COV}
            d["future_covariates"] = {k: None for k in COV}
        data.append(d)
    t = time.time()
    ft = pipe.fit(data, prediction_length=FT_PRED, finetune_mode="full", context_length=ctx_len,
                  learning_rate=lr, num_steps=steps, batch_size=batch, min_past=FT_MIN_PAST,
                  output_dir=str(OUT / "ckpt" / tag), remove_printer_callback=True, report_to=[],
                  save_strategy="no", logging_steps=max(steps // 5, 1), seed=0, dataloader_num_workers=0)
    import shutil
    shutil.rmtree(OUT / "ckpt" / tag, ignore_errors=True)          # 체크포인트(수백 MB)는 보관하지 않음
    return ft, {"n_series": len(data), "n_points": int(sum(len(d["target"]) for d in data)), "sec": time.time() - t}


def forecast_ft(variants=("ft_rv5_uni", "ft_rv5_cov", "ft_gk_uni", "ft_gk_cov")):
    import torch
    import yaml
    torch.manual_seed(0)
    from chronos import Chronos2Pipeline
    F = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))["fm"]
    R = pd.read_parquet(OUT / "fm_rv5.parquet")
    Gk = pd.read_parquet(OUT / "fm_gk.parquet")
    cuts = pd.read_parquet(OUT / "fm_cutoffs.parquet")
    codes = pd.read_parquet(OUT / "fm_targets.parquet").code.tolist()
    base = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    refits = F["refits"]
    dates = R.drop_duplicates("t").set_index("t").date
    out, meta = [], []
    for v in variants:
        cov = v.endswith("cov")
        for k, r in enumerate(refits):
            nxt = refits[k + 1] if k + 1 < len(refits) else 10 ** 9
            use = cuts[(cuts.i0 >= r) & (cuts.i0 < nxt)]
            if use.empty:
                continue
            if v.startswith("ft_rv5"):
                series = [g.sort_values("t")[g.t < r] for _, g in R.groupby("code")]
            else:
                d_r = dates[r]
                series = [g.sort_values("date")[g.date < d_r].tail(F["context_length"] * 8) for _, g in Gk.groupby("code")]
            pipe, info = _ft_train(base, series, cov, F["steps"], F["batch_size"], F["lr_full"], F["context_length"], f"{v}_{r}")
            meta.append({"variant": v, "refit": r, **info})
            print(v, r, info, flush=True)
            for c in use.itertuples():
                inp, futs = _ch_inputs(R, codes, c.i0, c.H, cov)
                H = max(len(f) for f in futs)
                _emit(out, futs, codes, _ch_predict(pipe, inp, H), c.i0, v, {"refit": r})
            pd.concat(out).to_parquet(OUT / "fm_ft.parquet")
            pd.DataFrame(meta).to_parquet(OUT / "fm_ft_meta.parquet")


def timing_test():
    import torch
    from chronos import Chronos2Pipeline
    R = pd.read_parquet(OUT / "fm_rv5.parquet")
    base = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    series = [g.sort_values("t")[g.t < 120] for _, g in R.groupby("code")]
    for cov in (False, True):
        _, info = _ft_train(base, series, cov, 10, 32, 1e-5, 256, f"timing_{cov}")
        print("cov" if cov else "uni", info, "threads", torch.get_num_threads(), flush=True)


# ---------------------------------------------------------------- 3) analysis (분석 환경)
def run():
    """분석 환경에서 호출되면 FM 환경 실행 명령만 안내 (환경 분리)."""
    py = ROOT / ".venv-fm" / "Scripts" / "python.exe"
    print(f"{py} -m src.rq2.phase1.ml_followup.exp3_fm forecast zs ft")


def analyze():
    from src.rq2.phase1.ml_followup import common as C
    from src.rq2.phase1.ml_followup import panel as P
    from src.rq2.phase1.ml_followup.exp5_audit import loo_audit
    parts = [pd.read_parquet(OUT / f) for f in ("fm_zs.parquet", "fm_ft.parquet") if (OUT / f).exists()]
    O = pd.concat(parts)
    rows = []
    for (u, i0, m), g in O.groupby(["code", "i0", "model"]):
        g = g.sort_values("h")
        for r in C.window_rows(g.y.values, g.pred.values):
            rows.append({"code": u, "i0": i0, "model": m, **r})
    W = pd.DataFrame(rows)
    W["err"] = W.ybar - W.predbar
    W["group"] = np.where(W.code.isin(P.UNITS3), "treated", "donor")
    W["is_real"] = W.i0 == P.real_index()
    # 같은 표적·가짜 개입일에서 1차 연구 방식 OLS(C_main) 와 비교 (exp1 local_pseudo 캐시)
    loc = pd.read_parquet(C.cpath("exp1", "local_pseudo"))
    loc = loc[loc.model.isin(["ols", "lgbm", "bsts"])].assign(err=lambda d: d.ybar - d.predbar)
    loc["group"] = np.where(loc.code.isin(P.UNITS3), "treated", "donor")
    loc["is_real"] = False
    key = W[~W.is_real][["code", "i0"]].drop_duplicates()
    loc = loc.merge(key, on=["code", "i0"])
    A = pd.concat([W, loc[W.columns.intersection(loc.columns)]])
    ps = A[~A.is_real & A.window.isin(P.HZ + ["transition", "post_effective"])]
    acc = ps.groupby(["group", "model", "window"]).agg(rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))),
                                                       mae=("err", lambda e: e.abs().mean()), bias=("err", "mean"),
                                                       sd=("err", "std"), n=("err", "size")).reset_index()
    C.save_table(acc, "exp3_fm_accuracy")
    fpr = loo_audit(ps[ps.group == "treated"], P.FCFG["rv5"]["cutoff_step"], methods=["neff", "strict"], label="treated")
    C.save_table(fpr, "exp3_fm_fpr_loo")
    # 효과 안정성: 인접 가짜 개입일 간 효과 추정치 변화(같은 표적) — FM 표본 경로가 매번 달라지는지
    st = ps.sort_values("i0").groupby(["code", "model", "window"]).err.apply(lambda e: np.std(np.diff(e.values))).rename(
        "sd_diff_adjacent").reset_index().groupby(["model", "window"]).sd_diff_adjacent.mean().reset_index()
    C.save_table(st, "exp3_fm_effect_stability")
    rng = np.random.default_rng(P.SEED)
    real = W[W.is_real & W.code.isin(P.UNITS3)]
    rr = []
    for (u, m, w), g in real.groupby(["code", "model", "window"]):
        q = ps[(ps.code == u) & (ps.model == m) & (ps.window == w)]
        a, b = P.WIN[w]
        est = float(g.err.iloc[0])
        o = C.null_p(est, q.err.values, q.i0.values, b - a, P.FCFG["rv5"]["cutoff_step"], "neff", 3, 4999, rng)
        rr.append({"code": u, "model": m, "window": w, "effect_pct": 100 * (np.exp(est) - 1), "p_neff": o["p"]})
    C.save_table(pd.DataFrame(rr), "exp3_fm_real_effect")
    if (OUT / "fm_ft_meta.parquet").exists():
        C.save_table(pd.read_parquet(OUT / "fm_ft_meta.parquet"), "exp3_fm_finetune_meta")
    return acc


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "export":
        export()
    elif a[0] == "forecast":
        if "zs" in a:
            forecast_zs()
        if "ft" in a:
            forecast_ft()
    elif a[0] == "timing":
        timing_test()
    elif a[0] == "analysis":
        analyze()
