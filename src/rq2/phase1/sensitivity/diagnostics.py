"""진단·한계 요약 (RQ2-S6-DIAG). 새 적합 없음 — 앞 실험 결과표와 기존 캐시만 읽는다.

1. 검정력: 설계별 post_effective null sd → 80% 검정력 MDE (%), −10/−20% 효과의 근사 검정력
2. donor 교환가능성: 처치 종목의 사전 적합 RMSE·변동성 수준이 donor 분포에서 어디에 있나
3. 실행 간 차이: 주 분석 BSTS vs ml_ext space 캐시 BSTS (같은 설정, 다른 실행) 차이
4. Exp2 선형 근사 검증 (half-shock 선형성, 실제 주입 재적합) / Exp5 구조 점검 요약
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.rq2.core.data import load_panel
from src.rq2.phase1.sensitivity import common as C

DES = {"BSTS_005930": "bsts", "BSTS_000660": "bsts", "SIMPLEX_005930": "simplex", "SIMPLEX_000660": "simplex",
       "DIFF_005930": "diff"}


def power() -> pd.DataFrame:
    rows = []
    for d, k in DES.items():
        nl = C.null_dist(k)
        sd, crit = nl.std(), C.crit95(nl)
        r = dict(design=d, null_n=len(nl), null_mean=nl.mean(), null_sd=sd, crit95=crit,
                 mde80_pct=float(C.pct(-(crit + norm.ppf(.8) * sd))), theta_pct=float(C.pct(C.theta(d))))
        for tp in (-10, -20, -30):
            tau = float(C.lg(tp))
            r[f"power_{tp}pct"] = float(np.mean(np.abs(nl - nl.mean() + tau) >= crit))
        rows.append(r)
    return pd.DataFrame(rows)


def exchangeability() -> pd.DataFrame:
    w = pd.read_parquet(C.EXT / "space" / "bsts_win.parquet")
    w = w[w.window == "post_effective"].set_index("code")
    lv = load_panel("rv5").loc[C.RQ2["sample"]["start"]:C.RQ2["sample"]["train_end"]]
    lvl = lv.mean()
    trend = lv.apply(lambda s: np.polyfit(np.arange(s.notna().sum()), s.dropna().values, 1)[0] * 21 if s.notna().sum() > 50 else np.nan)
    don = [c for c in w.index if c not in C.TREAT + [C.NEG]]
    rows = []
    for c in C.TREAT + [C.NEG]:
        rows.append(dict(code=c,
                         pre_rmse=w.pre_rmse.get(c), pre_rmse_pctile=float((w.loc[don, "pre_rmse"] < w.pre_rmse.get(c)).mean()),
                         pre_level=lvl.get(c), pre_level_pctile=float((lvl.reindex(don) < lvl.get(c)).mean()),
                         pre_trend_per_month=trend.get(c),
                         pre_trend_pctile=float((trend.reindex(don) < trend.get(c)).mean())))
    return pd.DataFrame(rows)


def run_to_run() -> pd.DataFrame:
    w = pd.read_parquet(C.EXT / "space" / "bsts_win.parquet")
    rows = []
    for c in C.TREAT:
        for win in ["transition", "post_effective"]:
            s = w[(w.code == c) & (w.window == win)].iloc[0]
            m = C.theta(f"BSTS_{c}", win)
            rows.append(dict(code=c, window=win, main=m, space_cache=s.ybar - s.predbar, diff=s.ybar - s.predbar - m))
    return pd.DataFrame(rows)


def checks() -> pd.DataFrame:
    rows = []
    R = pd.read_csv(C.TAB / "s2_response_by_model.csv", dtype={"code": str})
    P = R[R.window == "post_effective"]
    a1 = P[P.scenario == "C_joint"].set_index(["code", "model"]).A
    a2 = P[P.scenario == "C_joint@half"].set_index(["code", "model"]).A
    rows.append(dict(check="Exp2 반응계수 선형성 |A(u)−A(u/2)| 최대 (C_joint)", value=float((a1 - a2).abs().max())))
    RC = pd.read_csv(C.TAB / "s2_refit_check.csv", dtype={"code": str})
    rows.append(dict(check="Exp2 실제 주입 재적합 − 선형 근사 |차이| 최대 (log)", value=float(RC["diff"].abs().max())))
    try:
        ST = pd.read_csv(C.TAB / "s5_structural_check.csv", dtype={"code": str})
        rows.append(dict(check="Exp5 구조 주입 − 가산식 |차이| 최대 (log)", value=float(ST["diff"].abs().max())))
        rows.append(dict(check="Exp5 τ 복원 오차 |최대| (log)", value=float(ST.tau_error.abs().max())))
    except FileNotFoundError:
        pass
    return pd.DataFrame(rows)


def run():
    P, E, RR, CK = power(), exchangeability(), run_to_run(), checks()
    C.save_table(P, "s6_power")
    C.save_table(E, "s6_exchangeability")
    C.save_table(RR, "s6_run_to_run")
    C.save_table(CK, "s6_approximation_checks")
    C.LOG.add(id="RQ2-S6-DIAG", rq="RQ2", kind="diagnostic", purpose="검정력·교환가능성·실행 간 차이·근사 검증",
              data="결과표·ml_ext 캐시", model="-",
              result={"mde80_pct": P.set_index("design").mde80_pct.round(3).to_dict(),
                      "power_-20pct": P.set_index("design")["power_-20pct"].round(3).to_dict()},
              diagnostics={"exch": E.round(3).to_dict("records"), "checks": CK.round(4).to_dict("records")},
              seed=C.SEED)
    return dict(P=P, E=E, RR=RR, CK=CK)


if __name__ == "__main__":
    out = run()
    C.LOG.save()
    pd.set_option("display.width", 250)
    for k, v in out.items():
        print("==", k)
        print(v.round(4).to_string())
