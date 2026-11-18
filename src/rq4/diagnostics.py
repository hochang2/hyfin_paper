"""4단계: 진단. 주 모형(분리형 DID, base vs post) 잔차 기준.

각 문제를 '문제 → 원인 → 영향 → 해결 → 전후 결과 → 남은 한계' 로 t06_diagnostics.csv 에 기록.
검정: BP, White, Wooldridge(패널 AR1), Pesaran CD, 잔차·이상치·영향력, 함수형(log/asinh/PPML/수준), VIF, 나쁜 통제.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.rq4.common import CFG, FIG, PAL, INK2, ExpLog, plt_setup, save_table
from src.rq4.data_pipeline import load_market
from src.rq4.did import design, fit
from src.rq4.estim import FE, codes, demean
from src.rq4.panel import main_panel, window

LOG = ExpLog("06_diagnostics")


def covariates(d):
    """계열 지수 변동성(20일 실현, |수익률|) × 2X, VKOSPI × 2X. 규제의 영향을 받을 수 있어 강건성용."""
    mk = load_market()[["date", "r_k200", "r_krx_semi", "vkospi"]].sort_values("date").copy()
    mk["rv_idx"] = np.log(mk["r_k200"].rolling(20, min_periods=10).std() * np.sqrt(252))
    mk["rv_semi"] = np.log(mk["r_krx_semi"].rolling(20, min_periods=10).std() * np.sqrt(252))
    mk["absr_idx"] = np.log(mk["r_k200"].abs() + 1e-3)
    mk["absr_semi"] = np.log(mk["r_krx_semi"].abs() + 1e-3)
    d = d.merge(mk[["date", "rv_idx", "rv_semi", "absr_idx", "absr_semi", "vkospi"]], on="date", how="left")
    d["rv_fam"] = np.where(d["family"] == "semi", d["rv_semi"], d["rv_idx"])
    d["absr_fam"] = np.where(d["family"] == "semi", d["absr_semi"], d["absr_idx"])
    d["x_rv"] = d["lev2"] * d["rv_fam"]
    d["x_absr"] = d["lev2"] * d["absr_fam"]
    d["x_vk"] = d["lev2"] * np.log(d["vkospi"])
    return d


def lm_test(e2, Z):
    Z = sm.add_constant(Z, has_constant="add")
    r = sm.OLS(e2, Z).fit()
    lm = len(e2) * r.rsquared
    dfree = Z.shape[1] - 1
    return lm, dfree, stats.chi2.sf(lm, dfree)


def pesaran_cd(E: pd.DataFrame):
    E = E.dropna(axis=1, how="any")
    N, T = E.shape[1], E.shape[0]
    C = np.corrcoef(E.values.T)
    iu = np.triu_indices(N, 1)
    rho = C[iu]
    cd = np.sqrt(2 * T / (N * (N - 1))) * rho.sum()
    return cd, 2 * stats.norm.sf(abs(cd)), np.mean(np.abs(rho)), N, T


def wooldridge_ar(d, y="log_value"):
    """Wooldridge(2002) 패널 AR(1): 1차 차분 회귀 잔차 ê_it 를 ê_i,t-1 에 회귀, H0 계수 = −0.5."""
    d, X = design(d, True)
    d = d.sort_values(["code", "date"])
    g = d.groupby("code")
    dd = pd.DataFrame({"code": d["code"], "date": d["date"], "fd": d["fd"],
                       "dy": g[y].diff()})
    for x in X:
        dd["d_" + x] = g[x].diff()
    dd = dd.dropna()
    m = FE(dd, "dy", ["d_" + x for x in X], ["fd"])
    tmp = m.d.assign(e=m.e).sort_values(["code", "date"])
    tmp["e_lag"] = tmp.groupby("code")["e"].shift(1)
    tmp = tmp.dropna(subset=["e_lag"])
    r = sm.OLS(tmp["e"], tmp[["e_lag"]]).fit(cov_type="cluster", cov_kwds={"groups": codes(tmp["code"])})
    F = ((r.params["e_lag"] + 0.5) / r.bse["e_lag"]) ** 2
    G = tmp["code"].nunique()
    return r.params["e_lag"], F, stats.f.sf(F, 1, G - 1)


def ppml(d, split=True):
    d, X = design(d, split)
    D = pd.get_dummies(d[["code", "fd"]].astype(str), drop_first=True, dtype=float)
    Z = pd.concat([d[X].astype(float), D], axis=1)
    Z = sm.add_constant(Z)
    y = d["value"] / 1e8
    r = sm.GLM(y, Z, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": codes(d["code"])},
                                                       maxiter=200)
    return r.params[X], r.bse[X], r.pvalues[X]


def main():
    plt = plt_setup()
    d = window(main_panel())
    m, X = fit(d)
    e = m.e
    dd = m.d.assign(e=e)
    rows = []

    def rec(problem, test, stat, p, cause, effect, fix, before, after, limit):
        rows.append(dict(문제=problem, 검정=test, 통계량=stat, p=p, 원인=cause, 영향=effect, 해결=fix,
                         전=before, 후=after, 남은한계=limit))

    s_cr1 = m.summary("cr1", cluster="code")
    s_iid = m.summary("iid")
    s_dk = m.summary("dk", cluster="date")
    b_str = lambda s: f"β_B={s.loc['post_idx', 'coef']:.3f}(se {s.loc['post_idx', 'se']:.3f}), β_C={s.loc['post_semi', 'coef']:.3f}(se {s.loc['post_semi', 'se']:.3f})"

    # BP, White
    e2 = e ** 2
    Zbp = dd[X].astype(float).assign(lev2=dd["lev2"], semi=(dd["family"] == "semi").astype(int),
                                     post=dd["post"], trans=dd["trans"])
    lm, df_, p = lm_test(e2, Zbp)
    cells = pd.get_dummies(dd["lev2"].astype(str) + dd["family"] + dd["period"], drop_first=True, dtype=float)
    cells["logv_bar"] = dd.groupby("code")["log_value"].transform("mean")
    cells["logv_bar2"] = cells["logv_bar"] ** 2
    lmw, dfw, pw = lm_test(e2, cells)
    sd_by = dd.groupby(["lev2", "family"])["e"].std().round(3).to_dict()
    rec("이분산", "Breusch–Pagan", f"LM={lm:.1f}, df={df_}", p, "상품 규모·레버리지·계열마다 잔차 분산이 다름",
        "iid SE 가 과소/과대 → 검정 크기 왜곡", "이분산-강건 SE (HC1, CR1, DK, wild) 사용",
        f"iid: {b_str(s_iid)}", f"CR1: {b_str(s_cr1)}", f"잔차 sd (2X,계열): {sd_by}")
    rec("이분산(일반형)", "White (셀 더미+규모 제곱)", f"LM={lmw:.1f}, df={dfw}", pw, "같음", "같음", "같음", "", "", "")

    # Wooldridge AR(1)
    rho, F, pa = wooldridge_ar(d)
    rec("계열상관", "Wooldridge 패널 AR(1) (H0: 차분잔차 계수=−0.5)", f"계수={rho:.3f}, F={F:.1f}", pa,
        "거래대금은 지속성이 강함 (주문 흐름·관심의 지속)", "상품 내 오차 상관 → iid·날짜군집 SE 과소",
        "상품 군집(CR1)·wild·Driscoll–Kraay; 주 추론은 접힌 시계열 역사적 위약 (지속성을 분포에 내장)",
        f"HC1/iid se(β_C)={s_iid.loc['post_semi', 'se']:.3f}", f"CR1 se(β_C)={s_cr1.loc['post_semi', 'se']:.3f}",
        "클러스터 11(처치) → CR1 도 과소 가능. wild·위약으로 보완")

    # Pesaran CD
    E = dd.pivot(index="date", columns="code", values="e")
    cd, pcd, arho, N, T = pesaran_cd(E)
    raw = demean(d[["log_value"]].values, [codes(d["code"])])[:, 0]
    Er = d.assign(r=raw).pivot(index="date", columns="code", values="r")
    cd0, pcd0, arho0, _, _ = pesaran_cd(Er)
    rec("횡단면 의존", "Pesaran CD", f"FE 후 CD={cd:.1f} (평균|ρ|={arho:.2f}); 상품FE만 CD={cd0:.1f} (|ρ|={arho0:.2f})",
        pcd, "시장 공통충격 (변동성·투자심리)", "상품 군집 SE 가 공통충격을 무시하면 과소",
        "계열×일 FE 로 공통충격 흡수, Driscoll–Kraay·2-way·날짜군집 SE 병기",
        f"CD(상품 FE)={cd0:.1f}", f"CD(계열×일 FE)={cd:.1f}", "FE 후에도 남는 의존은 DK·위약으로만 처리")

    # 잔차·이상치·영향력
    z = (e - e.mean()) / e.std()
    out = dd.assign(z=z)
    big = out[np.abs(out["z"]) > 4]
    topd = big.groupby("date").size().sort_values(ascending=False).head(5)
    topc = big.groupby("code").size().sort_values(ascending=False).head(5)
    sk, ku = stats.skew(e), stats.kurtosis(e)
    rec("이상치·두꺼운 꼬리", "표준화 잔차 |z|>4", f"{len(big)}행 ({100 * len(big) / len(e):.2f}%), 왜도 {sk:.2f}, 첨도 {ku:.1f}",
        np.nan, f"상위 날짜 {topd.to_dict()}; 상위 상품 {topc.to_dict()}", "극단일이 기간 평균을 끌 수 있음",
        "삭제하지 않음. 원저화·LODO·도넛 창으로 민감도 (sensitivity.py)", "", "", "")

    # 함수형: log / asinh / PPML / level share
    m_as, _ = fit(d, y="asinh_value")
    pp_b, pp_se, pp_p = ppml(d)
    ff = (f"log: {b_str(s_cr1)}; asinh: β_B={m_as.b[0]:.3f}, β_C={m_as.b[2]:.3f}; "
          f"PPML: β_B={pp_b['post_idx']:.3f}(se {pp_se['post_idx']:.3f}), β_C={pp_b['post_semi']:.3f}(se {pp_se['post_semi']:.3f})")
    rec("함수형", "log vs asinh vs PPML", ff, pp_p["post_semi"],
        "로그는 상품 동일가중 % 변화, PPML 은 거래대금 큰 상품 가중", "추정대상이 달라 값이 다를 수 있음",
        "주 결과 log 유지, 대안 병기", "", "", "PPML 은 대형 ETF(122630 등)가 지배")
    # 로그 변환 타당성: 수준 잔차 왜도
    m_lv, _ = fit(d.assign(lv=d["value"] / 1e8), y="lv")
    rec("함수형(수준)", "수준 모형 잔차 왜도·첨도", f"수준 왜도 {stats.skew(m_lv.e):.1f}, 첨도 {stats.kurtosis(m_lv.e):.0f} vs log 왜도 {sk:.2f}",
        np.nan, "거래대금은 상품 간 1만 배 차이", "수준 모형은 소수 대형 상품이 지배", "log 사전 지정 유지", "", "", "")

    # VIF (공변량 모형)
    dc = covariates(d)
    dc, Xc = design(dc, True)
    Xv = Xc + ["x_rv", "x_vk"]
    dm = demean(dc[Xv].values, [codes(dc["code"]), codes(dc["fd"])])
    vifs = {}
    for j, nme in enumerate(Xv):
        oth = np.delete(dm, j, axis=1)
        r2 = sm.OLS(dm[:, j], oth).fit().rsquared
        vifs[nme] = 1 / (1 - r2)
    mc = FE(dc, "log_value", Xv, ["code", "fd"])
    sc = mc.summary("cr1", cluster="code")
    rec("다중공선성", "VIF (FE 흡수 후)", ", ".join(f"{k}={v:.1f}" for k, v in vifs.items()), np.nan,
        "변동성 공변량이 기간 더미와 같이 움직임 (기준구간 고변동성)", "공변량 넣으면 β 분산 증가",
        "공변량은 강건성 모형에서만", b_str(s_cr1), b_str(sc), "변동성은 규제의 결과일 수도 있음 (나쁜 통제 위험)")

    # 나쁜 통제
    dd2 = d.copy()
    aum = dd2["aum_won"].where(dd2["aum_won"] > 0)
    dd2["log_aum"] = np.log(aum)
    mb, _ = fit(dd2.dropna(subset=["log_aum"]), extra=["log_aum"])
    sb = mb.summary("cr1", cluster="code")
    rec("나쁜 통제", "log 순자산(사후 결정 변수) 추가", f"log_aum 계수 {sb.loc['log_aum', 'coef']:.3f}", sb.loc["log_aum", "p"],
        "순자산은 처치(수요 이동) 후 유입으로 바뀌는 결과변수", "통제하면 효과의 일부를 흡수 → 편의",
        "주 모형에서 제외 (사전 지정)", b_str(s_cr1), b_str(sb), "")

    diag = pd.DataFrame(rows)
    save_table(diag, "t06_diagnostics")
    for i, r in diag.iterrows():
        LOG.add(id=f"G{i + 1:02d}", phase="4_diagnostics", kind="diagnostic", purpose=r["문제"],
                model=r["검정"], estimate=r["통계량"], p=r["p"], diagnostics=f"원인: {r['원인']} | 영향: {r['영향']}",
                failure_revision=f"해결: {r['해결']} | 전: {r['전']} | 후: {r['후']}", interpretation=r["남은한계"])
    LOG.save()

    # 잔차 그림
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for i, (f, lab) in enumerate([("idx", "지수"), ("semi", "반도체")]):
        x = dd[dd.family == f]
        for l, ls in [(1, "-"), (0, ":")]:
            r = x[x.lev2 == l].groupby("date")["e"].mean()
            axes[0].plot(r, color=PAL[i], ls=ls, lw=1.2, label=f"{lab} {'2X' if l else '1X'}")
    axes[0].axhline(0, color=INK2, lw=0.7)
    axes[0].set_title("일별 평균 잔차 (FE 모형)")
    axes[0].legend(fontsize=7)
    axes[1].hist(z, bins=80, color=PAL[0])
    axes[1].set_yscale("log")
    axes[1].set_title("표준화 잔차 분포 (로그 빈도)")
    fig.savefig(FIG / "f04_residuals.png")
    plt.close(fig)
    print(diag[["문제", "검정", "통계량", "p"]].to_string())
    print(diag[["문제", "전", "후"]].to_string())


if __name__ == "__main__":
    main()
