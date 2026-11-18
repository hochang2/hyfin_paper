"""RQ3 그림. 스타일은 RQ1·RQ2 와 같게 맞춘다."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "violet": "#6250d6",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "band": "#86b6ef"}
NAMES = {"005930": "삼성전자", "000660": "SK하이닉스", "005935": "삼성전자우"}

plt.rcParams.update({"font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["ink2"],
                     "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
                     "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "figure.dpi": 130,
                     "savefig.bbox": "tight", "lines.linewidth": 1.4})


def _events(ax, tl, first_origin):
    for k, lab, y in [("listing", "상장", 1.0), ("announce", "발표", 1.0), ("effective", "시행", 0.9)]:
        ax.axvline(tl[k], color=C["ink2"], ls=":", lw=0.9)
        ax.text(tl[k], y, f" {lab}", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=C["ink2"])
    ax.axvline(first_origin, color=C["orange"], ls="--", lw=0.9)
    ax.text(first_origin, 0.9, "첫 예측 원점 ", transform=ax.get_xaxis_transform(), va="top", ha="right",
            fontsize=8, color=C["orange"])


def cum_loss_diff(oos: dict[str, dict[str, pd.DataFrame]], tl: dict, out_png):
    """누적 (e_B² − e_C²): 올라가면 C 가 B 보다 잘 맞힌 구간. 참고로 A 대비 B 도 그린다."""
    codes = list(oos)
    fig, axes = plt.subplots(1, len(codes), figsize=(6 * len(codes), 3.8), sharey=True)
    for ax, c in zip(axes, codes):
        f = oos[c]
        eb, ec, ea = [(f[m].y - f[m].yhat) ** 2 for m in "BCA"]
        ax.plot(f["C"].index, (eb - ec).cumsum(), color=C["orange"], label="B − C (주 비교: X 추가 효과)")
        ax.plot(f["C"].index, (ea - eb).cumsum(), color=C["ink2"], lw=0.9, ls="--", label="A − B (R 추가 효과, 보조)")
        ax.axhline(0, color=C["ink"], lw=0.7)
        for k, lab in [("announce", "발표"), ("effective", "시행")]:
            ax.axvline(tl[k], color=C["ink2"], ls=":", lw=0.9)
            ax.text(tl[k], 1.0, f" {lab}", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=C["ink2"])
        ax.set_title(f"{NAMES.get(c, c)} ({c})", fontsize=11, loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("누적 제곱오차 차이 (ln RV)")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("표본외 누적 손실 차이 (0 위 = 뒤 모형이 더 잘 맞힘)", fontsize=12, x=0.01, y=1.04, ha="left")
    fig.savefig(out_png)
    plt.close(fig)


def placebo_hist(null: dict, real: dict, out_png, title: str = "위약: 규모 비율 S 를 규제 구간 안에서 5일 블록으로 섞은 1000번",
                 xlabel: str = "Clark-West t (C vs B)"):
    """구간 내 블록 위약의 Clark-West t 분포와 실제 값."""
    codes = list(null)
    fig, axes = plt.subplots(1, len(codes), figsize=(5.5 * len(codes), 3.4))
    for ax, c in zip(axes, codes):
        ax.hist(null[c], bins=40, color=C["band"], edgecolor="white")
        ax.axvline(real[c], color=C["orange"], lw=1.6, label=f"실제 {real[c]:.2f}")
        ax.axvline(pd.Series(null[c]).quantile(0.95), color=C["ink2"], ls=":", lw=1, label="위약 95%")
        ax.set_title(f"{NAMES.get(c, c)} ({c})", fontsize=11, loc="left")
        ax.set_xlabel(xlabel)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=12, x=0.01, y=1.04, ha="left")
    fig.savefig(out_png)
    plt.close(fig)


def coef_path(fc: dict, tl: dict, out_png):
    """확장 창 원점별 R·X 계수."""
    codes = list(fc)
    fig, axes = plt.subplots(1, len(codes), figsize=(6 * len(codes), 3.4), sharey=True)
    for ax, c in zip(axes, codes):
        f = fc[c]
        ax.plot(f.index, f.b_R, color=C["ink2"], label="R 계수")
        ax.plot(f.index, f.b_X, color=C["orange"], label="X 계수")
        ax.axhline(0, color=C["ink"], lw=0.7)
        for k, lab in [("announce", "발표"), ("effective", "시행")]:
            ax.axvline(tl[k], color=C["ink2"], ls=":", lw=0.9)
            ax.text(tl[k], 1.0, f" {lab}", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=C["ink2"])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(f"{NAMES.get(c, c)} ({c})", fontsize=11, loc="left")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("모형 C 확장 창 계수 경로", fontsize=12, x=0.01, y=1.04, ha="left")
    fig.savefig(out_png)
    plt.close(fig)


def robustness_forest(rob: pd.DataFrame, out_png):
    """사양별 R²oos(C|B). 점 색: CW 한쪽 p < 0.05 이면 주황."""
    codes = list(dict.fromkeys(rob.code))
    specs = list(dict.fromkeys(rob.spec))
    fig, axes = plt.subplots(1, len(codes), figsize=(5.2 * len(codes), 0.38 * len(specs) + 1.2), sharey=True)
    for ax, c in zip(axes, codes):
        r = rob[rob.code == c].set_index("spec").loc[specs]
        y = range(len(specs))[::-1]
        col = [C["orange"] if p < 0.05 else C["blue"] for p in r.cw_p]
        ax.scatter(100 * r.r2_oos, list(y), color=col, zorder=3, s=22)
        ax.axvline(0, color=C["ink"], lw=0.7)
        ax.set_yticks(list(y))
        ax.set_yticklabels(specs, fontsize=8)
        ax.set_xlabel("R²oos (C vs B, %)")
        ax.set_title(f"{NAMES.get(c, c)} ({c})", fontsize=11, loc="left")
    fig.suptitle("강건성: X 추가의 표본외 개선 (주황 = CW p<0.05)", fontsize=12, x=0.01, y=1.02, ha="left")
    fig.savefig(out_png)
    plt.close(fig)


def data_overview(designs: dict[str, pd.DataFrame], tl: dict, first_origin, out_png):
    """종목별 3단: ln RV, 규모 비율 S, 수익률 충격 R 과 리밸런싱 압력 X (상장 후 구간 중심).
    예측 결과가 아니라 설명변수 원자료다."""
    codes = list(designs)
    fig, axes = plt.subplots(3, len(codes), figsize=(6 * len(codes), 8), sharex=True)
    start = tl["listing"] - pd.Timedelta(days=45)
    for j, c in enumerate(codes):
        d = designs[c].loc[start:]
        ax = axes[0, j]
        ax.plot(d.index, d.har_d, color=C["ink"], lw=1.0)
        ax.set_title(f"{NAMES.get(c, c)} ({c})", fontsize=11, loc="left")
        ax.set_ylabel("ln RV" if j == 0 else "")
        ax = axes[1, j]
        ax.plot(d.index, d.S, color=C["blue"])
        ax.set_ylabel("S = ΣL(L−1)·AUM(t−1) / 거래대금" if j == 0 else "", fontsize=8)
        ax = axes[2, j]
        ax.plot(d.index, d.R, color=C["ink2"], lw=0.9, label="R = ln(1+100|r|)")
        ax.plot(d.index, d.X, color=C["orange"], lw=1.1, label="X = ln(1+100|r|·S)")
        if j == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper left")
        for a in axes[:, j]:
            _events(a, tl, first_origin)
            a.set_xlim(start, d.index.max())
    fig.suptitle("RQ3 설명변수 원자료 (상장 45일 전부터)", fontsize=12, x=0.01, ha="left")
    fig.savefig(out_png)
    plt.close(fig)
