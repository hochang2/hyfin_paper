"""RQ4 공통: 경로·설정·실험 로그·캐시·표 저장.

원칙
· DB 는 `hyfin.proc` 만 읽는다. 원자료는 바꾸지 않는다. 중간 자료는 CACHE(parquet)에만 쓴다.
· 설계(집단·기간·결과변수·주 추론·검정 가족)는 config.yaml 에 추정 전에 고정했다. 바꾸면 로그에 남긴다.
· 결과가 유의하지 않아도 그대로 남긴다. 실패도 status=failed 로 남긴다.
"""
from __future__ import annotations

import json
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = HERE / "results"
TAB = RES / "tables"
FIG = RES / "figures"
CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
CACHE = ROOT / CFG["cache_dir"]
SEED = int(CFG["seed"])
TL = {k: pd.Timestamp(v) for k, v in CFG["timeline"].items()}

for p in (RES, TAB, FIG, CACHE):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 실험 로그
FIELDS = ["id", "phase", "kind", "hypothesis", "purpose", "treated", "control", "period", "outcome",
          "controls", "model", "estimand", "estimate", "ci_low", "ci_high", "se", "p", "p_adj", "correction",
          "inference", "diagnostics", "failure_revision", "robustness", "sensitivity", "interpretation",
          "status", "runtime_s", "seed", "timestamp"]


class ExpLog:
    """한 실험 = 한 행. 모듈별 CSV 조각(results/log_parts/*.csv)에 쓰고 synthesis 가 합친다.
    모듈을 따로 다시 돌려도 다른 모듈 로그가 지워지지 않게 하려는 구조 (재시작 가능)."""

    def __init__(self, part: str):
        self.part = part
        self.path = RES / "log_parts" / f"{part}.csv"
        self.rows: list[dict] = []
        self._t0 = time.time()

    def add(self, **kw) -> str:
        row = {f: kw.get(f, "") for f in FIELDS}
        for f in FIELDS:
            v = row[f]
            if isinstance(v, (dict, list, tuple)):
                row[f] = json.dumps(v, ensure_ascii=False, default=str)
            elif isinstance(v, (float, np.floating)):
                row[f] = float(v)
        row["timestamp"] = datetime.now().isoformat(timespec="seconds")
        row["status"] = row["status"] or "ok"
        row["seed"] = row["seed"] or SEED
        if row["runtime_s"] == "":
            row["runtime_s"] = round(time.time() - self._t0, 1)
        self._t0 = time.time()
        self.rows.append(row)
        return row["id"]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows, columns=FIELDS).to_csv(self.path, index=False, encoding="utf-8-sig")


def merge_logs() -> pd.DataFrame:
    parts = sorted((RES / "log_parts").glob("*.csv"))
    order = ["01_universe", "02_descriptive", "03_did", "04_event_study", "05_ddd", "06_diagnostics",
             "07_placebo", "08_robustness", "09_sensitivity", "10_mechanism", "11_synthesis"]
    parts = sorted(parts, key=lambda p: order.index(p.stem) if p.stem in order else 99)
    df = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in parts], ignore_index=True)
    df.to_csv(RES / "experiment_log.csv", index=False, encoding="utf-8-sig")
    md = ["# RQ4 실험 로그", "", f"총 {len(df)}건. 생성 {datetime.now():%Y-%m-%d %H:%M}. "
          "열 정의는 README 참조. 실패·반대 결과도 모두 남김.", ""]
    for _, r in df.iterrows():
        md.append(f"## {r['id']} [{r['phase']}/{r['kind']}] — {r['status']}")
        for k, lab in [("hypothesis", "가설"), ("purpose", "목적"), ("treated", "처치"), ("control", "대조"),
                       ("period", "기간"), ("outcome", "결과변수"), ("controls", "통제"), ("model", "모형"),
                       ("estimand", "추정대상")]:
            if r[k]:
                md.append(f"- **{lab}**: {r[k]}")
        if r["estimate"]:
            md.append(f"- **추정치**: {r['estimate']} [{r['ci_low']}, {r['ci_high']}], se={r['se']}, "
                      f"p={r['p']}, 보정 p={r['p_adj']} ({r['correction']}); 추론: {r['inference']}")
        for k, lab in [("diagnostics", "진단"), ("failure_revision", "실패·수정"), ("robustness", "강건성"),
                       ("sensitivity", "민감도"), ("interpretation", "해석")]:
            if r[k]:
                md.append(f"- **{lab}**: {r[k]}")
        md.append(f"- 실행 {r['runtime_s']}s, seed {r['seed']}, {r['timestamp']}")
        md.append("")
    (RES / "experiment_log.md").write_text("\n".join(md), encoding="utf-8")
    return df


# ---------------------------------------------------------------- 표·캐시
def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    p = TAB / f"{name}.csv"
    df.to_csv(p, index=index, encoding="utf-8-sig")
    return p


def cache_path(name: str) -> Path:
    return CACHE / f"{name}.parquet"


def cached(name: str, fn, force: bool = False) -> pd.DataFrame:
    p = cache_path(name)
    if p.exists() and not force:
        return pd.read_parquet(p)
    df = fn()
    df.to_parquet(p, index=False)
    return df


def pct(x):
    """log 차이 → % 변화."""
    return 100 * np.expm1(np.asarray(x, float))


def fmt(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return str(x)


# ---------------------------------------------------------------- 그림 (정적 PNG, 보고서용)
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#52514e"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plt_setup():
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    warnings.filterwarnings("ignore")
    for f in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams.update({"axes.unicode_minus": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                         "font.size": 9, "axes.titlesize": 10, "axes.titlecolor": INK, "legend.frameon": False,
                         "savefig.dpi": 150, "savefig.bbox": "tight"})
    return plt


def vlines(ax, keys=("listing", "announce", "effective", "premium_rule")):
    lab = {"listing": "상장", "announce": "발표", "effective": "시행", "premium_rule": "괴리율"}
    for k in keys:
        ax.axvline(TL[k], color=INK2, lw=0.8, ls=":")
        ax.text(TL[k], 1.0, lab.get(k, k), transform=ax.get_xaxis_transform(), fontsize=7, color=INK2,
                ha="center", va="bottom")
