"""실험 로그 (RQ2·RQ4 공용). 한 실험 = 한 행. CSV(기계용) + Markdown(사람용).

필드: id, rq, kind(baseline/extension/robustness/diagnostic/placebo), purpose, data, model, settings,
      diagnostics, robustness, result, interpretation, status(ok/failed/skipped), seed, timestamp
실패도 status=failed 로 남긴다.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

FIELDS = ["id", "rq", "kind", "purpose", "data", "model", "settings", "diagnostics", "robustness",
          "result", "interpretation", "status", "seed", "timestamp"]


class ExpLog:
    def __init__(self, path_csv: Path):
        self.path = Path(path_csv)
        self.rows: list[dict] = []

    def add(self, **kw) -> str:
        row = {f: kw.get(f, "") for f in FIELDS}
        for f in FIELDS:
            v = row[f]
            if isinstance(v, (dict, list, tuple)):
                row[f] = json.dumps(v, ensure_ascii=False, default=str)
        row["timestamp"] = datetime.now().isoformat(timespec="seconds")
        row["status"] = row["status"] or "ok"
        self.rows.append(row)
        return row["id"]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.rows, columns=FIELDS)
        df.to_csv(self.path, index=False, encoding="utf-8-sig")
        md = [f"# 실험 로그 ({self.path.stem})", "", f"총 {len(df)}건. 생성: {datetime.now():%Y-%m-%d %H:%M}", ""]
        for _, r in df.iterrows():
            md += [f"## {r['id']} [{r['kind']}] — {r['status']}",
                   f"- **목적**: {r['purpose']}", f"- **데이터**: {r['data']}", f"- **모형**: {r['model']}",
                   f"- **설정**: `{r['settings']}`"]
            for k, lab in [("diagnostics", "진단"), ("robustness", "강건성"), ("result", "결과"),
                           ("interpretation", "해석")]:
                if r[k]:
                    md.append(f"- **{lab}**: {r[k]}")
            md.append("")
        self.path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
