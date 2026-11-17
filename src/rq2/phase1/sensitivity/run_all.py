"""RQ2 최종 민감도 분석 전체 실행:  py -3.13 -m src.rq2.phase1.sensitivity.run_all [--only s1,s2,...]

순서가 중요하다: Exp5 는 Exp1·Exp2 결과표를, 진단은 Exp2·Exp5 결과표를 읽는다.
재적합은 data/interim/rq_cache/sensitivity/fits 에 캐시되므로 두 번째 실행부터는 몇 분 안에 끝난다.
"""
from __future__ import annotations

import argparse
import time
import traceback

from src.rq2.phase1.sensitivity import common as C

STAGES = [
    ("s1", "src.rq2.phase1.sensitivity.bias_sensitivity"),
    ("s2", "src.rq2.phase1.sensitivity.contamination"),
    ("s3", "src.rq2.phase1.sensitivity.confounding"),
    ("s4", "src.rq2.phase1.sensitivity.event_window"),
    ("s5", "src.rq2.phase1.sensitivity.joint_sensitivity"),
    ("s6", "src.rq2.phase1.sensitivity.diagnostics"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    import importlib
    for key, mod in STAGES:
        if only and key not in only:
            continue
        t = time.time()
        print(f"[{key}] {mod} ...", flush=True)
        try:
            importlib.import_module(mod).run()
            print(f"[{key}] ok {time.time() - t:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            C.LOG.add(id=f"RQ2-{key.upper()}-FAILED", rq="RQ2", kind="robustness", purpose=mod, status="failed",
                      interpretation=repr(e), seed=C.SEED)
    C.LOG.save()


if __name__ == "__main__":
    main()
