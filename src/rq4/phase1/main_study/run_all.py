"""RQ4 전체 재실행: py -3.13 -m src.rq4.phase1.main_study.run_all  (총 약 12~15분, 캐시가 있으면 DB 접속 없음)"""
from __future__ import annotations

import importlib
import time

MODULES = ["product_universe", "descriptive", "did", "event_study", "ddd", "diagnostics", "placebo",
           "robustness", "sensitivity", "mechanism", "synthesis"]


def main():
    for m in MODULES:
        t = time.time()
        print(f"=== {m} ===", flush=True)
        importlib.import_module(f"src.rq4.phase1.main_study.{m}").main()
        print(f"--- {m}: {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
