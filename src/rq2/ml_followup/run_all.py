"""후속 실험 일괄 실행: py -3.13 -m src.rq2.ml_followup.run_all [단계 ...]

단계는 캐시(data/interim/rq_cache/ml_followup)를 쓰므로 다시 돌려도 이미 끝난 적합은 건너뛴다.
"""
from __future__ import annotations

import sys
import time
import traceback

STEPS = {
    "exp5_fit": lambda: (__import__("src.rq2.ml_followup.exp5_audit", fromlist=["x"]).run_rv5_space(),
                         __import__("src.rq2.ml_followup.exp5_audit", fromlist=["x"]).run_gk_space(),
                         __import__("src.rq2.ml_followup.exp5_audit", fromlist=["x"]).run_gk_dense()),
    "exp5_analysis": lambda: __import__("src.rq2.ml_followup.exp5_audit", fromlist=["x"]).run_analysis(),
    "exp1_local": lambda: __import__("src.rq2.ml_followup.exp1_global", fromlist=["x"]).run_local_pseudo(),
    "exp1_pseudo": lambda: __import__("src.rq2.ml_followup.exp1_global", fromlist=["x"]).run_pseudo(),
    "exp1_k": lambda: __import__("src.rq2.ml_followup.exp1_global", fromlist=["x"]).run_k_subsets(),
    "exp1_real": lambda: __import__("src.rq2.ml_followup.exp1_global", fromlist=["x"]).run_real(),
    "exp6_sim": lambda: __import__("src.rq2.ml_followup.exp6_dgp", fromlist=["x"]).run_sim(),
    "exp4_rv5": lambda: __import__("src.rq2.ml_followup.exp4_sample", fromlist=["x"]).run_rv5_lengths(),
    "exp4_nt": lambda: __import__("src.rq2.ml_followup.exp4_sample", fromlist=["x"]).run_nt(),
    "exp4_gk": lambda: __import__("src.rq2.ml_followup.exp4_sample", fromlist=["x"]).run_gk_lengths(),
    "exp7_run": lambda: __import__("src.rq2.ml_followup.exp7_donor", fromlist=["x"]).run(),
    "exp2_run": lambda: __import__("src.rq2.ml_followup.exp2_transfer", fromlist=["x"]).run(),
    "exp3_run": lambda: __import__("src.rq2.ml_followup.exp3_fm", fromlist=["x"]).run(),
    "analysis": lambda: [__import__(f"src.rq2.ml_followup.{m}", fromlist=["x"]).analyze()
                         for m in ("exp1_global", "exp2_transfer", "exp4_sample", "exp6_dgp", "exp7_donor", "exp3_fm")],
    "figures": lambda: [f() for f in __import__("src.rq2.ml_followup.figures", fromlist=["x"]).ALL],
}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(STEPS)
    for s in todo:
        t = time.time()
        print(f"=== {s} start {time.strftime('%H:%M:%S')}", flush=True)
        try:
            STEPS[s]()
            print(f"=== {s} done {time.time() - t:.0f}s", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"=== {s} FAILED", flush=True)
