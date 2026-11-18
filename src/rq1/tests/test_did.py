"""DiD 구현 검증 (모의자료). 실제 데이터는 쓰지 않는다."""
import numpy as np
import pandas as pd

from src.rq1.core import did


def _sim(effect=0.3, n_units=40, n_pre=60, n_post=20, seed=0, drop_one=False):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-01", periods=n_pre + n_post)
    codes = ["T"] + [f"C{i:02d}" for i in range(n_units)]
    a = rng.normal(0, 0.5, len(codes))
    lam = rng.normal(0, 0.3, len(dates)).cumsum() * 0.2
    rows = []
    for i, c in enumerate(codes):
        for t, d in enumerate(dates):
            post = int(t >= n_pre)
            y = a[i] + lam[t] + (effect if (c == "T" and post) else 0) + rng.normal(0, 0.2)
            rows.append((d, c, y, post, t - n_pre, (t - n_pre) // 5))
    df = pd.DataFrame(rows, columns=["date", "code", "y", "post", "rel", "bin"])
    if drop_one:
        df = df.drop(index=df.index[100])
    return df, codes[1:]


def test_recovers_effect_balanced():
    df, ctrl = _sim(effect=0.3)
    b = did.did_beta(df, "T", ctrl)
    assert abs(b - 0.3) < 0.08


def test_balanced_equals_simple_diff_in_means():
    df, ctrl = _sim(effect=0.2, seed=1)
    w = df.pivot(index="date", columns="code", values="y")
    post = df.drop_duplicates("date").set_index("date")["post"]
    d = w["T"] - w[ctrl].mean(axis=1)
    simple = d[post == 1].mean() - d[post == 0].mean()
    assert abs(did.did_beta(df, "T", ctrl) - simple) < 1e-8
    assert abs(did.did_hac(df, "T", ctrl)["beta_hac"] - simple) < 1e-8


def test_unbalanced_runs():
    df, ctrl = _sim(effect=0.3, drop_one=True)
    assert abs(did.did_beta(df, "T", ctrl) - 0.3) < 0.08


def test_null_permutation_not_extreme():
    df, ctrl = _sim(effect=0.0, seed=2)
    b = did.did_beta(df, "T", ctrl)
    null = did.placebo_space(df, ctrl)["D"].values
    assert did.perm_p(b, null) > 0.05


def test_event_study_bins():
    df, ctrl = _sim(effect=0.4, n_pre=60, n_post=20)
    d, cols = did.design(df, "T", ctrl, event_study=True)
    b = did.twfe(d, cols)
    post_bins = [c for c in cols if int(c[1:]) >= 0]
    assert all(abs(b[c] - 0.4) < 0.2 for c in post_bins)
    assert "b-9" not in cols   # 기준 구간


def test_holm():
    h = did.holm({"a": 0.01, "b": 0.04})
    assert h == {"a": 0.02, "b": 0.04}
    h = did.holm({"a": 0.03, "b": 0.02})
    assert h["b"] == 0.04 and h["a"] == 0.04
