"""예측·검정 구현 검증 (모의자료). 실제 데이터는 쓰지 않는다."""
import numpy as np
import pandas as pd

from src.rq3.core import forecast as F


def _sim(beta_x=0.0, n=240, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-11-03", periods=n)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = 0.5 * x1 + beta_x * x2 + rng.normal(size=n)
    return pd.DataFrame({"y_next": y, "x1": x1, "x2": x2}, index=idx), idx[n - 76]


def test_no_lookahead_training_excludes_origin_row():
    d, fo = _sim()
    f = F.oos(d, ["x1"], fo)
    # 원점 t 의 학습 행 수 = t 보다 앞선 행 수
    pos = {t: i for i, t in enumerate(d.index)}
    assert all(f.n_train.loc[t] == pos[t] for t in f.index)
    # 원점 행의 y 를 크게 바꿔도 그 원점의 예측은 그대로
    d2 = d.copy()
    t0 = f.index[0]
    d2.loc[t0, "y_next"] += 1e6
    assert np.isclose(F.oos(d2, ["x1"], fo).yhat.iloc[0], f.yhat.iloc[0])


def test_cw_detects_true_predictor():
    # 한 seed 는 우연히 진짜 모형도 표본외에서 질 수 있다 → 여러 seed 의 검출률로 본다
    hits = 0
    for k in range(1, 11):
        d, fo = _sim(beta_x=0.6, seed=k)
        s, b = F.oos(d, ["x1"], fo), F.oos(d, ["x1", "x2"], fo)
        hits += F.clark_west(s.y, s.yhat, b.yhat)["p_one_sided"] < 0.05
    assert hits >= 8


def test_cw_size_under_null():
    rej = 0
    for k in range(200):
        d, fo = _sim(beta_x=0.0, seed=100 + k)
        s, b = F.oos(d, ["x1"], fo), F.oos(d, ["x1", "x2"], fo)
        rej += F.clark_west(s.y, s.yhat, b.yhat)["p_one_sided"] < 0.05
    assert 0.01 <= rej / 200 <= 0.11


def test_rolling_and_constant_column_dropped():
    d, fo = _sim()
    d["c"] = 1.0
    f = F.oos(d, ["x1", "c"], fo, window="rolling", rolling=60)
    assert (f.n_train == 60).all() and f.b_c.isna().all() and f.yhat.notna().all()


def test_block_shuffle_stays_within_groups():
    rng = np.random.default_rng(0)
    s = pd.Series(np.arange(23, dtype=float))
    g = np.array(["a"] * 12 + ["b"] * 4 + ["c"] * 7)
    for _ in range(20):
        out = F.block_shuffle(s, g, 5, rng)
        for k in "abc":
            assert sorted(out[g == k]) == sorted(s[g == k])
    # 블록 안의 순서는 유지된다 (5일 블록 0..4 는 연속으로 남는다)
    out = F.block_shuffle(s, np.array(["a"] * 23), 5, np.random.default_rng(1)).to_numpy()
    i = int(np.where(out == 0)[0][0])
    assert list(out[i:i + 5]) == [0, 1, 2, 3, 4]


def test_ols_hac_matches_ols_coef():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 2))
    y = 1 + X @ [0.5, -0.2] + rng.normal(size=200)
    b, se = F.ols_hac(y, X, 3)
    assert np.allclose(b, np.linalg.lstsq(np.column_stack([np.ones(200), X]), y, rcond=None)[0])
    assert (se > 0).all()


def test_holm():
    assert F.holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    h = F.holm({"a": 0.03, "b": 0.02})
    assert h["a"] == h["b"] == 0.04
