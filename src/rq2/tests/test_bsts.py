"""BSTS 구현 점검: 알려진 효과를 넣은 모의자료에서 효과를 되찾는지, 효과 0 일 때 구간이 0을 포함하는지."""
import numpy as np

from src.rq2.core.bsts import fit_bsts


def _sim(effect, seed, T0=180, T1=60, k=6):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T0 + T1, k)).cumsum(0) * 0.1 + rng.standard_normal((T0 + T1, k))
    level = np.cumsum(rng.standard_normal(T0 + T1) * 0.02)
    y = level + 0.8 * X[:, 0] - 0.5 * X[:, 1] + rng.standard_normal(T0 + T1) * 0.3
    y[T0:] += effect
    return y[:T0], X[:T0], y[T0:], X[T0:]


def test_recovers_effect():
    yp, Xp, yq, Xq = _sim(-0.5, 1)
    r = fit_bsts(yp, Xp, Xq, niter=1500, burn=500, seed=0)
    eff = (yq[None, :] - r.pred_draws).mean(1)
    lo, hi = np.quantile(eff, [0.025, 0.975])
    assert lo < -0.5 < hi
    assert hi < 0
    ip = r.inclusion_prob
    assert ip[0] > 0.9 and ip[1] > 0.9


def test_null_coverage():
    cover = 0
    for s in range(8):
        yp, Xp, yq, Xq = _sim(0.0, 100 + s)
        r = fit_bsts(yp, Xp, Xq, niter=1000, burn=300, seed=s)
        eff = (yq[None, :] - r.pred_draws).mean(1)
        lo, hi = np.quantile(eff, [0.025, 0.975])
        cover += lo < 0 < hi
    assert cover >= 6


def test_constant_column():
    """학습구간에서 상수인 열이 있어도 실패하지 않고 그 열은 선택되지 않아야 한다."""
    yp, Xp, yq, Xq = _sim(0.0, 7)
    Xp = np.column_stack([Xp, np.zeros(len(Xp))])
    Xq = np.column_stack([Xq, np.ones(len(Xq))])
    r = fit_bsts(yp, Xp, Xq, niter=400, burn=100, seed=0)
    assert r.inclusion_prob[Xp.shape[1] - 1] == 0


def test_no_regressors():
    """회귀변수 0개(C0, local level 만)도 돌아가야 한다."""
    import numpy as np
    from src.rq2.core.bsts import fit_bsts
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(0, 0.05, 120)) + rng.normal(0, 0.3, 120)
    r = fit_bsts(y, np.empty((120, 0)), np.empty((20, 0)), niter=300, burn=100, seed=0)
    assert np.isfinite(np.asarray(r.pred_draws)).all() if hasattr(r, "pred_draws") else True
