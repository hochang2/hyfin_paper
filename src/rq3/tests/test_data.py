"""RQ3 데이터 점검: 상품 구성, 상장 전 0, 정보 시점(미래 정보 차단), 예측 대상 정렬. DB 캐시를 쓴다."""
import numpy as np
import pandas as pd
import pytest

from src.rq3.main_study import data as D

LISTING = D.CFG["sample"]["listing"]


@pytest.fixture(scope="module", params=["005930", "000660"])
def code(request):
    return request.param


def test_products_9_each():
    p = D.raw_products().drop_duplicates("code")
    assert len(p) == 18
    for u in ["005930", "000660"]:
        q = p[p.underlying == u]
        assert len(q) == 9
        assert sorted(q.leverage.tolist()) == [-2] + [2] * 8
        assert (q.kind == "ETN").sum() == 1


def test_sample_days_and_first_origin():
    days = D.trading_days()
    assert len(days) == 261 and days[0] == D.CFG["sample"]["start"] and days[-1] == D.CFG["sample"]["end"]
    assert (days >= LISTING).sum() == 106
    assert D.first_origin() == pd.Timestamp("2026-07-08")


def test_zero_before_listing_positive_after(code):
    d = D.design(code)
    pre, post = d[d.index < LISTING], d[d.index >= LISTING]
    assert (pre[["S", "X", "D"]] == 0).all().all()
    assert (post.D == 1).all() and (post.S > 0).all()


def test_y_next_is_next_day_log_rv(code):
    d = D.design(code)
    assert np.allclose(d.y_next.iloc[:-1].values, d.har_d.iloc[1:].values)
    assert np.isnan(d.y_next.iloc[-1])


def test_scale_uses_previous_day_aum(code):
    """S_t 는 t−1 일 AUM 만 쓴다 (상장일만 당일). 수동 계산과 비교."""
    p = D.raw_products()
    p = p[(p.underlying == code) & (p.date >= LISTING)]
    days = D.trading_days()
    post = days[days >= LISTING]
    val = D.stock_daily(code)["value"]
    S = D.scale(code)
    for i in [0, 1, 50, len(post) - 1]:
        t = post[i]
        a_day = post[i - 1] if i > 0 else t
        a = p[p.date == a_day]
        manual = (a.leverage * (a.leverage - 1) * a.aum_calc).sum() / val[t]
        assert np.isclose(S[t], manual)


def test_scale_ignores_future_aum(code):
    """t 이후 AUM 을 바꿔도 S_t 는 그대로여야 한다."""
    base = D.scale(code)
    orig = D.raw_products
    cut = pd.Timestamp("2026-08-14")

    def bumped():
        df = orig()
        df.loc[df.date > cut, "aum_calc"] *= 10
        return df
    D.raw_products = bumped
    try:
        new = D.scale(code)
    finally:
        D.raw_products = orig
    days = D.trading_days()
    nxt = days[days > cut][0]      # cut 다음 날은 전일(cut) AUM 을 쓰므로 그대로
    assert np.allclose(base[:nxt], new[:nxt])
    assert not np.allclose(base[nxt:].iloc[1:], new[nxt:].iloc[1:])


def test_negative_control_uses_samsung_pressure():
    d = D.design("005935", "005930")
    s = D.design("005930")
    assert np.allclose(d.X, s.X) and np.allclose(d.S, s.S)
    assert not np.allclose(d.R, s.R)
