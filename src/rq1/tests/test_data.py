"""RQ1 패널 점검: 표본 경계, 역할 구분, 누출 방지. DB 캐시(data/interim/rq_cache)를 쓴다."""
import pandas as pd
import pytest

from src.rq1.main_study import data as D


@pytest.fixture(scope="module")
def df():
    return D.panel_long("rv5")


def test_sample_ends_before_announcement(df):
    # 07-16(발표) 이후 자료가 들어오면 RQ2 기간이 섞인다
    assert df.date.max() <= D.CFG["sample"]["treat_end"]
    assert df.date.max() < D.CFG["timeline"]["announce"]
    assert df.date.min() >= D.CFG["sample"]["start"]


def test_pre_post_days(df):
    t = df[df.code == "005930"]
    assert (t.post == 0).sum() == 155
    assert (t.post == 1).sum() == 35
    assert t.loc[t.rel == 0, "date"].iloc[0] == D.CFG["timeline"]["listing"]


def test_post_bins_cover_treatment_exactly(df):
    t = df[df.code == "005930"]
    assert sorted(t.loc[t.post == 1, "bin"].unique()) == list(range(0, 7))
    assert t.loc[t.post == 0, "bin"].max() == -1


def test_roles_disjoint_and_sizes():
    g = D.groups()
    codes = [c for v in g.values() for c in v]
    assert len(codes) == len(set(codes))
    assert len(g["donor"]) == 175
    assert len(g["neg_semis"]) == 7
    assert "005935" not in g["donor"] and not set(g["treated"]) & set(g["donor"])


def test_no_zero_fill(df):
    # 결측을 0 으로 채우지 않았는지 (log RV 는 유한한 음수 범위)
    assert df.y.notna().all()
    assert (df.y < 0).all()


def test_treated_and_negative_controls_complete(df):
    n = df[df.role != "donor"].groupby("code").size()
    assert (n == 190).all()
