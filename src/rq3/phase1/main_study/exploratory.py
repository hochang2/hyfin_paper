"""RQ3 5단계 탐색: 보조 ETF 변수와 ML. 모든 결과는 탐색적이며 주 결론을 바꾸지 않는다.

- 원점 t 마다 index < t 인 행으로만 학습한다 (forecast.oos 와 같은 규칙).
- 선형 정규화 모형(Lasso, ENet)의 표준화·벌점 선택은 원점마다 학습 구간 안의 시계열 CV 로만 정한다.
- 트리 모형 하이퍼파라미터는 A 의 RQ2 ML 확장 기본값을 그대로 고정한다 (튜닝 없음: 작은 표본에서 튜닝 이득이
  표본 밖으로 이어지지 않았다는 RQ2 결과를 따름).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def make_model(kind: str, seed: int):
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "lasso":
        from sklearn.linear_model import LassoCV
        return make_pipeline(StandardScaler(), LassoCV(cv=TimeSeriesSplit(5), n_alphas=30, max_iter=20000))
    if kind == "enet":
        from sklearn.linear_model import ElasticNetCV
        return make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=TimeSeriesSplit(5),
                                                            n_alphas=30, max_iter=20000))
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=10, max_features=0.6,
                                     random_state=seed, n_jobs=1)
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=15,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                                 n_jobs=1, verbose=-1)
    if kind == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=2, subsample=0.8,
                                colsample_bytree=0.8, random_state=seed, n_jobs=1)
    raise ValueError(kind)


def oos_ml(d: pd.DataFrame, cols: list[str], first_origin: pd.Timestamp, kind: str, seed: int) -> pd.DataFrame:
    d = d.dropna(subset=["y_next"])
    X, y, idx = d[cols].to_numpy(float), d["y_next"].to_numpy(float), d.index
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in np.where(idx >= first_origin)[0]:
            m = make_model(kind, seed)
            m.fit(X[:i], y[:i])
            rows.append({"date": idx[i], "y": y[i], "yhat": float(m.predict(X[i:i + 1])[0])})
    return pd.DataFrame(rows).set_index("date")
