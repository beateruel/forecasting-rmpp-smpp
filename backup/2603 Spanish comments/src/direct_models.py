
# -*- coding: utf-8 -*-
"""Modelos directos para pronosticar SMPP usando RMPP como exógena.
Incluye: SARIMAX-exog y XGB con exógenas.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from .logger import get_logger
log = get_logger(__name__)



class SARIMAXExog:
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, y: pd.Series, X: pd.DataFrame):
        self.model_ = SARIMAX(
            y, exog=X, order=self.order, seasonal_order=self.seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
        return self

    def predict(self, steps: int, X_future: pd.DataFrame) -> np.ndarray:
        return self.model_.forecast(steps=steps, exog=X_future)


class XGBExog:
    def __init__(self, random_state=42):
        self.reg = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror',
            random_state=random_state, n_jobs=2
        )
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)
        self.reg.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.reg.predict(X)
