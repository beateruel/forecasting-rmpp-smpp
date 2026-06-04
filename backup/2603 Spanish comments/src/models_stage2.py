
# -*- coding: utf-8 -*-
"""Modelos Stage 2 para mapear RMPP→SMPP.
Incluye: Regresión Lineal y XGB Regressor.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from .logger import get_logger
log = get_logger(__name__)



class LinearStage2:
    def __init__(self):
        self.reg = LinearRegression()
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)
        self.reg.fit(X, y)
        log.info(f"Fit Linear Stage2 ")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        log.info(f"Predict Linear Stage2 ")
        return self.reg.predict(X)


class XGBStage2:
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
        log.info(f"Fit XGB Stage2 ")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        log.info(f"Predict XGB Stage2 ")
        return self.reg.predict(X)
