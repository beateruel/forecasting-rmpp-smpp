
# -*- coding: utf-8 -*-
"""Modelos Stage 1 para pronosticar RMPP.
Incluye: SES, ARIMA sencillo, RF, XGB, LGBM.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from .logger import get_logger
log = get_logger(__name__)



class SESModel:
    def fit(self, y: pd.Series):
        log.info(f"Fit modelo SES ")
        self.model_ = SimpleExpSmoothing(y, initialization_method='estimated').fit()
        return self

    def predict(self, steps: int) -> np.ndarray:
        log.info(f"Predict modelo SES ")
        return self.model_.forecast(steps)


class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        self.order = order

    def fit(self, y: pd.Series):
        log.info(f"Fit ARIMA ")
        self.model_ = ARIMA(y, order=self.order).fit()
        return self

    def predict(self, steps: int) -> np.ndarray:
        log.info(f"Predict ARIMA ")
        return self.model_.forecast(steps=steps)


class RFModel:
    def __init__(self, random_state=42):
        self.reg = RandomForestRegressor(n_estimators=300, random_state=random_state)
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)
        self.reg.fit(X, y)
        log.info(f"Fit RF ")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        log.info(f"Predict RF ")
        return self.reg.predict(X)


class XGBModel:
    def __init__(self, random_state=42):
        self.reg = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror',
            random_state=random_state, n_jobs=2
        )
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)
        self.reg.fit(X, y)
        log.info(f"Fit XGB ")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        log.info(f"Predict XGB ")
        return self.reg.predict(X)


class LGBMModel:
    def __init__(self, random_state=42):
        self.reg = LGBMRegressor(
            random_state=random_state, 
            verbosity=-1,        
            num_leaves=15,       #  complejidad
            max_depth=3,         # árboles profundidad
            min_data_in_leaf=20  # mínimo por hoja
        )
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)
        self.reg.fit(X, y)
        log.info(f"Fit LGB ")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        log.info(f"Predict LGB ")
        return self.reg.predict(X)
