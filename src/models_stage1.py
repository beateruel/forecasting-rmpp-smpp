
# -*- coding: utf-8 -*-

"""
    Stage 1 forecasting models for predicting RMPP.

    This module implements several alternative models used in Stage 1 of the
    pipeline, where the goal is to forecast upstream raw material prices (RMPP).

    Included models:
        • SESModel:   Simple Exponential Smoothing.
        • ARIMAModel: Classical ARIMA(p, d, q).
        • RFModel:    Random Forest regressor using lagged features.
        • XGBModel:   XGBoost regressor using lagged features.
        • LGBMModel:  LightGBM regressor using lagged features.

    All models are trained on lagged RMPP values (and optionally rolling features)
    and produce multi-step forecasts using a recursive approach.
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


# -------------------------------------------------------------------
# Simple Exponential Smoothing (statistical baseline)
# -------------------------------------------------------------------

class SESModel:    
    """
        Wrapper for Simple Exponential Smoothing to forecast RMPP.

        Attributes
        ----------
        model_ : fitted SES model instance
    """

    def fit(self, y: pd.Series):
        #log.info(f"Fitting SES model ")
        self.model_ = SimpleExpSmoothing(y, initialization_method='estimated').fit()
        return self

    def predict(self, steps: int) -> np.ndarray:
        #log.info(f"Predicting with SES model ")
        return self.model_.forecast(steps)

# -------------------------------------------------------------------
# ARIMA(p, d, q) model (classical time-series model)
# -------------------------------------------------------------------
class ARIMAModel:
    """
        Wrapper for a simple ARIMA model to forecast RMPP.

        Attributes
        ----------
        order : tuple(int, int, int)
        ARIMA order (p, d, q).

        model_ : fitted ARIMA model instance
    """
    def __init__(self, order=(1, 1, 1)):
        self.order = order

    def fit(self, y: pd.Series):
         #log.info(f"Fitting ARIMA MODEL")
        self.model_ = ARIMA(y, order=self.order).fit()
        return self

    def predict(self, steps: int) -> np.ndarray:
         #log.info(f"Predicting with ARIMA model ")
        return self.model_.forecast(steps=steps)


class RFModel:
    """
        Random Forest model for RMPP forecasting using lagged features.

        Attributes
        ----------
        reg : RandomForestRegressor
            Underlying sklearn regressor.

        feature_names_ : list[str]
            Names of features used during training.
    """
    
    def __init__(self, random_state=42):
        self.reg = RandomForestRegressor(n_estimators=300, random_state=random_state)
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)
        self.reg.fit(X, y)
         #log.info(f"Fiting RF ")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
         #log.info(f"Predicting with RF model ")
        return self.reg.predict(X)


# -------------------------------------------------------------------
# XGBoost Regressor (non-linear ML model)
# -------------------------------------------------------------------

class XGBModel:

    """
        XGBoost model for RMPP forecasting using lagged predictors.

        Attributes
        ----------
        reg : XGBRegressor
            Underlying gradient boosting regressor.

        feature_names_ : list[str]
            Names of features used during training.
    """

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
         #log.info(f"Fiting XGB ")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
         #log.info(f"Predicting with XGB model ")
        return self.reg.predict(X)


# -------------------------------------------------------------------
# LightGBM Regressor (fast gradient boosting model)
# -------------------------------------------------------------------

class LGBMModel:
    def __init__(self, random_state=42):
        self.reg = LGBMRegressor(
            random_state=random_state, 
            verbosity=-1,        
            num_leaves=15,       
            max_depth=3,         
            min_data_in_leaf=20  
        )
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)
        self.reg.fit(X, y)
         #log.info(f"Fiting LGB ")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
         #log.info(f"Predicting with LGB model ")
        return self.reg.predict(X)
