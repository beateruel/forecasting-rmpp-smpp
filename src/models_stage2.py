
# -*- coding: utf-8 -*-

"""
Stage 2 models for mapping upstream raw material prices (RMPP)
to supplier prices (SMPP).

This module defines two models:
    • LinearStage2 :  Linear Regression model.
    • XGBStage2    :  XGBoost regressor.

Both models learn a supervised mapping SMPP_t = f(features_t),
where the features typically consist of lagged RMPP values and,
optionally, Stage 1 RMPP forecasts.

The script exposes a simple interface:
    fit(X, y)    -> train the model
    predict(X)   -> generate SMPP predictions
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from .logger import get_logger
log = get_logger(__name__)


# -----------------------------------------------------------
# Linear Regression model for Stage 2
# -----------------------------------------------------------
class LinearStage2:
    """
        Linear regression model for mapping RMPP-based features to SMPP.

        Attributes
        ----------
        reg : LinearRegression
            Underlying linear regression estimator.

        feature_names_ : list[str]
            Names of the features used during training.
    """

    def __init__(self):
        self.reg = LinearRegression()
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)
        self.reg.fit(X, y)
        #log.info(f"Fitting Linear Stage2 ")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        #log.info(f"Predicting with Linear model in Stage2 ")
        return self.reg.predict(X)

# -----------------------------------------------------------
# XGBoost model for Stage 2
# -----------------------------------------------------------
class XGBStage2:

    """
        XGBoost regression model for non-linear SMPP forecasting
        using RMPP-based features.

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
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)
        self.reg.fit(X, y)
        #log.info(f"Fiting XGB Stage2 ")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        #log.info(f"Predicting with XGB model in Stage2 ")
        return self.reg.predict(X)
