
# -*- coding: utf-8 -*-
"""
    Direct forecasting models for predicting SMPP using only RMPP as exogenous regressors.
    This module includes:
        • SARIMAXExog:  SARIMAX with external regressors (lagged RMPP).
        • XGBExog:      XGBoost regressor using RMPP lags as features.

    These models correspond to the “direct” approach in the pipeline.
    they bypass Stage 1 and map historical RMPP → SMPP directly.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from .logger import get_logger
log = get_logger(__name__)

class SARIMAXExog:
    """
        Wrapper around Statsmodels' SARIMAX for SMPP forecasting with RMPP lags
        as exogenous inputs.
        Attributes
        ----------
        order : tuple(int, int, int)
            ARIMA (p, d, q) order.
        seasonal_order : tuple(int, int, int, int)
            Seasonal component (P, D, Q, s); set to zeros by default.
        model_ : fitted SARIMAX model instance
    """
    def __init__(self, order=(0, 1, 0), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order
    """
       Fit the SARIMAX model to the SMPP series using RMPP lags as exogenous variables.
        Parameters
        ----------
        y : pd.Series
            Target SMPP series.
        X : pd.DataFrame
            Exogenous regressors (lagged RMPP features).

        Returns
        -------
        self
    """

    def fit(self, y: pd.Series, X: pd.DataFrame):
        self.model_ = SARIMAX(
            y, exog=X, order=self.order, seasonal_order=self.seasonal_order,trend="t",
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(
                disp=False,
                method="powell",
                maxiter=2000
            )
        return self

    def predict(self, steps: int, X_future: pd.DataFrame) -> np.ndarray:
        return self.model_.forecast(steps=steps, exog=X_future)

class XGBExog:   
    """
            Wrapper around XGBoost for direct SMPP forecasting using lagged RMPP
            as exogenous regressors.

            This model represents the “direct” approach, where SMPP is predicted
            from historical RMPP features without an intermediate Stage 1 forecast.

            Attributes
            ----------
            reg : XGBRegressor
                Underlying XGBoost regression model.

            feature_names_ : list[str]
                Names of the exogenous features used during training.
        """


    def __init__(self, random_state=42):
        self.reg = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror',
            random_state=random_state, n_jobs=2
        )
        self.feature_names_: list[str] = []
    
        """
            Fit the XGBoost model to the SMPP series using lagged RMPP as
            exogenous predictors.

            Parameters
            ----------
            X : pd.DataFrame
                Exogenous regressors (lagged RMPP, rolling features, etc.).

            y : pd.Series
                Target SMPP series.

            Returns
            -------
            self
        """

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)
        self.reg.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.reg.predict(X)
