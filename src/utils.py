
# -*- coding: utf-8 -*-

"""
    Common utility functions used across the pipeline.

    This module provides small, reusable helpers for:
        • Date parsing and ordering.
        • Monthly resampling of time series.
        • Interpolation and gap filling.
        • Train/test splitting based on dates.
        • Feature construction (lags and rolling means).
        • Basic error metrics (MAPE, RMSE, MAE).    
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List

from .logger import get_logger
log = get_logger(__name__)



# -------------------------------------------------------------
# DATE HANDLING
# -------------------------------------------------------------
def ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:    
    """
        Convert a column to datetime format (day-first allowed), sort chronologically,
        and reset index. Ensures all downstream components work with a clean time index.
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
    df = df.sort_values(date_col).reset_index(drop=True)

    return df


# -------------------------------------------------------------
# MONTHLY RESAMPLING
# -------------------------------------------------------------

def resample_monthly(df: pd.DataFrame, date_col: str, how: str = 'mean') -> pd.DataFrame:
    
    """
        Resample a time series to month-end ('ME') frequency.

        Parameters
        ----------
        how : str
            Aggregation method per month. Supported: 'mean', 'first', 'last'.

        Returns
        -------
        DataFrame with monthly frequency.
    """
    df = df.set_index(date_col).sort_index()
    agg = {'mean': 'mean', 'last': 'last', 'first': 'first'}.get(how, 'mean')
    df = getattr(df.resample('ME'), agg)()
    return df.reset_index()


# -------------------------------------------------------------
# INTERPOLATION STRATEGY
# -------------------------------------------------------------

def apply_interpolations(df: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    
    """
    Apply a sequence of interpolation methods to fill missing values.
    Supported methods:
        'ffill'  -> forward-fill (LOCF)
        'bfill'  -> backward-fill
        'linear' -> linear interpolation

    Methods are applied in the order they appear in 'methods'.
    """
    x = df.copy()
    for m in methods:
        if m.lower() in ('ffill', 'pad'):
            x = x.ffill()
        elif m.lower() in ('bfill', 'backfill'):
            x = x.bfill()
        elif m.lower() == 'linear':
            x = x.interpolate(method='linear')
        else:
            x = x.interpolate(method='linear')
    return x


# -------------------------------------------------------------
# TRAIN/TEST SPLIT BASED ON DATE
# -------------------------------------------------------------

def train_test_mask(dates: pd.Series, train_end: str) -> Tuple[np.ndarray, np.ndarray]:
    """
        Generate boolean masks for train/test split based on a cutoff date.
    """

    train_end = pd.to_datetime(train_end)
    train_mask = dates <= train_end
    test_mask = dates > train_end
    return train_mask.values, test_mask.values


# -------------------------------------------------------------
# FEATURE CONSTRUCTION
# ------------------------------------------------------------
def make_lags(series: pd.Series, lags: List[int], prefix: str) -> pd.DataFrame:
    """
        Construct lagged features for a time series.

        Example:
            lag 1 -> series.shift(1)
            lag 3 -> series.shift(3)
    """

    df = pd.DataFrame({prefix: series})
    for L in lags:
        df[f"{prefix}_lag{L}"] = series.shift(L)
    return df


def make_rollmean(series: pd.Series, windows: List[int], prefix: str) -> pd.DataFrame:
    
    """
     Construct rolling-mean features for the given window sizes.
    """

    df = pd.DataFrame(index=series.index)
    for w in windows:
        df[f"{prefix}_roll{w}"] = series.rolling(w).mean()
    return df

# -------------------------------------------------------------
# ERROR METRICS
# -------------------------------------------------------------
def mape(y_true, y_pred) -> float:    
    """
        Mean Absolute Percentage Error (MAPE) with safe denominator
        (replaces values close to zero with 1e-8 to avoid division issues).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))
