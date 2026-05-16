
# -*- coding: utf-8 -*-
"""Utilidades comunes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List

from .logger import get_logger
log = get_logger(__name__)



def ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
    df = df.sort_values(date_col).reset_index(drop=True)

    return df


def resample_monthly(df: pd.DataFrame, date_col: str, how: str = 'mean') -> pd.DataFrame:
    df = df.set_index(date_col).sort_index()
    agg = {'mean': 'mean', 'last': 'last', 'first': 'first'}.get(how, 'mean')
    df = getattr(df.resample('ME'), agg)()
    return df.reset_index()


def apply_interpolations(df: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
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


def train_test_mask(dates: pd.Series, train_end: str) -> Tuple[np.ndarray, np.ndarray]:
    train_end = pd.to_datetime(train_end)
    train_mask = dates <= train_end
    test_mask = dates > train_end
    return train_mask.values, test_mask.values


def make_lags(series: pd.Series, lags: List[int], prefix: str) -> pd.DataFrame:
    df = pd.DataFrame({prefix: series})
    for L in lags:
        df[f"{prefix}_lag{L}"] = series.shift(L)
    return df


def make_rollmean(series: pd.Series, windows: List[int], prefix: str) -> pd.DataFrame:
    df = pd.DataFrame(index=series.index)
    for w in windows:
        df[f"{prefix}_roll{w}"] = series.rolling(w).mean()
    return df


def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def rmse(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))
