
# -*- coding: utf-8 -*-
"""Evaluación, métricas, intervalos y DM test."""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .utils import mape, rmse, mae

from .logger import get_logger
log = get_logger(__name__)



def metrics_table(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'MAPE_%': mape(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
    }

def bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.20,
    n_boot: int = 1000,
    random_state: int = 42,
    recenter: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Intervalos de predicción empíricos por bootstrap de residuales.
    Devuelve (lower, upper) percentiles (alpha/2, 1-alpha/2).
    Si recenter=True, resta la media del residuo antes del remuestreo.
    """
    rng = np.random.default_rng(random_state)
    resid = y_true - y_pred
    if recenter:
        resid = resid - np.mean(resid)

    draws = rng.choice(resid, size=(n_boot, resid.shape[0]), replace=True)
    sims = y_pred + draws
    lower = np.percentile(sims, 100 * (alpha / 2), axis=0)
    upper = np.percentile(sims, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


def bootstrap_intervals_old(y_true: np.ndarray, y_pred: np.ndarray, alpha=0.2,
                        n_boot=200, random_state=42) -> Tuple[np.ndarray, np.ndarray]:
    """Intervalos de predicción empíricos por bootstrap de residuales.
    Devuelve (lower, upper) percentiles (alpha/2, 1-alpha/2).
    """
    rng = np.random.default_rng(random_state)
    resid = y_true - y_pred
    draws = rng.choice(resid, size=(n_boot, resid.shape[0]), replace=True)
    sims = y_pred + draws
    lower = np.percentile(sims, 100 * (alpha / 2), axis=0)
    upper = np.percentile(sims, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2) -> Dict[str, float]:
    """Diebold-Mariano para comparar e1 vs e2 (errores), aproximación robusta.
    Fallback a Normal(0,1) si SciPy no está instalado.
    """
    e1 = np.asarray(e1)
    e2 = np.asarray(e2)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    mean_d = d.mean()
    # Newey-West var con h-1 rezagos
    T = len(d)
    if T < 3:
        return {"DM": np.nan, "p_value": np.nan}
    gamma0 = np.var(d, ddof=1)
    cov = 0.0
    for lag in range(1, h):
        cov += 2 * (1 - lag / h) * np.cov(d[:-lag], d[lag:])[0, 1]
    S = gamma0 + cov
    if S <= 0:
        return {"DM": np.nan, "p_value": np.nan}
    dm_stat = mean_d / np.sqrt(S / T)

    # p-value
    try:
        from scipy.stats import t  # type: ignore
        pval = 2 * (1 - t.cdf(np.abs(dm_stat), df=T - 1))
    except Exception:
        # aproximación normal
        from math import erf, sqrt
        pval = 2 * (1 - 0.5 * (1 + erf(np.abs(dm_stat) / np.sqrt(2))))
    return {"DM": float(dm_stat), "p_value": float(pval)}
