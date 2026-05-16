
# -*- coding: utf-8 -*-
"""
    Evaluation utilities: metrics, empirical prediction intervals, 
    and Diebold-Mariano test.

    This module provides tools for:
        • Model evaluation using MAPE, RMSE and MAE.
        • Bootstrap-based empirical prediction intervals.
        • Diebold–Mariano accuracy comparison between forecasts.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .utils import mape, rmse, mae

from .logger import get_logger
log = get_logger(__name__)



# -------------------------------------------------------------
# BASIC METRICS TABLE
# -------------------------------------------------------------
# Computes a dictionary of standard evaluation metrics.
# Provides consistency across components that expect unified output.

def metrics_table(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'MAPE_%': mape(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
    }


# -------------------------------------------------------------
# BOOTSTRAP PREDICTION INTERVALS
# -------------------------------------------------------------
# Generates empirical prediction intervals via residual bootstrap.
# Supports residual recentering to remove mean bias before sampling.
# Returns lower/upper bounds using percentile-based intervals.
def bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.20,
    n_boot: int = 1000,
    random_state: int = 42,
    recenter: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
   
    rng = np.random.default_rng(random_state)
    resid = y_true - y_pred

    
    # Optional residual recentering for bias correction
    if recenter:
        resid = resid - np.mean(resid)

    
    # Bootstrap sampling of residuals
    draws = rng.choice(resid, size=(n_boot, resid.shape[0]), replace=True)
    sims = y_pred + draws

    
    # Percentile-based interval estimation
    lower = np.percentile(sims, 100 * (alpha / 2), axis=0)
    upper = np.percentile(sims, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


# -------------------------------------------------------------
# DIEBOLD–MARIANO TEST (FORECAST COMPARISON)
# -------------------------------------------------------------
# Compares prediction accuracy between two models using a robust
# implementation of the DM test with Newey–West variance correction.
# Falls back to Normal(0,1) p-values if SciPy is unavailable.
def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2) -> Dict[str, float]:
    
    """
            Diebold–Mariano accuracy test for forecasting errors.

            Parameters
            ----------
            e1, e2 : np.ndarray
                Error series from the competing forecasts.
            h : int
                Forecast horizon (affects NW variance estimator).
            power : int
                Power applied to errors (power=2 → squared errors).

            Returns
            -------
            dict with:
                'DM'      : DM statistic
                'p_value' : two-sided p-value
        """

    e1 = np.asarray(e1)
    e2 = np.asarray(e2)

    
    # Loss differential: |e1|^p - |e2|^p
    d = np.abs(e1) ** power - np.abs(e2) ** power
    mean_d = d.mean()
    T = len(d)
    if T < 3:
        
        # Not enough data to compute a reliable test
        return {"DM": np.nan, "p_value": np.nan}

    
    # Newey–West variance estimator
    gamma0 = np.var(d, ddof=1)
    cov = 0.0
    for lag in range(1, h):
        cov += 2 * (1 - lag / h) * np.cov(d[:-lag], d[lag:])[0, 1]
    S = gamma0 + cov
    if S <= 0:
        return {"DM": np.nan, "p_value": np.nan}
    dm_stat = mean_d / np.sqrt(S / T)


    # p-value computation (t distribution if available)
    try:
        from scipy.stats import t  # type: ignore
        pval = 2 * (1 - t.cdf(np.abs(dm_stat), df=T - 1))
    except Exception:
        # Normal approximation fallback
        from math import erf, sqrt
        pval = 2 * (1 - 0.5 * (1 + erf(np.abs(dm_stat) / np.sqrt(2))))
    return {"DM": float(dm_stat), "p_value": float(pval)}


# -------------------------------------------------------------
# MODIFIED DIEBOLD–MARIANO TEST (HARVEY–LEYBOURNE–NEWBOLD, 1997)
# -------------------------------------------------------------
# This small-sample correction should be preferred whenever the 
# out-of-sample evaluation window is short (e.g., Stage 2 where 
# only h = 1 forecasts are available). The HLN adjustment improves 
# the size and validity of the statistical test under small T.
def dm_test_modified(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2) -> Dict[str, float]:
    """
        Modified Diebold–Mariano test (Harvey, Leybourne & Newbold, 1997).
        This version corrects the small-sample bias of the standard DM test.
        Recommended when the number of out-of-sample forecast errors is small.

        Parameters
        ----------
        e1, e2 : np.ndarray
            Error series from the two competing forecasting methods.
        h : int
            Forecast horizon (affects the Newey–West variance correction).
        power : int
            Power applied to the loss function (power=2 → squared errors).

        Returns
        -------
        dict
            'DM'      : Modified DM statistic (HLN correction applied).
            'p_value' : Two‑sided p‑value for the accuracy comparison.
    """

    # Convert inputs to numpy arrays
    e1 = np.asarray(e1)
    e2 = np.asarray(e2)

    # Loss differential series: d_t = |e1|^power – |e2|^power
    d = np.abs(e1) ** power - np.abs(e2) ** power
    mean_d = d.mean()
    T = len(d)

    # Not enough observations to compute a meaningful test
    if T < 3:
        return {"DM": np.nan, "p_value": np.nan}

    # ---------------------------------------------------------
    # Newey–West long‑run variance estimator for the 
    # loss‑differential series. Required to handle temporal 
    # autocorrelation, especially when h > 1.
    # ---------------------------------------------------------
    gamma0 = np.var(d, ddof=1)
    cov = 0.0
    for lag in range(1, h):
        cov += 2 * (1 - lag / h) * np.cov(d[:-lag], d[lag:])[0, 1]

    S = gamma0 + cov
    if S <= 0:
        return {"DM": np.nan, "p_value": np.nan}

    # Standard (asymptotic) DM statistic
    dm_stat = mean_d / np.sqrt(S / T)

    # ---------------------------------------------------------
    # HLN small-sample correction factor:
    #   DM_mod = DM * sqrt( (T + 1 - 2h + h(h-1)/T) / T )
    # This reduces size distortions when T is small.
    # ---------------------------------------------------------
    h_term = (T + 1 - 2*h + (h * (h - 1) / T))
    hln_factor = np.sqrt(h_term / T)
    dm_mod = dm_stat * hln_factor

    # ---------------------------------------------------------
    # p‑value computation:
    # Prefer Student-t distribution, fall back to Normal if
    # SciPy is not available.
    # ---------------------------------------------------------
    try:
        from scipy.stats import t
        pval = 2 * (1 - t.cdf(np.abs(dm_mod), df=T - 1))
    except Exception:
        from math import erf, sqrt
        pval = 2 * (1 - 0.5 * (1 + erf(np.abs(dm_mod) / np.sqrt(2))))

    return {"DM": float(dm_mod), "p_value": float(pval)}
