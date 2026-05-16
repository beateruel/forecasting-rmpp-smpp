
# -*- coding: utf-8 -*-

"""
    Feature engineering for RMPP-based models.

    This module generates the exogenous features used in Stage 1 (ML models)
    and Stage 2. It creates lagged versions of the RMPP series and
    rolling-mean features based on user-defined window sizes.

    The function does not implement any learning logic; it only constructs
    the feature matrix used later in the pipeline.
"""

from __future__ import annotations
import pandas as pd
from typing import List
from .utils import make_lags, make_rollmean

from .logger import get_logger
log = get_logger(__name__)



def build_features(df: pd.DataFrame, lags: List[int], roll_windows: List[int]) -> pd.DataFrame:
    """
        Build feature matrix for RMPP by generating lag features and rolling means.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing at least the column 'RMPP'.
    
        lags : list of int
            List of lag lengths to generate lagged RMPP features (e.g. [1, 2, 3, 6]).
    
        roll_windows : list of int
            Window sizes for rolling mean features (e.g. [3, 6]).

        Returns
        -------
        pd.DataFrame
            Original dataframe joined with the generated lag and rolling-mean features.
    """

    out = df.copy()
    log.info("Generating RMPP-based features")

    # Create lagged versions of RMPP (e.g. RMPP_lag_1, RMPP_lag_2, ...)
    lag_df = make_lags(out['RMPP'], lags, prefix='RMPP')
    log.info(f"Lags: {lags}")

    # Create rolling-mean features (e.g. RMPP_rm_3, RMPP_rm_6, ...)
    roll_df = make_rollmean(out['RMPP'], roll_windows, prefix='RMPP')
    log.info(f"Rolling windows: {roll_windows}")

    # Join both sets of features to the original dataframe.
    # Note: For Stage 1 ML, only lags will be used to avoid leakage
    #       (handled inside the pipeline). Rolling features remain available
    #       for Stage 2 and analysis.

    out = out.join(lag_df.drop(columns=['RMPP'])).join(roll_df)
    return out
