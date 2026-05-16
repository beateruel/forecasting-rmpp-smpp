
# -*- coding: utf-8 -*-
"""Ingeniería de variables: lags y medias móviles."""
from __future__ import annotations
import pandas as pd
from typing import List
from .utils import make_lags, make_rollmean

from .logger import get_logger
log = get_logger(__name__)



def build_features(df: pd.DataFrame, lags: List[int], roll_windows: List[int]) -> pd.DataFrame:
    out = df.copy()
    log.info("Generando features de RMPP")

    lag_df = make_lags(out['RMPP'], lags, prefix='RMPP')
    log.info(f"Lags: {lags}")
    roll_df = make_rollmean(out['RMPP'], roll_windows, prefix='RMPP')
    # Mantener tanto lags como rolling para stage2 y análisis; 
    # Para Stage1-ML evitaremos leakage usando sólo lags (ver pipeline.py)
    log.info(f"Rolling windows: {roll_windows}")
    out = out.join(lag_df.drop(columns=['RMPP'])).join(roll_df)
    return out
