# -*- coding: utf-8 -*-
"""
Loading and cleaning routines for the RMPP (raw material price) and SMPP (supplier price)
time series used throughout the pipeline.

"""
from __future__ import annotations
import pandas as pd
from .logger import get_logger
log = get_logger(__name__)

from .utils import ensure_datetime, resample_monthly, apply_interpolations

def load_and_clean(raw_csv: str,
                   supp_csv: str,
                   date_col: str,
                   raw_col: str,
                   supp_col: str,
                   resample: str = 'M',
                   interpolations=('ffill', 'linear')) -> pd.DataFrame:   

    """
        Load, align, clean and interpolate the two core input time series:
        RMPP (raw material price series)
        SMPP (supplier price series)
        Steps:
            1. Load raw CSV files.
            2. Ensure date column is parsed correctly.
            3. Rename input columns to standard internal names ('RMPP', 'SMPP').
            4. Resample both series to monthly frequency (month-end).
            5. Outer-merge them to produce a joint calendar.
            6. Apply interpolation strategy (forward-fill + linear, by default).
            7. Enforce no trailing NaNs (useful for incomplete supplier data).
            8. Rename date column to 'Date' for consistency downstream.

        Returns
        -------
        DataFrame
            Cleaned and aligned dataset with columns:
                Date, RMPP, SMPP
    """
    # Load CSV inputs (using semicolon delimiter by project convention)
    raw = pd.read_csv(raw_csv, delimiter=';')
    sup = pd.read_csv(supp_csv, delimiter=';')

    if date_col in raw.columns:
        log.info(f"RAW (RMPP) original date range: {raw[date_col].min()} → {raw[date_col].max()}")
    else:
        log.warning(f"RAW dataset does not contain date column '{date_col}'. Columns present: {list(raw.columns)}")

    if date_col in sup.columns:
        log.info(f"SUPPLIER (SMPP) original date range: {sup[date_col].min()} → {sup[date_col].max()}")
    else:
        log.warning(f"SUPPLIER dataset does not contain date column: '{date_col}'. Columns present: {list(sup.columns)}")

    # Convert date column to datetime (supports day-first formats such as dd/mm/yyyy)
    raw = ensure_datetime(raw, date_col)
    sup = ensure_datetime(sup, date_col)

    # Keep only required columns and standardize names
    raw = raw[[date_col, raw_col]].rename(columns={raw_col: 'RMPP'})
    sup = sup[[date_col, supp_col]].rename(columns={supp_col: 'SMPP'})

    # Resample mensual to the end of the month ('ME')
    raw_m = resample_monthly(raw, date_col, how='mean')
    sup_m = resample_monthly(sup, date_col, how='mean')

    log.info(f"RAW (RMPP) after resampling: {raw_m[date_col].min()} → {raw_m[date_col].max()}")
    log.info(f"SUPPLIER (SMPP) after resampling: {sup_m[date_col].min()} → {sup_m[date_col].max()}")

    # Merge both time series into a shared monthly calendar
    df = pd.merge(raw_m, sup_m, on=date_col, how='outer').sort_values(date_col)
        
    # Apply interpolation strategy:
    # - forward-fill (LOCF)
    # - then linear interpolation for larger gaps

    df[['RMPP', 'SMPP']] = apply_interpolations(df[['RMPP', 'SMPP']], list(interpolations))

     # Guarantee no NaN remains at the tail (important for short SMPP series)
    df['RMPP'] = df['RMPP'].ffill()
    df['SMPP'] = df['SMPP'].ffill()

    # Standardize date column name for downstream processing
    df = df.rename(columns={date_col: 'Date'})

    log.info(f"Final cleaned data: {df['Date'].min().date()} → {df['Date'].max().date()}")
    log.debug(f"Last 5 rows:\n{df.tail(5)}")
    return df

