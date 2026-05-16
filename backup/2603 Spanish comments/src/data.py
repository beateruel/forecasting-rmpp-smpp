# -*- coding: utf-8 -*-
"""Carga y limpieza de datos RMPP (materia prima) y SMPP (proveedor)."""

from __future__ import annotations
import pandas as pd

# ✅ Añadir el logger (esto faltaba y causaba el error)
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
    # Carga
    raw = pd.read_csv(raw_csv, delimiter=';')
    sup = pd.read_csv(supp_csv, delimiter=';')

    # Logs de rangos crudos (si la columna existe)
    if date_col in raw.columns:
        log.info(f"RAW (RMPP) rango original: {raw[date_col].min()} → {raw[date_col].max()}")
    else:
        log.warning(f"RAW no contiene la columna de fecha '{date_col}'. Columnas: {list(raw.columns)}")

    if date_col in sup.columns:
        log.info(f"SUPPLIER (SMPP) rango original: {sup[date_col].min()} → {sup[date_col].max()}")
    else:
        log.warning(f"SUPPLIER no contiene la columna de fecha '{date_col}'. Columnas: {list(sup.columns)}")

    # Normalización de fechas (con dayfirst=True para dd/mm/yyyy)
    raw = ensure_datetime(raw, date_col)
    sup = ensure_datetime(sup, date_col)

    # Selección y renombre de columnas
    raw = raw[[date_col, raw_col]].rename(columns={raw_col: 'RMPP'})
    sup = sup[[date_col, supp_col]].rename(columns={supp_col: 'SMPP'})

    # Resample mensual a fin de mes ('ME')
    raw_m = resample_monthly(raw, date_col, how='mean')
    sup_m = resample_monthly(sup, date_col, how='mean')

    log.info(f"RAW (RMPP) rango tras resample: {raw_m[date_col].min()} → {raw_m[date_col].max()}")
    log.info(f"SUPPLIER (SMPP) rango tras resample: {sup_m[date_col].min()} → {sup_m[date_col].max()}")

    # Merge outer y orden temporal
    df = pd.merge(raw_m, sup_m, on=date_col, how='outer').sort_values(date_col)

    # Interpolaciones (LOCF + lineal por defecto)
    df[['RMPP', 'SMPP']] = apply_interpolations(df[['RMPP', 'SMPP']], list(interpolations))

    # Asegurar cola sin NaN (opcional, útil para prolongar hasta fin de serie)
    df['RMPP'] = df['RMPP'].ffill()
    df['SMPP'] = df['SMPP'].ffill()

    # Renombrar la columna de fecha a 'Date' para consistencia aguas abajo
    df = df.rename(columns={date_col: 'Date'})

    log.info(f"Rango final después de limpieza: {df['Date'].min().date()} → {df['Date'].max().date()}")
    log.debug(f"Últimas 5 filas:\n{df.tail(5)}")
    return df

