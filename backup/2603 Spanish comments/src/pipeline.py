
# -*- coding: utf-8 -*-
"""Orquestador del pipeline (P1–P8) con rolling-origin evaluation."""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path

from .data import load_and_clean
from .features import build_features
from .models_stage1 import SESModel, ARIMAModel, RFModel, XGBModel, LGBMModel
from .models_stage2 import LinearStage2, XGBStage2
from .direct_models import SARIMAXExog, XGBExog
from .utils import train_test_mask, make_lags
from .evaluation import metrics_table, bootstrap_intervals, dm_test

from .logger import get_logger
log = get_logger(__name__)



def _recursive_forecast_ml(series: pd.Series, model, lags: List[int], horizon: int) -> float:
    """Pronóstico h-steps-ahead usando modelo ML con lags, actualizando con predicciones."""
    s = series.copy()
    for _ in range(horizon):
        feats = {f"RMPP_lag{L}": s.iloc[-L] for L in lags}
        Xh = pd.DataFrame([feats])
        yhat = float(model.predict(Xh)[0])
        s = pd.concat([s, pd.Series([yhat])], ignore_index=True)
    return s.iloc[-1]


def rolling_origin_stage1(df: pd.DataFrame, lags: List[int], roll_w: List[int],
                          horizons: List[int], train_end: str,
                          models: List[str]) -> Dict[str, Dict[int, pd.DataFrame]]:
    """Devuelve para cada modelo y horizonte un DataFrame con columnas [Date, y_true, y_pred]."""
    
    log.info(f"Rango final después de limpieza: {df['Date'].min().date()} → {df['Date'].max().date()}")
    log.info(f"Últimos 5 (RMPP,SMPP):\n{df[['Date','RMPP','SMPP']].tail(5)}")

    results: Dict[str, Dict[int, pd.DataFrame]] = {}
    # construir features para ML
    feat_df = build_features(df, lags=lags, roll_windows=roll_w)
    train_mask, test_mask = train_test_mask(df['Date'], train_end)
    dates = df['Date']

    for mname in models:
        model_results: Dict[int, List[Dict]] = {h: [] for h in horizons}
        for t_idx in np.where(test_mask)[0]:
            # ventana de entrenamiento: hasta t_idx-1
            train_idx = np.arange(0, t_idx)
            y_train = df.loc[train_idx, 'RMPP']

            if mname == 'SES':
                ses = SESModel().fit(y_train)                
                for h in horizons:
                        if t_idx + h - 1 >= len(df):
                            continue

                        log.info(f"Predict SES | horizon={h}")
                        forecast = ses.predict(h)
                        yhat = forecast.iloc[-1]   # ✅ FIX CLAVE
                        y_true = df.loc[t_idx + h - 1, 'RMPP']

                        model_results[h].append({
                            "Date": dates.iloc[t_idx + h - 1],
                            "y_true": y_true,
                            "y_pred": yhat
                        })


            elif mname == 'ARIMA':                
                log.info("Fit modelo ARIMA")
                ar = ARIMAModel(order=(1,1,1)).fit(y_train)

                for h in horizons:
                    if t_idx + h - 1 >= len(df):
                        continue

                    log.info(f"Predict ARIMA | horizon={h}")
                    forecast = ar.predict(h)
                    yhat = forecast.iloc[-1]  
                    y_true = df.loc[t_idx + h - 1, 'RMPP']

                    model_results[h].append({
                        "Date": dates.iloc[t_idx + h - 1],
                        "y_true": y_true,
                        "y_pred": yhat
                    })


            else:
                # Modelos ML: entrenar SOLO con lags para que el forecast recursivo sea consistente
                X_all = feat_df[[c for c in feat_df.columns if c.startswith('RMPP_lag')]]
                y_all = df['RMPP']
                X_tr = X_all.loc[train_idx].dropna()
                y_tr = y_all.loc[X_tr.index]
                if len(X_tr) < 24:
                    continue
                if mname == 'RF':
                    mdl = RFModel()
                elif mname == 'XGB':
                    mdl = XGBModel()
                elif mname == 'LGBM':
                    mdl = LGBMModel()
                else:
                    raise ValueError('Modelo Stage1 no soportado')
                mdl.fit(X_tr, y_tr)
                for h in horizons:
                    if t_idx + h - 1 >= len(df):
                        continue
                    # forecast recursivo con lags
                    yhat = _recursive_forecast_ml(df.loc[:t_idx - 1, 'RMPP'], mdl, lags, h)
                    y_true = df.loc[t_idx + h - 1, 'RMPP']
                    model_results[h].append({
                        "Date": dates.iloc[t_idx + h - 1], "y_true": y_true, "y_pred": yhat
                    })

        results[mname] = {h: pd.DataFrame(v) for h, v in model_results.items() if len(v) > 0}
    return results

def rolling_origin_stage2(
    df: pd.DataFrame,
    stage1_preds: Dict[str, Dict[int, pd.DataFrame]],
    lags: List[int],
    horizons: List[int],
    min_train: int = 6,
    models: List[str] = ("Linear", "XGB")
) -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    Rolling-origin Stage 2 (RMPP → SMPP) SIN split fijo.
    Para cada pipeline y horizonte:
      - entrena hasta t-1
      - predice en t
    """

    results: Dict[str, Dict[int, pd.DataFrame]] = {}

    for s1_name, hdict in stage1_preds.items():
        for s2_name in models:

            pipe_id = f"{s1_name}->{s2_name}"
            results[pipe_id] = {}

            for h, s1_df in hdict.items():

                # Normalizar fechas de Stage 1 a fin de mes
                tmp = s1_df.copy()
                tmp["Date"] = tmp["Date"].dt.to_period("M").dt.to_timestamp("M")
                tmp = tmp.rename(columns={"y_pred": "forecast_RMPP"})

                # Merge con SMPP y RMPP historizados (con LOCF previo)
                tmp = (
                    tmp.merge(
                        df[["Date", "RMPP", "SMPP"]],
                        on="Date",
                        how="left"
                    )
                    .sort_values("Date")
                    .reset_index(drop=True)
                )

                # Construir lags de RMPP
                for L in lags:
                    tmp[f"RMPP_lag{L}"] = tmp["RMPP"].shift(L)

                tmp = tmp.dropna().reset_index(drop=True)

                if len(tmp) < min_train + 1:
                    continue

                rows = []

                feature_cols = [f"RMPP_lag{L}" for L in lags] + ["forecast_RMPP"]

                for idx in range(min_train, len(tmp)):

                    X_tr = tmp.loc[:idx - 1, feature_cols]
                    y_tr = tmp.loc[:idx - 1, "SMPP"]

                    X_te = tmp.loc[[idx], feature_cols]
                    y_te = tmp.loc[idx, "SMPP"]

                    if s2_name == "Linear":
                        mdl = LinearStage2()
                    elif s2_name == "XGB":
                        mdl = XGBStage2()
                    else:
                        raise ValueError("Modelo Stage 2 no soportado")

                    mdl.fit(X_tr, y_tr)
                    y_hat = mdl.predict(X_te)[0]

                    rows.append({
                        "Date": tmp.loc[idx, "Date"],
                        "y_true": y_te,
                        "y_pred": y_hat
                    })

                if rows:
                    results[pipe_id][h] = pd.DataFrame(rows)

    return results


def rolling_origin_stage2_old(df: pd.DataFrame, stage1_preds: Dict[str, Dict[int, pd.DataFrame]],
                          lags: List[int], horizons: List[int], train_end: str,
                          models: List[str]) -> Dict[str, Dict[int, pd.DataFrame]]:
    """Para cada pipeline (modelo Stage1 + modelo Stage2) y horizonte, retorna DataFrames con SMPP y pred.
    Features Stage2: forecast_RMPP_h, RMPP_lag{1..k}
    """
    results: Dict[str, Dict[int, pd.DataFrame]] = {}

    for s1_name, hdict in stage1_preds.items():
        for s2_name in models:
            pipe_id = f"{s1_name}->{s2_name}"
            results[pipe_id] = {}
            for h, s1_df in hdict.items():
               
                # s1_df viene de Stage 1: normalizamos a fin de mes para que case con df_stage2
                s1_df = s1_df.copy()
                s1_df['Date'] = s1_df['Date'].dt.to_period('M').dt.to_timestamp('M')

                # Construcción del dataset Stage 2 (usaremos 'forecast_RMPP' como nombre claro)
                tmp = pd.DataFrame({
                    'Date': s1_df['Date'],
                    'forecast_RMPP': s1_df['y_pred'].values
                })                
                
                # Merge con df_stage2 (que ya trae RMPP y SMPP con LOCF)
                tmp = (
                    tmp.merge(df[['Date', 'RMPP', 'SMPP']], on='Date', how='left')
                       .dropna()
                       .reset_index(drop=True)
                )

                               
                log.warning(
                    f"{pipe_id} | h={h} | filas tras merge: {len(tmp)}"
                )
                log.info(f"[Stage2:{s1_name}->{s2_name}|h={h}] fechas antes de dropna: {tmp['Date'].min().date()} → {tmp['Date'].max().date()} (n={len(tmp)})")
                tmp = tmp.sort_values('Date').reset_index(drop=True)
                tmp = tmp.dropna().reset_index(drop=True)
                log.info(f"[Stage2:{s1_name}->{s2_name}|h={h}] fechas después de dropna: {tmp['Date'].min().date()} → {tmp['Date'].max().date()} (n={len(tmp)})")
                for L in lags:
                    tmp[f'RMPP_lag{L}'] = tmp['RMPP'].shift(L)
                tmp = tmp.dropna().reset_index(drop=True)

                # Split temporal usando train_end
                train_mask = tmp['Date'] <= pd.to_datetime(train_end)
                X = tmp[[f'RMPP_lag{L}' for L in lags]].copy()
                X[f'forecast_RMPP_h{h}'] = tmp['forecast_RMPP']
                y = tmp['SMPP']
                X_tr, y_tr = X[train_mask], y[train_mask]
                X_te, y_te = X[~train_mask], y[~train_mask]
                if len(X_tr) < 12 or len(X_te) == 0:
                    continue
                if s2_name == 'Linear':
                    mdl = LinearStage2()
                elif s2_name == 'XGB':
                    mdl = XGBStage2()
                else:
                    raise ValueError('Modelo Stage2 no soportado')
                mdl.fit(X_tr, y_tr)
                yhat = mdl.predict(X_te)
                results[pipe_id][h] = pd.DataFrame({
                    'Date': tmp.loc[~train_mask, 'Date'],
                    'y_true': y_te.values,
                    'y_pred': yhat
                })
    
    log.info(
        "Stage2 completado | filas por pipeline: "
        + str({k: sum(len(vv) for vv in v.values()) for k, v in results.items()})
    )

    return results


def rolling_origin_direct(df: pd.DataFrame, lags: List[int], horizons: List[int],
                          train_end: str, models: List[str]) -> Dict[str, Dict[int, pd.DataFrame]]:
    results: Dict[str, Dict[int, pd.DataFrame]] = {}

    for mname in models:
        results[mname] = {}
        # construir lags de RMPP como exógenas
        X_exog_full = pd.concat([make_lags(df['RMPP'], lags, 'RMPP').drop(columns=['RMPP'])], axis=1)
        data = pd.concat([df[['Date', 'SMPP']], X_exog_full], axis=1)
        log.info(f"[Direct:{mname}] fechas antes dropna: {data['Date'].min().date()} → {data['Date'].max().date()} (n={len(data)})")
        data = data.dropna().reset_index(drop=True)
        log.info(f"[Direct:{mname}] fechas después dropna: {data['Date'].min().date()} → {data['Date'].max().date()} (n={len(data)})")
        train_mask = data['Date'] <= pd.to_datetime(train_end)
        for h in horizons:
            rows = []
            # rolling-origin en el conjunto de test
            test_idx = data.index[~train_mask]
            for idx in test_idx:
                # entrenar con datos hasta idx-1
                tr_idx = data.index[(data.index < idx)]
                if len(tr_idx) < 12:
                    continue
                y_tr = data.loc[tr_idx, 'SMPP']
                X_tr = data.loc[tr_idx, data.columns.str.startswith('RMPP_lag')]
                # construir X_future para horizonte h (usar la fila target en idx+h-1)
                target_row = idx + h - 1
                if target_row >= len(data):
                    continue
                X_future = data.loc[[target_row], X_tr.columns]
                if mname == 'SARIMAX': 
                    log.info(f"Fit SARIMAX direct | idx={idx} | horizon={h}")
                    mdl = SARIMAXExog(order=(1,1,1), seasonal_order=(0,0,0,0)).fit(y_tr, X_tr)
                    forecast = mdl.predict(steps=1, X_future=X_future)
                    yhat = forecast.iloc[0]
                elif mname == 'XGB_EXOG':
                    mdl = XGBExog()
                    mdl.fit(X_tr, y_tr)
                    yhat = mdl.predict(X_future)[0]
                else:
                    raise ValueError('Modelo directo no soportado')
                y_true = data.loc[target_row, 'SMPP']
                rows.append({'Date': data.loc[target_row, 'Date'], 'y_true': y_true, 'y_pred': yhat})
            if rows:
                results[mname][h] = pd.DataFrame(rows)
    return results


def _df_safe(rows, cols_order=None):
    df = pd.DataFrame(rows or [])
    if cols_order:
        # añade columnas que falten y reordena
        for c in cols_order:
            if c not in df.columns:
                df[c] = pd.Series(dtype='float64' if c not in ('Pipeline ID','Tipo','Notas','Stage1 Model','Stage2 Model') else 'object')
        df = df[cols_order]
    return df

def run_pipeline(config_path: str = 'config.json') -> Dict:
    
    log.info("=== INICIO PIPELINE ===")
    log.info("Cargando configuración")

    with open(config_path, 'r') as f:
        cfg = json.load(f)

    df = load_and_clean(
        cfg['input']['raw_prices_csv'],
        cfg['input']['supplier_prices_csv'],
        cfg['input']['date_col'],
        cfg['input']['raw_col'],
        cfg['input']['supplier_col'],
        resample=cfg['processing']['frequency'],
        interpolations=cfg['processing']['interpolation']
    )

    
    log.info(f"Horizontes: {cfg['horizons']}")
    log.info(f"Stage1 models: {cfg['stage1_models']}")
    log.info(f"Stage2 models: {cfg['stage2_models']}")

    # Renombrar columna de fecha a 'Date' para consistencia
    if 'Date' not in df.columns:
        df = df.rename(columns={cfg['input']['date_col']: 'Date'})
    
    log.info("Ejecutando Stage 1 (RMPP)")    

    # Stage 1
    s1_results = rolling_origin_stage1(
        df=df.copy(),
        lags=cfg['features']['lags'],
        roll_w=cfg['features']['rolling_means'],
        horizons=cfg['horizons'],
        train_end=cfg['processing']['train_end'],
        models=cfg['stage1_models']
    )

    log.info(f"Modelos: {list(s1_results.keys())}")

    log.info("Ejecutando Stage 2 (SMPP)")
    # Stage 2 (two-stage pipelines)
    
    # 1) Aseguramos que el calendario es mensual (MS o fin de mes) y continuo
    df_stage2 = df.copy()
    df_stage2['Date'] = df_stage2['Date'].dt.to_period('M').dt.to_timestamp('M')  # fin de mes consistente
    
    # 2) Resample mensual sobre Date como índice, y forward-fill para SMPP y RMPP
    df_stage2 = (
        df_stage2
          .set_index('Date')
          .asfreq('ME')                     # mensual continuo
          .ffill()                         # LOCF explícito
          .reset_index()
    )

    log.info(f"Stage2 | df_stage2 tras LOCF: {df_stage2.shape[0]} filas")
    # s2_results = rolling_origin_stage2(
    #     df=df_stage2.copy(),
    #     stage1_preds=s1_results,
    #     lags=cfg['features']['lags'],
    #     horizons=cfg['horizons'],
    #     train_end=cfg['processing']['train_end'],
    #     models=cfg['stage2_models']
    # )
    
    s2_results = rolling_origin_stage2(
        df=df_stage2.copy(),
        stage1_preds=s1_results,
        lags=cfg['features']['lags'],
        horizons=cfg['horizons'],
        models=cfg['stage2_models']
    )

    log.info(f"Pipelines generados: {list(s2_results.keys())}")


   
    #Direct baselines
    log.info("Ejecutando modelos directos")
     # Usar h=1 por coherencia con Stage 2 (limitación estructural); configurable en config.json
    direct_horizons = cfg.get('direct_horizons', [1])  # por defecto solo h=1
    direct_results = rolling_origin_direct(
        df=df.copy(),
        lags=cfg['features']['lags'],
        horizons=direct_horizons,
        train_end=cfg['processing']['train_end'],
        models=cfg['direct_models']
    )
    log.info(f"Direct models: {list(direct_results.keys())}")

   

    # Construir tablas de resultados
    summary_rows, stage1_rows, stage2_rows, dm_rows = [], [], [], []

    # Stage1 metrics
    for mname, hdict in s1_results.items():
        for h, dd in hdict.items():
            if len(dd) == 0:
                continue
            mt = metrics_table(dd['y_true'], dd['y_pred'])
            stage1_rows.append({
                'Modelo Stage1': mname,
                'Horizonte': h,
                'RMPP MAPE': mt['MAPE_%'],
                'RMPP RMSE': mt['RMSE'],
                'RMPP MAE': mt['MAE'],
                'R²': np.nan,
                'Notas': ''
            })

    # Stage2 metrics and Summary
    
    map_name_to_pid = {
        'SES->Linear': 'P1',
        'ARIMA->Linear': 'P2',
        'RF->Linear': 'P3',
        'XGB->Linear': 'P4',
        'LGBM->Linear': 'P5',
        'XGB->XGB': 'P6',
    }


    for pipe, hdict in s2_results.items():        
        pipe_norm = pipe.replace('-&gt;', '->')  # por si entra escapado
        pid = map_name_to_pid.get(pipe_norm, pipe_norm)
        for h, dd in hdict.items():
            mt = metrics_table(dd['y_true'], dd['y_pred'])
            summary_rows.append({
                'Pipeline ID': pid,
                'Tipo': 'Two-stage',
                'Horizonte': h,
                'SMPP MAPE (%)': mt['MAPE_%'],
                'SMPP RMSE': mt['RMSE'],
                'SMPP MAE': mt['MAE'],
                'RMPP MAPE (%)': np.nan,
                'PI Coverage 80%': np.nan,
                'PI Coverage 95%': np.nan,
                'DM p-value vs baseline': np.nan,
                'Notas': ''
            })
            stage2_rows.append({
                'Pipeline ID': pid,
                'Stage1 Model': pipe.split('->')[0],
                'Stage2 Model': pipe.split('->')[1],
                'Horizonte': h,
                'SMPP MAPE': mt['MAPE_%'],
                'SMPP RMSE': mt['RMSE'],
                'SMPP MAE': mt['MAE'],
                'Notas': ''
            })

    # Direct baselines P7 y P8
    base_map = {'SARIMAX': 'P7', 'XGB_EXOG': 'P8'}
    for mname, hdict in direct_results.items():
        pid = base_map.get(mname, mname)
        for h, dd in hdict.items():
            mt = metrics_table(dd['y_true'], dd['y_pred'])
            summary_rows.append({
                'Pipeline ID': pid,
                'Tipo': 'Direct',
                'Horizonte': h,
                'SMPP MAPE (%)': mt['MAPE_%'],
                'SMPP RMSE': mt['RMSE'],
                'SMPP MAE': mt['MAE'],
                'RMPP MAPE (%)': np.nan,
                'PI Coverage 80%': np.nan,
                'PI Coverage 95%': np.nan,
                'DM p-value vs baseline': 'baseline',
                'Notas': ''
            })

    # Ejemplo de DM test: P4 vs P7 si existen
    for h in cfg['horizons']:
        if 'XGB->Linear' in s2_results and h in s2_results['XGB->Linear'] and 'SARIMAX' in direct_results and h in direct_results['SARIMAX']:
            a = s2_results['XGB->Linear'][h]
            b = direct_results['SARIMAX'][h]
            # Alinear por fecha
            cmp = pd.merge(a, b, on='Date', suffixes=('_A', '_B'))
            eA = cmp['y_true_A'] - cmp['y_pred_A']
            eB = cmp['y_true_B'] - cmp['y_pred_B']
            res = dm_test(eA.values, eB.values, h=1, power=1)
            dm_rows.append({
                'Comparación': 'P4 vs P7',
                'Horizonte': h,
                'Modelo A': 'XGB→LR',
                'Modelo B': 'SARIMAX-exog',
                'Loss Function': 'MAE',
                'DM Statistic': res['DM'],
                'p-value': res['p_value'],
                'Significativo?': 'Sí' if (isinstance(res['p_value'], float) and res['p_value'] < 0.05) else 'No'
            })

    # Guardar predicciones por pipeline a CSV
    os.makedirs('output', exist_ok=True)
    # Stage1
    for mname, hdict in s1_results.items():
        for h, dd in hdict.items():
            dd.to_csv(os.path.join('output', f'stage1_{mname}_h{h}.csv'), index=False)
    # Stage2
    for pipe, hdict in s2_results.items():
        for h, dd in hdict.items():
            dd.to_csv(os.path.join('output', f'stage2_{pipe.replace("->", "_")}_h{h}.csv'), index=False)
    # Direct
    for mname, hdict in direct_results.items():
        for h, dd in hdict.items():
            dd.to_csv(os.path.join('output', f'direct_{mname}_h{h}.csv'), index=False)
    
    log.info("Guardando resultados en output/")
    # Excel con hojas
    
    with pd.ExcelWriter('output/results.xlsx', engine='openpyxl') as writer:
        # Summary_Table
        df_summary = _df_safe(
            summary_rows,
            cols_order=['Pipeline ID','Tipo','Horizonte','SMPP MAPE (%)','SMPP RMSE','SMPP MAE','RMPP MAPE (%)','PI Coverage 80%','PI Coverage 95%','DM p-value vs baseline','Notas']
        )
        if set(['Pipeline ID','Horizonte']).issubset(df_summary.columns):
            df_summary = df_summary.sort_values(['Pipeline ID','Horizonte'])
        df_summary.to_excel(writer, sheet_name='Summary_Table', index=False)

        # Stage1_Results
        df_s1 = _df_safe(
            stage1_rows,
            cols_order=['Modelo Stage1','Horizonte','RMPP MAPE','RMPP RMSE','RMPP MAE','R²','Notas']
        )
        if set(['Modelo Stage1','Horizonte']).issubset(df_s1.columns):
            df_s1 = df_s1.sort_values(['Modelo Stage1','Horizonte'])
        df_s1.to_excel(writer, sheet_name='Stage1_Results', index=False)

        # Stage2_Results
        df_s2 = _df_safe(
            stage2_rows,
            cols_order=['Pipeline ID','Stage1 Model','Stage2 Model','Horizonte','SMPP MAPE','SMPP RMSE','SMPP MAE','Notas']
        )
        if set(['Pipeline ID','Horizonte']).issubset(df_s2.columns):
            df_s2 = df_s2.sort_values(['Pipeline ID','Horizonte'])
        df_s2.to_excel(writer, sheet_name='Stage2_Results', index=False)

        # DM_tests
        df_dm = _df_safe(
            dm_rows,
            cols_order=['Comparación','Horizonte','Modelo A','Modelo B','Loss Function','DM Statistic','p-value','Significativo?']
        )
        df_dm.to_excel(writer, sheet_name='DM_tests', index=False)

        # PI_Coverage (placeholder) y Notes_Log
        pd.DataFrame(columns=['Pipeline ID','Horizonte','Residual Method','PI 80% Coverage (%)','PI 95% Coverage (%)','Notas']).to_excel(writer, sheet_name='PI_Coverage', index=False)
        pd.DataFrame(columns=['Fecha','Tarea realizada','Dataset','Modelos tocados','Problemas','Solución']).to_excel(writer, sheet_name='Notes_Log', index=False)

        
    # === Uncertainty: Prediction Intervals, Coverage & Sharpness ===
    log.info("Calculando Prediction Intervals (80%/95%), cobertura y sharpness...")
    from .uncertainty import discover_prediction_files, aggregate_coverage_across_files

    # Cargar configuración de incertidumbre (o defaults)
    cfg_unc = cfg.get("uncertainty", {
        "n_boot": 1000,
        "alpha_levels": [0.20, 0.05],
        "recenter": True,
        "random_state": 42,
        "make_plots": True,
    })

    base_output = Path("output")
    files = discover_prediction_files(base_output=base_output)
    cov_df, width_df = aggregate_coverage_across_files(
        files=files, cfg_unc=cfg_unc, base_output=base_output
    )

    # Escribir PI_Coverage y PI_Stats en el Excel existente reemplazando sheets
    from pandas import ExcelWriter
    with ExcelWriter(base_output / "results.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        if not cov_df.empty:
            cov_df = cov_df.sort_values(["Pipeline ID", "Horizonte"]).reset_index(drop=True)
        cov_df.to_excel(writer, sheet_name="PI_Coverage", index=False)
        width_df.to_excel(writer, sheet_name="PI_Stats", index=False)

    log.info("PIs y cobertura escritos en output/pis/ y output/results.xlsx (PI_Coverage, PI_Stats)")


    
    log.info("=== PIPELINE FINALIZADO OK ===")
    return {
        'stage1': s1_results,
        'stage2': s2_results,
        'direct': direct_results
    }
