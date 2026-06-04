
# -*- coding: utf-8 -*-
"""
Orchestrates the entire forecasting pipeline (P1–P7), including
data loading, feature creation, model training and evaluation,
rolling-origin validation, uncertainty estimation and result export.
"""

from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path

# Local project modules providing:
# - Data loading and cleaning
# - Feature engineering
# - Stage 1 forecasting models (RMPP)
# - Stage 2 forecasting models (SMPP)
# - Direct baseline models with exogenous regressors
# - Utility functions for mask creation, lag generation, etc.
# - Evaluation metrics and DM tests
# - Logging infrastructure
from .data import load_and_clean
from .features import build_features
from .models_stage1 import SESModel, ARIMAModel, RFModel, XGBModel, LGBMModel
from .models_stage2 import LinearStage2, XGBStage2
from .direct_models import SARIMAXExog, XGBExog
from .utils import train_test_mask, make_lags
from .evaluation import metrics_table, bootstrap_intervals, dm_test, dm_test_modified
from .uncertainty import discover_prediction_files, aggregate_coverage_across_files

from .logger import get_logger
log = get_logger(__name__)

#-----------------------------------------------------------------
#Complementary functions 
#-----------------------------------------------------------------
def _recursive_forecast_ml(series: pd.Series, model, lags: List[int], horizon: int) -> float:
    """    
        Performs recursive h-step forecasting for ML models that rely only on lagged values.
        Workflow:
            1. Copy the input series (RMPP history).
            2. For each step up to the forecast horizon:
               a. Extract lag features from the last observed or predicted values.
               b. Predict the next value using the trained ML model.
               c. Append the prediction to the temporary series to enable further-step prediction.
            3. Return the final predicted value corresponding to the requested horizon.
    """
    s = series.copy()
    for _ in range(horizon):
        feats = {f"RMPP_lag{L}": s.iloc[-L] for L in lags}
        Xh = pd.DataFrame([feats])
        yhat = float(model.predict(Xh)[0])
        s = pd.concat([s, pd.Series([yhat])], ignore_index=True)
    return s.iloc[-1]

#-----------------------------------------------------------------------
def rolling_origin_stage1(df: pd.DataFrame, lags: List[int], roll_w: List[int],
                          horizons: List[int], min_train:int, train_end: str,
                          models: List[str]) -> Dict[str, Dict[int, pd.DataFrame]]:
    """  
     Performs rolling-origin evaluation for Stage 1 forecasting models (predicting RMPP).
     Output structure: results[model_name][horizon] = DataFrame(Date, y_true, y_pred)
        Steps:
        1. Generate ML lag and rolling-window features.
        2. Compute train/test masks using the configured date split.
        3. For each test point:
            a. Fit each Stage 1 model on all data up to t-1.
            b. Produce forecasts for each horizon.
            c. Store predictions aligned with their corresponding future dates.

        Special handling:
        - SES and ARIMA have direct forecast methods.
        - ML models rely on recursive forecasting using only lag features.    
    """
    
    #log.info(f"Rolling origin - final range after cleaning: {df['Date'].min().date()} → {df['Date'].max().date()}")
    #log.info(f"Last 5 (RMPP,SMPP):\n{df[['Date','RMPP','SMPP']].tail(5)}")

    results: Dict[str, Dict[int, pd.DataFrame]] = {}
    # build features for ML
    feat_df = build_features(df, lags=lags, roll_windows=roll_w)
    train_mask, test_mask = train_test_mask(df['Date'], train_end)
    dates = df['Date']

    for mname in models:
        model_results: Dict[int, List[Dict]] = {h: [] for h in horizons}
        for t_idx in np.where(test_mask)[0]:
            # training window: to t_idx-1
            train_idx = np.arange(0, t_idx)
            y_train = df.loc[train_idx, 'RMPP']

            if mname == 'SES':
                #log.info("Fit model SES")
                ses = SESModel().fit(y_train)                
                for h in horizons:
                        if t_idx + h - 1 >= len(df):
                            continue

                        #log.info(f"Predict SES | horizon={h}")
                        forecast = ses.predict(h)
                        yhat = forecast.iloc[-1]   #FIX CLAVE
                        y_true = df.loc[t_idx + h - 1, 'RMPP']

                        model_results[h].append({
                            "Date": dates.iloc[t_idx + h - 1],
                            "y_true": y_true,
                            "y_pred": yhat
                        })


            elif mname == 'ARIMA':                
                #log.info("Fit model ARIMA")
                ar = ARIMAModel(order=(1,1,1)).fit(y_train)

                for h in horizons:
                    if t_idx + h - 1 >= len(df):
                        continue

                    #log.info(f"Predict ARIMA | horizon={h}")
                    forecast = ar.predict(h)
                    yhat = forecast.iloc[-1]  
                    y_true = df.loc[t_idx + h - 1, 'RMPP']

                    model_results[h].append({
                        "Date": dates.iloc[t_idx + h - 1],
                        "y_true": y_true,
                        "y_pred": yhat
                    })


            else:
                # ML Models: train ONLY with lags for a consistent recursive forecast
                X_all = feat_df[[c for c in feat_df.columns if c.startswith('RMPP_lag')]]
                y_all = df['RMPP']
                X_tr = X_all.loc[train_idx].dropna()
                y_tr = y_all.loc[X_tr.index]
                if len(X_tr) < min_train:
                    continue
                if mname == 'RF':
                    mdl = RFModel()
                elif mname == 'XGB':
                    mdl = XGBModel()
                elif mname == 'LGBM':
                    mdl = LGBMModel()
                else:
                    raise ValueError('Stage1 models not supported')
                mdl.fit(X_tr, y_tr)
                for h in horizons:
                    if t_idx + h - 1 >= len(df):
                        continue
                    # recursive forecast with lags
                    yhat = _recursive_forecast_ml(df.loc[:t_idx - 1, 'RMPP'], mdl, lags, h)
                    y_true = df.loc[t_idx + h - 1, 'RMPP']
                    model_results[h].append({
                        "Date": dates.iloc[t_idx + h - 1], "y_true": y_true, "y_pred": yhat
                    })

        results[mname] = {h: pd.DataFrame(v) for h, v in model_results.items() if len(v) > 0}
    return results

#-----------------------------------------------------------------------
def rolling_origin_stage2(
    df: pd.DataFrame,
    stage1_preds: Dict[str, Dict[int, pd.DataFrame]],
    lags: List[int],
    horizons: List[int],
    min_train: int = 6,
    models: List[str] = ("Linear", "XGB")
) -> Dict[str, Dict[int, pd.DataFrame]]:
    """   
      Performs rolling-origin evaluation for the second stage of the pipeline.
      Stage 2 predicts SMPP using:
        1) Historical RMPP lags
        2) Forecasted RMPP from Stage 1

      Stage 2 pipelines follow this process:
        For each pipeline (Stage1 model combined with Stage2 model):
            For each horizon:
                Normalize dates of Stage 1 predictions to end-of-month.
                Merge Stage 1 RMPP forecasts with true SMPP and RMPP values.
                Create lag features of RMPP.
                For each index after the minimum training size:
                    Train the Stage 2 model using all data up to t−1.
                    Predict SMPP for time t.
                    Store predictions with their corresponding date.

     Output structure:
        results[pipeline_name][horizon] = DataFrame(Date, y_true, y_pred)
    """
    results: Dict[str, Dict[int, pd.DataFrame]] = {}

    for s1_name, hdict in stage1_preds.items():
        for s2_name in models:

            pipe_id = f"{s1_name}->{s2_name}"
            results[pipe_id] = {}

            for h, s1_df in hdict.items():
                # Normalize dates for Stage 1 to the end of the monts
                tmp = s1_df.copy()
                tmp["Date"] = tmp["Date"].dt.to_period("M").dt.to_timestamp("M")
                tmp = tmp.rename(columns={"y_pred": "forecast_RMPP"})

                # Merge with SMPP and histroical RMPP  (with previous LOCF)
                tmp = (
                    tmp.merge(
                        df[["Date", "RMPP", "SMPP"]],
                        on="Date",
                        how="left"
                    )
                    .sort_values("Date")
                    .reset_index(drop=True)
                )

                # Build lags for RMPP
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
                    #elif s2_name == "XGB":
                     #   mdl = XGBStage2()
                    else:
                        raise ValueError("Stage 2 model no supported")

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

#----------------------------------------------------------------------------------------
def rolling_origin_direct(df: pd.DataFrame, lags: List[int], horizons: List[int],min_train:int,
                          train_end: str, models: List[str]) -> Dict[str, Dict[int, pd.DataFrame]]:
    results: Dict[str, Dict[int, pd.DataFrame]] = {}
    """
        Performs rolling-origin evaluation for direct baseline models that predict SMPP
        using exogenous regressors (lagged RMPP values).

        This implements single-step or multi-step direct forecasting:
          For each model (SARIMAX, XGB_EXOG):
            For each horizon:
                Construct exogenous matrix consisting of lagged RMPP values.
                Use rolling-origin methodology:
                    At each test index t:
                        Train the model with all data up to t−1.
                        Predict SMPP at time t+h−1 using exogenous features associated
                        with that future month.
                        Store the predicted and true values.

        Output structure:
            results[model_name][horizon] = DataFrame(Date, y_true, y_pred)
        """


    for mname in models:
        results[mname] = {}
        # build lags of RMPP as exogenous
        X_exog_full = pd.concat([make_lags(df['RMPP'], lags, 'RMPP').drop(columns=['RMPP'])], axis=1)
        data = pd.concat([df[['Date', 'SMPP']], X_exog_full], axis=1)
        
        #log.info(f"[Direct:{mname}] dates before dropna: {data['Date'].min().date()} → {data['Date'].max().date()} (n={len(data)})")
        data = data.dropna().reset_index(drop=True)
        
        #log.info(f"[Direct:{mname}] dates after dropna: {data['Date'].min().date()} → {data['Date'].max().date()} (n={len(data)})")
        train_mask = data['Date'] <= pd.to_datetime(train_end)
        for h in horizons:
            rows = []
            # rolling-origin in the whole test
            test_idx = data.index[~train_mask]
            for idx in test_idx:
                # trins with data up to idx-1
                tr_idx = data.index[(data.index < idx)]
                if len(tr_idx) < min_train:
                    continue
                y_tr = data.loc[tr_idx, 'SMPP']
                X_tr = data.loc[tr_idx, data.columns.str.startswith('RMPP_lag')]
                # build X_future for horizont h (use the raw target en idx+h-1)
                target_row = idx + h - 1
                if target_row >= len(data):
                    continue
                X_future = data.loc[[target_row], X_tr.columns]
                if mname == 'SARIMAX': 
                    #log.info(f"Fit SARIMAX direct | idx={idx} | horizon={h}")
                    mdl = SARIMAXExog(order=(1,1,1), seasonal_order=(0,0,0,0)).fit(y_tr, X_tr)
                    forecast = mdl.predict(steps=1, X_future=X_future)
                    yhat = forecast.iloc[0]
                elif mname == 'XGB_EXOG':
                   # log.info(f"Fit XGB direct | idx={idx} | horizon={h}")
                    mdl = XGBExog()
                    mdl.fit(X_tr, y_tr)
                    yhat = mdl.predict(X_future)[0]
                else:
                    raise ValueError('Direct not supported')
                y_true = data.loc[target_row, 'SMPP']
                rows.append({'Date': data.loc[target_row, 'Date'], 'y_true': y_true, 'y_pred': yhat})
            results[mname][h] = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['Date','y_true','y_pred'])
    return results
#--------------------------------------------------------------------------------------------------------------------
def _df_safe(rows, cols_order=None):
    """
        Safely converts a list of row dictionaries into a DataFrame, ensuring:
            1. The DataFrame is created even if rows is empty.
            2. All expected columns (cols_order) exist in the final DataFrame.
            3. Columns appear in the order specified by cols_order.
            4. Missing columns are added with appropriate empty dtypes.

        Parameters
        ----------
        rows : list of dict
            Each dictionary represents one row of data.
            If None or empty, an empty DataFrame will be created.

        cols_order : list of str, optional
            Defines the required columns and their order.
            Any column missing from the DataFrame will be created.

        Returns
        -------
        DataFrame
            A DataFrame with guaranteed column presence and ordering.
    """
    df = pd.DataFrame(rows or [])
    if cols_order:
        # it adds the missing columns and sort them
        for c in cols_order:
            if c not in df.columns:
                df[c] = pd.Series(dtype='float64' if c not in ('Pipeline ID','Type','Notes','Stage1 Model','Stage2 Model') else 'object')
        df = df[cols_order]
    return df

#----------------------------------------------------------------------------
# RUN PIPELINE
#----------------------------------------------------------------------------

def run_pipeline(config_path: str = 'config.json') -> Dict:        
    """
    Main orchestrator for the full pipeline.

    Responsibilities:
        1. Load configuration and datasets.
        2. Clean and preprocess raw data.
        3. Execute Stage 1 forecasting (RMPP).
        4. Standardize calendar and prepare data for Stage 2.
        5. Execute Stage 2 forecasting (SMPP).
        6. Execute direct baselines (SARIMAX and XGB with exogenous RMPP lags).
        7. Compute evaluation metrics for all models.
        8. Save predictions and evaluation tables to CSV and Excel.
        9. Compute Prediction Intervals and add them to the Excel output.

    Returns:
        Dictionary with Stage 1, Stage 2, and direct model predictions.
    """    
    log.info("=== START PIPELINE ===")
    log.info("Loading config")

    # Load configuration JSON with all settings (data paths, model lists, horizons, etc.)
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    print("\n===== CONFIG FILE =====")
    print(cfg['input']['supplier_prices_csv'])
    #input("Press Enter to continue...")
    
    # Load and clean raw input data:
    # Includes optional resampling and interpolation options defined in the config file.
    df = load_and_clean(
        cfg['input']['raw_prices_csv'],
        cfg['input']['supplier_prices_csv'],
        cfg['input']['date_col'],
        cfg['input']['raw_col'],
        cfg['input']['supplier_col'],
        resample=cfg['processing']['frequency'],
        interpolations=cfg['processing']['interpolation']
    )
    
    # Detect automatically the first valid date for Stage 2
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    # first date where SMPP and RMPP exist simultanously
    first_valid_stage2_date = df_sorted.dropna(subset=["SMPP", "RMPP"])["Date"].min()
    
    # Adjust train_end automatically
    cfg['processing']['train_end'] = str(first_valid_stage2_date.date())
    log.info(f"Auto-adjusted train_end for Stage2: {cfg['processing']['train_end']}")

        
    log.info(f"Horizonts: {cfg['horizons']}")
    log.info(f"Stage1 models: {cfg['stage1_models']}")
    log.info(f"Stage2 models: {cfg['stage2_models']}")

    # Normalize date column name if necessary
    if 'Date' not in df.columns:
        df = df.rename(columns={cfg['input']['date_col']: 'Date'})
    
    log.info("Executing Stage 1 (RMPP)")    

    # ----------------------------------------------------------
    # Stage 1: Predict RMPP using SES, ARIMA, RF, XGB, LGBM
    # ----------------------------------------------------------
    s1_results = rolling_origin_stage1(
        df=df.copy(),
        lags=cfg['features']['lags'],
        roll_w=cfg['features']['rolling_means'],
        horizons=cfg['horizons'],
        min_train = cfg.get('training', {}).get('min_train_stage1', 24),
        train_end=cfg['processing']['train_end'],
        models=cfg['stage1_models']
    )

    log.info(f"Models: {list(s1_results.keys())}")
    
    log.info("Harmonizing Stage 2 (SMPP)")        
    # ----------------------------------------------------------
    # Stage 2 requires consistent monthly index and forward-fill
    # ----------------------------------------------------------   
    df_stage2 = df.copy()
    df_stage2['Date'] = df_stage2['Date'].dt.to_period('M').dt.to_timestamp('M')    
    
    df_stage2 = (
        df_stage2
          .set_index('Date')
          .asfreq('ME')                    # monthly end frequency
          .ffill()                         # fill missing months using last observation
          .reset_index()
    )

    log.info(f"Stage2 | df_stage2 after LOCF: {df_stage2.shape[0]} filas")
    log.info("Executing Stage 2 (SMPP)") 
        
    # ----------------------------------------------------------
    # Stage 2 forecasting: RMPP forecast → SMPP forecast
    # ----------------------------------------------------------    
    s2_results = rolling_origin_stage2(
        df=df_stage2.copy(),
        stage1_preds=s1_results,
        lags=cfg['features']['lags'],
        horizons=cfg['horizons'],
        min_train=cfg.get('training', {}).get('min_train_stage2', 6),
        models=cfg['stage2_models']
    )

    log.info(f"Pipelines generated: {list(s2_results.keys())}")
    
    # ----------------------------------------------------------
    # Direct baselines (SARIMAXEXOG, XGB_EXOG)
    # ----------------------------------------------------------
    log.info("Executing direct models")    
    direct_horizons = cfg.get('direct_horizons', [1])  
    direct_results = rolling_origin_direct(
        df=df.copy(),
        lags=cfg['features']['lags'],
        horizons=direct_horizons,
        min_train = cfg.get('training', {}).get('min_train_direct', 12),
        train_end=cfg['processing']['train_end'],
        models=cfg['direct_models']
    )
    log.info(f"Direct models: {list(direct_results.keys())}")

    # ----------------------------------------------------------
    # Save Stage 1, Stage 2 and Direct predictions to CSV files
    # ----------------------------------------------------------   
    # Stage1
    os.makedirs('output/predictions', exist_ok=True)
    for mname, hdict in s1_results.items():
        for h, dd in hdict.items():
            dd.to_csv(os.path.join('output/predictions', f'stage1_{mname}_h{h}.csv'), index=False)
    # Stage2
    for pipe, hdict in s2_results.items():
        for h, dd in hdict.items():
            dd.to_csv(os.path.join('output/predictions', f'stage2_{pipe.replace("->", "_")}_h{h}.csv'), index=False)
    # Direct
    for mname, hdict in direct_results.items():
        for h, dd in hdict.items():
            dd.to_csv(os.path.join('output/predictions', f'direct_{mname}_h{h}.csv'), index=False)  
    
    # ----------------------------------------------------------
    # Build evaluation tables: Stage 1, Stage 2, Direct, PIs, DM
    # ----------------------------------------------------------
    #summary_rows, stage1_rows, stage2_rows, dm_rows = [], [], [], []  
    
    stage1_rows= []
    # Stage1 metrics
    for mname, hdict in s1_results.items():
        for h, dd in hdict.items():
            if len(dd) == 0:
                continue
            mt = metrics_table(dd['y_true'], dd['y_pred'])
            stage1_rows.append({
                'Model Stage1': mname,
                'Horizont': h,
                'RMPP MAPE (%)': mt['MAPE_%'],
                'RMPP RMSE': mt['RMSE'],
                'RMPP MAE': mt['MAE']            
                
            })

    # Map Stage 2 pipelines to predefined IDs (P1–P6)    
    map_name_to_pid = {
        'SES->Linear': 'P1',
        'ARIMA->Linear': 'P2',
        'RF->Linear': 'P3',
        'XGB->Linear': 'P4',
        'LGBM->Linear': 'P5',     
    }
    #stage2 metrics 
    stage2_rows= []
    for pipe, hdict in s2_results.items():        
        pipe_norm = pipe.replace('-&gt;', '->')  # just in case the input is with escape
        pid = map_name_to_pid.get(pipe_norm, pipe_norm)
        for h, dd in hdict.items():
            mt = metrics_table(dd['y_true'], dd['y_pred'])
            stage2_rows.append({
                'Pipeline ID': pid,
                'Stage1 Model': pipe.split('->')[0],
                'Stage2 Model': pipe.split('->')[1],
                'Horizont': h,
                'SMPP MAPE': mt['MAPE_%'],
                'SMPP RMSE': mt['RMSE'],
                'SMPP MAE': mt['MAE']
            })

    # Direct metrics
    base_map = {'SARIMAX': 'P6', 'XGB_EXOG': 'P7'}
    direct_rows= []
    for mname, hdict in direct_results.items():
        pid = base_map.get(mname, mname)
        for h, dd in hdict.items():
            mt = metrics_table(dd['y_true'], dd['y_pred'])
            direct_rows.append({
                'Pipeline ID': pid,
                'Type': 'Direct',
                'Horizont': h,
                'SMPP MAPE (%)': mt['MAPE_%'],
                'SMPP RMSE': mt['RMSE'],
                'SMPP MAE': mt['MAE']
            })

    
    # Prediction intervals and uncertainty analysis
    log.info("Calculating Prediction Intervals (80%/95%), coverate y sharpness...")
    cfg_unc = cfg.get("uncertainty", {
        "n_boot": 1000,
        "alpha_levels": [0.20, 0.05],
        "recenter": True,
        "random_state": 42,
        "make_plots": True,
    })

    base_output = Path("output/predictions")
    files = discover_prediction_files(base_output=base_output)
    cov_df, width_df = aggregate_coverage_across_files(
        files=files, cfg_unc=cfg_unc, base_output=base_output
    )
    
    # Modified Diebold Mariano Test: validate if 2 stages model area statically better than direct
    dm_rows, dm_modified_rows= [], []
    baseline_names = ['SARIMAX', 'XGB_EXOG']
    for baseline_name in baseline_names:
        baseline = direct_results.get(baseline_name, {})

        for pipe, hdict in s2_results.items():
            pipe_norm = pipe.replace('->', '->')
            pid = map_name_to_pid.get(pipe_norm, pipe_norm)
            for h, df_pipe in hdict.items():

                if h not in baseline:
                    continue

                df_base = baseline[h]
                # Merge by date
                cmp = pd.merge(df_pipe, df_base, on='Date', suffixes=('_A', '_B'))

                # Minumum requirement
                if len(cmp) < 10:
                    continue

                eA = cmp['y_true_A'] - cmp['y_pred_A']
                eB = cmp['y_true_B'] - cmp['y_pred_B']

                res_dm_mofified = dm_test_modified(eA.values, eB.values, h=h, power=1)

                dm_modified_rows.append({
                    'Pipeline ID': pid,
                    'Horizont': h,
                    'Model A': pipe_norm,
                    'Model B': baseline_name,
                    'Loss Function': 'MAE',
                    'DM Statistic': res_dm_mofified['DM'],
                    'DM p-value vs baseline': res_dm_mofified['p_value'],
                    'Significative?': 'Yes' if res_dm_mofified['p_value'] < 0.05 else 'No'
                })
                res_dm = dm_test(eA.values, eB.values, h=h, power=1)

                dm_rows.append({
                    'Pipeline ID': pid,
                    'Horizont': h,
                    'Model A': pipe_norm,
                    'Model B': baseline_name,
                    'Loss Function': 'MAE',
                    'DM Statistic': res_dm['DM'],
                    'DM p-value vs baseline': res_dm['p_value'],
                    'Significative?': 'Yes' if res_dm['p_value'] < 0.05 else 'No'
                })
        df_dm_modified = pd.DataFrame(dm_modified_rows)
        df_dm = pd.DataFrame(dm_rows)

    # =========================================================
    # Summary table
    # =========================================================

    df_stage2 = pd.DataFrame(stage2_rows)
    df_direct = pd.DataFrame(direct_rows)

    # Stage2
    df_stage2_main = df_stage2[[
        'Pipeline ID','Stage1 Model','Stage2 Model','Horizont',
        'SMPP MAE','SMPP MAPE'
    ]].copy()

    df_stage2_main['Model'] = df_stage2_main['Stage1 Model'] + "->" + df_stage2_main['Stage2 Model']

    # Direct
    df_direct_main = df_direct[[
        'Pipeline ID','Horizont','SMPP MAE','SMPP MAPE (%)'
    ]].copy()

    df_direct_main = df_direct_main.rename(columns={'SMPP MAPE (%)':'SMPP MAPE'})
    df_direct_main['Model'] = df_direct_main['Pipeline ID']

    # Combine
    df_all = pd.concat([
        df_stage2_main[['Pipeline ID','Model','Horizont','SMPP MAE','SMPP MAPE']],
        df_direct_main[['Pipeline ID','Model','Horizont','SMPP MAE','SMPP MAPE']]
    ], ignore_index=True)

    # Baselines
    baseline_sarimax = df_all[df_all['Pipeline ID']=='P6'][['Horizont','SMPP MAE']].rename(columns={'SMPP MAE':'MAE_SARIMAX'})
    baseline_xgb = df_all[df_all['Pipeline ID']=='P7'][['Horizont','SMPP MAE']].rename(columns={'SMPP MAE':'MAE_XGB_EXOG'})

    df_all = df_all.merge(baseline_sarimax, on='Horizont', how='left')
    df_all = df_all.merge(baseline_xgb, on='Horizont', how='left')

    # Relative performance
    df_all['Delta_vs_SARIMAX_%'] = 100*(df_all['SMPP MAE'] - df_all['MAE_SARIMAX'])/df_all['MAE_SARIMAX']
    df_all['Delta_vs_XGB_%'] = 100*(df_all['SMPP MAE'] - df_all['MAE_XGB_EXOG'])/df_all['MAE_XGB_EXOG']

    # Best model per horizon
    df_all['Best'] = df_all.groupby('Horizont')['SMPP MAE'].transform(lambda x: x == x.min())
    df_all['Best'] = df_all['Best'].map({True:'✓', False:''})

    # Final formatting
    df_main_table = df_all.sort_values(['Horizont','SMPP MAE'])
    df_main_table = df_all[~df_all['Pipeline ID'].isin(['P6','P7'])].copy()

    # =========================================================
    # BUILD SIGNIFICANCE TABLE (summary)
    # =========================================================

    df_dm['Test'] = 'DM'
    df_dm_modified['Test'] = 'Modified_DM'

    df_sig = pd.concat([df_dm, df_dm_modified], ignore_index=True)

    col_model_a = 'Model A' if 'Model A' in df_sig.columns else 'Model A'
    col_model_b = 'Model B' if 'Model B' in df_sig.columns else 'Model B'

    df_sig_table = df_sig[[
        'Pipeline ID',
        col_model_a,
        'Horizont',
        col_model_b,
        'Test',
        'Significative?'
    ]].copy()

    df_sig_table = df_sig_table.rename(columns={
        col_model_a: 'Model A',
        col_model_b: 'Baseline'
    })

    df_sig_table = df_sig_table.pivot_table(
        index=['Pipeline ID','Model A','Horizont','Baseline'],
        columns='Test',
        values='Significative?',
        aggfunc='first'
    ).reset_index()

    

    # Write PI results back into the Excel report
    from pandas import ExcelWriter
    with pd.ExcelWriter('output/results.xlsx', engine='openpyxl') as writer:
        #summary tables for reporting
        df_main_table.to_excel(writer, sheet_name='Main_Table', index=False)
        df_sig_table.to_excel(writer, sheet_name='Significance_Table', index=False)
        #prediction intervals
        cov_df.to_excel(writer, sheet_name='PI_Coverage', index=False)
        #prediction stats ************************** REVISAR NO SE QUE HACE PARA QUE SIRVE
        width_df.to_excel(writer, sheet_name='PI_Stats', index=False) 
       

        # Stage1 metrics
        pd.DataFrame(stage1_rows).to_excel(writer, sheet_name='Stage1_Results', index=False)
        # Stage2 metrics
        pd.DataFrame(stage2_rows).to_excel(writer, sheet_name='Stage2_Results', index=False)
        # Direct metrics
        pd.DataFrame(direct_rows).to_excel(writer, sheet_name='Direct_Results', index=False)
        #Diebold Mariano test modified
        df_dm_modified.to_excel(writer, sheet_name='DM_tests_modified', index=False)
        #Diebold Mariano test
        df_dm.to_excel(writer, sheet_name='DM_tests', index=False)
        
        


    log.info("=== PIPELINE FINSHES OK ===")
    # Final return dictionary
    return {
            'stage1': s1_results,
            'stage2': s2_results,
            'direct':direct_results
        }

  