
# Forecasting RMPP→SMPP (Two‑Stage & Direct)

This repository implements the forecasting pipelines described in the paper:

**"Multi‑Stage Forecasting of Supplier Material Prices under Upstream–Downstream Dependency and Data Scarcity"**

It compares:

- **Two-stage pipelines (P1–P5)**  
  (SES / ARIMA / RF / XGB / LGBM → Linear)

- **Direct baselines (P6–P7)**  
  (SARIMAX, XGB with exogenous variables)

All models are evaluated using **rolling-origin multi-horizon forecasting**.

---

## Methodology

The pipeline follows these steps:

1. Data loading and preprocessing  
2. Monthly alignment + interpolation (forward-fill / linear)  
3. Feature engineering (lagged RMPP features)  
4. Multi-stage or direct forecasting  
5. Rolling-origin evaluation  
6. Performance evaluation:
   - MAE, RMSE, MAPE  
   - Prediction intervals (bootstrap)  
   - Diebold–Mariano statistical tests  

---

## How to run

1. Place your input CSVs in `input/` and configure:

```json
{
  "input": {
    "raw_prices_csv": "input/RMPP_input.csv",
    "supplier_prices_csv": "input/SMPP_input.csv",
    "date_col": "Date",
    "raw_col": "column_1",
    "supplier_col": "column"
  },
  "processing": {
    "frequency": "M",
    "interpolation": ["ffill", "linear"],
    "train_end": "2024-06-30"
  },
  "features": {
    "lags": [1,2,3,6],
    "rolling_means": [3,6]
  },
  "horizons": [1,3,6],
  "stage1_models": ["SES","ARIMA","RF","XGB","LGBM"],
  "stage2_models": ["Linear"],
  "direct_models": ["SARIMAX","XGB_EXOG"]
}
```
2. Run the pipeline

pip install -r requirements.txt
python run.py --config config.json

---

## Check the outputs

output/
  ├── predictions/        # CSV per model and horizon
  └── results.xlsx       # aggregated results

results.xlsx includes:
-  Summary_Table
-  Stage1_Results
-  Stage2_Results
-  PI_Coverage
-  DM_tests
-  Notes_Log

---

 ## Model design
Two-stage pipelines explicitly model upstream–downstream dependency
- Stage 1: RMPP forecasting
- Stage 2: SMPP prediction using forecasted upstream values
- Direct models use only observed lagged RMPP
- Multi-step forecasting is implemented via recursive strategy for ML models

----

## Reproducibility
- Rolling-origin evaluation avoids information leakage  
- Models are retrained at each forecast origin  
- Features are constructed using only past data  
- Random seeds are fixed where applicable

---

## Data availability

The dataset used in this study is based on industrial data from a real SME use case.

Due to confidentiality constraints, the original supplier price data (SMPP) cannot be publicly shared. However, the repository includes:

- Upstream raw material price data (RMPP), obtained from public sources
- A modified version of the supplier price data

The modified SMPP dataset has been constructed to preserve the statistical properties and temporal structure of the original data (e.g., variability, sparsity, and update patterns), while removing any sensitive information.

This ensures that:
- The experimental setup can be fully reproduced  
- The behavior of the forecasting pipelines remains representative of the real use case  

Researchers interested in the original data may contact the authors subject to data access agreements.

