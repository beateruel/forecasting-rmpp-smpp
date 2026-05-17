# Forecasting RMPP→SMPP (Two-Stage & Direct)

Forecasting framework for supplier material prices under upstream–downstream dependency and data scarcity conditions.

This repository implements the forecasting pipelines presented in the paper:

> **"Multi-Stage Forecasting of Supplier Material Prices under Upstream–Downstream Dependency and Data Scarcity"**

The framework compares:

- **Two-stage pipelines (P1–P5)**  
  SES / ARIMA / RF / XGB / LGBM → Linear

- **Direct baselines (P6–P7)**  
  SARIMAX, XGB with exogenous variables

All models are evaluated using **rolling-origin multi-horizon forecasting**.

---

# Repository Structure

```bash
.
├── input/                # Input datasets
├── output/
│   ├── predictions/      # Forecast outputs per model/horizon
│   └── results.xlsx      # Aggregated evaluation results
├── config.json           # Experiment configuration
├── run.py                # Main execution script
├── requirements.txt
└── README.md
```

---

# Methodology

The forecasting workflow consists of:

1. Data loading and preprocessing  
2. Monthly alignment and interpolation  
3. Feature engineering using lagged RMPP variables  
4. Multi-stage or direct forecasting  
5. Rolling-origin evaluation  
6. Performance assessment:
   - MAE
   - RMSE
   - MAPE
   - Prediction intervals (bootstrap)
   - Diebold–Mariano statistical tests

---

# Model Design

## Two-stage forecasting pipelines

The proposed pipelines explicitly model the upstream–downstream dependency between:

- **RMPP** → Raw Material Market Prices  
- **SMPP** → Supplier Material Purchase Prices

### Stage 1
Forecast upstream raw material prices (RMPP).

### Stage 2
Predict supplier prices (SMPP) using forecasted upstream values.

---

## Direct forecasting baselines

Direct models predict SMPP directly from observed lagged RMPP variables without explicit upstream forecasting.

---

## Multi-step forecasting

Machine learning models implement recursive multi-step forecasting strategies for longer horizons.

---

# Configuration

Configure experiments through `config.json`.

Example:

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

---

# Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run the Pipeline

```bash
python run.py --config config.json
```

---

# Outputs

The pipeline generates:

```bash
output/
├── predictions/
└── results.xlsx
```

The `results.xlsx` file includes:

- Summary_Table
- Stage1_Results
- Stage2_Results
- PI_Coverage
- DM_tests
- Notes_Log

---

# Reproducibility

The framework has been designed to ensure reproducible forecasting experiments:

- Rolling-origin evaluation avoids information leakage
- Models are retrained at each forecast origin
- Features use only historical information
- Random seeds are fixed where applicable

---

# Data Availability

The dataset used in this study is based on industrial data from a real SME use case.

Due to confidentiality constraints, the original supplier price dataset (SMPP) cannot be publicly shared.

This repository includes:

- Public upstream raw material price data (RMPP)
- A modified version of the supplier price dataset

The modified SMPP dataset preserves the statistical properties and temporal structure of the original data, including:

- variability,
- sparsity,
- update patterns,
- temporal dynamics.

This allows:

- reproducibility of the experimental setup,
- representative benchmarking of forecasting pipelines,
- methodological validation without disclosure of sensitive industrial information.

Researchers interested in the original data may contact the authors subject to data access agreements.

---

# Citation

If you use this repository in academic work, please cite:

```bibtex
@article{royo2026_multistage,
  title={Multi-Stage Forecasting of Supplier Material Prices under Upstream–Downstream Dependency and Data Scarcity},
  author={Royo, Beatriz and others},
  year={2026}
}
```

---

# Acknowledgements

This work was developed at Fundación Zaragoza Logistics Center (ZLC)
within the Horizon Europe project R3GROUP.

This project has received funding from the European Union’s Horizon Europe research and innovation programme under Grant Agreement No. 101091869.

Project reference:
https://cordis.europa.eu/project/id/101091869

---

# Authors

- Beatriz Royo  
  Fundación Zaragoza Logistics Center (ZLC)

---

# License

This project is licensed under the MIT License. See the `LICENSE` file for details.