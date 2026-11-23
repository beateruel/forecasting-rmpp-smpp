# Forecasting RMPP→SMPP (Two-Stage & Direct)

Forecasting framework for supplier material prices under upstream–downstream dependency and data-scarce industrial environments.

This repository contains the complete implementation of the forecasting framework proposed in the manuscript:

**"Forecasting Supplier Prices under Data Scarcity: A Two-Stage Framework and Horizon-Dependent Trade-offs"**

The framework compares direct and two-stage forecasting architectures under realistic industrial conditions characterised by limited supplier price observations and strong upstream–downstream dependencies. In addition to conventional forecast evaluation, it includes a decision-oriented assessment of the ability to anticipate procurement-relevant supplier price increases.

## Models compared

- **Two-stage pipelines (P1–P5):** SES / ARIMA / RF / XGB / LGBM → Linear
- **Direct baselines (P6–P7):** SARIMAX, XGB with exogenous variables (XGB_EXOG)

All models are evaluated with rolling-origin, multi-horizon forecasting (h = 1, 3 and 6 months).

## Repository structure

```
.
├── input/                # Public RMPP data and SMPP data template (proprietary SMPP not included)
├── output/
│   ├── predictions/      # Forecast outputs per model/horizon
│   └── results.xlsx      # Aggregated evaluation results
├── config.json           # Experiment configuration
├── run.py                # Main execution script
├── requirements.txt
├── LICENSE
└── README.md
```

## Methodology

The forecasting workflow consists of:

1. Data loading and preprocessing
2. Monthly alignment and interpolation
3. Feature engineering using lagged RMPP variables
4. Two-stage or direct forecasting
5. Rolling-origin evaluation
6. Performance assessment:
   - MAE, RMSE, MAPE
   - Prediction intervals (residual bootstrap)
   - Diebold–Mariano tests (with the Harvey small-sample correction)
7. Decision-oriented evaluation:
   - Procurement-relevant price-threshold detection
   - Correct and missed threshold crossings
   - Recall
   - Exact Clopper–Pearson confidence intervals
   - Exact McNemar paired tests

## Model design

**Two-stage forecasting pipelines.** The pipelines explicitly model the upstream–downstream dependency between:

- RMPP: Raw Material Market Prices (upstream)
- SMPP: Supplier Material Purchase Prices (downstream)

*Stage 1:* forecast upstream raw material prices (RMPP).
*Stage 2:* predict supplier prices (SMPP) using the forecasted upstream values.

**Direct forecasting baselines.** Direct models predict SMPP directly from observed lagged RMPP variables, without explicit upstream forecasting.

**Multi-step forecasting.** Machine learning models implement recursive multi-step forecasting strategies for longer horizons.

**Decision-oriented evaluation.** In addition to conventional point-forecast evaluation, the framework assesses whether forecasts anticipate supplier price increases that exceed procurement-relevant decision thresholds. The 5% threshold is the primary decision threshold in the industrial use case; 3% and 10% are included as sensitivity scenarios. Threshold-crossing performance is evaluated with raw event counts and recall. For the primary 5% threshold, exact Clopper–Pearson confidence intervals and exact McNemar paired tests account for the limited number of observed crossing events and compare detection performance across forecasting architectures (SARIMAX, ARIMA→Linear and RF→Linear as representative configurations).

## Configuration

Experiments are configured through `config.json`. Example:

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
    "lags": [1, 2, 3, 6],
    "rolling_means": []
  },
  "horizons": [1, 3, 6],
  "stage1_models": ["SES", "ARIMA", "RF", "XGB", "LGBM"],
  "stage2_models": ["Linear"],
  "direct_models": ["SARIMAX", "XGB_EXOG"]
}
```

The `rolling_means` field is optional and supported by the pipeline, but it was left empty (`[]`) in the experiments reported in the paper, to keep the feature set simple and reduce overfitting risk in this data-scarce setting (see the feature-engineering subsection of the manuscript). Users may enable rolling means (e.g., `[3, 6]`) in their own experiments.

## Installation

```
pip install -r requirements.txt
```

## Run the pipeline

```
python run.py --config config.json
```

## Outputs

The pipeline generates:

```
output/
├── predictions/                         # Forecast outputs per model and horizon
├── results.xlsx                         # Aggregated forecasting evaluation results
└── procurement_impact_common_timecoverage/
    ├── procurement_impact_common_timecoverage_summary.csv
    ├── procurement_impact_common_timecoverage_detail.csv
    ├── procurement_impact_common_timecoverage_dates.csv
    ├── procurement_impact_common_timecoverage_analysis.xlsx
    ├── ci.xlsx
    └── mcnemar.xlsx
```

`results.xlsx` includes the sheets: Summary_Table, Stage1_Results, Stage2_Results, PI_Coverage, DM_tests and Notes_Log.

The `procurement_impact_common_timecoverage` folder contains the decision-oriented post-processing analysis:

- `summary.csv`: threshold-crossing metrics by model, horizon and threshold
- `detail.csv`: observation-level threshold-crossing results
- `dates.csv`: common forecast target dates used across models
- `analysis.xlsx`: consolidated summary, detail and common-date sheets
- `ci.xlsx`: exact 95% Clopper–Pearson confidence intervals for recall at the 5% threshold
- `mcnemar.xlsx`: exact paired McNemar comparisons between representative forecasting configurations

## Reproducibility

The framework is designed to ensure reproducible forecasting experiments:

- Rolling-origin evaluation avoids information leakage
- Models are retrained at each forecast origin
- Features use only historical information
- Random seeds are fixed where applicable
- Decision-oriented comparisons are performed on matched forecast origins
- Threshold definitions and representative model comparisons are fixed consistently across horizons

## Data availability

The framework was developed and evaluated using a real industrial case study involving supplier material purchase prices (SMPP) and upstream raw material market prices (RMPP).

The upstream RMPP series is publicly available from the U.S. Energy Information Administration (EIA). The downstream SMPP series consists of proprietary supplier price records obtained through an industrial collaboration and is subject to confidentiality agreements, so the original SMPP dataset cannot be publicly released.

To support transparency and reproducibility, this repository provides:

- Complete source code for all forecasting pipelines
- Experimental configuration files
- Data preprocessing procedures
- Feature-engineering workflow
- Rolling-origin evaluation framework
- Prediction-interval generation procedures
- Statistical comparison procedures based on the Diebold–Mariano test
- Decision-oriented threshold-crossing evaluation procedures
- Exact confidence-interval and paired statistical testing procedures

The repository enables full reproduction of the computational workflow and experimental protocol. Researchers may apply the framework to alternative datasets to validate and extend the proposed methodology. The results reported in the associated manuscript were obtained with the proprietary industrial dataset described in the paper.

## Citation

*Author details are withheld during double-anonymised peer review and will be added upon acceptance.*

```
@article{anonymous2026_twostage,
  title  = {Forecasting Supplier Prices under Data Scarcity: A Two-Stage Framework and Horizon-Dependent Trade-offs},
  author = {Anonymous},
  year   = {2026}
}
```

## Acknowledgements and authors

Withheld during double-anonymised peer review.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
