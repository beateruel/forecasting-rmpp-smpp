# Forecasting RMPP→SMPP (Two-stage & Direct)

Este repositorio implementa los **pipelines P1–P8** del plan de 2 semanas (SES/ARIMA/RF/XGB/LGBM → Linear/XGB y baselines directos SARIMAX/XGB-exog), con **rolling-origin evaluation** y exporte a `output/results.xlsx`.

## Cómo ejecutar

1. Coloca tus CSV en `input/` y ajusta `config.json`:
```json
{
  "input": {
    "raw_prices_csv": "input/Oil_original.csv",
    "supplier_prices_csv": "input/PBT_historical_prices.csv",
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
  "horizons": [1,3,6,10],
  "stage1_models": ["SES","ARIMA","RF","XGB","LGBM"],
  "stage2_models": ["Linear","XGB"],
  "direct_models": ["SARIMAX","XGB_EXOG"],
  "random_state": 42
}
```

2. Ejecuta:
```bash
python run.py --config config.json
```

3. Salidas:
- CSVs por modelo/horizonte en `output/`
- `output/results.xlsx` con hojas: `Summary_Table`, `Stage1_Results`, `Stage2_Results`, `PI_Coverage`, `DM_tests`, `Notes_Log`.

## Notas
- La implementación de ARIMA usa `(1,1,1)` por simplicidad; puedes ajustar.
- Los modelos ML usan lags de `RMPP`. Para multi-step se usa **forecast recursivo**.
- **SARIMAX** directo usa lags de RMPP como exógenas.
- Se evita leakage con splits temporales y construcción de lags.
