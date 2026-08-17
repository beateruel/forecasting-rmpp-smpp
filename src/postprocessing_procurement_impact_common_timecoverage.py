# -*- coding: utf-8 -*-
"""
Post-processing analysis of procurement relevance using COMMON TIME COVERAGE.

This script does NOT retrain or modify any forecasting model.
It evaluates downstream SMPP forecasts only on forecast target dates that are
available for ALL compared models within each horizon. This ensures that
decision-support metrics are calculated on exactly the same realized events.

Default thresholds: 3%, 5%, 10%.

For a forecast targeting month t+h:

    reference price  = SMPP_t
    actual change    = (SMPP_{t+h} - SMPP_t) / SMPP_t
    predicted change = (SMPP_hat_{t+h} - SMPP_t) / SMPP_t

A threshold crossing is defined only for PRICE INCREASES:

    actual_crossing    = actual_change > threshold
    predicted_crossing = predicted_change > threshold

Thresholds are hypothetical decision scenarios and do not represent disclosed
contractual conditions of the case company.

Outputs
-------
output/procurement_impact_common_timecoverage/
    procurement_impact_common_timecoverage_summary.csv
    procurement_impact_common_timecoverage_detail.csv
    procurement_impact_common_timecoverage_dates.csv
    procurement_impact_common_timecoverage_analysis.xlsx   (if openpyxl installed)

The summary reports, by model / horizon / threshold:
    - MAE
    - N on common target dates
    - actual crossings
    - predicted crossings
    - correct crossings
    - missed crossings
    - false alerts
    - decision accuracy
    - crossing recall
    - crossing precision
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(root: Path) -> dict:
    config_path = root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_monthly_smpp(
    smpp_csv: Path,
    date_col: str,
    supplier_col: str,
    end_date: pd.Timestamp,
) -> pd.Series:
    if not smpp_csv.exists():
        raise FileNotFoundError(
            f"SMPP input file not found at: {smpp_csv}\n"
            "Place SMPP_input.csv in the input/ folder."
        )

    df = pd.read_csv(smpp_csv, sep=";")

    required = {date_col, supplier_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {smpp_csv.name}: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[date_col, supplier_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df[supplier_col] = pd.to_numeric(df[supplier_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    monthly = (
        df.set_index(date_col)[supplier_col]
        .resample("ME")
        .mean()
    )

    full_end = pd.Timestamp(end_date) + pd.offsets.MonthEnd(0)
    full_index = pd.date_range(
        start=monthly.index.min(),
        end=full_end,
        freq="ME",
    )
    monthly = monthly.reindex(full_index).ffill()
    monthly.name = "SMPP_reference_series"
    return monthly


PREDICTION_PATTERN = re.compile(
    r"^(?P<family>direct|stage2)_(?P<model>.+)_h(?P<horizon>\d+)\.csv$",
    flags=re.IGNORECASE,
)


def discover_supplier_prediction_files(predictions_dir: Path) -> list[dict]:
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found at: {predictions_dir}")

    records = []
    for path in sorted(predictions_dir.glob("*.csv")):
        match = PREDICTION_PATTERN.match(path.name)
        if not match:
            continue

        family = match.group("family").lower()
        raw_model = match.group("model")
        horizon = int(match.group("horizon"))

        if family == "direct":
            architecture = "Direct"
            model = raw_model
        else:
            architecture = "Two-stage"
            model = raw_model.replace("_Linear", "->Linear")

        records.append(
            {
                "path": path,
                "architecture": architecture,
                "model": model,
                "horizon": horizon,
            }
        )

    if not records:
        raise FileNotFoundError(
            f"No direct/stage2 prediction CSV files found in {predictions_dir}"
        )

    return records


def read_prediction_frame(meta: dict) -> pd.DataFrame:
    df = pd.read_csv(meta["path"])
    required = {"Date", "y_true", "y_pred"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{meta['path'].name} is missing {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[["Date", "y_true", "y_pred"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Date"] = df["Date"] + pd.offsets.MonthEnd(0)
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    return df.dropna(subset=["Date", "y_true", "y_pred"]).sort_values("Date")


def compute_common_target_dates(
    prediction_files: list[dict],
) -> dict[int, pd.DatetimeIndex]:
    """
    For each horizon, intersect target dates across ALL compared models.
    """
    common_dates = {}

    horizons = sorted({int(m["horizon"]) for m in prediction_files})
    for h in horizons:
        metas = [m for m in prediction_files if int(m["horizon"]) == h]
        date_sets = []

        for meta in metas:
            df = read_prediction_frame(meta)
            date_sets.append(set(pd.DatetimeIndex(df["Date"])))

        if not date_sets:
            continue

        common = set.intersection(*date_sets)
        common_dates[h] = pd.DatetimeIndex(sorted(common))

    return common_dates


def safe_ratio(num: float, den: float) -> float:
    if pd.isna(num) or pd.isna(den) or np.isclose(den, 0.0):
        return np.nan
    return float(num / den)


def analyse_prediction_file(
    meta: dict,
    smpp_monthly: pd.Series,
    thresholds: Iterable[float],
    common_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_prediction_frame(meta)

    # Enforce exactly the same target dates for every model in the horizon.
    df = df[df["Date"].isin(common_dates)].copy()

    h = int(meta["horizon"])

    base_rows = []
    for _, row in df.iterrows():
        target_date = pd.Timestamp(row["Date"]) + pd.offsets.MonthEnd(0)
        reference_date = target_date - pd.offsets.MonthEnd(h)

        reference_price = (
            float(smpp_monthly.loc[reference_date])
            if reference_date in smpp_monthly.index
            else np.nan
        )

        actual_change = safe_ratio(
            float(row["y_true"]) - reference_price,
            reference_price,
        )
        predicted_change = safe_ratio(
            float(row["y_pred"]) - reference_price,
            reference_price,
        )

        base_rows.append(
            {
                "Architecture": meta["architecture"],
                "Model": meta["model"],
                "Horizon": h,
                "Coverage": "common_timecoverage",
                "ReferenceDate": reference_date,
                "TargetDate": target_date,
                "ReferencePrice": reference_price,
                "ActualPrice": float(row["y_true"]),
                "PredictedPrice": float(row["y_pred"]),
                "AbsoluteError": abs(float(row["y_true"]) - float(row["y_pred"])),
                "ActualChange": actual_change,
                "PredictedChange": predicted_change,
            }
        )

    base = pd.DataFrame(base_rows).dropna(
        subset=["ReferencePrice", "ActualChange", "PredictedChange"]
    )

    if base.empty:
        return pd.DataFrame(), pd.DataFrame()

    detail_parts = []
    summary_rows = []

    for theta in thresholds:
        x = base.copy()
        x["Threshold"] = float(theta)

        x["ActualCrossing"] = x["ActualChange"] > theta
        x["PredictedCrossing"] = x["PredictedChange"] > theta

        x["CorrectDecision"] = x["ActualCrossing"] == x["PredictedCrossing"]
        x["CorrectCrossing"] = x["ActualCrossing"] & x["PredictedCrossing"]
        x["MissedCrossing"] = x["ActualCrossing"] & ~x["PredictedCrossing"]
        x["FalseAlert"] = ~x["ActualCrossing"] & x["PredictedCrossing"]

        detail_parts.append(x)

        n = len(x)
        actual_crossings = int(x["ActualCrossing"].sum())
        predicted_crossings = int(x["PredictedCrossing"].sum())
        correct_crossings = int(x["CorrectCrossing"].sum())
        missed = int(x["MissedCrossing"].sum())
        false_alerts = int(x["FalseAlert"].sum())

        recall = (
            correct_crossings / actual_crossings
            if actual_crossings > 0 else np.nan
        )
        precision = (
            correct_crossings / predicted_crossings
            if predicted_crossings > 0 else np.nan
        )

        summary_rows.append(
            {
                "Coverage": "common_timecoverage",
                "Architecture": meta["architecture"],
                "Model": meta["model"],
                "Horizon": h,
                "Threshold": float(theta),
                "N": n,
                "MAE": float(x["AbsoluteError"].mean()),
                "ActualCrossings": actual_crossings,
                "PredictedCrossings": predicted_crossings,
                "CorrectCrossings": correct_crossings,
                "MissedCrossings": missed,
                "FalseAlerts": false_alerts,
                "DecisionAccuracy": float(x["CorrectDecision"].mean()),
                "CrossingRecall": recall,
                "CrossingPrecision": precision,
            }
        )

    return (
        pd.concat(detail_parts, ignore_index=True),
        pd.DataFrame(summary_rows),
    )


def export_results(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    coverage_dates: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / "procurement_impact_common_timecoverage_detail.csv"
    summary_path = output_dir / "procurement_impact_common_timecoverage_summary.csv"
    dates_path = output_dir / "procurement_impact_common_timecoverage_dates.csv"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    coverage_dates.to_csv(dates_path, index=False)

    print(f"[OK] Detail CSV : {detail_path}")
    print(f"[OK] Summary CSV: {summary_path}")
    print(f"[OK] Dates CSV  : {dates_path}")

    try:
        import openpyxl  # noqa: F401

        excel_path = output_dir / "procurement_impact_common_timecoverage_analysis.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="summary", index=False)
            detail.to_excel(writer, sheet_name="detail", index=False)
            coverage_dates.to_excel(writer, sheet_name="common_dates", index=False)
        print(f"[OK] Excel      : {excel_path}")
    except ImportError:
        print(
            "[INFO] openpyxl is not installed; Excel export skipped. "
            "The CSV files contain the complete analysis."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Procurement-impact post-processing using common time coverage."
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.03, 0.05, 0.10],
        help="Hypothetical price-increase thresholds as decimals.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help="Default: output/predictions",
    )
    parser.add_argument(
        "--smpp-file",
        type=str,
        default=None,
        help="Default: supplier file specified in config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Default: output/procurement_impact_common_timecoverage",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_config(root)

    predictions_dir = (
        Path(args.predictions_dir)
        if args.predictions_dir
        else root / "output" / "predictions"
    )
    smpp_file = (
        Path(args.smpp_file)
        if args.smpp_file
        else root / config["input"]["supplier_prices_csv"]
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "output" / "procurement_impact_common_timecoverage"
    )

    thresholds = sorted(set(float(t) for t in args.thresholds))
    if any(t <= 0 for t in thresholds):
        raise ValueError("Thresholds must be positive decimals.")

    prediction_files = discover_supplier_prediction_files(predictions_dir)
    common_dates = compute_common_target_dates(prediction_files)

    if not common_dates:
        raise RuntimeError("Could not determine common target dates.")

    latest_target = max(max(dates) for dates in common_dates.values() if len(dates) > 0)

    smpp_monthly = build_monthly_smpp(
        smpp_csv=smpp_file,
        date_col=config["input"]["date_col"],
        supplier_col=config["input"]["supplier_col"],
        end_date=latest_target,
    )

    all_detail = []
    all_summary = []
    date_rows = []

    for h, dates in sorted(common_dates.items()):
        print(f"[INFO] h={h}: {len(dates)} common target dates across all models")
        for d in dates:
            date_rows.append(
                {
                    "Coverage": "common_timecoverage",
                    "Horizon": h,
                    "TargetDate": d,
                }
            )

    for meta in prediction_files:
        h = int(meta["horizon"])
        if h not in common_dates or len(common_dates[h]) == 0:
            continue

        detail, summary = analyse_prediction_file(
            meta=meta,
            smpp_monthly=smpp_monthly,
            thresholds=thresholds,
            common_dates=common_dates[h],
        )
        if not detail.empty:
            all_detail.append(detail)
        if not summary.empty:
            all_summary.append(summary)

    if not all_summary:
        raise RuntimeError("No common-time-coverage results were generated.")

    detail_df = pd.concat(all_detail, ignore_index=True)
    summary_df = pd.concat(all_summary, ignore_index=True)
    coverage_dates_df = pd.DataFrame(date_rows)

    summary_df = summary_df.sort_values(
        ["Horizon", "Threshold", "Architecture", "Model"]
    ).reset_index(drop=True)

    detail_df = detail_df.sort_values(
        ["Horizon", "Threshold", "Architecture", "Model", "TargetDate"]
    ).reset_index(drop=True)

    summary_df["Threshold_pct"] = summary_df["Threshold"] * 100
    summary_df["DecisionAccuracy_pct"] = summary_df["DecisionAccuracy"] * 100
    summary_df["CrossingRecall_pct"] = summary_df["CrossingRecall"] * 100
    summary_df["CrossingPrecision_pct"] = summary_df["CrossingPrecision"] * 100

    detail_df["Threshold_pct"] = detail_df["Threshold"] * 100
    detail_df["ActualChange_pct"] = detail_df["ActualChange"] * 100
    detail_df["PredictedChange_pct"] = detail_df["PredictedChange"] * 100

    export_results(detail_df, summary_df, coverage_dates_df, output_dir)

    print("\nAnalysis completed using COMMON TIME COVERAGE.")
    print(
        "All models within a given horizon are evaluated on exactly the same "
        "forecast target dates."
    )


if __name__ == "__main__":
    main()