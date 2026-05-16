
    # -*- coding: utf-8 -*-
"""
    Uncertainty utilities used for constructing empirical prediction intervals (PIs)
    via residual bootstrap, evaluating interval quality, discovering prediction files,
    and generating fan-chart visualizations.

    This module provides:
        • Filename parsers to extract model metadata (stage, horizon, pipeline).
        • Bootstrap-based prediction intervals for any y_true / y_pred pair.
        • Coverage and sharpness metrics, including 95% Wilson confidence intervals.
        • Tools to process and enrich prediction CSVs with PIs and plots.
        • Aggregation routines to summarize uncertainty performance across all pipelines.
"""
from __future__ import annotations
import os, re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .logger import get_logger
from .evaluation import bootstrap_intervals  # extended version used by pipeline
log = get_logger(__name__)


# ---------------------------------------------------------------------
# Filename patterns used to recognize and classify pipeline outputs
# The patterns are aligned with the naming rules used in pipeline.py.
# ---------------------------------------------------------------------

# Example: stage1_ARIMA_h3.csv
PAT_STAGE1 = re.compile(r"stage1_(?P<model>[^_]+)_h(?P<h>\d+)\.csv$", re.IGNORECASE)


# Example: stage2_SES_Linear_h2.csv     (pipeline.py replaces '->' with '_')
PAT_STAGE2 = re.compile(r"stage2_(?P<pipeline_und>.+)_h(?P<h>\d+)\.csv$", re.IGNORECASE)


# Example: direct_SARIMAX_h1.csv
PAT_DIRECT = re.compile(r"direct_(?P<model>.+)_h(?P<h>\d+)\.csv$", re.IGNORECASE)

# Mapping from pipeline name → pipeline ID (P1..P7), as used in reports
MAP_PIPELINE_TO_PID = {
    "SES->Linear": "P1",
    "ARIMA->Linear": "P2",
    "RF->Linear": "P3",
    "XGB->Linear": "P4",
    "LGBM->Linear": "P5", 
}
MAP_DIRECT_TO_PID = {"SARIMAX": "P6", "XGB_EXOG": "P7"}


def _extract_meta_from_filename(path: Path) -> Dict[str, str]:
    
    """
        Parse metadata (stage, model/pipeline identifier, forecast horizon)
        from a filename following the standardized naming convention.

        Returns a dictionary containing:
            • stage: 'stage1', 'stage2', or 'direct'
            • entity: model or pipeline (textual name)
            • pipeline_id: standardized ID (P1..P8) when applicable
            • h: forecast horizon extracted from the filename
        """

    #Stage 1 files
    fname = path.name
    m1 = PAT_STAGE1.search(fname)
    if m1:
        model = m1.group("model")
        h = int(m1.group("h"))
        return {"stage": "stage1", "entity": model, "pipeline_id": model, "h": h}

    #Stage 2 files
    m2 = PAT_STAGE2.search(fname)
    if m2:
        pipe_und = m2.group("pipeline_und")
        # Reconstruct the pipeline name: SES_Linear -> SES->Linear
        pipe = pipe_und.replace("_", "->")
        h = int(m2.group("h"))
        pid = MAP_PIPELINE_TO_PID.get(pipe, pipe)  # si no está mapeado, usar texto
        return {"stage": "stage2", "entity": pipe, "pipeline_id": pid, "h": h}

    #Direct model
    m3 = PAT_DIRECT.search(fname)
    if m3:
        model = m3.group("model")
        h = int(m3.group("h"))
        pid = MAP_DIRECT_TO_PID.get(model, model)
        return {"stage": "direct", "entity": model, "pipeline_id": pid, "h": h}

    
    # Unknown or misnamed file
    return {"stage": "unknown", "entity": fname, "pipeline_id": fname, "h": -1}


def compute_pis_for_dataframe(
    df: pd.DataFrame,
    levels: Iterable[float] = (0.20, 0.05),
    n_boot: int = 1000,
    recenter: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
        
    """
        Compute bootstrap-based prediction intervals for a DataFrame containing
        'y_true' and 'y_pred' columns.

        For each alpha (e.g., 0.20 -> 80% PI), the function:
            1. Calls the empirical residual bootstrap routine.
            2. Creates lower/upper bound columns (pi80_lo, pi80_hi, ...).
            3. Computes the interval width.

        Returns an enriched DataFrame with PI bounds and widths.
        """
    assert {"y_true", "y_pred"}.issubset(df.columns), "Required columns: y_true, y_pred"
    out = df.copy()
    # Generat intervals for each alpha (p.ej. 0.20 -> 80%)
    for alpha in levels:
        lo, hi = bootstrap_intervals(
            y_true=out["y_true"].values,
            y_pred=out["y_pred"].values,
            alpha=alpha,
            n_boot=n_boot,
            random_state=random_state,
            recenter=recenter,
        )
        pct = int(round((1 - alpha) * 100))
        out[f"pi{pct}_lo"] = lo
        out[f"pi{pct}_hi"] = hi
        out[f"width{pct}"] = out[f"pi{pct}_hi"] - out[f"pi{pct}_lo"]
    return out

def _wilson_interval(k, n, z=1.96):  # 95%
    if n == 0: return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    half = z * sqrt((p*(1-p)/n) + (z**2/(4*n**2))) / denom
    return (center - half, center + half)


def coverage_and_sharpness(df_with_pi: pd.DataFrame) -> Dict[str, float]:
    
    """
        Compute empirical coverage (%) and mean prediction‑interval width for all
        available PIs in the DataFrame.

        For each percentile level (80%, 95%), the function:
            1. Checks whether lower/upper interval columns exist.
            2. Computes empirical coverage: fraction of y_true inside the PI.
            3. Computes mean interval width.
            4. Computes a 95% Wilson confidence interval for the coverage estimate.

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
                coverage_80, coverage_95,
                coverage_80_[lo95|hi95], coverage_95_[lo95|hi95],
                mean_width_80, mean_width_95,
                n_obs.
        """

    res: Dict[str, float] = {"n_obs": float(len(df_with_pi))}
    for pct in (80, 95):
        lo, hi = f"pi{pct}_lo", f"pi{pct}_hi"
        if lo in df_with_pi.columns and hi in df_with_pi.columns:
            inside = (df_with_pi["y_true"] >= df_with_pi[lo]) & (df_with_pi["y_true"] <= df_with_pi[hi])
            k, n = int(inside.sum()), int(inside.size)
            res[f"coverage_{pct}"] = float(inside.mean() * 100.0)
            res[f"mean_width_{pct}"] = float((df_with_pi[hi] - df_with_pi[lo]).mean())
            lo_ci, hi_ci = _wilson_interval(k, n, 1.96)
            res[f"coverage_{pct}_lo95"] = float(lo_ci * 100 if not np.isnan(lo_ci) else np.nan)
            res[f"coverage_{pct}_hi95"] = float(hi_ci * 100 if not np.isnan(hi_ci) else np.nan)
    return res

def process_prediction_file(
    path_in: Path,
    out_dir: Path,
    levels=(0.20, 0.05),
    n_boot: int = 1000,
    recenter: bool = True,
    random_state: int = 42,
    make_plots: bool = True,
    plot_dir: Path | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, str]]:
    
    """
        Load a prediction CSV, compute empirical bootstrap prediction intervals,
        save an enriched CSV, and (optionally) generate a fan‑chart visualization.

        Steps
        -----
            1. Extract metadata (stage, pipeline, horizon) from filename.
            2. Load y_true / y_pred with associated Date.
            3. Compute prediction intervals using empirical residual bootstrap.
            4. Save enriched CSV ending in _with_PI.csv.
            5. Generate optional PI fan‑chart (80% and 95%) for uncertainty reporting.
            6. Compute coverage and sharpness statistics.

        Returns
        -------
        (df_pi, stats, meta):
            df_pi : DataFrame with PI bounds and widths added
            stats : coverage / interval‑width metrics
            meta  : metadata extracted from the filename
        """

    meta = _extract_meta_from_filename(path_in)
    df = pd.read_csv(path_in, parse_dates=["Date"])
    df_pi = compute_pis_for_dataframe(
        df, levels=levels, n_boot=n_boot, recenter=recenter, random_state=random_state
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path_in.stem}_with_PI.csv"
    df_pi.to_csv(out_path, index=False)
    #log.info(f"[PIs] written: {out_path}")


    # Optional fan‑chart visualization
    if make_plots:
        plot_dir = plot_dir or (out_dir.parent / "figs_pis")
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.plot(df_pi["Date"], df_pi["y_true"], color="#333", lw=1.8, label="Real")
        ax.plot(df_pi["Date"], df_pi["y_pred"], color="#0072B2", lw=1.6, label="Prediction")
        if {"pi80_lo", "pi80_hi"}.issubset(df_pi.columns):
            ax.fill_between(df_pi["Date"], df_pi["pi80_lo"], df_pi["pi80_hi"],
                            color="#0072B2", alpha=0.18, label="PI 80%")
        if {"pi95_lo", "pi95_hi"}.issubset(df_pi.columns):
            ax.fill_between(df_pi["Date"], df_pi["pi95_lo"], df_pi["pi95_hi"],
                            color="#56B4E9", alpha=0.12, label="PI 95%")
        title = f"{meta['stage'].upper()} | {meta['entity']} | h={meta['h']}"
        ax.set_title(title)
        ax.set_xlabel("Date"); ax.set_ylabel("Price")
        ax.legend(loc="best"); ax.grid(alpha=0.25, linestyle="--")
        fig.tight_layout()
        fig_path = plot_dir / f"pis_{meta['stage']}_{meta['entity'].replace('->','-')}_h{meta['h']}.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
        #log.info(f"[PIs] Figure: {fig_path}")

    stats = coverage_and_sharpness(df_pi)
    return df_pi, stats, meta


def discover_prediction_files(base_output: Path) -> List[Path]:
    
    """
        Discover all prediction CSV files generated by pipeline.py:
            • stage2_*.csv
            • direct_*.csv

        Files ending with '_with_PI.csv' are excluded to avoid double processing.

        Returns
        -------
        List[Path]
            Sorted list of files ready for PI computation.
        """
    files = []
    #for pat in ("stage1_*.csv", "stage2_*.csv", "direct_*.csv"):
    for pat in ("stage2_*.csv", "direct_*.csv"):
        files.extend((base_output).glob(pat))
    # filtra los _with_PI para no re-procesarlos
    files = [p for p in files if not p.name.endswith("_with_PI.csv")]
    return sorted(files)


def aggregate_coverage_across_files(
    files: List[Path],
    cfg_unc: Dict,
    base_output: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
        Process all prediction files, compute prediction intervals, and aggregate
        interval‑quality metrics into summary tables for reporting.

        Produces two tables:
            1. coverage_table : empirical coverage and Wilson CI for each pipeline/model.
            2. width_table    : distribution statistics of PI widths (p50, p90).

        This function is used to build uncertainty dashboards and Excel reports.
    """
    pis_dir = base_output / "pis"
    cov_rows, width_rows = [], []
    for p in files:
        df_pi, stats, meta = process_prediction_file(
            path_in=p,
            out_dir=pis_dir,
            levels=tuple(cfg_unc.get("alpha_levels", (0.20, 0.05))),
            n_boot=int(cfg_unc.get("n_boot", 1000)),
            recenter=bool(cfg_unc.get("recenter", True)),
            random_state=int(cfg_unc.get("random_state", 42)),
            make_plots=bool(cfg_unc.get("make_plots", True)),
        )

        # Summary row for coverage and mean width
        cov_rows.append({          
            "Pipeline ID": meta["pipeline_id"],
            "Stage/Model": meta["entity"],
            "Horizont": meta["h"],
            "Residual Method": f"Empirical bootstrap (recenter={bool(cfg_unc.get('recenter', True))})",
            "PI 80% Coverage (%)": stats.get("coverage_80", np.nan),
            "PI 80% CI95% lo": stats.get("coverage_80_lo95", np.nan),
            "PI 80% CI95% hi": stats.get("coverage_80_hi95", np.nan),
            "PI 95% Coverage (%)": stats.get("coverage_95", np.nan),
            "PI 95% CI95% lo": stats.get("coverage_95_lo95", np.nan),
            "PI 95% CI95% hi": stats.get("coverage_95_hi95", np.nan),
            "Mean Width 80": stats.get("mean_width_80", np.nan),
            "Mean Width 95": stats.get("mean_width_95", np.nan),
            "N obs": stats.get("n_obs", np.nan),
            "Note n<8": "Indicative" if stats.get("n_obs", 0) < 8 else "",

        })

        # Width distribution statistics (p50, p90)
        for pct in (80, 95):
            lo, hi = f"pi{pct}_lo", f"pi{pct}_hi"
            if {lo, hi}.issubset(df_pi.columns):
                width = (df_pi[hi] - df_pi[lo]).dropna()
                width_rows.append({
                    "Pipeline ID": meta["pipeline_id"],
                    "Stage/Model": meta["entity"],
                    "Horizonte": meta["h"],
                    "PI": f"{pct}%",
                    "p50 width": float(width.quantile(0.50)),
                    "p90 width": float(width.quantile(0.90)),
                    "count": int(width.size),
                })

   
    cov_df = pd.DataFrame(cov_rows)
    width_df = pd.DataFrame(width_rows)

     # Ensure numeric ordering of Pipeline IDs.
    def _sort_pid(pid):
        import re
        m = re.match(r"P(\d+)", str(pid))
        return int(m.group(1)) if m else float("inf")

    cov_df = cov_df.sort_values(by="Pipeline ID", key=lambda s: s.map(_sort_pid))
    width_df = width_df.sort_values(by="Pipeline ID", key=lambda s: s.map(_sort_pid))

    return cov_df, width_df
