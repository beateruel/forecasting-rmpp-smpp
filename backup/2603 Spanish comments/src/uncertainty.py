
# -*- coding: utf-8 -*-
"""
Uncertainty utilities: empirical residual bootstrap PIs, coverage & sharpness,
file discovery and plotting.
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
from .evaluation import bootstrap_intervals  # usaremos la versión extendida
log = get_logger(__name__)

# --- Helpers de meta extraídos del patrón de nombres que ya genera pipeline.py ---
# stage1_{MODEL}_h{H}.csv
PAT_STAGE1 = re.compile(r"stage1_(?P<model>[^_]+)_h(?P<h>\d+)\.csv$", re.IGNORECASE)
# stage2_{PIPELINE_W_UNDERSCORES}_h{H}.csv   (pipeline.py reemplaza '->' por '_' al guardar)
PAT_STAGE2 = re.compile(r"stage2_(?P<pipeline_und>.+)_h(?P<h>\d+)\.csv$", re.IGNORECASE)
# direct_{MODEL}_h{H}.csv
PAT_DIRECT = re.compile(r"direct_(?P<model>.+)_h(?P<h>\d+)\.csv$", re.IGNORECASE)

# Mapeo coherente con pipeline.py para etiquetas P1..P8
MAP_PIPELINE_TO_PID = {
    "SES->Linear": "P1",
    "ARIMA->Linear": "P2",
    "RF->Linear": "P3",
    "XGB->Linear": "P4",
    "LGBM->Linear": "P5",
    "XGB->XGB": "P6",
}
MAP_DIRECT_TO_PID = {"SARIMAX": "P7", "XGB_EXOG": "P8"}


def _extract_meta_from_filename(path: Path) -> Dict[str, str]:
    """Detecta stage, id, horizonte, y asigna Pipeline ID cuando aplique."""
    fname = path.name
    m1 = PAT_STAGE1.search(fname)
    if m1:
        model = m1.group("model")
        h = int(m1.group("h"))
        return {"stage": "stage1", "entity": model, "pipeline_id": model, "h": h}

    m2 = PAT_STAGE2.search(fname)
    if m2:
        pipe_und = m2.group("pipeline_und")
        # reconstruir 'SES->Linear' desde 'SES_Linear'
        pipe = pipe_und.replace("_", "->")
        h = int(m2.group("h"))
        pid = MAP_PIPELINE_TO_PID.get(pipe, pipe)  # si no está mapeado, usar texto
        return {"stage": "stage2", "entity": pipe, "pipeline_id": pid, "h": h}

    m3 = PAT_DIRECT.search(fname)
    if m3:
        model = m3.group("model")
        h = int(m3.group("h"))
        pid = MAP_DIRECT_TO_PID.get(model, model)
        return {"stage": "direct", "entity": model, "pipeline_id": pid, "h": h}

    return {"stage": "unknown", "entity": fname, "pipeline_id": fname, "h": -1}


def compute_pis_for_dataframe(
    df: pd.DataFrame,
    levels: Iterable[float] = (0.20, 0.05),
    n_boot: int = 1000,
    recenter: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Añade columnas de PI y anchuras a un DataFrame con [y_true, y_pred]."""
    assert {"y_true", "y_pred"}.issubset(df.columns), "Se requieren y_true, y_pred"
    out = df.copy()
    # Genera bandas por cada alpha solicitado (p.ej. 0.20 -> 80%)
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
    Devuelve cobertura empírica (%), anchura media y tamaño muestral (n_obs).
    Incluye intervalo de confianza (Wilson, 95%) para cada cobertura reportada.
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
    """Lee un CSV de predicciones, añade PIs, guarda CSV enriquecido y (opcional) figura."""
    meta = _extract_meta_from_filename(path_in)
    df = pd.read_csv(path_in, parse_dates=["Date"])
    df_pi = compute_pis_for_dataframe(
        df, levels=levels, n_boot=n_boot, recenter=recenter, random_state=random_state
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path_in.stem}_with_PI.csv"
    df_pi.to_csv(out_path, index=False)
    log.info(f"[PIs] Escrito: {out_path}")

    # (Opcional) fan-chart
    if make_plots:
        plot_dir = plot_dir or (out_dir.parent / "figs_pis")
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.plot(df_pi["Date"], df_pi["y_true"], color="#333", lw=1.8, label="Real")
        ax.plot(df_pi["Date"], df_pi["y_pred"], color="#0072B2", lw=1.6, label="Predicción")
        if {"pi80_lo", "pi80_hi"}.issubset(df_pi.columns):
            ax.fill_between(df_pi["Date"], df_pi["pi80_lo"], df_pi["pi80_hi"],
                            color="#0072B2", alpha=0.18, label="PI 80%")
        if {"pi95_lo", "pi95_hi"}.issubset(df_pi.columns):
            ax.fill_between(df_pi["Date"], df_pi["pi95_lo"], df_pi["pi95_hi"],
                            color="#56B4E9", alpha=0.12, label="PI 95%")
        title = f"{meta['stage'].upper()} | {meta['entity']} | h={meta['h']}"
        ax.set_title(title)
        ax.set_xlabel("Fecha"); ax.set_ylabel("Precio")
        ax.legend(loc="best"); ax.grid(alpha=0.25, linestyle="--")
        fig.tight_layout()
        fig_path = plot_dir / f"pis_{meta['stage']}_{meta['entity'].replace('->','-')}_h{meta['h']}.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
        log.info(f"[PIs] Figura: {fig_path}")

    stats = coverage_and_sharpness(df_pi)
    return df_pi, stats, meta


def discover_prediction_files(base_output: Path) -> List[Path]:
    """Encuentra todos los CSV de predicciones estándar en output/."""
    files = []
    for pat in ("stage1_*.csv", "stage2_*.csv", "direct_*.csv"):
        files.extend((base_output).glob(pat))
    # filtra los _with_PI para no re-procesarlos
    files = [p for p in files if not p.name.endswith("_with_PI.csv")]
    return sorted(files)


def aggregate_coverage_across_files(
    files: List[Path],
    cfg_unc: Dict,
    base_output: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Procesa todos los CSV, devuelve (coverage_table, width_stats) listos para Excel."""
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
        # Cobertura y anchura media
        cov_rows.append({          
            "Pipeline ID": meta["pipeline_id"],
            "Stage/Model": meta["entity"],
            "Horizonte": meta["h"],
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
            "Nota n<8": "Indicativa" if stats.get("n_obs", 0) < 8 else "",

        })
        # Stats de anchura (p50/p90)
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
    return cov_df, width_df
