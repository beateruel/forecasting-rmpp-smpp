import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path



def bar_metrics(resultsDict):
    df = pd.DataFrame.from_dict(resultsDict)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pallette = plt.cm.get_cmap("tab20c", len(df.columns))
    colors = [pallette(x) for x in range(len(df.columns))]
    color_dict = dict(zip(df.columns, colors))
    fig = plt.figure(figsize=(20, 15))

    # MAE plot
    fig.add_subplot(2, 2, 1)
    df.loc["mae"].sort_values().plot(
        kind="bar",
        colormap="Paired",
        color=[color_dict.get(x, "#333333") for x in df.loc["mae"].sort_values().index],
    )
    plt.legend()
    plt.title("MAE Metric, lower is better")
    fig.add_subplot(2, 2, 2)
    df.loc["rmse"].sort_values().plot(
        kind="bar",
        colormap="Paired",
        color=[
            color_dict.get(x, "#333333") for x in df.loc["rmse"].sort_values().index
        ],
    )
    plt.legend()
    plt.title("RMSE Metric, lower is better")
    fig.add_subplot(2, 2, 3)
    df.loc["mape"].sort_values().plot(
        kind="bar",
        colormap="Paired",
        color=[
            color_dict.get(x, "#333333") for x in df.loc["mape"].sort_values().index
        ],
    )
    plt.legend()
    plt.title("MAPE Metric, lower is better")
    fig.add_subplot(2, 2, 4)
    df.loc["r2"].sort_values(ascending=False).plot(
        kind="bar",
        colormap="Paired",
        color=[
            color_dict.get(x, "#333333")
            for x in df.loc["r2"].sort_values(ascending=False).index
        ],
    )
    plt.legend()
    plt.title("R2 Metric, higher is better")
    plt.tight_layout()
    plt.savefig("results/metrics.png")
    plt.show()

# -*- coding: utf-8 -*-

"""
Plotting utilities for model comparison with prediction intervals.
"""


def plot_comparison_with_baseline(
    baseline_file: str,
    compare_file: str,
    pi: int = 95,
    title: str | None = None
):
    """
    Plot SMPP forecast comparison between a direct model and a pipeline model,
    including empirical prediction intervals.

    Parameters
    ----------
    baseline_file : str
        Path to baseline CSV file (e.g. direct_SARIMAX_h6_with_PI.csv)

    compare_file : str
        Path to model CSV to compare (e.g. stage2_ARIMA_Linear_h6_with_PI.csv)

    pi : int, optional
        Prediction interval level (80 or 95). Default = 95

    title : str, optional
        Custom plot title
    """

    if pi not in (80, 95):
        raise ValueError("pi must be 80 or 95")

    # -------------------------------------------------
    # Load CSV files
    # -------------------------------------------------
    df_base = pd.read_csv(baseline_file, parse_dates=["Date"])
    df_cmp = pd.read_csv(compare_file, parse_dates=["Date"])

    # -------------------------------------------------
    # Merge by date (robust alignment)
    # -------------------------------------------------
    df = (
        df_cmp.merge(
            df_base,
            on="Date",
            suffixes=("_cmp", "_base"),
            how="inner"
        )
        .sort_values("Date")
    )

    # -------------------------------------------------
    # Infer names and horizon from filenames
    # -------------------------------------------------
    base_name = Path(baseline_file).stem.replace("_with_PI", "")
    cmp_name  = Path(compare_file).stem.replace("_with_PI", "")

    h_match = re.search(r"_h(\d+)", compare_file)
    horizon = f"h = {h_match.group(1)}" if h_match else ""

    # PI columns
    pi_lo = f"pi{pi}_lo"
    pi_hi = f"pi{pi}_hi"

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    plt.figure(figsize=(12, 6))

    # Observed SMPP
    plt.plot(
        df["Date"],
        df["y_true_cmp"],
        color="black",
        linewidth=2,
        label="Observed SMPP"
    )

    # Candidate model
    plt.plot(
        df["Date"],
        df["y_pred_cmp"],
        color="tab:blue",
        linewidth=2,
        label=cmp_name
    )

    plt.fill_between(
        df["Date"],
        df[f"{pi_lo}_cmp"],
        df[f"{pi_hi}_cmp"],
        color="tab:blue",
        alpha=0.15,
        label=f"{pi}% PI ({cmp_name})"
    )

    # Baseline model
    plt.plot(
        df["Date"],
        df["y_pred_base"],
        color="tab:orange",
        linestyle="--",
        linewidth=2,
        label=base_name
    )

    plt.fill_between(
        df["Date"],
        df[f"{pi_lo}_base"],
        df[f"{pi_hi}_base"],
        color="tab:orange",
        alpha=0.15,
        label=f"{pi}% PI ({base_name})"
    )

    # -------------------------------------------------
    # Formatting
    # -------------------------------------------------
    if title is None:
        title = f"SMPP forecast comparison ({horizon})"

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("SMPP")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()