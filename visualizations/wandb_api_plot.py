#!/usr/bin/env python3
"""
Plot the full val_mIoU_subpart learning curves for runs 204, 205, 208, 209,
highlighting the “hyperbolic” config flag.

Usage:
    python plot_wandb_miou.py          # saves to plots/wandb_api.png
"""
import re
from pathlib import Path
from typing import List

import numpy as np
import wandb                      # expects WANDB_API_KEY env variable
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
TARGET_IDS = {175, 180, 181, 182, 203}  # dim 64  (almost)
# TARGET_IDS = {204, 205, 208, 209}  # dim 16
# TARGET_IDS = {210, 211, 212, 213}  # dim 4
METRIC     = "val_mIoU_subpart"
STEP_KEY   = "trainer/global_step"      # change if your step key differs
PROJECT    = "inverse_rl/hyperbolic-segmentation"
OUT_FILE   = Path("plots/wandb_api.png")
# --------------------------------------------------------------------------- #


def fetch_history() -> pd.DataFrame:
    """
    Return a single DataFrame containing the history of `METRIC`
    for the four runs, plus run metadata (run_number, run_name, hyperbolic).
    """
    api = wandb.Api()

    # Build a regex to match “…-204” OR “…-205” OR “…-208” OR “…-209”
    regex_tail = "(" + "|".join(str(i) for i in TARGET_IDS) + r")$"

    runs = api.runs(
        PROJECT,
        filters={"display_name": {"$regex": regex_tail}},
    )

    frames: List[pd.DataFrame] = []

    for run in runs:
        try:
            run_id = int(run.name.split("-")[-1])
        except ValueError:
            # Shouldn't happen thanks to the filter, but guard anyway
            continue

        hyper = run.config.get("hyperbolic", None)

        # Download only the columns we need. This keeps it light.
        hist = run.history(keys=[STEP_KEY, METRIC], pandas=True)
        if hist.empty:
            continue

        # Keep just the two columns, drop NaNs, and annotate metadata
        hist = hist[[STEP_KEY, METRIC]].dropna()
        hist["run_number"] = run_id
        hist["run_name"]   = run.name
        hist["hyperbolic"] = hyper
        hist["ratio_loss_weight"] = run.config.get("ratio_loss_weight", 0)
        if hyper:
            effective_dim = int(re.search(r'weights_(\d+)_only', run.config["embeddings_path"]).group(1))
        else:
            effective_dim = run.config["head_dim"]
        hist["effective_dim"] = effective_dim

        frames.append(hist)

    if not frames:
        raise RuntimeError("No matching runs with the requested metric history.")

    df = pd.concat(frames, ignore_index=True)
    if len(np.unique(df["effective_dim"])) > 1:
        print("Warning: more than one effective_dim found in the history.")
        print(df["effective_dim"])

    return df


def plot_history(df: pd.DataFrame, out_path: Path | None = None) -> None:
    """Plot learning curves with Seaborn (Accent palette)."""
    sns.set(style="whitegrid")      # clean background
    sns.set_palette("Accent")       # requested palette
    plt.figure(figsize=(10, 6))

    # order runs left‑to‑right by run_number in the legend
    df = df.sort_values(["run_number", STEP_KEY])

    ax = sns.lineplot(
        data=df.query("ratio_loss_weight == 0"),
        x=STEP_KEY,
        y=METRIC,
        hue="run_number",           # colour = individual run
        style="hyperbolic",         # solid vs dashed
        dashes={True: "", False: (3, 3)},  # ← solid for hyperbolic, dashed otherwise
        markers=False,
        linewidth=2,
    )

    sns.lineplot(
        data=df.query("ratio_loss_weight > 0"),
        x=STEP_KEY,
        y=METRIC,
        hue="run_number",  # colour = individual run
        style="hyperbolic",  # solid vs dashed
        dashes={True: "", False: (3, 3)},  # ← solid for hyperbolic, dashed otherwise
        markers=False,
        linewidth=3.5,
    )

    ax.set_title(f"{METRIC} for dim={df['effective_dim'].iloc[0]}. With Ratio Loss is thicker.")
    ax.set_xlabel(STEP_KEY.replace("_", " "))
    ax.set_ylabel(METRIC)
    ax.legend(title="Run # / hyperbolic")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
        print(f"Plot saved to {out_path.resolve()}")
    else:
        plt.show()


def main() -> None:
    df = fetch_history()
    plot_history(df, OUT_FILE)


if __name__ == "__main__":
    main()
