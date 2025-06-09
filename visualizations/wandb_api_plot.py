from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
import wandb                      # expects WANDB_API_KEY env variable
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# TARGET_IDS = {175, 180, 181, 182, 203}  # dim 64  (almost)
# TARGET_IDS = {204, 205, 208, 209}  # dim 16
# TARGET_IDS = {210, 211, 212, 213}  # dim 4

# TITLE = "Zero-shot segmentation performance"
# RUN_GROUPS = [
#     ([384, 385], "Euclidian. dim=4"),
#     ([386, 387], "Standard hier. dim=4"),
#     ([390, 388], "Part-first hier. dim=4"),
#     ([391, 389], "Part-first hier. dim=8"),
# ]

TITLE = "Embedding loss power ablation"
RUN_GROUPS = [
    ([177], "Power=1"),
    ([179, 271, 273], "Power=3"),
    ([181, 182], "Power=5"),
]


PLOT_TYPE  = "bar"                     # "line" or "bar"
METRIC     = "val_mIoU_part"            # metric for line plots
BAR_METRIC = "test_mIoU_subpart"        # metric for bar charts
STEP_KEY   = "trainer/global_step"      # change if your step key differs
PROJECT    = "inverse_rl/hyperbolic-segmentation"
# --------------------------------------------------------------------------- #


@dataclass
class RunGroup:
    """Represents a group of runs to be analyzed together"""
    run_ids: List[int]
    name: str


def get_groups() -> List[RunGroup]:
    """Convert the RUN_GROUPS configuration into RunGroup objects"""
    if RUN_GROUPS is None:
        # Get all unique run IDs from the groups
        all_ids = []
        if RUN_GROUPS:
            for ids, _ in RUN_GROUPS:
                all_ids.extend(ids)
        return [RunGroup(all_ids, "all")]
    
    return [RunGroup(ids, name) for ids, name in RUN_GROUPS]


def fetch_history(groups: List[RunGroup]) -> pd.DataFrame:
    """
    Return a single DataFrame containing the history of metrics
    for all runs, plus run metadata and group information.
    """
    api = wandb.Api()

    # Collect all run IDs
    all_ids = set()
    for group in groups:
        all_ids.update(group.run_ids)

    # Build regex for all IDs
    regex_tail = "(" + "|".join(str(i) for i in all_ids) + r")$"

    runs = api.runs(
        PROJECT,
        filters={"display_name": {"$regex": regex_tail}},
    )

    frames: List[pd.DataFrame] = []

    # Create mapping of run_id to group_name
    id_to_group = {}
    for group in groups:
        for run_id in group.run_ids:
            id_to_group[run_id] = group.name

    # Determine which metrics to fetch
    if PLOT_TYPE == "bar":
        metrics_to_fetch = [BAR_METRIC]
        if STEP_KEY not in metrics_to_fetch:
            metrics_to_fetch.append(STEP_KEY)
    else:
        metrics_to_fetch = [STEP_KEY, METRIC]

    for run in runs:
        run_id = int(run.name.split("-")[-1])

        hyper = run.config.get("hyperbolic", None)

        hist = run.history(keys=metrics_to_fetch, pandas=True)
        if hist.empty:
            continue

        hist = hist[metrics_to_fetch].dropna()
        hist["run_number"] = run_id
        hist["run_name"] = run.name
        hist["hyperbolic"] = hyper
        hist["ratio_loss_weight"] = run.config.get("ratio_loss_weight", 0)
        # if hyper:
        #     effective_dim = int(re.search(r'weights_(\d+)_only', run.config["embeddings_path"]).group(1))
        # else:
        #     effective_dim = run.config["head_dim"]
        # hist["effective_dim"] = effective_dim
        hist["group"] = id_to_group.get(run_id, "ungrouped")

        frames.append(hist)

    if not frames:
        raise RuntimeError("No matching runs with the requested metric history.")

    return pd.concat(frames, ignore_index=True)


def plot_history(df: pd.DataFrame, groups: List[RunGroup], out_path: Path | None = None) -> None:
    """Plot learning curves or bar charts with group statistics if groups are provided."""
    sns.set(style="whitegrid")
    sns.set_palette(None)
    plt.figure(figsize=(10, 6))

    if PLOT_TYPE == "bar":
        plot_bar_chart(df, groups)
    else:
        plot_line_chart(df, groups)

    if out_path:
        plt.savefig(out_path, dpi=300)
        print(f"Plot saved to {out_path.resolve()}")
    else:
        plt.show()


def plot_line_chart(df: pd.DataFrame, groups: List[RunGroup]) -> None:
    """Plot line charts for learning curves."""
    for group in groups:
        group_data = df[df["group"] == group.name]

        # Calculate statistics per step
        stats = group_data.groupby(STEP_KEY)[METRIC].agg(['mean', 'std']).reset_index()

        # Plot mean line
        plt.plot(stats[STEP_KEY], stats['mean'], label=group.name, linewidth=2)

        # Plot standard deviation band
        plt.fill_between(
            stats[STEP_KEY],
            stats['mean'] - stats['std'],
            stats['mean'] + stats['std'],
            alpha=0.2
        )

    plt.legend(title="Groups")

    plt.title(TITLE)
    plt.xlabel(STEP_KEY.replace("_", " "))
    plt.ylabel(METRIC)
    plt.tight_layout()


def plot_bar_chart(df: pd.DataFrame, groups: List[RunGroup]) -> None:
    """Plot bar charts showing final/best metric values."""
        # Plot group statistics - mean and std for each group
    group_stats = []
    group_names = []
    
    for group in groups:
        group_data = df[df["group"] == group.name]
        # Get max value for each run, then calculate group statistics
        run_maxes = group_data.groupby("run_number")[BAR_METRIC].max()
        group_stats.append({
            'mean': run_maxes.mean(),
            'std': run_maxes.std() if len(run_maxes) > 1 else 0
        })
        group_names.append(group.name)
    
    means = [stat['mean'] for stat in group_stats]
    stds = [stat['std'] for stat in group_stats]
    
    # Create bars with different colors
    colors = sns.color_palette(None, len(group_names))
    bars = plt.bar(range(len(group_names)), means, yerr=stds, capsize=5, color=colors)
    plt.xticks(range(len(group_names)), group_names, rotation=45)
    plt.ylabel(f"{BAR_METRIC} (mean ± std)")

    plt.title(TITLE)
    plt.tight_layout()


def main() -> None:
    groups = get_groups()
    df = fetch_history(groups)
    run_names = np.array(df['run_number'].unique()).tolist()
    out_file = Path(f"plots/{TITLE} {run_names}.png")
    plot_history(df, groups, out_file)


if __name__ == "__main__":
    main()
