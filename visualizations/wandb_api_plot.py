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
#     ([393, 394], "Quadruped. Euclidian. dim=4"),
#     ([395, 396], "Quadruped. Euclidian. dim=16"),
#     ([386, 387], "Quadruped. Standard hier. dim=4"),
#     ([390, 388], "Quadruped. Part-first hier. dim=4"),
#     ([391, 389], "Quadruped. Part-first hier. dim=8"),
#
#     ([504, 503], "Bird. Euclidian. dim=4"),
#     ([505, 506], "Bird. Euclidian. dim=16"),
#     ([499, 500], "Bird. Standard hier. dim=4"),
#     ([501, 502], "Bird. Part-first hier. dim=4"),
#     ([507, 508], "Bird. Part-first hier. dim=8"),
# ]

TITLE = "Embedding loss power ablation"
RUN_GROUPS = [
    ([143, 177], "Power=1, random, dim 64"),
    ([179, 271, 273], "Power=3, random, dim 64"),
    ([181, 182], "Power=5, random, dim 64"),
    ([495, 496], "Power=3, random, dim 8"),
    ([212, 213], "Power=3, random, dim 4"),
    ([466, 467, 472, 473], "Power=3, tree-aware, dim 8"),
    ([487, 488], "Power=1, tree-aware, dim 64"),
    ([486, 485], "Power=3, tree-aware, dim 64"),
]

# TITLE = "Part-first vs standard hierarchy"
# RUN_GROUPS = [
#     ([160, 174], "Baseline, dim=64"),
#     ([272, 274], "Standard hierarchy, dim=64"),
#     ([292, 293], "Part-first hierarchy, dim=64"),
#     ([342], "Baseline, dim=4"),
#     ([313, 314], "Standard hierarchy, dim=4"),
#     ([311, 312], "Part-first hierarchy, dim=4"),
# ]

# TITLE = "PASCAL VOC 2012 OOD performance"
# RUN_GROUPS = [
#     ([427, 428, 434, 435], "Euclidian"),
#     ([430, 431, 432], "Hyperbolic"),  # 2 runs for 432
# ]

# TITLE = "Tau parameter"
# RUN_GROUPS = [
#     ([446, 447, 448, 449, 454, 455, 456, 457, 458], "Tau ablation"),
# ]

# TITLE = "Shared encoder"
# RUN_GROUPS: Optional[List[tuple]] = [
#     ([442, 443, 481, 482], "Baseline"),
#     ([475, 476, 477, 478], "Ours"),
# ]

# TITLE = "Subpart only"
# RUN_GROUPS: Optional[List[tuple]] = [
#     ([484, 483], "Euclidian"),
#     ([466, 467, 472, 473], "Hyperbolic"),
# ]

# For hyperparameter plots - set PLOT_TYPE to "hyperparameter" and configure below
# TITLE = "Effect of Tau on validation mIoU"
# HYPERPARAMETER_RUNS = [123, 124, 125, 126, 127]  # runs with different tau values
# HYPERPARAMETER_KEY = "tau"  # config key for the hyperparameter
# HYPERPARAMETER_METRIC = "val_mIoU_part"  # metric to plot on Y-axis

PLOT_TYPE  = "bar"                     # "line", "bar", or "hyperparameter"
METRIC     = "test_mIoU_subpart"            # metric for line plots
BAR_METRIC = "test_mIoU_subpart"        # metric for bar charts
HYPERPARAMETER_KEY = "tau"             # hyperparameter config key for hyperparameter plots
HYPERPARAMETER_METRIC = "test_mIoU_whole"  # metric for hyperparameter plots
HYPERPARAMETER_RUNS = [461, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474]               # list of run IDs for hyperparameter plots
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
    if PLOT_TYPE == "hyperparameter":
        all_ids = set(HYPERPARAMETER_RUNS)
    else:
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
    if PLOT_TYPE != "hyperparameter":
        for group in groups:
            for run_id in group.run_ids:
                id_to_group[run_id] = group.name

    # Determine which metrics to fetch
    if PLOT_TYPE == "bar":
        metrics_to_fetch = [STEP_KEY, BAR_METRIC]
    elif PLOT_TYPE == "hyperparameter":
        metrics_to_fetch = [STEP_KEY, HYPERPARAMETER_METRIC]
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
        
        # Add hyperparameter value for hyperparameter plots
        if PLOT_TYPE == "hyperparameter":
            hyperparameter_value = run.config.get(HYPERPARAMETER_KEY, None)
            if hyperparameter_value is None:
                print(f"Warning: Run {run_id} missing hyperparameter '{HYPERPARAMETER_KEY}', skipping...")
                continue
            hist[HYPERPARAMETER_KEY] = hyperparameter_value
        
        # hist["ratio_loss_weight"] = run.config.get("ratio_loss_weight", 0)
        # if hyper:
        #     effective_dim = int(re.search(r'weights_(\d+)_only', run.config["embeddings_path"]).group(1))
        # else:
        #     effective_dim = run.config["head_dim"]
        # hist["effective_dim"] = effective_dim
        
        if PLOT_TYPE != "hyperparameter":
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
    elif PLOT_TYPE == "hyperparameter":
        plot_hyperparameter_chart(df)
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

        run_values = group_data.groupby("run_name").agg({
            BAR_METRIC: 'max'
        })
        group_stats.append({
            'mean': run_values[BAR_METRIC].mean(),
            'std': run_values[BAR_METRIC].std() if len(run_values) > 1 else 0
        })
        group_names.append(group.name)
        print(f"{group.name} - {BAR_METRIC}: ${group_stats[-1]['mean']:.4f} \pm {group_stats[-1]['std']:.4f}$")
    
    means = [stat['mean'] for stat in group_stats]
    stds = [stat['std'] for stat in group_stats]
    
    # Create bars with different colors
    colors = sns.color_palette(None, len(group_names))
    bars = plt.bar(range(len(group_names)), means, yerr=stds, capsize=5, color=colors)
    plt.xticks(range(len(group_names)), group_names, rotation=45)
    plt.ylabel(f"{BAR_METRIC} (mean ± std)")

    plt.title(TITLE)
    plt.tight_layout()


def plot_hyperparameter_chart(df: pd.DataFrame) -> None:
    """Plot hyperparameter vs metric chart."""
    # Get the best (max) metric value for each run
    run_best_values = df.groupby("run_number").agg({
        HYPERPARAMETER_METRIC: 'max',
        HYPERPARAMETER_KEY: 'first'  # hyperparameter value should be constant per run
    }).reset_index()
    
    # Sort by hyperparameter value for a clean line plot
    run_best_values = run_best_values.sort_values(HYPERPARAMETER_KEY)
    print(f"{HYPERPARAMETER_KEY} = {run_best_values[HYPERPARAMETER_KEY].tolist()}")
    print(f"{HYPERPARAMETER_METRIC} = {run_best_values[HYPERPARAMETER_METRIC].tolist()}")
    
    # Plot the points and line
    plt.plot(run_best_values[HYPERPARAMETER_KEY], run_best_values[HYPERPARAMETER_METRIC], 
             'o-', linewidth=2, markersize=8)
    
    # Add value labels on points
    for _, row in run_best_values.iterrows():
        plt.annotate(f'{row[HYPERPARAMETER_METRIC]:.3f}', 
                    (row[HYPERPARAMETER_KEY], row[HYPERPARAMETER_METRIC]),
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.xlabel(HYPERPARAMETER_KEY)
    plt.ylabel(f"Best {HYPERPARAMETER_METRIC}")
    plt.title(TITLE)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def main() -> None:
    groups = get_groups()
    df = fetch_history(groups)

    if PLOT_TYPE == "hyperparameter":
        run_names = HYPERPARAMETER_RUNS
        out_file = Path(f"plots/{TITLE} {run_names}.png")
    else:
        run_names = np.array(df['run_number'].unique()).tolist()
        out_file = Path(f"plots/{TITLE} {run_names}.png")
    
    plot_history(df, groups, out_file)


if __name__ == "__main__":
    main()
