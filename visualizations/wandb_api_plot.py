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
# Define run groups as tuples of (run_ids, group_name)
# Set to None to plot individual runs without grouping
RUN_GROUPS = [
    ([162, 165], "baseline"),
    ([272, 274], "standard hierarchy"),
    ([292, 293], "part-first hierarchy"),
]
# RUN_GROUPS = None  # Uncomment to plot individual runs

METRIC     = "val_mIoU_part"
STEP_KEY   = "trainer/global_step"      # change if your step key differs
PROJECT    = "inverse_rl/hyperbolic-segmentation"
OUT_FILE   = Path("plots/wandb_api.png")
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
    Return a single DataFrame containing the history of `METRIC`
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

    for run in runs:
        run_id = int(run.name.split("-")[-1])

        hyper = run.config.get("hyperbolic", None)

        hist = run.history(keys=[STEP_KEY, METRIC], pandas=True)
        if hist.empty:
            continue

        hist = hist[[STEP_KEY, METRIC]].dropna()
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
    """Plot learning curves with group statistics if groups are provided."""
    sns.set(style="whitegrid")
    sns.set_palette("Accent")
    plt.figure(figsize=(10, 6))

    if len(groups) == 1 and groups[0].name == "all":
        # Plot individual runs
        df = df.sort_values(["run_number", STEP_KEY])
        
        # Plot runs without ratio loss
        sns.lineplot(
            data=df.query("ratio_loss_weight == 0"),
            x=STEP_KEY,
            y=METRIC,
            hue="run_number",
            style="hyperbolic",
            dashes={True: "", False: (3, 3)},
            markers=False,
            linewidth=2,
        )

        # Plot runs with ratio loss
        sns.lineplot(
            data=df.query("ratio_loss_weight > 0"),
            x=STEP_KEY,
            y=METRIC,
            hue="run_number",
            style="hyperbolic",
            dashes={True: "", False: (3, 3)},
            markers=False,
            linewidth=3.5,
        )
        
        plt.legend(title="Run # / hyperbolic")
    else:
        # Plot group statistics
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

    plt.title(f"{METRIC} of runs {np.array(df['run_number'].unique()).tolist()}")
    plt.xlabel(STEP_KEY.replace("_", " "))
    plt.ylabel(METRIC)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=300)
        print(f"Plot saved to {out_path.resolve()}")
    else:
        plt.show()


def main() -> None:
    groups = get_groups()
    df = fetch_history(groups)
    plot_history(df, groups, OUT_FILE)


if __name__ == "__main__":
    main()
