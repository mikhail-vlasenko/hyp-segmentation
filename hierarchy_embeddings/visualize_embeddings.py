from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from hierarchy_embeddings.utils import load_hierarchy

ATTR_CHAIN = "embeddings.weight.tensor"  # where the (N, 2) tensor lives
CATEGORIES = [
    "Background",
    "Quadruped", "Biped", "Fish", "Bird", "Snake",
    "Reptile", "Car", "Bicycle", "Boat", "Aeroplane", "Bottle",
]

# ─────────────────────────── Utilities ─────────────────────────────────────

def _resolve_attr(obj, chain: str):
    """Follow dotted *chain* through attributes **or** dict keys."""
    cur = obj
    for part in chain.split('.'):  # type: ignore[assignment]
        cur = cur[part] if isinstance(cur, dict) else getattr(cur, part)
    return cur


def load_embeddings(path: Path) -> np.ndarray:
    tensor = _resolve_attr(torch.load(path, map_location="cpu"), ATTR_CHAIN)
    arr = tensor.detach().cpu().numpy()
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{path}: expected shape (N, 2), got {arr.shape}")
    return arr


# ───────────────────────── Plotting core ───────────────────────────────────

def plot_on_circle(embeds: np.ndarray, hierarchy, out_file: Path):
    radius = np.linalg.norm(embeds, axis=1).max()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=600)  # high‑res output
    ax.set_aspect("equal", "box")

    # reference circle
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), color="grey", linewidth=1)

    # Draw edges first so they appear behind points
    for edge in hierarchy.edges():
        source_idx = edge[0]
        target_idx = edge[1]
        ax.plot([embeds[source_idx, 0], embeds[target_idx, 0]], 
                [embeds[source_idx, 1], embeds[target_idx, 1]], 
                color='lightgrey', linewidth=0.5, zorder=1)

    # points
    n = len(CATEGORIES) - 1
    cmap = cm.get_cmap("rainbow", n)  # rainbow spectrum
    for i, (x, y) in enumerate(embeds):
        node_name = hierarchy.nodes[i]["label"]
        idx = None
        for cat in CATEGORIES:
            if cat in node_name:
                idx = CATEGORIES.index(cat) - 1  # -1 to skip "Background"
                break
        if idx is None:
            color = "black"  # default color for unclassified nodes
            marker = "x"  # different marker for unclassified nodes
        elif idx == -1:
            color = "black"  # Background color
            marker = 'o'
        else:
            color = cmap(idx)
            marker = 'o'  # use circle marker for all other nodes
        ax.scatter(x, y, color=color, s=30, zorder=3, marker=marker)
        ax.annotate(
            node_name,  # node name
            (x, y),
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=4,  # much smaller text
        )

    margin = radius * 1.1
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_file)
    print(f"Saved {out_file}")


# ────────────────────────────── Main ───────────────────────────────────────

def main():
    # Load hierarchy
    hierarchy = load_hierarchy("spin_dataset", "spin_hierarchy_part-first")
    
    # Load embeddings
    path = Path(
        "hierarchy_embeddings/hierarchies/hierarchy_embeddings/experiments/"
        "spin_dataset/spin_hierarchy_part-first/2025-05-22_163008"
    )
    embeds = load_embeddings(path / "HierarchyEmbedding_weights_2.pth")
    
    # Plot with edges
    plot_on_circle(embeds, hierarchy, path / "embedding_circle.png")


if __name__ == "__main__":
    main()
