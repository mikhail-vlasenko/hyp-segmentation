from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch

ATTR_CHAIN = "embeddings.weight.tensor"  # where the (N, 2) tensor lives

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

def plot_on_circle(embeds: np.ndarray, out_file: Path):
    radius = np.linalg.norm(embeds, axis=1).max()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=600)  # high‑res output
    ax.set_aspect("equal", "box")

    # reference circle
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), color="grey", linewidth=1)

    # points
    n = embeds.shape[0]
    cmap = cm.get_cmap("rainbow", n)  # rainbow spectrum
    for i, (x, y) in enumerate(embeds):
        ax.scatter(x, y, color=cmap(i / (n - 1) if n > 1 else 0.5), s=30, zorder=3)
        ax.annotate(
            str(i),
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
    PATH = Path(
        "hierarchy_embeddings/hierarchies/hierarchy_embeddings/experiments/"
        "spin_dataset/spin_hierarchy/2025-04-28_232541/HierarchyEmbedding_weights_2.pth"
    )
    embeds = load_embeddings(PATH)
    plot_on_circle(embeds, Path("visualizations/plots/embedding_circle.png"))


if __name__ == "__main__":
    main()
