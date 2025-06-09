from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from hierarchy_embeddings.utils import load_hierarchy

ATTR_CHAIN = "embeddings.weight.tensor"  # where the (N, 3) tensor lives
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
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected 2D array, got {arr.ndim}D")
    if arr.shape[1] not in [2, 3]:
        raise ValueError(f"{path}: expected shape (N, 2) or (N, 3), got {arr.shape}")
    return arr


def rotate_3d(points: np.ndarray, angle_degrees: float, axis: str = 'z') -> np.ndarray:
    """Rotate 3D points around specified axis."""
    angle_rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    if axis == 'z':
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
    elif axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
    else:
        raise ValueError(f"Invalid axis: {axis}. Use 'x', 'y', or 'z'")
    
    return points @ rotation_matrix.T


# ───────────────────────── Plotting helpers ───────────────────────────────────

def get_node_color_and_marker(node_name: str, cmap):
    """Get color and marker for a node based on its name."""
    idx = None
    for cat in CATEGORIES:
        if cat in node_name:
            idx = CATEGORIES.index(cat) - 1  # -1 to skip "Background"
            break
    
    if idx is None:
        return "black", "x"  # default for unclassified nodes
    elif idx == -1:
        return "black", 'o'  # Background
    else:
        return cmap(idx), 'o'


def draw_reference_circle(ax, radius: float, alpha: float = 1.0):
    """Draw a reference circle on the given axis."""
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), 
            color="grey", linewidth=1, alpha=alpha)


def draw_edges(ax, embeds_2d: np.ndarray, hierarchy):
    """Draw edges between connected nodes."""
    for edge in hierarchy.edges():
        source_idx, target_idx = edge[0], edge[1]
        ax.plot([embeds_2d[source_idx, 0], embeds_2d[target_idx, 0]], 
                [embeds_2d[source_idx, 1], embeds_2d[target_idx, 1]], 
                color='lightgrey', linewidth=0.5, zorder=1)


def plot_nodes(ax, embeds_2d: np.ndarray, hierarchy, cmap, embeds_3d: np.ndarray = None, radius: float = None, annotate=True):
    """Plot nodes with colors, markers, and annotations."""
    for i, (x, y) in enumerate(embeds_2d):
        node_name = hierarchy.nodes[i]["label"]
        color, marker = get_node_color_and_marker(node_name, cmap)
        
        # Calculate point size (for 3D depth perception)
        if embeds_3d is not None and radius is not None:
            z = embeds_3d[i, 2]
            size = 30 + 20 * (z / radius)
            size = max(10, size)
            alpha = 0.8
        else:
            size = 30
            alpha = 1.0
        
        ax.scatter(x, y, color=color, s=size, zorder=3, marker=marker, alpha=alpha)
        if annotate and "-" not in node_name:  # no subpart annotations
            ax.annotate(
                node_name,
                (x, y),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=4 if embeds_3d is None else 3,
            )


def setup_axis(ax, radius: float, title: str = None):
    """Setup axis properties with margins and styling."""
    margin = radius * 1.1
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)


# ───────────────────────── Plotting core ───────────────────────────────────

def plot_on_circle_2d(embeds: np.ndarray, hierarchy, out_file: Path):
    """Plot 2D embeddings on a circle."""
    radius = np.linalg.norm(embeds, axis=1).max()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=600)
    
    # Get colormap
    n = len(CATEGORIES) - 1
    cmap = cm.get_cmap("rainbow", n)
    
    # Draw components
    draw_reference_circle(ax, radius)
    draw_edges(ax, embeds, hierarchy)
    plot_nodes(ax, embeds, hierarchy, cmap)
    setup_axis(ax, radius)
    
    fig.tight_layout()
    fig.savefig(out_file)
    print(f"Saved {out_file}")


def plot_3d_views(embeds: np.ndarray, hierarchy, out_path: Path):
    num_views = 18
    rows = 3
    views_per_row = (num_views // rows)
    fig, axes = plt.subplots(rows, num_views//rows, figsize=(18, 9), dpi=300)
    
    radius = np.linalg.norm(embeds, axis=1).max()
    rotation_angles = np.linspace(0, 360, num=num_views, endpoint=False)
    view_titles = [f"View {i} ({angle}°)" for i, angle in enumerate(rotation_angles)]
    
    # Get colormap
    n = len(CATEGORIES) - 1
    cmap = cm.get_cmap("rainbow", n)
    
    for view_idx, (angle, title) in enumerate(zip(rotation_angles, view_titles)):
        ax = axes[view_idx // views_per_row][view_idx % views_per_row]
        
        # Rotate and project to 2D
        rotated_embeds = rotate_3d(embeds, angle, axis='y')
        
        # Filter to only show embeddings with non-negative 3rd component
        visible_mask = rotated_embeds[:, 2] >= 0
        visible_indices = np.where(visible_mask)[0]
        filtered_embeds = rotated_embeds[visible_mask]
        embeds_2d = filtered_embeds[:, :2]
        
        # Create index mapping for filtered data
        old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(visible_indices)}
        
        # Filter hierarchy to only include edges between visible nodes
        filtered_edges = []
        for edge in hierarchy.edges():
            source_idx, target_idx = edge[0], edge[1]
            if source_idx in old_to_new_idx and target_idx in old_to_new_idx:
                filtered_edges.append((old_to_new_idx[source_idx], old_to_new_idx[target_idx]))
        
        # Create filtered hierarchy-like object for drawing
        class FilteredHierarchy:
            def __init__(self, edges, nodes, visible_indices):
                self._edges = edges
                self._nodes = {old_to_new_idx[old_idx]: hierarchy.nodes[old_idx] 
                              for old_idx in visible_indices if old_idx in old_to_new_idx}
            def edges(self):
                return self._edges
            @property
            def nodes(self):
                return self._nodes
        
        filtered_hierarchy = FilteredHierarchy(filtered_edges, hierarchy.nodes, visible_indices)
        
        # Draw components
        draw_reference_circle(ax, radius, alpha=0.5)
        draw_edges(ax, embeds_2d, filtered_hierarchy)
        plot_nodes(ax, embeds_2d, filtered_hierarchy, cmap, filtered_embeds, radius, annotate=False)
        setup_axis(ax, radius, title)
    
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved 3D views to {out_path}")


def plot_on_circle(embeds: np.ndarray, hierarchy, out_file: Path):
    """Main plotting function that handles both 2D and 3D embeddings."""
    if embeds.shape[1] == 2:
        plot_on_circle_2d(embeds, hierarchy, out_file)
    elif embeds.shape[1] == 3:
        base_name = out_file.stem
        out_path_3d = out_file.parent / f"{base_name}_3d_views.png"
        plot_3d_views(embeds, hierarchy, out_path_3d)
    else:
        raise ValueError(f"Unsupported embedding dimension: {embeds.shape[1]}")


# ────────────────────────────── Main ───────────────────────────────────────

def main():
    # Load hierarchy
    hierarchy = load_hierarchy("spin_dataset", "spin_hierarchy")
    
    # Load embeddings
    path = Path(
        "hierarchy_embeddings/hierarchies/hierarchy_embeddings/experiments/"
        "spin_dataset/spin_hierarchy/2025-05-11_151726"
    )
    embeds = load_embeddings(path / "HierarchyEmbedding_weights_2.pth")
    
    # Plot (handles both 2D and 3D automatically)
    plot_on_circle(embeds, hierarchy, path / "embedding_circle.png")


if __name__ == "__main__":
    main()
