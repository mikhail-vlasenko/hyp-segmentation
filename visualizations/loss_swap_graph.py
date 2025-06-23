import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Tuple, List
import networkx as nx
import math

from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors import ManifoldTensor
from tqdm import tqdm

from hierarchy_embeddings.losses import distortion_loss

# Set style for better plots
plt.style.use('default')
sns.set_palette("RdBu_r")
Y_MIN = 0.5
X_LIM = 0.15
POWER = 3.0
SECTOR_ANGLE = np.pi / 6
CLIP_MAX = 0.2
CLIP_MIN = 0.1
side_height = 0.82
top_offset = 0.07
middle_height = 0.75
middle_offset = 0.05
GRID_SIZE = 100


MOVABLE_POS = [top_offset, side_height]

def setup_static_embeddings() -> Tuple[torch.Tensor, List[str]]:
    """Set up the 5 static embeddings in hyperbolic space within the unit disk, plus the root."""
    # Place embeddings within the Poincare disk (radius < 1)
    # Root (R) is at the center (0, 0)
    embeddings = torch.tensor([
        [-middle_offset, middle_height],  # center bottom (CB)
        [middle_offset, middle_height],   # right bottom (RB)
        [0.02, 0.83],   # center top (CT)
        [-top_offset, side_height],  # left top (LT)
        [0.0, 0.5],   # root R at center
    ], dtype=torch.float32)

    labels = ['center_bottom', 'right_bottom', 'center_top', 'left_top', 'root']
    return embeddings, labels


def construct_graph() -> Tuple[nx.Graph, List[Tuple[int, int]]]:
    """
    Construct the graph based on the tree structure:

    LT    M    CT
      \   /    |
       CB      RB
        \      /
           R

    Where: CB=center_bottom, RB=right_bottom, CT=center_top, LT=left_top, M=movable, R=root
    Node indices: [CB=0, RB=1, CT=2, LT=3, R=4, M=5]
    """
    G = nx.Graph()

    # Add nodes - now including root explicitly
    nodes = ['CB', 'RB', 'CT', 'LT', 'R', 'M']
    G.add_nodes_from(range(len(nodes)))

    # Add edges based on the tree structure
    edges = [
        (3, 0),  # LT -> CB
        (5, 0),  # M -> CB
        (2, 1),  # CT -> RB
        (0, 4),  # CB -> R
        (1, 4),  # RB -> R
    ]

    G.add_edges_from(edges)

    return G, edges


def get_distance_matrix_from_graph(G: nx.Graph) -> torch.Tensor:
    """Compute shortest path distances from the graph structure."""
    # Compute all pairs shortest paths
    distances = dict(nx.all_pairs_shortest_path_length(G))

    # Convert to matrix format (including root node R which is now index 4)
    # Keep all 6 nodes: [CB=0, RB=1, CT=2, LT=3, R=4, M=5]
    n_nodes = 6
    distance_matrix = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)

    for i in range(n_nodes):
        for j in range(n_nodes):
            distance_matrix[i, j] = distances[i][j]

    return distance_matrix


def get_graph_edges_only() -> List[Tuple[int, int]]:
    """Get only the edges that exist in the graph."""
    G, edges = construct_graph()
    return edges


def compute_loss_for_position(movable_pos: torch.Tensor, static_embeddings: torch.Tensor,
                              distance_matrix: torch.Tensor, manifold: PoincareBall, verbose=False) -> float:
    """Compute the distortion loss for a given movable embedding position."""
    # Combine static embeddings with movable position (movable is at index 5)
    all_embeddings = torch.cat([static_embeddings, movable_pos.unsqueeze(0)], dim=0)
    
    # Get all pairwise combinations of nodes (excluding self-pairs)
    n_nodes = all_embeddings.shape[0]
    pairs = []
    target_distances = []
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:  # Exclude self-pairs
                pairs.append((i, j))
                target_distances.append(distance_matrix[i, j].item())
    
    # Structure the data for distortion_loss: [B=1, N=num_pairs, 2, D]
    batch_size = 1
    num_pairs = len(pairs)
    embedding_dim = all_embeddings.shape[1]
    
    # Create pair embeddings tensor: [1, num_pairs, 2, embedding_dim]
    pair_embeddings = torch.zeros(batch_size, num_pairs, 2, embedding_dim)
    pair_distances = torch.tensor(target_distances).unsqueeze(0)  # [1, num_pairs]
    
    for idx, (i, j) in enumerate(pairs):
        pair_embeddings[0, idx, 0, :] = all_embeddings[i]  # source node
        pair_embeddings[0, idx, 1, :] = all_embeddings[j]  # target node
        if verbose:
            print(f"Pair {idx}: ({i} -> {j}) with graph distance {pair_distances[0, idx].item()}")

    # Create ManifoldTensor
    manifold_embeddings = ManifoldTensor(
        data=pair_embeddings,
        manifold=manifold,
        man_dim=-1,
    )

    if verbose:
        embedding_dists = manifold.dist(x=manifold_embeddings[:, :, 0, :], y=manifold_embeddings[:, :, 1, :])
        print("Hyperbolic distances:")
        print(embedding_dists[0])
        print("Graph distances:")
        print(pair_distances[0])
    
    # Compute loss
    loss = distortion_loss(manifold_embeddings, pair_distances, power=POWER)
    return loss.item()


def is_in_poincare_disk(x: float, y: float, margin: float = 0.01) -> bool:
    """Check if point is within the Poincare disk with some margin."""
    return (x**2 + y**2) < (1.0 - margin)**2


def compute_loss_map_core(coords_x: np.ndarray, coords_y: np.ndarray) -> Tuple[np.ndarray, torch.Tensor, List[str], torch.Tensor, PoincareBall]:
    """Core function to compute loss map given coordinate arrays."""
    static_embeddings, labels = setup_static_embeddings()

    # Construct graph and get distance matrix
    G, _ = construct_graph()
    distance_matrix = get_distance_matrix_from_graph(G)

    # Initialize the hyperbolic manifold
    manifold = PoincareBall(c=Curvature(value=1.0))

    # Initialize loss map
    loss_map = np.full(coords_x.shape, np.nan)

    print(f"Computing loss map for {coords_x.shape[0]}x{coords_x.shape[1]} grid...")
    
    for i in tqdm(range(coords_x.shape[0])):
        for j in range(coords_x.shape[1]):
            x = coords_x[i, j]
            y = coords_y[i, j]
            
            if is_in_poincare_disk(x, y):
                movable_pos = torch.tensor([x, y], dtype=torch.float32)
                loss = compute_loss_for_position(movable_pos, static_embeddings, distance_matrix, manifold)
                if np.isnan(loss):
                    print(f"NaN loss at position: ({x:.3f}, {y:.3f})")
                loss_map[i, j] = loss
            else:
                print(f"Skipping out-of-disk position: ({x:.3f}, {y:.3f})")

    return loss_map, static_embeddings, labels, distance_matrix, manifold


def generate_loss_map(sector_angle: float = np.pi / 2):
    """Generate the loss map by sampling positions in a sector of the hyperbolic disk."""

    # Generate coordinates in a sector of the disk
    # Use polar coordinates and convert to Cartesian
    max_radius = 0.87  # Very close to unit disk boundary
    # Add extra points for better coverage at boundaries
    angles = np.linspace(-sector_angle/2 + np.pi / 2, sector_angle/2 + np.pi / 2, GRID_SIZE)
    radii = np.linspace(Y_MIN, max_radius, GRID_SIZE)
    
    # Create coordinate meshgrids
    angle_grid, radius_grid = np.meshgrid(angles, radii, indexing='ij')
    coords_x = radius_grid * np.cos(angle_grid)
    coords_y = radius_grid * np.sin(angle_grid)
    
    # Compute loss map using core function
    loss_map, static_embeddings, labels, distance_matrix, manifold = compute_loss_map_core(coords_x, coords_y)

    return loss_map, coords_x, coords_y, static_embeddings, labels, distance_matrix, manifold


def generate_loss_map_region(x_min: float, x_max: float, y_min: float, y_max: float):
    """Generate a loss map for a specific rectangular region."""
    print(f"Setting up zoomed loss map for region ({x_min:.2f}-{x_max:.2f}, {y_min:.2f}-{y_max:.2f})...")
    
    # Generate coordinates in the specified rectangular region
    x_coords = np.linspace(x_min, x_max, GRID_SIZE)
    y_coords = np.linspace(y_min, y_max, GRID_SIZE)
    
    # Create coordinate meshgrids
    coords_x, coords_y = np.meshgrid(x_coords, y_coords, indexing='xy')
    
    # Compute loss map using core function
    loss_map, static_embeddings, labels, distance_matrix, manifold = compute_loss_map_core(coords_x, coords_y)

    return loss_map, coords_x, coords_y, static_embeddings, labels, distance_matrix, manifold


def plot_loss_map_with_graph():
    """Create and plot the loss map with graph edges in hyperbolic space."""
    # Generate main loss map
    loss_map, coords_x, coords_y, static_embeddings, labels, distance_matrix, manifold = generate_loss_map(sector_angle=SECTOR_ANGLE)
    
    # Generate zoomed loss map for region
    zoom_x_min, zoom_x_max = 0.0, 0.1
    zoom_y_min, zoom_y_max = 0.75, 0.85
    zoom_loss_map, zoom_coords_x, zoom_coords_y, _, _, _, _ = generate_loss_map_region(
        zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max
    )

    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # ========== MAIN PLOT (LEFT) ==========
    # Plot heatmap using pcolormesh for irregular grids
    if np.isnan(loss_map).any():
        print("Warning: loss map contains NaN values, check computation.")
    # Mask out NaN values
    masked_loss_map = np.ma.masked_invalid(loss_map)
    masked_loss_map = np.clip(masked_loss_map, CLIP_MIN, CLIP_MAX)
    im1 = ax1.pcolormesh(coords_x, coords_y, masked_loss_map, shading='auto', cmap='RdBu_r', alpha=0.8)

    # Add colorbar for main plot
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Distortion Loss')

    # Draw the unit circle boundary
    circle = plt.Circle((0, 0), 1.0, fill=False, color='black', linewidth=2, linestyle='--', alpha=0.5)
    ax1.add_patch(circle)

    # Plot static embeddings with different colors for root
    for i, (pos, label) in enumerate(zip(static_embeddings, labels)):
        ax1.scatter(pos[0], pos[1], c='black', s=100, marker='o',
                   edgecolors='black', linewidths=2, zorder=5)

    # Plot movable embedding at a reasonable position within the sector
    movable_pos = torch.tensor(MOVABLE_POS)  # within the sector
    ax1.scatter(movable_pos[0], movable_pos[1], c='purple', s=300, marker='*',
               zorder=5)

    # Add graph edges - plot all edges including those to/from root
    all_positions = torch.cat([static_embeddings, movable_pos.unsqueeze(0)], dim=0)
    edges = get_graph_edges_only()

    for i, j in edges:
        pos1, pos2 = all_positions[i], all_positions[j]
        ax1.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                'k--', alpha=0.6, linewidth=2, zorder=3)

        # Add distance labels on edges
        mid_x, mid_y = (pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2
        distance = distance_matrix[i, j].item()

    ax1.set_xlabel('X Coordinate (Poincaré Disk)')
    ax1.set_ylabel('Y Coordinate (Poincaré Disk)')
    ax1.set_title('Hyperbolic Distortion Loss Map')
    ax1.set_aspect('equal')
    
    # Set limits to show the unit disk properly
    ax1.set_xlim(-X_LIM, X_LIM)
    ax1.set_ylim(Y_MIN, 0.9)

    # Add zoom region rectangle
    zoom_rect = plt.Rectangle((zoom_x_min, zoom_y_min), 
                             zoom_x_max - zoom_x_min, 
                             zoom_y_max - zoom_y_min,
                             linewidth=2, edgecolor='white', facecolor='none', zorder=6)
    ax1.add_patch(zoom_rect)

    # ========== ZOOMED PLOT (RIGHT) ==========
    # Plot zoomed heatmap
    if np.isnan(zoom_loss_map).any():
        print("Warning: zoomed loss map contains NaN values, check computation.")
    # Mask out NaN values  
    masked_zoom_loss_map = np.ma.masked_invalid(zoom_loss_map)
    masked_zoom_loss_map = np.clip(masked_zoom_loss_map, 0.1, 0.15)
    im2 = ax2.pcolormesh(zoom_coords_x, zoom_coords_y, masked_zoom_loss_map, shading='auto', cmap='RdBu_r', alpha=0.8)

    # Add colorbar for zoomed plot
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Distortion Loss (Zoomed)')

    # Plot static embeddings that fall within the zoom region
    for i, (pos, label) in enumerate(zip(static_embeddings, labels)):
        if zoom_x_min <= pos[0] <= zoom_x_max and zoom_y_min <= pos[1] <= zoom_y_max:
            ax2.scatter(pos[0], pos[1], c='black', s=100, marker='o',
                       edgecolors='black', linewidths=2, zorder=5)

    # Plot movable embedding if it's in the zoom region
    if zoom_x_min <= movable_pos[0] <= zoom_x_max and zoom_y_min <= movable_pos[1] <= zoom_y_max:
        ax2.scatter(movable_pos[0], movable_pos[1], c='purple', s=300, marker='*',
                   zorder=5)

    # Add graph edges that pass through the zoom region
    for i, j in edges:
        pos1, pos2 = all_positions[i], all_positions[j]
        # Check if edge passes through or intersects the zoom region
        if ((zoom_x_min <= pos1[0] <= zoom_x_max and zoom_y_min <= pos1[1] <= zoom_y_max) or
            (zoom_x_min <= pos2[0] <= zoom_x_max and zoom_y_min <= pos2[1] <= zoom_y_max)):
            ax2.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                    'k--', alpha=0.6, linewidth=2, zorder=3)

    ax2.set_xlabel('X Coordinate (Poincaré Disk)')
    ax2.set_ylabel('Y Coordinate (Poincaré Disk)')
    ax2.set_title(f'Zoomed View: ({zoom_x_min:.1f}-{zoom_x_max:.1f}, {zoom_y_min:.2f}-{zoom_y_max:.2f})')
    ax2.set_aspect('equal')
    
    # Set limits for zoomed view
    ax2.set_xlim(zoom_x_min, zoom_x_max)
    ax2.set_ylim(zoom_y_min, zoom_y_max)

    plt.tight_layout()
    plt.savefig('plots/loss_swap_graph.png', dpi=300, bbox_inches='tight')

    # Print some statistics
    valid_losses = loss_map[~np.isnan(loss_map)]
    if len(valid_losses) > 0:
        print(f"\nMain Loss Map Statistics:")
        print(f"Minimum loss: {np.min(valid_losses):.4f}")
        print(f"Maximum loss: {np.max(valid_losses):.4f}")
        print(f"Mean loss: {np.mean(valid_losses):.4f}")

        # Find position of minimum loss
        min_idx = np.unravel_index(np.nanargmin(loss_map), loss_map.shape)
        min_x = coords_x[min_idx]
        min_y = coords_y[min_idx]
        print(f"Position of minimum loss: ({min_x:.3f}, {min_y:.3f})")
    else:
        print("No valid loss values computed.")

    # Print zoomed statistics
    valid_zoom_losses = zoom_loss_map[~np.isnan(zoom_loss_map)]
    if len(valid_zoom_losses) > 0:
        print(f"\nZoomed Loss Map Statistics:")
        print(f"Minimum loss: {np.min(valid_zoom_losses):.4f}")
        print(f"Maximum loss: {np.max(valid_zoom_losses):.4f}")
        print(f"Mean loss: {np.mean(valid_zoom_losses):.4f}")

        # Find position of minimum loss in zoomed region
        zoom_min_idx = np.unravel_index(np.nanargmin(zoom_loss_map), zoom_loss_map.shape)
        zoom_min_x = zoom_coords_x[zoom_min_idx]
        zoom_min_y = zoom_coords_y[zoom_min_idx]
        print(f"Position of minimum loss in zoomed region: ({zoom_min_x:.3f}, {zoom_min_y:.3f})")
    else:
        print("No valid loss values computed in zoomed region.")


if __name__ == "__main__":
    plot_loss_map_with_graph()
