import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from torch.utils.data import DataLoader

from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors import ManifoldTensor

from hierarchy_embeddings.hierarchy_datasets import HierarchyEmbeddingDataset
from hierarchy_embeddings.utils import load_hierarchy


def compute_distortion_matrix(hierarchy_embedding, hierarchy, dataset, manifold):
    """Compute the distortion matrix for a given embedding."""
    # Find the root and get BFS ordering
    root = [n for n, d in hierarchy.in_degree() if d == 0][0]
    node_order = [root] + [t for _, t in nx.bfs_edges(hierarchy, root)]
    
    # Re-order the embeddings according to the BFS
    embeddings = ManifoldTensor(
        data=hierarchy_embedding.embeddings.weight.tensor[node_order],
        manifold=manifold,
        man_dim=-1,
    )
    
    # Compute embedding distances
    embeddings_dist_mat = manifold.dist(embeddings[:, None, :], embeddings).detach()
    
    # Get graph distances
    graph_dist_mat = dataset.sampler.dist_matrix[node_order, :][:, node_order]
    
    # Compute distortion matrix
    # Handle the case where graph_dist_mat is 0 (diagonal elements)
    mask_zero_target = (graph_dist_mat == 0)
    
    # For non-zero target distances, compute relative distortion
    distortion_mat = torch.zeros_like(embeddings_dist_mat)
    non_zero_mask = ~mask_zero_target
    distortion_mat[non_zero_mask] = ((embeddings_dist_mat[non_zero_mask] - graph_dist_mat[non_zero_mask]) / graph_dist_mat[non_zero_mask]).abs()
    
    # For zero target distances, distortion should be 0 (since both should be 0)
    distortion_mat[mask_zero_target] = 0
    
    return distortion_mat, node_order


def plot_distortion_difference(diff_mat, node_permutation, output_path):
    """Plot the difference in distortion matrices."""
    df_cm = pd.DataFrame(
        diff_mat[:len(node_permutation), :len(node_permutation)],
        node_permutation,
        node_permutation,
    )

    fig = plt.figure(figsize=(12, 10), dpi=300)

    # Use a diverging colormap to show positive and negative differences
    sn.heatmap(df_cm, cmap='RdBu_r', shading='auto', center=0)

    plt.title('Difference in Distortions (Loss Power 3 - Standard)')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')

    plt.tight_layout()
    
    fig.savefig(output_path)
    plt.clf()
    print(f"Distortion difference plot saved to: {output_path}")


if __name__ == "__main__":
    # Paths to the embedding files
    common = "/home/misha/projects/hyp-segmentation/hierarchy_embeddings/hierarchies/hierarchy_embeddings/experiments/spin_dataset/spin_hierarchy/"
    standard_path = common + "2025-06-09_184349/HierarchyEmbedding_weights_16.pth"
    loss_power3_path = common + "2025-06-09_183232/HierarchyEmbedding_weights_16.pth"
    
    # Load the hierarchy
    hierarchy = load_hierarchy(dataset="spin_dataset", hierarchy_name="spin_hierarchy")
    
    # Create dataset to get distance matrix
    dataset = HierarchyEmbeddingDataset(
        hierarchy=hierarchy,
        num_negs=64,  # Default values from generate_embeddings.py
        edge_sample_from="both",
        edge_sample_strat="uniform",
    )
    
    # Initialize the manifold (using default curvature from generate_embeddings.py)
    manifold = PoincareBall(c=Curvature(value=1.0))
    
    # Load both embedding models
    print("Loading standard embeddings...")
    standard_embedding = torch.load(standard_path, map_location='cpu')
    
    print("Loading loss power 3 embeddings...")
    loss_power3_embedding = torch.load(loss_power3_path, map_location='cpu')
    
    # Compute distortion matrices for both embeddings
    print("Computing distortion matrix for standard embeddings...")
    standard_distortion, node_order = compute_distortion_matrix(
        standard_embedding, hierarchy, dataset, manifold
    )
    
    print("Computing distortion matrix for loss power 3 embeddings...")
    loss_power3_distortion, _ = compute_distortion_matrix(
        loss_power3_embedding, hierarchy, dataset, manifold
    )
    
    # Compute the difference (loss power 3 - standard)
    distortion_difference = loss_power3_distortion - standard_distortion

    plot_distortion_difference(distortion_difference, node_order, "../visualizations/plots/distortion_difference.png")

    # Print some statistics
    print(f"\nDistortion difference statistics:")
    print(f"Mean difference: {distortion_difference.mean().item():.4f}")
    print(f"Std difference: {distortion_difference.std().item():.4f}")
    print(f"Min difference: {distortion_difference.min().item():.4f}")
    print(f"Max difference: {distortion_difference.max().item():.4f}")
