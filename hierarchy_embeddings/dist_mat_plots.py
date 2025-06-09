import os

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sn

import torch


def create_dist_mat_plots(
    dists: torch.Tensor,
    node_permutation: list[int],
    output_dir: str = "",
    file_name: str = "prototype_edge_distances"
) -> torch.Tensor:
    df_cm = pd.DataFrame(
        dists[:len(node_permutation), :len(node_permutation)],
        node_permutation,
        node_permutation,
    )

    fig = plt.figure(figsize=(10, 8), dpi=300)
    sn.set(font_scale=0.5)
    sn.heatmap(df_cm)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{file_name}.png"))
    plt.clf()


def create_dist_diff_mat_plots(
    embeddings_dist_mat: torch.Tensor,
    graph_dist_mat: torch.Tensor,
    node_permutation: list[int],
    output_dir: str = "",
) -> None:
    diff_mat = ((embeddings_dist_mat - graph_dist_mat) / graph_dist_mat).abs()
    diff_mat[torch.isnan(diff_mat)] = embeddings_dist_mat[torch.isnan(diff_mat)]
    df_cm = pd.DataFrame(
        diff_mat[:len(node_permutation), :len(node_permutation)],
        node_permutation,
        node_permutation,
    )

    fig = plt.figure(figsize=(10, 8), dpi=300)
    sn.set(font_scale=0.5)
    sn.heatmap(df_cm, vmin=0, vmax=1.5)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"prototype_edge_distortions.png"))
    plt.clf()
