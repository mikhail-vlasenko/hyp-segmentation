import argparse
from datetime import datetime
import json
import os

import networkx as nx

import torch
from torch.utils.data import DataLoader

from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.optim import RiemannianSGD
from hypll.tensors import ManifoldTensor

from hierarchy_embeddings import HierarchyEmbedding
from hierarchy_embeddings.losses import (
    distortion_loss,
    poincare_embeddings_loss,
)
from hierarchy_embeddings.hierarchy_datasets import (
    HierarchyEmbeddingDataset,
)
from hierarchy_embeddings.dist_mat_plots import (
    create_dist_mat_plots,
    create_dist_diff_mat_plots,
)
from hierarchy_embeddings.plot_tree import plot_hierarchy_tree
from hierarchy_embeddings.utils import load_hierarchy


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["cifar100", "cub-200-2011"],
        default="cifar100",
        help="Dataset corresponding to the hierarchy",
    )
    parser.add_argument(
        "--hierarchy-name",
        type=str,
        default="cifar_hierarchy",
        help="Name of the file storing the hierarchy",
    )
    parser.add_argument(
        "--num-negs",
        type=int,
        default=64,
        help="Number of negative edges that are too be sampled for each positive edge",
    )
    parser.add_argument(
        "--sample-from",
        type=str,
        choices=["both", "source", "target"],
        default="both",
        help="Which side of the edges is to be corrupted when sampling negative edges",
    )
    parser.add_argument(
        "--sample-strat",
        type=str,
        choices=["uniform", "siblings"],
        default="uniform",
        help="Strategy for corrupting nodes of edges (see sampler files for more info)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate", )
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs for training")
    parser.add_argument("-c", "--curvature", type=float, default=1.0,
                        help="Curvature of the manifold (before containing function)", )
    parser.add_argument("-e", "--embedding-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--pretrain-lr", type=float, default=5.0,
                        help="Learning rate used for pretraining embeddings with PE loss", )
    parser.add_argument("--pretrain-epochs", type=int, default=100,
                        help="Number of epochs used for pretraining embeddings with PE loss", )
    parser.add_argument("--burn-in-lr-mult", type=float, default=0.1,
                        help="Burn-in LR multiplier used during burn-in phase of pretraining", )
    parser.add_argument("--burn-in-epochs", type=int, default=20,
                        help="Number of epochs used for burn-in phase of pretraining", )

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    args = get_arg_parser()

    # Load the hierarchy and wrap a dataset around it
    hierarchy = load_hierarchy(dataset=args.dataset, hierarchy_name=args.hierarchy_name)

    plot_hierarchy_tree(hierarchy, title="CIFAR100 Hierarchy")

    dataset = HierarchyEmbeddingDataset(
        hierarchy=hierarchy,
        num_negs=args.num_negs,
        edge_sample_from=args.sample_from,
        edge_sample_strat=args.sample_strat,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Create the experiment directory
    now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    cwd = os.path.dirname(os.path.realpath(__file__))
    exp_dir = os.path.join(
        cwd,
        "hierarchies",
        "hierarchy_embeddings",
        "experiments",
        args.dataset,
        args.hierarchy_name,
        now,
    )
    os.makedirs(exp_dir)

    # Initialize the Poincare ball
    manifold = PoincareBall(c=Curvature(value=args.curvature))

    # Initialize the embeddings
    hierarchy_embedding = HierarchyEmbedding(
        hierarchy=hierarchy,
        embedding_dim=args.embedding_dim,
        manifold=manifold,
    )
    # --------------------------------------------------------------------------------------------
    # Create the pretraining optimizer
    optimizer = RiemannianSGD(
        params=hierarchy_embedding.parameters(),
        lr=args.pretrain_lr,
        momentum=0,
        weight_decay=0,
    )

    # Perform pretraining with the Poincare embeddings method from Nickel & Kiela
    for epoch in range(args.pretrain_epochs):
        # Use a reduced learning rate during a few "burn-in" epochs
        if epoch < args.burn_in_epochs:
            optimizer.param_groups[0]["lr"] = args.pretrain_lr * args.burn_in_lr_mult
        else:
            optimizer.param_groups[0]["lr"] = args.pretrain_lr

        for idx, batch in enumerate(dataloader):
            edges = batch["edges"]
            edge_label_targets = batch["edge_label_targets"]

            optimizer.zero_grad()

            embeddings = hierarchy_embedding(input=edges)
            loss = poincare_embeddings_loss(
                embeddings=embeddings, targets=edge_label_targets
            )

            if epoch % 20 == 0:
                print(epoch, loss.item())

            loss.backward()
            optimizer.step()

    # Rescale the embeddings, similar to the entailment cone paper by Ganea et al.
    with torch.no_grad():
        hierarchy_embedding.embeddings.weight.tensor.mul_(0.8)
    # --------------------------------------------------------------------------------------------
    # Create the training optimizer
    optimizer = RiemannianSGD(
        params=hierarchy_embedding.parameters(),
        lr=args.lr,
        momentum=0,
        weight_decay=0,
    )

    # Perform training with the distortion loss
    for epoch in range(args.epochs):
        for idx, batch in enumerate(dataloader):
            edges = batch["edges"]
            dist_targets = batch["dist_targets"]

            optimizer.zero_grad()

            embeddings = hierarchy_embedding(input=edges)

            loss = distortion_loss(
                embeddings=embeddings,
                dist_targets=dist_targets,
            )

            if epoch % 20 == 0:
                print(epoch, loss.item())
            loss.backward()
            optimizer.step()

    # Find the root of the hierarchy and perform BFS to find a nice ordering for making plots
    root = [n for n, d in hierarchy.in_degree() if d == 0][0]
    node_order = [root] + [t for _, t in nx.bfs_edges(hierarchy, root)]

    # Re-order the embeddings according to the BFS
    embeddings = ManifoldTensor(
        data=hierarchy_embedding.embeddings.weight.tensor[node_order],
        manifold=manifold,
        man_dim=-1,
    )
    dists = manifold.dist(embeddings[:, None, :], embeddings).detach()

    # Plot the distances between the embeddings
    create_dist_mat_plots(
        dists=dists,
        node_permutation=node_order,
        output_dir=exp_dir,
        file_name="prototype_edge_distances",
    )

    # Plot the distances between the nodes in the original tree
    create_dist_mat_plots(
        dists=dataset.sampler.dist_matrix[node_order, :][:, node_order],
        node_permutation=node_order,
        output_dir=exp_dir,
        file_name="target_distances",
    )

    # Plot the relative distortion of the embeddings
    create_dist_diff_mat_plots(
        embeddings_dist_mat=dists,
        graph_dist_mat=dataset.sampler.dist_matrix[node_order, :][:, node_order],
        node_permutation=node_order,
        output_dir=exp_dir,
    )

    # Store the embeddings
    torch.save(
        obj=hierarchy_embedding.state_dict(),
        f=os.path.join(
            exp_dir,
            f"{hierarchy_embedding.__class__.__name__}_weights_{args.embedding_dim}.pth",
        ),
    )

    # Store the training configuration
    config_file = os.path.join(exp_dir, "config.json")
    with open(config_file, "w") as file:
        json.dump(vars(args), file, indent=4)
