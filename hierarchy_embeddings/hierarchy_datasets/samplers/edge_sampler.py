from functools import partial
from typing import Literal

import networkx as nx

import torch

# TODO: Use a decorator to register samplers together with a parser to select from library
from .sample_funcs import (
    edge_corrupt_both_sampler,
    edge_corrupt_source_sampler,
    edge_corrupt_target_sampler,
    edge_sample_uniform,
    edge_sample_prioritize_siblings,
)


class EdgeSampler:
    """See HierarchyEmbeddingDataset for an explanation of the arguments of this class.
    """

    edge_sample_from_dict = {
        "both": edge_corrupt_both_sampler,
        "source": edge_corrupt_source_sampler,
        "target": edge_corrupt_target_sampler,
    }

    edge_sample_strat_dict = {
        "uniform": edge_sample_uniform,
        "siblings": edge_sample_prioritize_siblings,
    }

    def __init__(
        self,
        hierarchy: nx.DiGraph,
        num_negs: int,
        edge_sample_from: Literal["both", "source", "target"] = "both",
        edge_sample_strat: Literal["uniform", "siblings"] = "uniform",
    ) -> None:
        self.hierarchy = hierarchy
        self.num_negs = num_negs
        self.edge_sample_from = edge_sample_from
        self.edge_sample_strat = edge_sample_strat

        # Need the undirected version of the tree for some stuff
        self.undirected_hierarchy = self.hierarchy.to_undirected()
        n = self.undirected_hierarchy.number_of_nodes()

        # Create a symmetric matrix containing the pairwise distances between the hierarchy nodes
        self.dist_matrix = torch.empty([n, n])
        # source is a node id and target_dict is structured as: {target: dist(source, target)}
        for source, target_dict in nx.shortest_path_length(self.undirected_hierarchy):
            # Sorting the dictionary items sorts the dict by target id
            distances_sorted_by_node_id = [d for n, d in sorted(target_dict.items())]
            # So now we can place this ordered row directly into our symmtric matrix
            self.dist_matrix[source, :] = torch.tensor(distances_sorted_by_node_id)

        # Create the sampling function from the hierarchy and other input arguments
        self.edge_sample_fn = partial(
            self.edge_sample_from_dict[edge_sample_from],
            hierarchy=self.hierarchy,
            num_negs=self.num_negs,
            sample_strat=self.edge_sample_strat_dict[edge_sample_strat],
        )

    def sample(self, rel: tuple[int, int]) -> dict[str, torch.Tensor]:
        # Sample edges and edge label targets
        edges, edge_label_targets = self.edge_sample_fn(rel=rel)

        # Initialize dictionary containing this sample
        sample = sample = {
            "edges": edges,
            "edge_label_targets": edge_label_targets,
        }

        # Add the distance targets by grabbing the right values from the dist_matrix
        sample["dist_targets"] = self.dist_matrix[
            sample["edges"][:, 0],
            sample["edges"][:, 1],
        ]

        return sample