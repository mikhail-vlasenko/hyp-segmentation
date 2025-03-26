from typing import Literal

import networkx as nx

import torch
from torch.utils.data import Dataset

from .samplers.edge_sampler import EdgeSampler


class HierarchyEmbeddingDataset(Dataset):
    """A dataset class for sampling edges from a networkx (directed) tree.

    Each item consists of a dictionary containing keys

        - edges: tensor of size [B, N + 1, 2] describing source, target pairs for edges
        where B is the batch size and N the number of negatives sampled per positive;
        - edge_label_targets: boolean tensor of size [B, N + 1] indicating whether the edge
        exists or not;
        - dist_targets: float tensor of size [B, N + 1] indicating the shortest path distance
        between the source and target.

    edge_sample_from: 
        - both: can corrupt either the source or the target of the existing edge;
        - source: can only corrupt the source of the existing edge;
        - target: can only corrupt the target of the existing edge.

    edge_sample_strat:
        - uniform: sample uniformly from all possible corruption choices.
        - siblings: prioritize sampling from siblings before other corruption options.
    """

    def __init__(
        self,
        hierarchy: nx.DiGraph,
        num_negs: int = 10,
        edge_sample_from: Literal["both", "source", "target"] = "both",
        edge_sample_strat: Literal["uniform", "siblings"] = "uniform",
    ):
        super(HierarchyEmbeddingDataset, self).__init__()
        self.hierarchy = hierarchy
        self.num_negs = num_negs
        self.edge_sample_from = edge_sample_from
        self.edge_sample_strat = edge_sample_strat

        self.sampler = EdgeSampler(
            hierarchy=self.hierarchy,
            num_negs=self.num_negs,
            edge_sample_from=edge_sample_from,
            edge_sample_strat=edge_sample_strat,
        )

        self.edges_list = list(hierarchy.edges())

    def __len__(self) -> int:
        return len(self.edges_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rel = self.edges_list[idx]
        sample = self.sampler.sample(rel=rel)
        return sample
