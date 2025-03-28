from torch import Tensor
from torch.nn import Module, init

from hypll.tensors import ManifoldTensor
from hypll.manifolds import Manifold
import hypll.nn as hnn

import networkx as nx


class HierarchyEmbedding(Module):
    """A wrapper class for embeddings of a hierarchy on a manifold.
    """
    
    def __init__(
        self,
        hierarchy: nx.DiGraph,
        embedding_dim: int,
        manifold: Manifold,
    ):
        super().__init__()
        self.hierarchy = hierarchy
        self.embedding_dim = embedding_dim
        self.manifold = manifold

        self.embeddings = hnn.HEmbedding(
            num_embeddings=hierarchy.number_of_nodes(),
            embedding_dim=embedding_dim,
            manifold=manifold,
        )

        self.reset_embeddings()

    def forward(self, input: Tensor) -> ManifoldTensor:
        return self.embeddings(input)
    
    def reset_embeddings(self) -> None:
        # Overwriting the initialization since the hypll init doesn't seem to work well
        init.uniform_(
            tensor=self.embeddings.weight.tensor,
            a=-0.001,
            b=0.001,
        )

    @staticmethod
    def load(state_dict: dict) -> "HierarchyEmbedding":
        embedding = HierarchyEmbedding(
            hierarchy=None,
            embedding_dim=None,
            manifold=None,
        )
        embedding.load_state_dict(state_dict)
        return embedding
