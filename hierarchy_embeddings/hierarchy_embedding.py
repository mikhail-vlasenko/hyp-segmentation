from torch import Tensor
from torch.nn import Module, init
import torch
import math

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
        # Initialize embeddings in a flattened circular tree structure
        # Find the root(s) of the hierarchy
        roots = [n for n, d in self.hierarchy.in_degree() if d == 0]
        if not roots:
            raise ValueError("Hierarchy has no root")
        
        # Perform DFS traversal to get node ordering
        node_order = []
        visited = set()
        
        def dfs(node):
            if node not in visited:
                visited.add(node)
                node_order.append(node)
                # Visit children in a consistent order
                children = sorted(list(self.hierarchy.successors(node)))
                for child in children:
                    dfs(child)
        
        # Start DFS from all roots
        for root in sorted(roots):
            dfs(root)
        
        # Add any remaining nodes that weren't reached (in case of disconnected components)
        for node in sorted(self.hierarchy.nodes()):
            if node not in visited:
                node_order.append(node)
        
        # Initialize embeddings in a circular pattern
        num_nodes = len(node_order)
        
        with torch.no_grad():
            for i, node in enumerate(node_order):
                # Calculate angle for circular placement
                angle = 2 * math.pi * i / num_nodes
                radius = 0.8
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)

                if self.embedding_dim == 2:
                    embedding = torch.tensor([x, y], dtype=torch.float32)
                else:
                    # For higher dimensions, fill first two dims with circle, rest with small values
                    embedding = torch.FloatTensor(self.embedding_dim).uniform_(-0.001, 0.001)
                    embedding[0] = radius * math.cos(angle)
                    embedding[1] = radius * math.sin(angle)
                    
                self.embeddings.weight.tensor[node] = embedding

    @staticmethod
    def load(state_dict: dict) -> "HierarchyEmbedding":
        embedding = HierarchyEmbedding(
            hierarchy=None,
            embedding_dim=None,
            manifold=None,
        )
        embedding.load_state_dict(state_dict)
        return embedding
