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
        
        # Create a mapping from node to its position in the ordering
        node_to_idx = {node: i for i, node in enumerate(node_order)}
        
        # Initialize embeddings in a circular pattern
        num_nodes = len(node_order)
        
        with torch.no_grad():
            for i, node in enumerate(node_order):
                # Calculate angle for circular placement
                angle = 2 * math.pi * i / num_nodes
                
                if self.embedding_dim == 2:
                    # For 2D embeddings, place directly on circle
                    radius = 0.8  # Stay within Poincare ball bounds
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    self.embeddings.weight.tensor[node] = torch.tensor([x, y], dtype=torch.float32)
                
                elif self.embedding_dim == 3:
                    # For 3D embeddings, place on circle in xy-plane with z varying by depth
                    radius = 0.7
                    # Calculate depth in hierarchy for z-coordinate
                    try:
                        # Try to get shortest path from root to determine depth
                        depth = 0
                        for root in roots:
                            try:
                                path_length = nx.shortest_path_length(self.hierarchy, root, node)
                                depth = max(depth, path_length)
                            except nx.NetworkXNoPath:
                                continue
                        z = 0.1 * depth  # Scale depth to small z values
                    except:
                        z = 0.0
                    
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    self.embeddings.weight.tensor[node] = torch.tensor([x, y, z], dtype=torch.float32)
                
                else:
                    # For higher dimensions, fill first two dims with circle, rest with small values
                    embedding = torch.zeros(self.embedding_dim, dtype=torch.float32)
                    radius = 0.7
                    embedding[0] = radius * math.cos(angle)
                    embedding[1] = radius * math.sin(angle)
                    
                    # Fill remaining dimensions with small random values
                    if self.embedding_dim > 2:
                        embedding[2:] = torch.randn(self.embedding_dim - 2) * 0.1
                    
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
