import json
import os

import networkx as nx


def load_hierarchy(dataset: str, hierarchy_name: str) -> nx.DiGraph:
    hierarchy_dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(hierarchy_dir, dataset, f"{hierarchy_name}.json")
    with open(file) as f:
        hierarchy_data = json.load(f)

    return nx.node_link_graph(hierarchy_data)


def get_parent(node: int, hierarchy: nx.DiGraph):
    """Returns the parent of the given node under the assumption that the hierarchy is a tree"""
    return next(hierarchy.predecessors(node))


# Testing
if __name__ == "__main__":
    hierarchy = load_hierarchy(
        dataset="cifar100",
        hierarchy_name="cifar_hierarchy",
    )
    print(hierarchy)