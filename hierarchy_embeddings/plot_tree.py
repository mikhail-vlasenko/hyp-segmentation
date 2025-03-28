import matplotlib.pyplot as plt
import networkx as nx


def plot_hierarchy_tree(hierarchy: nx.DiGraph, title: str = "Hierarchy Tree", save_path: str = None):
    """
    Plots the hierarchy in a tree-shaped layout using matplotlib.
    Args:
        hierarchy (nx.DiGraph): The hierarchy to visualize.
        title (str): Title of the plot.
        save_path (str): If provided, saves the figure to this path.
    """
    def get_tree_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        children = list(G.successors(root))
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos[child] = (nextx, vert_loc - vert_gap)
                pos = get_tree_pos(G, root=child, width=dx, vert_gap=vert_gap,
                                   vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root)
        return pos

    root = [n for n, d in hierarchy.in_degree() if d == 0][0]
    pos = get_tree_pos(hierarchy, root)

    plt.figure(figsize=(12, 10), dpi=300)
    nx.draw(hierarchy, pos, with_labels=True, node_size=300, font_size=6, arrows=False)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.clf()

