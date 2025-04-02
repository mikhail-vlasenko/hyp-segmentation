from pprint import pprint

import torch
import torch.nn as nn
import networkx as nx


def compute_distance_matrix(graph: nx.DiGraph):
    """
    Computes the pairwise shortest path distances between nodes in the graph.

    Args:
        graph (nx.DiGraph): The input graph.

    Returns:
        dist_matrix (torch.Tensor): A tensor of shape (num_nodes, num_nodes) where
            dist_matrix[i, j] is the shortest path length from node i to node j.
        nodes (list): List of nodes in the graph corresponding to the indices in dist_matrix.
    """
    nodes = list(graph.nodes())
    num_nodes = len(nodes)
    # Initialize matrix with infinity for unreachable pairs.
    dist_matrix = torch.full((num_nodes, num_nodes), float('inf'))

    # make graph undirected
    graph = graph.to_undirected()

    # Compute shortest path lengths from each source node.
    for i, source in enumerate(nodes):
        lengths = nx.single_source_shortest_path_length(graph, source)
        for j, target in enumerate(nodes):
            if target in lengths:
                dist_matrix[i, j] = lengths[target]

    assert torch.isinf(dist_matrix).sum() == 0, "Graph is not connected."
    # # Replace infinities with a maximum value (e.g. num_nodes) so the metric is bounded.
    # dist_matrix[dist_matrix == float('inf')] = num_nodes
    return dist_matrix, nodes


class GraphDistanceSoftIoU(nn.Module):
    def __init__(self, distance_matrix: torch.Tensor, decay_fn=None):
        """
        Args:
            distance_matrix (torch.Tensor): A tensor of shape (num_classes, num_classes)
                where distance_matrix[i, j] is the graph distance between class i and j.
            decay_fn (callable, optional): A function f(d) that converts a distance into a similarity score.
                If None, a default f(d)=1/(1+d) is used.
        """
        super(GraphDistanceSoftIoU, self).__init__()
        # Register the precomputed distance matrix as a buffer.
        self.register_buffer("distance_matrix", distance_matrix)
        # Use a decay function that converts distances to similarity scores.
        if decay_fn is None:
            self.decay_fn = lambda d: 1.0 / (1.0 + d)
        else:
            self.decay_fn = decay_fn

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Computes a graph-distance–aware mIoU.

        For each class c, we compute:
          - soft_intersection_c = sum_{pixels with target c} f(D(c, prediction))
          - union_c = (number of pixels with target c) + (number of pixels predicted as c) - soft_intersection_c

        The per-class IoU is then the ratio of soft_intersection_c / union_c,
        and the mIoU is the mean over all classes with union > 0.

        Args:
            preds (torch.Tensor): Tensor of predicted class indices (shape: [N, H, W]).
            targets (torch.Tensor): Tensor of ground truth class indices (shape: [N, H, W]).
            num_classes (int): Total number of classes.

        Returns:
            torch.Tensor: The distance-aware mIoU.
        """
        iou_list = []
        # Loop over each class to compute a soft IoU.
        for c in range(num_classes):
            # Create a binary mask for ground truth pixels belonging to class c.
            target_mask = (targets == c).float()

            # For each pixel in the prediction, look up the distance between the true class c and the predicted class.
            # This results in a tensor of the same shape as preds.
            distances = self.distance_matrix[c, preds]
            # Convert distances to similarity scores (closer predictions yield higher scores).
            soft_scores = self.decay_fn(distances)

            # For pixels where the ground truth is class c, add up the similarity scores.
            soft_intersection = (soft_scores * target_mask).sum()

            # Standard union calculation: |target| + |prediction| - intersection.
            # Here, |target| is the number of pixels with true class c.
            target_area = target_mask.sum()
            # For the prediction area, we simply count all pixels predicted as class c (hard count).
            pred_area = (preds == c).float().sum()
            union = target_area + pred_area - soft_intersection

            # If there is any area in the union, compute IoU for class c.
            if union > 0:
                iou_c = soft_intersection / union
                iou_list.append(iou_c)

        # mIoU is the average IoU over classes that are present.
        if len(iou_list) == 0:
            return torch.tensor(0.0, device=preds.device)
        mIoU = sum(iou_list) / len(iou_list)
        return mIoU


# def _jaccard_index_reduce(
#     confmat: Tensor,
#     average: Optional[Literal["micro", "macro", "weighted", "none", "binary"]],
#     ignore_index: Optional[int] = None,
#     zero_division: float = 0.0,
# ) -> Tensor:
#     """Perform reduction of an un-normalized confusion matrix into jaccard score.
#
#     Args:
#         confmat: tensor with un-normalized confusionmatrix
#         average: reduction method
#
#             - ``'binary'``: binary reduction, expects a 2x2 matrix
#             - ``'macro'``: Calculate the metric for each class separately, and average the
#               metrics across classes (with equal weights for each class).
#             - ``'micro'``: Calculate the metric globally, across all samples and classes.
#             - ``'weighted'``: Calculate the metric for each class separately, and average the
#               metrics across classes, weighting each class by its support (``tp + fn``).
#             - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
#               the metric for every class.
#
#         ignore_index:
#             Specifies a target value that is ignored and does not contribute to the metric calculation
#         zero_division:
#             Value to replace when there is a division by zero. Should be `0` or `1`.
#
#     """
#     allowed_average = ["binary", "micro", "macro", "weighted", "none", None]
#     if average not in allowed_average:
#         raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")
#     confmat = confmat.float()
#     if average == "binary":
#         return _safe_divide(confmat[1, 1], (confmat[0, 1] + confmat[1, 0] + confmat[1, 1]), zero_division=zero_division)
#
#     ignore_index_cond = ignore_index is not None and 0 <= ignore_index < confmat.shape[0]
#     multilabel = confmat.ndim == 3
#     if multilabel:
#         num = confmat[:, 1, 1]
#         denom = confmat[:, 1, 1] + confmat[:, 0, 1] + confmat[:, 1, 0]
#     else:  # multiclass
#         num = torch.diag(confmat)
#         denom = confmat.sum(0) + confmat.sum(1) - num
#
#     if average == "micro":
#         num = num.sum()
#         denom = denom.sum() - (denom[ignore_index] if ignore_index_cond else 0.0)
#
#     jaccard = _safe_divide(num, denom, zero_division=zero_division)
#
#     if average is None or average == "none" or average == "micro":
#         return jaccard
#     if average == "weighted":
#         weights = confmat[:, 1, 1] + confmat[:, 1, 0] if confmat.ndim == 3 else confmat.sum(1)
#     else:
#         weights = torch.ones_like(jaccard)
#         if ignore_index_cond:
#             weights[ignore_index] = 0.0
#         if not multilabel:
#             weights[confmat.sum(1) + confmat.sum(0) == 0] = 0.0
#     return ((weights * jaccard) / weights.sum()).sum()


if __name__ == "__main__":
    # Create an example hierarchy graph.
    G = nx.DiGraph()
    G.add_edges_from([
        (0, 1),
        (0, 2),
        (1, 3),
        (1, 4),
        (2, 5),
        (2, 6),
    ])

    dist_matrix, node_list = compute_distance_matrix(G)

    # Instantiate the distance-aware soft IoU metric.
    soft_iou_metric = GraphDistanceSoftIoU(dist_matrix)

    # Create dummy predictions and targets for an image batch of shape (N, H, W).
    # For example, let's create a batch of 2 images, each 4x4.
    preds = torch.randint(0, len(G.nodes), (2, 4, 4))
    targets = torch.randint(0, len(G.nodes), (2, 4, 4))

    mIoU_value = soft_iou_metric(preds, targets, len(G.nodes))
    print("Graph Distance Soft mIoU:", mIoU_value.item())
