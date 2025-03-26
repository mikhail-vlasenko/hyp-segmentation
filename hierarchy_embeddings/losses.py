import torch

from hypll.tensors import ManifoldTensor


def distortion_loss(
    embeddings: ManifoldTensor, dist_targets: torch.Tensor
) -> torch.Tensor:
    """Our own distortion loss.

    The embeddings input consists of a tensor of shape [B, 1 + N, 2, D], where
        B: batch size;
        N: number of negative edges sampled per positive edge;
        2: comes from the 2 nodes involved per edge (so source and target);
        D: the embedding dimension.

    The dists_targets input is a tensor of size [B, 1 + N] containing the target distances
    for the edges corresponding to the pairs in the embeddings tensor.
    """
    manifold = embeddings.manifold
    embedding_dists = manifold.dist(x=embeddings[:, :, 0, :], y=embeddings[:, :, 1, :])
    losses = (embedding_dists - dist_targets).abs() / dist_targets
    return losses.mean()


def poincare_embeddings_loss(
    embeddings: ManifoldTensor, targets: torch.Tensor
) -> torch.Tensor:
    """The original loss function from the Poincare embeddings paper by Nickel & Kiela, 
    described in their paper: https://arxiv.org/pdf/1705.08039.pdf
    """
    manifold = embeddings.manifold
    dists = manifold.dist(x=embeddings[:, :, 0, :], y=embeddings[:, :, 1, :])
    logits = dists.neg().exp()
    numerator = torch.where(condition=targets, input=logits, other=0).sum(dim=-1)
    denominator = logits.sum(dim=-1)
    loss = (numerator / denominator).log().mean().neg()
    return loss
