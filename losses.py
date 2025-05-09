from dataclasses import dataclass

import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F


@dataclass
class LossParams:
    background_loss_weight: float = 0,
    focal_loss: bool = False,
    focal_loss_gamma: float = 0.7,
    ratio_loss_weight: float = 0.0,
    clamp_to: float = 2.0,
    norm_penalty_weight: float = 0.0


class PrototypeRatioLoss(nn.Module):
    def __init__(self, weight=1.0, eps=1e-4, clamp_to=2.0):
        """
        Args:
            weight (float): Weight for the ratio loss component.
            eps (float): Small constant to prevent division by zero.
        """
        super().__init__()
        self.weight = weight
        self.eps = eps
        self.clamp_to = clamp_to

    def forward(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape (B, C, *spatial), where C is the # of prototypes.
                    These are assumed to be NEGATIVE distances to each prototype.
            labels: LongTensor of shape (B, *spatial), with values in [0, C).
        Returns:
            A scalar tensor: the mean ratio-loss over all positions.
        """
        distances = -logits  # convert logits back to distances

        # --- flatten all non-prototype dims into a single batch dimension ---
        perm = [0] + list(range(2, logits.dim())) + [1]
        distances = distances.permute(*perm).contiguous()
        *spatial_dims, C = distances.shape
        N = int(torch.prod(torch.tensor(spatial_dims)))  # total number of positions

        # reshape to (N, C) and (N,)
        d_flat = distances.view(-1, C)
        l_flat = labels.view(-1)

        # gather correct distances
        idx = torch.arange(N, device=d_flat.device)
        correct_dists = d_flat[idx, l_flat]

        # build mask for incorrect prototypes
        mask = torch.ones_like(d_flat, dtype=torch.bool)
        mask[idx, l_flat] = False

        # for each position, find min distance among incorrect prototypes
        # masked_select gives a 1D tensor, so we view back into (N, C-1)
        min_incorrect, _ = (
            d_flat.masked_select(mask)
                  .view(N, C - 1)
                  .min(dim=1)
        )
        # avoid zero
        min_incorrect = torch.clamp(min_incorrect, min=self.eps)

        # compute per-position margin: max(2*correct - best-wrong, 0)
        margin = torch.relu((correct_dists * 2) - min_incorrect)
        ratio_loss = margin / min_incorrect

        # clamp each element to max=1 with proportional gradient scaling
        ratio_loss = ClampMaxGrad.apply(ratio_loss, self.clamp_to)

        # mean over positions
        ratio_loss = ratio_loss.mean()

        if torch.isnan(ratio_loss):
            raise ValueError("Ratio loss became NaN")

        return self.weight * ratio_loss


class ClampMaxGrad(Function):
    @staticmethod
    def forward(ctx, input, max_val):
        # save raw input for backward
        ctx.save_for_backward(input)
        ctx.max_val = float(max_val)
        # forward is hard clamp at max_val
        return input.clamp(max=max_val)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        max_val = ctx.max_val
        # compute factor = clamp(input, max_val) / input
        # for input > max_val: factor = max_val / input
        # for input <= max_val: factor = 1
        factor = torch.where(input > max_val,
                             max_val / input,
                             torch.ones_like(input))
        # scale the incoming gradient
        return grad_output * factor, None


def hinge_norm_penalty(h, threshold=0.5):
    sqnorm = h.pow(2).sum(dim=1)
    over = F.relu(sqnorm - threshold)
    return (over**2).mean()

