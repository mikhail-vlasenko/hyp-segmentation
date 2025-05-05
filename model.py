import copy
import os
import argparse
from typing import Union

import torch
import numpy as np
import lightning as L
from torch import nn
from torch.autograd import Function
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from focal_loss.focal_loss import FocalLoss

from torchmetrics.classification import MulticlassJaccardIndex

from hierarchy_embeddings.utils import load_hierarchy
from metric import GraphDistanceSoftIoU, compute_distance_matrix
from prototypes import create_prototypes
from segformer_head import HyperbolicSegformerDecodeHead
from utils import num_labels_for_granularity, background_class_for_granularity


class SegformerLightningModule(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        granularities: list[str],
        head_dim: Union[int, None],
        hyperbolic: bool,
        curvature: float,
        max_class_sep: bool,
        tau: float,
        background_loss_weight: float = 0,
        focal_loss: bool = False,
        focal_loss_gamma: float = 0.7,
        embeddings_path: str = None,
        ratio_loss_weight: float = 0.0,
        clamp_to: float = 2.0,
        norm_penalty_weight: float = 0.0,
    ):
        super().__init__()
        # Save all hyperparameters so they can be later accessed via self.hparams
        self.save_hyperparameters()

        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.backbone = self.model.segformer
        original_decode_head = self.model.decode_head

        self.lr = lr
        self.granularities = granularities
        self.focal_loss = focal_loss
        self.focal_loss_gamma = focal_loss_gamma
        if not self.focal_loss:
            self.background_loss_weight = background_loss_weight

        self.decode_heads = nn.ModuleDict({
            g: HyperbolicSegformerDecodeHead.from_segformer_decode_head(
                copy.deepcopy(original_decode_head),
                num_labels_for_granularity(g),
                head_dim,
                hyperbolic,
                curvature,
                max_class_sep,
                tau,
                embeddings_path,
            )
            for g in self.granularities
        })

        self.jaccards = nn.ModuleDict({
            g: MulticlassJaccardIndex(
                num_classes=num_labels_for_granularity(g),
                ignore_index=None
            )
            for g in self.granularities
        })
        self.no_bg_jaccards = nn.ModuleDict({
            g: MulticlassJaccardIndex(
                num_classes=num_labels_for_granularity(g),
                ignore_index=background_class_for_granularity(g)
            )
            for g in self.granularities
        })
        self.hierarchical_metrics = self.configure_hierarchical_metrics()

        # Prepare containers for test predictions and targets.
        self.test_preds = {g: [] for g in self.granularities}
        self.test_targets = {g: [] for g in self.granularities}

        self.ratio_loss_weight = ratio_loss_weight
        self.clamp_to = clamp_to
        self.norm_penalty_weight = norm_penalty_weight

    def logits_to_loss(self, logits, labels, reprs, num_classes, background_index, return_preds=False):
        # upsample logits to the images' original size
        upsampled_logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        if self.focal_loss:
            loss_fct = FocalLoss(gamma=self.focal_loss_gamma)
            upsampled_logits = upsampled_logits.permute(0, 2, 3, 1)
            upsampled_logits = nn.functional.softmax(upsampled_logits, dim=-1)
        else:
            loss_weights = torch.ones(num_classes).to(logits.device)
            loss_weights[background_index] = self.background_loss_weight
            loss_fct = CrossEntropyLoss(weight=loss_weights)

        loss = loss_fct(upsampled_logits, labels)
        if self.ratio_loss_weight > 0 and self.decode_heads[self.granularities[0]].max_class_sep:
            loss_fct2 = PrototypeRatioLoss(weight=self.ratio_loss_weight, clamp_to=self.clamp_to)
            loss += loss_fct2(upsampled_logits, labels)

        if self.norm_penalty_weight > 0 and self.decode_heads[self.granularities[0]].hyperbolic:
            norm_penalty = hinge_norm_penalty(reprs)
            loss += self.norm_penalty_weight * norm_penalty

        if return_preds:
            preds = torch.argmax(upsampled_logits, dim=-1 if self.focal_loss else 1)
            return {
                "loss": loss,
                "preds": preds,
            }
        return loss

    def forward(self, pixel_values):
        outputs = self.backbone(
            pixel_values,
            output_attentions=None,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=None,
        )
        encoder_hidden_states = outputs[1]
        result = {}
        # Compute logits for each granularity using the same backbone features
        for granularity in self.granularities:
            logits, reprs = self.decode_heads[granularity](encoder_hidden_states, return_repr=True)
            result[f"logits_{granularity}"] = logits
            result[f"reprs_{granularity}"] = reprs
        return result

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch["pixel_values"])

        loss = 0
        for granularity in self.granularities:
            logits = outputs[f"logits_{granularity}"]
            labels = batch[f"labels_{granularity}"]
            reprs = outputs[f"reprs_{granularity}"]
            loss += self.logits_to_loss(
                logits, labels, reprs,
                num_labels_for_granularity(granularity),
                background_class_for_granularity(granularity),
            )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def evaluation_step(self, batch, batch_idx, save_preds=False):
        outputs = self.forward(batch["pixel_values"])
        eval_loss = 0
        for granularity in self.granularities:
            logits = outputs[f"logits_{granularity}"]
            labels = batch[f"labels_{granularity}"]
            reprs = outputs[f"reprs_{granularity}"]
            result = self.logits_to_loss(
                logits, labels, reprs,
                num_labels_for_granularity(granularity),
                background_class_for_granularity(granularity),
                return_preds=True
            )
            eval_loss += result["loss"]
            self.jaccards[granularity].update(result["preds"], labels)
            self.no_bg_jaccards[granularity].update(result["preds"], labels)
            self.hierarchical_metrics[granularity].update(result["preds"], labels)
            if save_preds:
                self.test_preds[granularity].append(result["preds"].detach().cpu())
                self.test_targets[granularity].append(labels.detach().cpu())
        return eval_loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.evaluation_step(batch, batch_idx)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        test_loss = self.evaluation_step(batch, batch_idx, save_preds=True)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        return test_loss

    def on_test_epoch_end(self):
        self.on_eval_epoch_end("test")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        for granularity in self.granularities:
            # Aggregate predictions and targets across batches.
            preds = torch.cat(self.test_preds[granularity], dim=0).numpy().flatten()
            targets = torch.cat(self.test_targets[granularity], dim=0).numpy().flatten()
            share_correct = np.sum(preds == targets) / len(preds)
            num_classes = num_labels_for_granularity(granularity)
            # Compute confusion matrix. Ensure that all classes are represented.
            cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
            cm = np.log1p(cm)  # Apply log1p to avoid log(0)
            # set diagonal to 0
            np.fill_diagonal(cm, 0)

            fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
            cax = ax.matshow(cm, cmap=plt.cm.Reds)
            fig.colorbar(cax)
            ax.set_title(f"Confusion Matrix for {granularity}. Share correct: {share_correct:.2f}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks(np.arange(num_classes))
            ax.set_yticks(np.arange(num_classes))

            plt.savefig(f"confusion_matrix_{granularity}.png")
            plt.clf()
        # Reset the stored predictions and targets.
        self.test_preds = {g: [] for g in self.granularities}
        self.test_targets = {g: [] for g in self.granularities}

    def on_eval_epoch_end(self, prefix="val"):
        for granularity in self.granularities:
            metric = self.jaccards[granularity]
            no_bg_metric = self.no_bg_jaccards[granularity]
            hierarchical_metric = self.hierarchical_metrics[granularity]
            self.log(f"{prefix}_mIoU_{granularity}", metric.compute(), prog_bar=True)
            self.log(f"{prefix}_mIoU_no_bg_{granularity}", no_bg_metric.compute(), prog_bar=True)
            self.log(f"{prefix}_hierarchical_mIoU_{granularity}", hierarchical_metric.compute(), prog_bar=True)
            metric.reset()
            no_bg_metric.reset()
            hierarchical_metric.reset()

    def on_validation_epoch_end(self):
        self.on_eval_epoch_end("val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    @staticmethod
    def configure_hierarchical_metrics():
        hierarchy = load_hierarchy(None, None, "hierarchy_embeddings/spin_dataset/spin_hierarchy.json")
        dist_matrix, node_list = compute_distance_matrix(hierarchy)
        obj_dist_mat = dist_matrix[205:216, 205:216]

        def add_bg(some_dist_mat, start, end):
            # add the background class (0) to the object distance matrix.
            bg_to_obj = dist_matrix[0, start:end]
            some_dist_mat = torch.cat([bg_to_obj.unsqueeze(0), some_dist_mat], dim=0)
            obj_to_bg = dist_matrix[start:end, 0]
            with_bg_to_bg = torch.cat([dist_matrix[0, 0].unsqueeze(0), obj_to_bg], dim=0)
            some_dist_mat = torch.cat([with_bg_to_bg.unsqueeze(1), some_dist_mat], dim=1)
            return some_dist_mat

        obj_dist_mat = add_bg(obj_dist_mat, 205, 216)
        # save as image for debugging
        # import matplotlib.pyplot as plt
        # plt.imshow(obj_dist_mat.cpu().numpy())
        # plt.colorbar()
        # plt.title("Object distance matrix")
        # plt.savefig("obj_dist_mat.png")
        # plt.clf()

        part_dist_mat = dist_matrix[216:256, 216:256]
        part_dist_mat = add_bg(part_dist_mat, 216, 256)
        subpart_dist_mat = dist_matrix[0:204, 0:204]
        return nn.ModuleDict({
            "whole": GraphDistanceSoftIoU(obj_dist_mat),
            "part": GraphDistanceSoftIoU(part_dist_mat),
            "subpart": GraphDistanceSoftIoU(subpart_dist_mat),
        })


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
