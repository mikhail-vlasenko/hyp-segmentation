import copy
import os
import argparse
from typing import Union

import torch
import numpy as np
import lightning as L
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from focal_loss.focal_loss import FocalLoss

from torchmetrics.classification import MulticlassJaccardIndex


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
        # self.hierarchical_jaccards = nn.ModuleDict({})
        # Prepare containers for test predictions and targets.
        self.test_preds = {g: [] for g in self.granularities}
        self.test_targets = {g: [] for g in self.granularities}

    def logits_to_loss(self, logits, labels, num_classes, background_index, return_preds=False):
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
            logits = self.decode_heads[granularity](encoder_hidden_states)
            result[f"logits_{granularity}"] = logits
        return result

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch["pixel_values"])

        loss = 0
        for granularity in self.granularities:
            logits = outputs[f"logits_{granularity}"]
            labels = batch[f"labels_{granularity}"]
            loss += self.logits_to_loss(
                logits, labels,
                num_labels_for_granularity(granularity),
                background_class_for_granularity(granularity)
            )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def evaluation_step(self, batch, batch_idx, save_preds=False):
        outputs = self.forward(batch["pixel_values"])
        eval_loss = 0
        for granularity in self.granularities:
            logits = outputs[f"logits_{granularity}"]
            labels = batch[f"labels_{granularity}"]
            result = self.logits_to_loss(
                logits, labels,
                num_labels_for_granularity(granularity),
                background_class_for_granularity(granularity),
                return_preds=True
            )
            eval_loss += result["loss"]
            self.jaccards[granularity].update(result["preds"], labels)
            self.no_bg_jaccards[granularity].update(result["preds"], labels)
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
            num_classes = num_labels_for_granularity(granularity)
            # Compute confusion matrix. Ensure that all classes are represented.
            cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))

            # Create a figure for the confusion matrix.
            fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
            cax = ax.matshow(cm, cmap=plt.cm.Blues)
            fig.colorbar(cax)
            ax.set_title(f"Confusion Matrix for {granularity}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks(np.arange(num_classes))
            ax.set_yticks(np.arange(num_classes))
            # # Annotate each cell with its count.
            # for i in range(num_classes):
            #     for j in range(num_classes):
            #         ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')

            plt.savefig(f"confusion_matrix_{granularity}.png")
            plt.clf()
        # Reset the stored predictions and targets.
        self.test_preds = {g: [] for g in self.granularities}
        self.test_targets = {g: [] for g in self.granularities}

    def on_eval_epoch_end(self, prefix="val"):
        for granularity in self.granularities:
            metric = self.jaccards[granularity]
            no_bg_metric = self.no_bg_jaccards[granularity]
            self.log(f"{prefix}_mIoU_{granularity}", metric.compute(), prog_bar=True)
            self.log(f"{prefix}_mIoU_no_bg_{granularity}", no_bg_metric.compute(), prog_bar=True)
            metric.reset()
            no_bg_metric.reset()

    def on_validation_epoch_end(self):
        self.on_eval_epoch_end("val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
