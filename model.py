import copy
import os
import argparse
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
                None,
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
        # Compute logits from each decoding head using the same backbone features
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

    def evaluation_step(self, batch, batch_idx):
        outputs = self.forward(batch["pixel_values"])

        val_loss = 0
        for granularity in self.granularities:
            logits = outputs[f"logits_{granularity}"]
            labels = batch[f"labels_{granularity}"]
            result = self.logits_to_loss(
                logits, labels,
                num_labels_for_granularity(granularity),
                background_class_for_granularity(granularity),
                return_preds=True
            )
            val_loss += result["loss"]
            self.jaccards[granularity].update(result["preds"], labels)
            self.no_bg_jaccards[granularity].update(result["preds"], labels)
        return val_loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.evaluation_step(batch, batch_idx)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        test_loss = self.evaluation_step(batch, batch_idx)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        return test_loss

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

    def on_test_epoch_end(self):
        self.on_eval_epoch_end("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
