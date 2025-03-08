import copy
import os
import argparse
import torch
import numpy as np
import lightning as L
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

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
    ):
        super().__init__()
        # Save all hyperparameters so they can be later accessed via self.hparams
        self.save_hyperparameters()

        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.backbone = self.model.segformer
        original_decode_head = self.model.decode_head

        self.lr = lr
        self.granularities = granularities
        self.background_loss_weight = background_loss_weight
        self.ignore_background = background_loss_weight == 0
        if self.ignore_background:
            print("Ignoring background class in mIoU computation")

        self.decode_heads = nn.ModuleDict({
            g: HyperbolicSegformerDecodeHead.from_segformer_decode_head(
                copy.deepcopy(original_decode_head),
                num_labels_for_granularity(g),
                None,
                hyperbolic,
                curvature,
                max_class_sep,
                tau
            )
            for g in self.granularities
        })

        self.jaccards = nn.ModuleDict({
            g: MulticlassJaccardIndex(
                num_classes=num_labels_for_granularity(g),
                ignore_index=background_class_for_granularity(g) if self.ignore_background else None
            )
            for g in self.granularities
        })

    def logits_to_loss(self, logits, labels, num_classes, background_index, return_preds=False):
        # upsample logits to the images' original size
        upsampled_logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        loss_weights = torch.ones(num_classes).to(logits.device)
        loss_weights[background_index] = self.background_loss_weight
        loss_fct = CrossEntropyLoss(weight=loss_weights)

        loss = loss_fct(upsampled_logits, labels)
        if return_preds:
            preds = torch.argmax(upsampled_logits, dim=1)
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
        print("training step done")
        return loss

    def validation_step(self, batch, batch_idx):
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

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def on_validation_epoch_end(self):
        for granularity in self.granularities:
            metric = self.jaccards[granularity]
            miou = metric.compute()
            self.log(f"val_mIoU_{granularity}", miou, prog_bar=True)
            metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
