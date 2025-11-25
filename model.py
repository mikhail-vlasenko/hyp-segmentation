import copy
from dataclasses import replace
import json
from collections import defaultdict

import torch
import numpy as np
import lightning as L
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from focal_loss.focal_loss import FocalLoss

from torchmetrics.classification import MulticlassJaccardIndex

from hierarchy_embeddings.utils import load_hierarchy
from losses import LossParams, PrototypeRatioLoss, hinge_norm_penalty
from metric import GraphDistanceSoftIoU, compute_distance_matrix
from segformer_head import HyperbolicSegformerDecodeHead, HeadKwargs, HeadReturnType
from utils import num_labels_for_granularity, background_class_for_granularity


def prediction_idx_to_hierarchy_id(granularity, idx):
    """
    Convert prediction index to hierarchy node ID.
    
    Args:
        granularity: "subpart", "part", or "whole"
        idx: prediction class index
    
    Returns:
        int: hierarchy node ID
    """
    bg_idx = background_class_for_granularity(granularity)
    
    # Background always maps to 0 in hierarchy
    if idx == bg_idx:
        return 0
    
    if granularity == "subpart":
        # Level 3: direct mapping (0-203), background at 0
        return idx
    elif granularity == "part":
        # Level 2: classes 0-39 map to 216-255, background at 40
        return 216 + idx
    elif granularity == "whole":
        # Level 1: classes 0-10 map to 205-215, background at 11
        return 205 + idx
    else:
        raise ValueError(f"Unknown granularity: {granularity}")


def load_hierarchy_mappings(hierarchy_path):
    """
    Load hierarchy and create parent-child mappings.
    
    Returns:
        dict: {child_id: parent_id} for subpart→part and part→whole
    """
    with open(hierarchy_path, 'r') as f:
        hierarchy = json.load(f)
    
    # Build adjacency: parent -> children
    parent_to_children = defaultdict(list)
    for link in hierarchy['links']:
        parent_to_children[link['source']].append(link['target'])
    
    # Build child -> parent mapping
    child_to_parent = {}
    for link in hierarchy['links']:
        child_to_parent[link['target']] = link['source']
    
    return child_to_parent


class SegformerLightningModule(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        granularities: list[str],
        head_kwargs: HeadKwargs,
        loss_params: LossParams,
        num_epochs: int = 10,
        poly_decay_power: float = 1.0,
    ):
        super().__init__()
        # Save all hyperparameters so they can be later accessed via self.hparams
        self.save_hyperparameters()

        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.backbone = self.model.segformer
        original_decode_head = self.model.decode_head

        self.lr = lr
        self.granularities = granularities
        self.loss_params = loss_params
        self.num_epochs = num_epochs
        self.poly_decay_power = poly_decay_power

        self.decode_heads = nn.ModuleDict({
            g: HyperbolicSegformerDecodeHead.from_segformer_decode_head(
                copy.deepcopy(original_decode_head),
                replace(head_kwargs, primary_granularity=g),
            )
            for g in self.granularities
        })

        if len(self.granularities) > 1 and len(head_kwargs.embeddings_paths) > 0 and not head_kwargs.independent_heads:
            raise ValueError("Cannot use multiple heads with embeddings paths. (yet). Forward method will overwrite.")
        self.prediction_granularities = {
            *head_kwargs.embeddings_paths,  # implicitly .keys()
            *self.granularities
        }

        ignore_bg_metric = self.loss_params.background_loss_weight == 0.0
        if ignore_bg_metric:
            print("Background class will be ignored in metrics computation. ")
        self.jaccards = nn.ModuleDict({
            g: MulticlassJaccardIndex(
                num_classes=num_labels_for_granularity(g),
                ignore_index=background_class_for_granularity(g) if ignore_bg_metric else None,
            )
            for g in self.prediction_granularities
        })
        self.hierarchical_metrics = self.configure_hierarchical_metrics()

        # Load hierarchy mappings for spatial consistency
        self.child_to_parent = load_hierarchy_mappings("hierarchy_embeddings/spin_dataset/spin_hierarchy.json")
        
        # Containers for spatial consistency scores
        self.spatial_consistency_scores = {
            "subpart_to_part": [],
            "part_to_whole": []
        }

        # Prepare containers for test predictions and targets.
        self.test_preds = {g: [] for g in self.prediction_granularities}
        self.test_targets = {g: [] for g in self.prediction_granularities}

    def logits_to_loss(self, upsampled_logits, labels, reprs, num_classes, background_index):
        if self.loss_params.focal_loss:
            loss_fct = FocalLoss(gamma=self.loss_params.focal_loss_gamma)
            loss_logits = upsampled_logits.permute(0, 2, 3, 1)
            loss_logits = nn.functional.softmax(loss_logits, dim=-1)
            loss = loss_fct(loss_logits, labels)
        else:
            loss_weights = torch.ones(num_classes).to(upsampled_logits.device)
            loss_weights[background_index] = self.loss_params.background_loss_weight
            loss_fct = CrossEntropyLoss(weight=loss_weights)
            loss = loss_fct(upsampled_logits, labels)

        if self.loss_params.ratio_loss_weight > 0 and self.decode_heads[self.granularities[0]].max_class_sep:
            loss_fct2 = PrototypeRatioLoss(weight=self.loss_params.ratio_loss_weight, clamp_to=self.loss_params.clamp_to)
            loss += loss_fct2(upsampled_logits, labels)

        if self.loss_params.norm_penalty_weight > 0 and self.decode_heads[self.granularities[0]].hyperbolic:
            norm_penalty = hinge_norm_penalty(reprs)
            loss += self.loss_params.norm_penalty_weight * norm_penalty
        return loss

    def forward(self, pixel_values, eval_mode=False):
        outputs = self.backbone(
            pixel_values,
            output_attentions=None,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=None,
        )
        encoder_hidden_states = outputs[1]
        logits = {}
        reprs = {}
        # Compute logits for each granularity using the same backbone features
        for granularity in self.granularities:
            head_result: HeadReturnType = self.decode_heads[granularity](encoder_hidden_states, eval_mode)
            logits.update(head_result.logits)
            reprs[granularity] = head_result.repr
        return logits, reprs

    def training_step(self, batch, batch_idx):
        logits, reprs = self.forward(batch["pixel_values"])

        loss = 0
        for granularity in self.granularities:
            labels = batch[f"labels_{granularity}"]
            loss += self.logits_to_loss(
                self.upsample_logits(logits[granularity], labels), labels, reprs,
                num_labels_for_granularity(granularity),
                background_class_for_granularity(granularity),
            )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def evaluation_step(self, batch, batch_idx, save_preds=False):
        logits, reprs = self.forward(batch["pixel_values"], eval_mode=True)
        upsampled_logits = {}
        preds = {}
        for granularity, logit in logits.items():
            upsampled_logits[granularity] = self.upsample_logits(logit, batch[f"labels_{granularity}"])
            preds[granularity] = torch.argmax(upsampled_logits[granularity], dim=-1 if self.loss_params.focal_loss else 1)

        eval_loss = 0
        for granularity in self.granularities:
            eval_loss += self.logits_to_loss(
                upsampled_logits[granularity], batch[f"labels_{granularity}"], reprs,
                num_labels_for_granularity(granularity),
                background_class_for_granularity(granularity),
            )

            for computed_granularity in upsampled_logits:
                these_labels = batch[f"labels_{computed_granularity}"]
                these_preds = preds[computed_granularity]
                self.jaccards[computed_granularity].update(these_preds, these_labels)
                self.hierarchical_metrics[computed_granularity].update(these_preds, these_labels)
                # fixme: removed to save memory
                # if save_preds:
                #     self.test_preds[computed_granularity].append(these_preds.detach().cpu())
                #     self.test_targets[computed_granularity].append(these_labels.detach().cpu())
        
        # Compute spatial consistency scores
        if save_preds and "subpart" in preds and "part" in preds and "whole" in preds:
            self._compute_spatial_consistency(preds)
        # # === DEBUG DUMP & EXIT ===
        # if save_preds:
        #     import seaborn as sns
        #     import matplotlib.pyplot as plt
        #     import sys
        #     for idx in range(len(batch["pixel_values"])):  # iterate over batch
        #         for granularity in self.prediction_granularities:
        #             # take the second sample in batch
        #             pred_mask = preds[granularity][idx].detach().cpu().numpy().flatten()
        #             gt_mask   = batch[f"labels_{granularity}"][idx].detach().cpu().numpy().flatten()
        #
        #             num_classes = num_labels_for_granularity(granularity)
        #             class_ids = np.arange(num_classes)
        #
        #             histograms = False
        #             if histograms:
        #                 # compute counts
        #                 counts_pred = np.bincount(pred_mask, minlength=num_classes)
        #                 counts_gt = np.bincount(gt_mask, minlength=num_classes)
        #
        #                 # get seaborn palette
        #                 palette = sns.color_palette(None, num_classes)
        #
        #                 # 2 rows × 2 cols: [pred_img, gt_img] over [pred_hist, gt_hist]
        #                 fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
        #
        #                 # top-left: predicted mask
        #                 axs[0, 0].imshow(pred_mask.reshape(batch["labels_"+granularity][idx].shape), cmap="gray")
        #                 axs[0, 0].set_title(f"Predicted mask ({granularity})")
        #                 axs[0, 0].axis("off")
        #
        #                 # top-right: ground-truth mask
        #                 axs[0, 1].imshow(gt_mask.reshape(batch["labels_" + granularity][1].shape), cmap="gray")
        #                 axs[0, 1].set_title(f"Ground-truth mask ({granularity})")
        #                 axs[0, 1].axis("off")
        #
        #                 # bottom-left: pred distribution
        #                 sns.barplot(x=class_ids, y=counts_pred, hue=class_ids, palette=palette, legend=False, ax=axs[1, 0])
        #                 axs[1, 0].set_title(f"Pred counts ({granularity})")
        #                 axs[1, 0].set_xlabel("Class ID")
        #                 axs[1, 0].set_ylabel("Frequency")
        #                 axs[1, 0].set_yscale("log")
        #                 y_max = max(counts_pred.max(), counts_gt.max()) * 1.2
        #                 axs[1, 0].set_ylim(1, y_max)
        #
        #                 tick_labels = []
        #                 for i, count in enumerate(counts_pred):
        #                     if count > 0:
        #                         tick_labels.append(f"$\\mathbf{{{i}}}$")
        #                     else:
        #                         tick_labels.append(str(i))
        #                 axs[1, 0].set_xticklabels(tick_labels)
        #
        #                 # bottom-right: gt distribution
        #                 sns.barplot(x=class_ids, y=counts_gt, hue=class_ids, palette=palette, legend=False, ax=axs[1, 1])
        #                 axs[1, 1].set_title(f"GT counts ({granularity})")
        #                 axs[1, 1].set_xlabel("Class ID")
        #                 axs[1, 1].set_ylabel("Frequency")
        #                 axs[1, 1].set_yscale("log")
        #                 axs[1, 1].set_ylim(1, y_max)
        #
        #                 tick_labels = []
        #                 for i, count in enumerate(counts_gt):
        #                     if count > 0:
        #                         tick_labels.append(f"$\\mathbf{{{i}}}$")
        #                     else:
        #                         tick_labels.append(str(i))
        #                 axs[1, 1].set_xticklabels(tick_labels)
        #
        #                 fig.tight_layout()
        #                 plt.savefig(f"visualizations/plots/preds_hist/preds_gt_hist_{granularity}_{idx}.png")
        #                 plt.close(fig)
        #             else:
        #                 from PIL import Image
        #                 from visualizations.dataset_viewer import colorise_mask
        #                 from visualizations.dataset_viewer import blend
        #                 from visualizations.dataset_viewer import get_dataset
        #
        #                 mask_rgb = colorise_mask(pred_mask.reshape(batch["labels_"+granularity][idx].shape), granularity)
        #                 img = get_dataset().spin_api.get_image(batch_idx * batch["pixel_values"].shape[0] + idx)
        #                 resized_mask = mask_rgb.resize(
        #                     img.size,
        #                     resample=Image.NEAREST
        #                 )
        #                 blended = blend(img, resized_mask, alpha=0.5)
        #                 plt.imshow(blended)
        #                 plt.axis("off")
        #                 # plt.title(f"Predicted mask for {granularity} granularity")
        #                 plt.tight_layout()
        #                 plt.savefig(f"visualizations/plots/preds/preds_{granularity}_{idx}.png")
        #                 plt.close()
        #     sys.exit(0)
        return eval_loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.evaluation_step(batch, batch_idx)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        test_loss = self.evaluation_step(batch, batch_idx, save_preds=True)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        return test_loss

    def _compute_containment_scores(self, child_pred, parent_pred, child_granularity, parent_granularity):
        """
        Compute containment scores for child→parent relationship.
        
        Args:
            child_pred: predictions for child level
            parent_pred: predictions for parent level
            child_granularity: granularity name for child ("subpart" or "part")
            parent_granularity: granularity name for parent ("part" or "whole")
        
        Returns:
            list: Containment ratios for each child instance
        """
        batch_size = child_pred.shape[0]
        scores = []
        
        # Get background indices for each granularity
        child_bg = background_class_for_granularity(child_granularity)
        parent_bg = background_class_for_granularity(parent_granularity)
        
        for b in range(batch_size):
            unique_children = np.unique(child_pred[b])
            for child_idx in unique_children:
                if child_idx == child_bg:  # Skip background
                    continue
                
                # Convert prediction indices to hierarchy node IDs
                child_id = prediction_idx_to_hierarchy_id(child_granularity, int(child_idx))
                
                # Find parent in hierarchy
                if child_id not in self.child_to_parent:
                    continue
                parent_hierarchy_id = self.child_to_parent[child_id]
                
                # Convert parent hierarchy ID back to prediction index
                # Background (id=0) maps to background index for that granularity
                if parent_hierarchy_id == 0:
                    parent_idx = parent_bg
                elif parent_granularity == "part":
                    # Parts: hierarchy 216-255 -> prediction 0-39
                    parent_idx = parent_hierarchy_id - 216
                elif parent_granularity == "whole":
                    # Wholes: hierarchy 205-215 -> prediction 0-10
                    parent_idx = parent_hierarchy_id - 205
                else:
                    # Subparts: direct mapping
                    parent_idx = parent_hierarchy_id
                
                # Create binary masks and compute containment
                child_mask = (child_pred[b] == child_idx)
                parent_mask = (parent_pred[b] == parent_idx)
                child_size = child_mask.sum()
                
                if child_size > 0:
                    intersection_size = (child_mask & parent_mask).sum()
                    containment_ratio = intersection_size / child_size
                    scores.append(containment_ratio)
        
        return scores
    
    def _compute_spatial_consistency(self, preds):
        """
        Compute spatial consistency between hierarchical levels.
        """
        subpart_pred = preds["subpart"].detach().cpu().numpy()
        part_pred = preds["part"].detach().cpu().numpy()
        whole_pred = preds["whole"].detach().cpu().numpy()
        
        # Subpart → Part consistency
        subpart_to_part_scores = self._compute_containment_scores(
            subpart_pred, part_pred, "subpart", "part"
        )
        if len(subpart_to_part_scores) > 0:
            self.spatial_consistency_scores["subpart_to_part"].append(
                np.mean(subpart_to_part_scores)
            )
        
        # Part → Whole consistency
        part_to_whole_scores = self._compute_containment_scores(
            part_pred, whole_pred, "part", "whole"
        )
        if len(part_to_whole_scores) > 0:
            self.spatial_consistency_scores["part_to_whole"].append(
                np.mean(part_to_whole_scores)
            )

    def on_test_epoch_end(self):
        self.on_eval_epoch_end("test")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import numpy.ma as ma

        # fixme: removed to save memory
        # for granularity in self.granularities:
        #     # Aggregate predictions and targets across batches.
        #     preds = torch.cat(self.test_preds[granularity], dim=0).numpy().flatten()
        #     targets = torch.cat(self.test_targets[granularity], dim=0).numpy().flatten()
        #     share_correct = np.sum(preds == targets) / len(preds)
        #     num_classes = num_labels_for_granularity(granularity)
        #     # Compute confusion matrix. Ensure that all classes are represented.
        #     cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
        #     cm = np.log1p(cm)  # Apply log1p to avoid log(0)
        #     # set diagonal to 0
        #     np.fill_diagonal(cm, 0)
        #     cm_masked = ma.masked_where(cm == 0, cm)
        #
        #     fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
        #     cax = ax.matshow(cm_masked, cmap=plt.cm.Reds, vmin=0.0)
        #     fig.colorbar(cax)
        #     ax.set_title(f"Log-Scale Confusion Matrix for {granularity}. Share correct: {share_correct:.2f}")
        #     ax.set_xlabel("Predicted")
        #     ax.set_ylabel("True")
        #     plt.tight_layout()
        #
        #     plt.savefig(f"confusion_matrix_{granularity}.png")
        #     plt.clf()
        # Reset the stored predictions and targets.
        self.test_preds = {g: [] for g in self.prediction_granularities}
        self.test_targets = {g: [] for g in self.prediction_granularities}

    def on_eval_epoch_end(self, prefix="val"):
        for granularity in self.prediction_granularities:
            metric = self.jaccards[granularity]
            hierarchical_metric = self.hierarchical_metrics[granularity]
            self.log(f"{prefix}_mIoU_{granularity}", metric.compute(), prog_bar=True)
            self.log(f"{prefix}_hierarchical_mIoU_{granularity}", hierarchical_metric.compute(), prog_bar=True)
            metric.reset()
            hierarchical_metric.reset()
        
        # Log spatial consistency scores
        if len(self.spatial_consistency_scores["subpart_to_part"]) > 0:
            score = np.mean(self.spatial_consistency_scores["subpart_to_part"])
            self.log(f"{prefix}_spatial_consistency_subpart_to_part", score, prog_bar=True)
        
        if len(self.spatial_consistency_scores["part_to_whole"]) > 0:
            score = np.mean(self.spatial_consistency_scores["part_to_whole"])
            self.log(f"{prefix}_spatial_consistency_part_to_whole", score, prog_bar=True)
        
        # Reset spatial consistency scores
        self.spatial_consistency_scores = {
            "subpart_to_part": [],
            "part_to_whole": []
        }

    def on_validation_epoch_end(self):
        self.on_eval_epoch_end("val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        
        # Polynomial learning rate scheduler
        def poly_lr_lambda(epoch):
            return (1 - epoch / self.num_epochs) ** self.poly_decay_power
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @staticmethod
    def upsample_logits(logits, labels):
        return nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )

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
