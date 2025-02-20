import copy
import os
import argparse
import torch
import numpy as np
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.nn import CrossEntropyLoss

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
import torchvision.transforms.functional as TF

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from torchmetrics.classification import MulticlassJaccardIndex

from spin import SPIN
from segformer_head import HyperbolicSegformerDecodeHead
from PIL import Image


def background_class_for_granularity(granularity):
    return {"whole": 158, "part": 40, "subpart": 0}[granularity]

def num_labels_for_granularity(granularity):
    return {"whole": 159, "part": 41, "subpart": 204}[granularity]  # classes + 1 background

class RandomCropAndFlip:
    """
    A custom transform that applies a random crop and random horizontal flip
    to both an image and its segmentation map.
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image, segmentation_map):
        """
        Can take multiple segmentation maps as input. Transforms them all in the same way.
        """
        # image.size => (width, height)
        w, h = image.size
        crop_w, crop_h = int(self.crop_size[0] * w), int(self.crop_size[1] * h)

        # Random crop
        i, j, h_, w_ = transforms.RandomCrop.get_params(image, (crop_h, crop_w))
        image = TF.crop(image, i, j, h_, w_)
        if isinstance(segmentation_map, list):
            segmentation_map = [TF.crop(s, i, j, h_, w_) for s in segmentation_map]
        else:
            segmentation_map = TF.crop(segmentation_map, i, j, h_, w_)

        # Random horizontal flip
        if np.random.random() > 0.5:
            image = TF.hflip(image)
            if isinstance(segmentation_map, list):
                segmentation_map = [TF.hflip(s) for s in segmentation_map]
            else:
                segmentation_map = TF.hflip(segmentation_map)

        return image, segmentation_map


class SPINSegmentationDataset(Dataset):
    def __init__(self, annotation_dir, image_dir, split, granularity, processor, crop_size=None):
        self.spin_api = SPIN(
            annotation_dir=annotation_dir,
            image_dir=image_dir,
            split=split,
            download=False,
        )
        self.granularity = granularity
        self.processor = processor
        self.image_ids = self.spin_api.getImgIds()
        self.split = split

        if self.split == "train" and crop_size is not None:
            self.transform = RandomCropAndFlip(crop_size=crop_size)
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_ids)

    def get_segmentation_map(self, image_id, granularity):
        segmentation_map = self.spin_api.rasterize_coco_segmentations(
            self.spin_api.__getattribute__(granularity + "s"),
            image_id,
            background_class=background_class_for_granularity(granularity),
        )
        segmentation_map = Image.fromarray(segmentation_map.astype("uint8"))
        return segmentation_map

    def __getitem__(self, idx):
        # todo: infer the train signal from one segmentation map rather than constructing 3 of them
        #   requires the object class to be general (quadruped instead of dog)
        image_id = self.image_ids[idx]

        # Load image
        image = self.spin_api.get_image(image_id)

        if self.granularity == "all":
            # Load all segmentation maps
            segmentation_map_whole = self.get_segmentation_map(image_id, "whole")
            segmentation_map_part = self.get_segmentation_map(image_id, "part")
            segmentation_map_subpart = self.get_segmentation_map(image_id, "subpart")
        else:
            segmentation_map = self.get_segmentation_map(image_id, self.granularity)

        # Apply custom transforms (random crop/flip) if training
        if self.transform is not None:
            if self.granularity == "all":
                image, [segmentation_map_whole, segmentation_map_part, segmentation_map_subpart] = self.transform(
                    image, [segmentation_map_whole, segmentation_map_part, segmentation_map_subpart]
                )
            else:
                image, segmentation_map = self.transform(image, segmentation_map)

        if self.granularity == "all":
            inputs = {}
            wholes = self.processor(
                images=image,
                segmentation_maps=segmentation_map_whole,
                return_tensors="pt",
            )
            inputs["pixel_values"] = wholes["pixel_values"]
            inputs["labels_whole"] = wholes["labels"]
            inputs["labels_part"] = self.processor(
                images=image,
                segmentation_maps=segmentation_map_part,
                return_tensors="pt",
            )["labels"]
            inputs["labels_subpart"] = self.processor(
                images=image,
                segmentation_maps=segmentation_map_subpart,
                return_tensors="pt",
            )["labels"]
        else:
            inputs = self.processor(
                images=image,
                segmentation_maps=segmentation_map,
                return_tensors="pt",
            )

        # Remove batch dimension (since processor adds it)
        inputs = {k: v.squeeze() for k, v in inputs.items()}

        return inputs


class SPINDataModule(L.LightningDataModule):
    """
    A LightningDataModule to handle train/val dataset creation
    and produce the corresponding dataloaders.
    """

    def __init__(
        self,
        annotation_dir: str,
        image_dir: str,
        granularity: str,
        processor: SegformerImageProcessor,
        batch_size: int = 8,
        crop_size=(0.8, 0.8),
        num_workers=4,
    ):
        super().__init__()
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
        self.granularity = granularity
        self.processor = processor
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Create train/val datasets
        if stage == "fit" or stage is None:
            self.train_dataset = SPINSegmentationDataset(
                self.annotation_dir,
                self.image_dir,
                split="train",
                processor=self.processor,
                granularity=self.granularity,
                crop_size=self.crop_size,
            )
            self.val_dataset = SPINSegmentationDataset(
                self.annotation_dir,
                self.image_dir,
                split="val",
                granularity=self.granularity,
                processor=self.processor,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


class SegformerLightningModule(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
    ):
        super().__init__()
        # Save all hyperparameters so they can be later accessed via self.hparams
        self.save_hyperparameters()

        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.backbone = self.model.segformer
        original_decode_head = self.model.decode_head

        # Create three separate decode heads by reusing the original one as a template.
        self.decode_head_whole = HyperbolicSegformerDecodeHead.from_segformer_decode_head(
            copy.deepcopy(original_decode_head), num_labels_for_granularity("whole"), None, hyperbolic=False
        )
        self.decode_head_part = HyperbolicSegformerDecodeHead.from_segformer_decode_head(
            copy.deepcopy(original_decode_head), num_labels_for_granularity("part"), None, hyperbolic=False
        )
        self.decode_head_subpart = HyperbolicSegformerDecodeHead.from_segformer_decode_head(
            copy.deepcopy(original_decode_head), num_labels_for_granularity("subpart"), None, hyperbolic=False
        )

        self.jaccard_whole = MulticlassJaccardIndex(
            num_classes=num_labels_for_granularity("whole"), ignore_index=background_class_for_granularity("whole")
        )
        self.jaccard_part = MulticlassJaccardIndex(
            num_classes=num_labels_for_granularity("part"), ignore_index=background_class_for_granularity("part")
        )
        self.jaccard_subpart = MulticlassJaccardIndex(
            num_classes=num_labels_for_granularity("subpart"), ignore_index=background_class_for_granularity("subpart")
        )
        self.lr = lr

    def logits_to_loss(self, logits, labels, ignore_index, return_preds=False):
        # upsample logits to the images' original size
        upsampled_logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
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
        # Compute logits from each decoding head using the same backbone features
        logits_whole = self.decode_head_whole(encoder_hidden_states)
        logits_part = self.decode_head_part(encoder_hidden_states)
        logits_subpart = self.decode_head_subpart(encoder_hidden_states)
        return {
            "logits_whole": logits_whole,
            "logits_part": logits_part,
            "logits_subpart": logits_subpart,
        }

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels_whole = batch["labels_whole"]
        labels_part = batch["labels_part"]
        labels_subpart = batch["labels_subpart"]

        outputs = self.forward(pixel_values)
        logits_whole = outputs["logits_whole"]
        logits_part = outputs["logits_part"]
        logits_subpart = outputs["logits_subpart"]

        loss_whole = self.logits_to_loss(logits_whole, labels_whole, background_class_for_granularity("whole"))
        loss_part = self.logits_to_loss(logits_part, labels_part, background_class_for_granularity("part"))
        loss_subpart = self.logits_to_loss(logits_subpart, labels_subpart, background_class_for_granularity("subpart"))
        total_loss = loss_whole + loss_part + loss_subpart  # add coefs?

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels_whole = batch["labels_whole"]
        labels_part = batch["labels_part"]
        labels_subpart = batch["labels_subpart"]

        outputs = self.forward(pixel_values)
        logits_whole = outputs["logits_whole"]
        logits_part = outputs["logits_part"]
        logits_subpart = outputs["logits_subpart"]

        result_whole = self.logits_to_loss(logits_whole, labels_whole, background_class_for_granularity("whole"), return_preds=True)
        result_part = self.logits_to_loss(logits_part, labels_part, background_class_for_granularity("part"), return_preds=True)
        result_subpart = self.logits_to_loss(logits_subpart, labels_subpart, background_class_for_granularity("subpart"), return_preds=True)
        val_loss = result_whole["loss"] + result_part["loss"] + result_subpart["loss"]

        # Update metrics for each granularity
        self.jaccard_whole.update(result_whole["preds"], labels_whole)
        self.jaccard_part.update(result_part["preds"], labels_part)
        self.jaccard_subpart.update(result_subpart["preds"], labels_subpart)

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def on_validation_epoch_end(self):
        miou_whole = self.jaccard_whole.compute()
        miou_part = self.jaccard_part.compute()
        miou_subpart = self.jaccard_subpart.compute()
        self.log("val_mIoU_whole", miou_whole, prog_bar=True)
        self.log("val_mIoU_part", miou_part, prog_bar=True)
        self.log("val_mIoU_subpart", miou_subpart, prog_bar=True)
        self.jaccard_whole.reset()
        self.jaccard_part.reset()
        self.jaccard_subpart.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train SegFormer for Semantic Segmentation")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--model_name", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512", help="Name of the pre-trained model")
    parser.add_argument("--granularity", type=str, default="subpart", choices=["whole", "part", "subpart", "all"], required=True,
                        help="Level of segmentation granularity: whole, part, or subpart")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--crop_size", type=float, nargs=2, default=(0.8, 0.8), help="Crop size as a fraction of image dimensions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="Accumulate gradient batches before updating weights")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_annotation_dir = os.path.join(args.dataset_dir, "annotations")
    dataset_image_dir = os.path.join(args.dataset_dir, "images")

    L.seed_everything(args.seed, workers=True)

    processor = SegformerImageProcessor.from_pretrained(args.model_name)
    processor.do_reduce_labels = False

    spin_dm = SPINDataModule(
        annotation_dir=dataset_annotation_dir,
        image_dir=dataset_image_dir,
        granularity=args.granularity,
        processor=processor,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        num_workers=args.num_workers,
    )

    segformer_module = SegformerLightningModule(
        model_name=args.model_name,
        lr=args.learning_rate,
    )

    wandb_logger = WandbLogger(
        project="hyperbolic-segmentation",
        log_model=True,
    )

    wandb_logger.log_hyperparams(vars(args))

    trainer = L.Trainer(
        logger=wandb_logger,
        max_epochs=args.num_epochs,
        accelerator="auto",
        devices="auto",
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    trainer.fit(segformer_module, spin_dm)

    save_dir = "segformer-finetuned-spin"
    os.makedirs(save_dir, exist_ok=True)

    segformer_module.model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
