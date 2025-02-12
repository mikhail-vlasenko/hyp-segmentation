import os
import argparse
import torch
import numpy as np
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
import torchvision.transforms.functional as TF

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from torchmetrics.classification import MulticlassJaccardIndex

from spin import SPIN
from segformer_head import HyperbolicSegformerDecodeHead
from PIL import Image


class RandomCropAndFlip:
    """
    A custom transform that applies a random crop and random horizontal flip
    to both an image and its segmentation map.
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image, segmentation_map):
        # image.size => (width, height)
        w, h = image.size
        crop_w, crop_h = int(self.crop_size[0] * w), int(self.crop_size[1] * h)

        # Random crop
        i, j, h_, w_ = transforms.RandomCrop.get_params(image, (crop_h, crop_w))
        image = TF.crop(image, i, j, h_, w_)
        segmentation_map = TF.crop(segmentation_map, i, j, h_, w_)

        # Random horizontal flip
        if np.random.random() > 0.5:
            image = TF.hflip(image)
            segmentation_map = TF.hflip(segmentation_map)

        return image, segmentation_map


class SPINSegmentationDataset(Dataset):
    def __init__(self, annotation_dir, image_dir, split, processor, crop_size=None):
        self.spin_api = SPIN(
            annotation_dir=annotation_dir,
            image_dir=image_dir,
            split=split,
            download=False,
        )
        self.processor = processor
        self.image_ids = self.spin_api.getImgIds()
        self.split = split

        if self.split == "train" and crop_size is not None:
            self.transform = RandomCropAndFlip(crop_size=crop_size)
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Load image
        image = self.spin_api.get_image(image_id)

        # Generate segmentation map
        segmentation_map = self.spin_api.rasterize_coco_segmentations(
            self.spin_api.subparts, image_id, background_class=0
        )
        segmentation_map = Image.fromarray(segmentation_map.astype("uint8"))

        # Apply custom transforms (random crop/flip) if training
        if self.transform is not None:
            image, segmentation_map = self.transform(image, segmentation_map)

        # Process for SegFormer
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
        processor: SegformerImageProcessor,
        batch_size: int = 8,
        crop_size=(0.8, 0.8),
        num_workers=4,
    ):
        super().__init__()
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
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
                crop_size=self.crop_size,
            )
            self.val_dataset = SPINSegmentationDataset(
                self.annotation_dir,
                self.image_dir,
                split="val",
                processor=self.processor,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class SegformerLightningModule(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Processor is separate, used for dataset collations
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        # Replace the decode head with the custom hyperbolic head
        self.model.decode_head = HyperbolicSegformerDecodeHead.from_segformer_decode_head(
            self.model.decode_head, num_labels, None, hyperbolic=False
        )

        # Metric: mIoU
        self.jaccard = MulticlassJaccardIndex(
            num_classes=num_labels,
            ignore_index=0,
        )
        self.lr = lr

    def forward(self, pixel_values, labels=None):
        # Standard forward for huggingface model
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        outputs = self.forward(pixel_values, labels)
        loss = outputs.loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        outputs = self.forward(pixel_values, labels=labels)

        val_loss = outputs.loss
        # Upsample logits to original size
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        preds = torch.argmax(upsampled_logits, dim=1)

        # Update Jaccard (mIoU) metric
        self.jaccard.update(preds, labels)

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def on_validation_epoch_end(self):
        miou = self.jaccard.compute()
        self.log("val_mIoU", miou, prog_bar=True)
        # Reset for next epoch
        self.jaccard.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train SegFormer for Semantic Segmentation")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--model_name", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512", help="Name of the pre-trained model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--crop_size", type=float, nargs=2, default=(0.8, 0.8), help="Crop size as a fraction of image dimensions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_annotation_dir = os.path.join(args.dataset_dir, "annotations")
    dataset_image_dir = os.path.join(args.dataset_dir, "images")

    NUM_CLASSES = 204  # 203 classes + 1 background
    L.seed_everything(args.seed, workers=True)

    # Processor
    processor = SegformerImageProcessor.from_pretrained(args.model_name)
    processor.do_reduce_labels = False

    spin_dm = SPINDataModule(
        annotation_dir=dataset_annotation_dir,
        image_dir=dataset_image_dir,
        processor=processor,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        num_workers=args.num_workers,

    )

    segformer_module = SegformerLightningModule(
        model_name=args.model_name,
        num_labels=NUM_CLASSES,
        lr=args.learning_rate,
    )

    wandb_logger = WandbLogger(
        project="hyperbolic-segmentation",
        log_model=True  # Logs checkpoints
    )

    trainer = L.Trainer(
        logger=wandb_logger,
        max_epochs=args.num_epochs,
        accelerator="auto",
        devices="auto",
    )

    trainer.fit(segformer_module, spin_dm)

    save_dir = "segformer-finetuned-spin"
    os.makedirs(save_dir, exist_ok=True)

    segformer_module.model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
