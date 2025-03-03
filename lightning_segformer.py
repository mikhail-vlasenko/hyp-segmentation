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

from prototypes import create_prototypes
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

    def __call__(self, image, segmentation_maps):
        """
        Takes multiple segmentation maps as input. Transforms them all in the same way.
        """
        # image.size => (width, height)
        w, h = image.size
        crop_w, crop_h = int(self.crop_size[0] * w), int(self.crop_size[1] * h)

        # Random crop
        i, j, h_, w_ = transforms.RandomCrop.get_params(image, (crop_h, crop_w))
        image = TF.crop(image, i, j, h_, w_)
        segmentation_maps = [TF.crop(s, i, j, h_, w_) for s in segmentation_maps]

        # Random horizontal flip
        if np.random.random() > 0.5:
            image = TF.hflip(image)
            segmentation_maps = [TF.hflip(s) for s in segmentation_maps]

        return image, segmentation_maps


class SPINSegmentationDataset(Dataset):
    def __init__(self, annotation_dir, image_dir, split, granularities, processor, crop_size=None):
        self.spin_api = SPIN(
            annotation_dir=annotation_dir,
            image_dir=image_dir,
            split=split,
            download=False,
        )
        self.granularities = granularities
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
        image = self.spin_api.get_image(image_id)

        segmentation_maps = [
            self.get_segmentation_map(image_id, granularity) for granularity in self.granularities
        ]

        if self.transform:
            image, segmentation_maps = self.transform(image, segmentation_maps)

        inputs = {}
        for granularity, segmentation_map in zip(self.granularities, segmentation_maps):
            res = self.processor(
                images=image, segmentation_maps=segmentation_map, return_tensors="pt"
            )  # some computational overhead here for 2+ granularities
            inputs[f"pixel_values"] = res["pixel_values"]
            inputs[f"labels_{granularity}"] = res["labels"]

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
        granularities: list[str],
        processor: SegformerImageProcessor,
        batch_size: int = 8,
        crop_size=(0.8, 0.8),
        num_workers=4,
    ):
        super().__init__()
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
        self.granularities = granularities
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
                granularities=self.granularities,
                crop_size=self.crop_size,
            )
            self.val_dataset = SPINSegmentationDataset(
                self.annotation_dir,
                self.image_dir,
                split="val",
                granularities=self.granularities,
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
        granularities: list[str],
        hyperbolic: bool,
        curvature: float,
        max_class_sep: bool,
    ):
        super().__init__()
        # Save all hyperparameters so they can be later accessed via self.hparams
        self.save_hyperparameters()

        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.backbone = self.model.segformer
        original_decode_head = self.model.decode_head

        self.lr = lr
        self.granularities = granularities

        self.max_class_sep = max_class_sep
        if self.max_class_sep:
            assert self.granularities == ["subpart"]
            self.prototypes = create_prototypes(num_labels_for_granularity("subpart"))
            self.prototypes = torch.from_numpy(self.prototypes).float()
            dim = self.prototypes.shape[1]
            self.prototypes = self.prototypes.t().cuda()

        self.decode_heads = nn.ModuleDict({
            g: HyperbolicSegformerDecodeHead.from_segformer_decode_head(
                copy.deepcopy(original_decode_head),
                num_labels_for_granularity(g) if not self.max_class_sep else dim,
                None,
                hyperbolic,
                curvature,
            )
            for g in self.granularities
        })

        self.jaccards = nn.ModuleDict({
            g: MulticlassJaccardIndex(
                num_classes=num_labels_for_granularity(g), ignore_index=background_class_for_granularity(g)
            )
            for g in self.granularities
        })

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
        result = {}
        # Compute logits from each decoding head using the same backbone features
        for granularity in self.granularities:
            logits = self.decode_heads[granularity](encoder_hidden_states)
            if self.max_class_sep:
                # logits are of shape (batch, num_classes - 1, h, w)
                # prototypes are of shape (num_classes - 1, num_classes)
                # we want (batch, num_classes, h, w) on output
                logits = torch.einsum("bchw,cd->bdhw", logits, self.prototypes)
            result[f"logits_{granularity}"] = logits
        return result

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch["pixel_values"])

        loss = 0
        for granularity in self.granularities:
            logits = outputs[f"logits_{granularity}"]
            labels = batch[f"labels_{granularity}"]
            loss += self.logits_to_loss(logits, labels, background_class_for_granularity(granularity))

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch["pixel_values"])

        val_loss = 0
        for granularity in self.granularities:
            logits = outputs[f"logits_{granularity}"]
            labels = batch[f"labels_{granularity}"]
            result = self.logits_to_loss(logits, labels, background_class_for_granularity(granularity), return_preds=True)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train SegFormer for Semantic Segmentation")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--model_name", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512", help="Name of the pre-trained model")
    parser.add_argument("--granularity", type=str, default="subpart", choices=["whole", "part", "subpart", "all"], required=True,
                        help="Level of segmentation granularity: whole, part, or subpart")
    parser.add_argument("--hyperbolic", action="store_true", help="Use hyperbolic decode head if specified")
    parser.add_argument("--curvature", type=float, default=1., help="Hyperbolic curvature")
    parser.add_argument("--max_class_sep", action="store_true", help="Use maximum class separation prototypes pipeline")
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

    if args.granularity == "all":
        granularities = ["whole", "part", "subpart"]
    else:
        granularities = [args.granularity]

    processor = SegformerImageProcessor.from_pretrained(args.model_name)
    processor.do_reduce_labels = False

    spin_dm = SPINDataModule(
        annotation_dir=dataset_annotation_dir,
        image_dir=dataset_image_dir,
        granularities=granularities,
        processor=processor,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        num_workers=args.num_workers,
    )

    segformer_module = SegformerLightningModule(
        model_name=args.model_name,
        lr=args.learning_rate,
        granularities=granularities,
        hyperbolic=args.hyperbolic,
        curvature=args.curvature,
        max_class_sep=args.max_class_sep,
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
