import numpy as np
from spin import SPIN
from PIL import Image

import lightning as L
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
import torchvision.transforms.functional as TF
from transformers import SegformerImageProcessor

from utils import background_class_for_granularity


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
        self.test_dataset = SPINSegmentationDataset(
            self.annotation_dir,
            self.image_dir,
            split="test",
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

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
