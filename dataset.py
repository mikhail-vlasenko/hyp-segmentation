import json
from pathlib import Path
from pprint import pprint
from typing import List, Optional

import numpy as np
from spin import SPIN, file_to_object_mapping
from PIL import Image

import lightning as L
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
import torchvision.transforms.functional as TF
from transformers import SegformerImageProcessor

from hierarchy_embeddings.object_supercategory_mapping import OBJECT_SUPERCATEGORY_MAPPING
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
    def __init__(
            self,
            annotation_dir: str | Path,
            image_dir: str | Path,
            split: str,
            granularities: List[str],
            processor,
            crop_size: Optional[int] = None,
            remap_objects: bool = False,
    ):
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

        # Optional coarse mapping for whole‑object masks
        self._whole_lut = None
        if remap_objects:
            assert len(OBJECT_SUPERCATEGORY_MAPPING) == 158
            assert max(OBJECT_SUPERCATEGORY_MAPPING) == 10

            # background is last in whole, there are 11 supercategories
            # Build a NumPy LUT of length 159: 0‑157 from JSON, 158 → background
            self._whole_lut = np.asarray(OBJECT_SUPERCATEGORY_MAPPING + [11], dtype=np.int16)

    def __len__(self):
        return len(self.image_ids)

    def get_segmentation_map(self, image_id, granularity):
        segmentation_map = self.spin_api.rasterize_coco_segmentations(
            self.spin_api.__getattribute__(granularity + "s"),
            image_id,
            background_class=background_class_for_granularity(granularity),
        )

        # Optional remap for whole granularity --------------------------------
        if granularity == "whole" and self._whole_lut is not None:
            segmentation_map = self._whole_lut[segmentation_map]

        return Image.fromarray(segmentation_map.astype("uint8"))

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = self.spin_api.get_image(image_id)

        segmentation_maps = [
            self.get_segmentation_map(image_id, granularity)
            for granularity in self.granularities
        ]

        if self.transform:
            image, segmentation_maps = self.transform(image, segmentation_maps)

        inputs = {}
        for granularity, segmentation_map in zip(self.granularities, segmentation_maps):
            res = self.processor(
                images=image, segmentation_maps=segmentation_map, return_tensors="pt"
            )
            inputs["pixel_values"] = res["pixel_values"]
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
        remap_objects: bool = False,
    ):
        super().__init__()
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
        self.granularities = granularities
        self.processor = processor
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.num_workers = num_workers
        self.remap_objects = remap_objects

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
                remap_objects=self.remap_objects,
            )
        self.val_dataset = SPINSegmentationDataset(
            self.annotation_dir,
            self.image_dir,
            split="val",
            granularities=["whole", "part", "subpart"],  # just put all of them here, sometimes used in eval
            processor=self.processor,
            remap_objects=self.remap_objects,
        )
        self.test_dataset = SPINSegmentationDataset(
            self.annotation_dir,
            self.image_dir,
            split="test",
            granularities=["whole", "part", "subpart"],
            processor=self.processor,
            remap_objects=self.remap_objects,
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
