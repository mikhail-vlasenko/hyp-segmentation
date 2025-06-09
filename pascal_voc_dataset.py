import os
from pathlib import Path
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

import lightning as L
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
import torchvision.transforms.functional as TF
from transformers import SegformerImageProcessor

import torch

# Pascal VOC to SPIN "whole" granularity mapping
SPIN_WHOLES = [
    "Quadruped", "Biped", "Fish", "Bird", "Snake",
    "Reptile", "Car", "Bicycle", "Boat", "Aeroplane", "Bottle", "Background"
]

# Pascal VOC classes that map to SPIN wholes
PASCAL_VOC_TO_SPIN_MAPPING = {
    # Direct mappings
    0: 11,   # background -> Background
    1: 9,    # aeroplane -> Aeroplane  
    2: 7,    # bicycle -> Bicycle
    3: 3,    # bird -> Bird
    4: 8,    # boat -> Boat
    5: 10,   # bottle -> Bottle
    7: 6,    # car -> Car
    
    # Quadruped mappings (merge multiple Pascal classes to single SPIN class)
    8: 0,    # cat -> Quadruped
    10: 0,   # cow -> Quadruped
    12: 0,   # dog -> Quadruped
    13: 0,   # horse -> Quadruped
    17: 0,   # sheep -> Quadruped
    
    # Classes to ignore (set to 255 - ignore index)
    6: 255,   # bus
    9: 255,   # chair
    11: 255,  # diningtable
    14: 255,  # motorbike
    15: 255,  # person
    16: 255,  # pottedplant
    18: 255,  # sofa
    19: 255,  # train
    20: 255,  # tvmonitor
}

# Reverse mapping for creating Pascal VOC classes from SPIN
SPIN_TO_PASCAL_VOC_CLASSES = [
    "quadruped",    # 0: Quadruped (SPIN) -> quadruped (Pascal subset)
    None,           # 1: Biped (no direct Pascal equivalent)
    None,           # 2: Fish (no direct Pascal equivalent)  
    "bird",         # 3: Bird -> bird
    None,           # 4: Snake (no direct Pascal equivalent)
    None,           # 5: Reptile (no direct Pascal equivalent)
    "car",          # 6: Car -> car
    "bicycle",      # 7: Bicycle -> bicycle
    "boat",         # 8: Boat -> boat
    "aeroplane",    # 9: Aeroplane -> aeroplane
    "bottle",       # 10: Bottle -> bottle
    "background",   # 11: Background -> background
]

# Valid Pascal VOC subset classes (those that have SPIN equivalents)
PASCAL_VOC_SUBSET_CLASSES = [cls for cls in SPIN_TO_PASCAL_VOC_CLASSES if cls is not None]


def create_pascal_voc_label_mapping():
    """Create mapping array to convert Pascal VOC labels to SPIN-aligned labels."""
    # Create mapping array
    mapping = torch.full((256,), 255, dtype=torch.long)  # Default to ignore
    
    # Create reverse lookup for SPIN to Pascal subset indices
    spin_to_subset = {}
    subset_idx = 0
    for spin_idx, pascal_class in enumerate(SPIN_TO_PASCAL_VOC_CLASSES):
        if pascal_class is not None:
            spin_to_subset[spin_idx] = subset_idx
            subset_idx += 1
    
    # Apply mappings
    for pascal_idx, spin_idx in PASCAL_VOC_TO_SPIN_MAPPING.items():
        if spin_idx != 255 and spin_idx in spin_to_subset:
            mapping[pascal_idx] = spin_to_subset[spin_idx]
    
    return mapping


def create_pascal_voc_prototypes_from_spin(spin_prototypes):
    """
    Create Pascal VOC prototypes from SPIN "whole" granularity prototypes.
    
    Args:
        spin_prototypes: Tensor of shape [num_spin_classes, prototype_dim] 
                        representing SPIN "whole" granularity prototypes
    
    Returns:
        Tensor of shape [num_pascal_subset_classes, prototype_dim]
        representing prototypes for Pascal VOC subset classes
    """
    num_subset_classes = len(PASCAL_VOC_SUBSET_CLASSES)
    prototype_dim = spin_prototypes.shape[-1]
    hyperbolic = len(spin_prototypes.shape) == 3
    
    # Initialize Pascal VOC prototypes
    pascal_prototypes = torch.zeros(num_subset_classes, prototype_dim, dtype=spin_prototypes.dtype, device=spin_prototypes.device)
    
    # Map SPIN prototypes to Pascal subset prototypes
    subset_idx = 0
    for spin_idx, pascal_class in enumerate(SPIN_TO_PASCAL_VOC_CLASSES):
        if pascal_class is not None:
            pascal_prototypes[subset_idx] = spin_prototypes[spin_idx]
            subset_idx += 1
    
    if hyperbolic:
        pascal_prototypes = pascal_prototypes.unsqueeze(1)
    return pascal_prototypes


class RandomCropAndFlip:
    """
    A custom transform that applies a random crop and random horizontal flip
    to both an image and its segmentation map.
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image, segmentation_map):
        """
        Apply the same random crop and flip to both image and segmentation map.
        """
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


class PascalVOCSegmentationDataset(Dataset):
    """
    Pascal VOC 2012 Segmentation Dataset
    
    Expected directory structure:
    voc_root/
    ├── JPEGImages/          # RGB images
    ├── SegmentationClass/   # Segmentation masks
    └── ImageSets/
        └── Segmentation/
            ├── train.txt    # Training image IDs
            ├── val.txt      # Validation image IDs
            └── trainval.txt # Combined training and validation
    """
    
    # Pascal VOC 2012 has 20 object classes + 1 background class
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    def __init__(
        self,
        voc_root: str | Path,
        split: str,
        processor: SegformerImageProcessor,
        crop_size: Optional[Tuple[float, float]] = None,
        align_with_spin: bool = True,
    ):
        """
        Args:
            voc_root: Root directory of Pascal VOC dataset
            split: Dataset split ('train', 'val', 'trainval')
            processor: SegformerImageProcessor for preprocessing
            crop_size: Tuple of (height_ratio, width_ratio) for random cropping during training
        """
        self.voc_root = Path(voc_root)
        self.split = split
        self.processor = processor
        self.align_with_spin = align_with_spin
        
        if align_with_spin:
            self.num_classes = len(PASCAL_VOC_SUBSET_CLASSES)
            self.label_mapping = create_pascal_voc_label_mapping()
        else:
            self.num_classes = len(self.CLASSES)
            self.label_mapping = None
        
        # Load image IDs for the specified split
        split_file = self.voc_root / "ImageSets" / "Segmentation" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        # Set up transforms
        if self.split == "train" and crop_size is not None:
            self.transform = RandomCropAndFlip(crop_size=crop_size)
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = self.voc_root / "JPEGImages" / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")
        
        # Load segmentation mask
        mask_path = self.voc_root / "SegmentationClass" / f"{image_id}.png"
        segmentation_map = Image.open(mask_path)
        
        # Apply SPIN alignment if enabled
        if self.align_with_spin:
            # Convert to numpy for label mapping
            seg_array = np.array(segmentation_map)
            original_shape = seg_array.shape
            # Convert to torch tensor, flatten, apply mapping, then reshape back
            seg_tensor = torch.from_numpy(seg_array).flatten().long()
            seg_mapped = self.label_mapping[seg_tensor].numpy()
            seg_array = seg_mapped.reshape(original_shape)
            segmentation_map = Image.fromarray(seg_array.astype(np.uint8))
        
        # Apply transforms if specified
        if self.transform:
            image, segmentation_map = self.transform(image, segmentation_map)
        
        # Process with SegformerImageProcessor
        inputs = self.processor(
            images=image, 
            segmentation_maps=segmentation_map, 
            return_tensors="pt"
        )
        
        # Remove batch dimension (since processor adds it)
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        
        # Add granularity-specific labels for compatibility with SPIN model
        # Pascal VOC is treated as "whole" granularity level
        inputs["labels_whole"] = inputs["labels"]
        
        return inputs

    @classmethod
    def get_class_names(cls, align_with_spin: bool = False) -> List[str]:
        """Return list of class names."""
        if align_with_spin:
            return PASCAL_VOC_SUBSET_CLASSES.copy()
        return cls.CLASSES.copy()

    @classmethod
    def get_num_classes(cls, align_with_spin: bool = False) -> int:
        """Return number of classes including background."""
        if align_with_spin:
            return len(PASCAL_VOC_SUBSET_CLASSES)
        return len(cls.CLASSES)


class PascalVOCDataModule(L.LightningDataModule):
    """
    A LightningDataModule for Pascal VOC 2012 segmentation dataset.
    """

    def __init__(
        self,
        voc_root: str,
        processor: SegformerImageProcessor,
        batch_size: int = 8,
        crop_size: Tuple[float, float] = (0.8, 0.8),
        num_workers: int = 4,
    ):
        """
        Args:
            voc_root: Root directory of Pascal VOC dataset
            processor: SegformerImageProcessor for preprocessing
            batch_size: Batch size for dataloaders
            crop_size: Tuple of (height_ratio, width_ratio) for random cropping during training
            num_workers: Number of workers for data loading
        """
        super().__init__()
        self.voc_root = voc_root
        self.processor = processor
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """Set up train/val/test datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = PascalVOCSegmentationDataset(
                voc_root=self.voc_root,
                split="train",
                processor=self.processor,
                crop_size=self.crop_size,
                align_with_spin=True,
            )
        
        self.val_dataset = PascalVOCSegmentationDataset(
            voc_root=self.voc_root,
            split="val",
            processor=self.processor,
            align_with_spin=True,
        )
        
        # Pascal VOC doesn't have a separate test set, use val for testing
        self.test_dataset = self.val_dataset

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

    def get_class_names(self):
        """Return list of class names."""
        return PascalVOCSegmentationDataset.get_class_names(align_with_spin=True)

    def get_num_classes(self):
        """Return number of classes including background."""
        return PascalVOCSegmentationDataset.get_num_classes(align_with_spin=True)
