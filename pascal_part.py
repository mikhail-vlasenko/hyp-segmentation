import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import lightning as L
from scipy.io import loadmat
from collections import defaultdict

from torchvision.transforms import v2 as transforms
import torchvision.transforms.functional as TF


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


class PascalPartDataset(Dataset):
    """
    Pascal Part Dataset for part-level segmentation

    Expected directory structure:
    voc_root/
    ├── JPEGImages/          # RGB images
    ├── SegmentationClass/   # Class segmentation masks
    ├── Annotations_Part/    # Part annotations (.mat files)
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

    # Part index mapping (converted from MATLAB code)
    # Merged indexed parts (engine_1, engine_2, etc.) and left/right parts (lwing/rwing, etc.)
    # Also merged front/back and upper/lower variants where logically similar
    PART_INDEX_MAP = {
        1: {  # aeroplane
            'body': 1, 'stern': 2, 'wing': 3, 'tail': 4, 'engine': 5, 'wheel': 6,
            # Merge lwing/rwing -> wing, all engine_i -> engine, all wheel_i -> wheel
            'lwing': 3, 'rwing': 3,
            **{f'engine_{i}': 5 for i in range(1, 11)},
            **{f'wheel_{i}': 6 for i in range(1, 11)}
        },
        2: {  # bicycle
            'wheel': 1, 'fwheel': 1, 'bwheel': 1, 'saddle': 2, 'handlebar': 3, 'chainwheel': 4, 'headlight': 5,
            # Merge all headlight_i -> headlight
            **{f'headlight_{i}': 5 for i in range(1, 11)}
        },
        3: {  # bird
            'head': 1, 'eye': 2, 'beak': 3, 'torso': 4, 'neck': 5, 'wing': 6, 'leg': 7, 'foot': 8, 'tail': 9,
            # Merge leye/reye -> eye, lwing/rwing -> wing, lleg/rleg -> leg, lfoot/rfoot -> foot
            'leye': 2, 'reye': 2, 'lwing': 6, 'rwing': 6, 'lleg': 7, 'rleg': 7, 'lfoot': 8, 'rfoot': 8
        },
        4: {},  # boat - only silhouette mask
        5: {'cap': 1, 'body': 2},  # bottle
        6: {  # bus
            'frontside': 1, 'side': 2, 'backside': 3, 'roofside': 4, 'mirror': 5, 'liplate': 6, 'door': 7, 'wheel': 8, 'headlight': 9, 'window': 10,
            # Merge leftside/rightside -> side, leftmirror/rightmirror -> mirror, fliplate/bliplate -> liplate
            'leftside': 2, 'rightside': 2, 'leftmirror': 5, 'rightmirror': 5, 'fliplate': 6, 'bliplate': 6,
            # Merge all numbered parts
            **{f'door_{i}': 7 for i in range(1, 11)},
            **{f'wheel_{i}': 8 for i in range(1, 11)},
            **{f'headlight_{i}': 9 for i in range(1, 11)},
            **{f'window_{i}': 10 for i in range(1, 21)}
        },
        7: {},  # car - same as bus, will be copied
        8: {  # cat
            'head': 1, 'eye': 2, 'ear': 3, 'nose': 4, 'torso': 5, 'neck': 6, 'leg': 7, 'paw': 8, 'tail': 9,
            # Merge leye/reye -> eye, lear/rear -> ear
            'leye': 2, 'reye': 2, 'lear': 3, 'rear': 3,
            # Merge front/back legs -> leg, front/back paws -> paw
            'fleg': 7, 'bleg': 7, 'lfleg': 7, 'rfleg': 7, 'lbleg': 7, 'rbleg': 7,
            'fpa': 8, 'bpa': 8, 'lfpa': 8, 'rfpa': 8, 'lbpa': 8, 'rbpa': 8
        },
        9: {},  # chair - only silhouette mask
        10: {  # cow
            'head': 1, 'eye': 2, 'ear': 3, 'muzzle': 4, 'horn': 5, 'torso': 6, 'neck': 7, 'uleg': 8, 'lleg': 9, 'tail': 10,
            # Merge leye/reye -> eye, lear/rear -> ear, lhorn/rhorn -> horn
            'leye': 2, 'reye': 2, 'lear': 3, 'rear': 3, 'lhorn': 5, 'rhorn': 5,
            # Merge front/back upper legs -> uleg, front/back lower legs -> lleg
            'fuleg': 8, 'buleg': 8, 'lfuleg': 8, 'rfuleg': 8, 'lbuleg': 8, 'rbuleg': 8,
            'flleg': 9, 'blleg': 9, 'lflleg': 9, 'rflleg': 9, 'lblleg': 9, 'rblleg': 9
        },
        11: {},  # diningtable - only silhouette mask
        12: {},  # dog - same as cat + muzzle, will be copied
        13: {    # horse - same as cow but with hooves, and without horns
            'head': 1, 'eye': 2, 'ear': 3, 'muzzle': 4, 'torso': 5, 'neck': 6, 'uleg': 7, 'lleg': 8,
            'tail': 9, 'hoof': 10,
            # Merge leye/reye -> eye, lear/rear -> ear
            'leye': 2, 'reye': 2, 'lear': 3, 'rear': 3,
            # Merge front/back upper legs -> uleg, front/back lower legs -> lleg
            'fuleg': 7, 'buleg': 7, 'lfuleg': 7, 'rfuleg': 7, 'lbuleg': 7, 'rbuleg': 7,
            'flleg': 8, 'blleg': 8, 'lflleg': 8, 'rflleg': 8, 'lblleg': 8, 'rblleg': 8,
            'lfho': 10, 'rfho': 10, 'lbho': 10, 'rbho': 10
        },  
        14: {  # motorbike
            'wheel': 1, 'fwheel': 1, 'bwheel': 1, 'handlebar': 2, 'saddle': 3, 'headlight': 4,
            # Merge all headlight_i -> headlight
            **{f'headlight_{i}': 4 for i in range(1, 11)}
        },
        15: {  # person
            'head': 1, 'eye': 2, 'ear': 3, 'eyebrow': 4, 'nose': 5, 'mouth': 6, 'hair': 7, 'torso': 8, 'neck': 9,
            'arm': 10, 'hand': 11, 'leg': 12, 'foot': 13,
            # Merge left/right body parts
            'leye': 2, 'reye': 2, 'lear': 3, 'rear': 3, 'lebrow': 4, 'rebrow': 4,
            # Merge arm variants -> arm, leg variants -> leg
            'larm': 10, 'rlarm': 10, 'llarm': 10, 'uarm': 10, 'luarm': 10, 'ruarm': 10,
            'lhand': 11, 'rhand': 11,
            'lleg': 12, 'rlleg': 12, 'llleg': 12, 'uleg': 12, 'luleg': 12, 'ruleg': 12,
            'lfoot': 13, 'rfoot': 13
        },
        16: {'pot': 1, 'plant': 2},  # pottedplant
        17: {},  # sheep - same as cow, will be copied
        18: {},  # sofa - only silhouette mask
        19: {  # train
            'head': 1, 'hfrontside': 2, 'hside': 3, 'hbackside': 4, 'hroofside': 5, 'headlight': 6,
            'coach': 7, 'cfrontside': 7, 'cside': 7, 'cbackside': 7, 'croofside': 7,
            # Merge hleftside/hrightside -> hside, cleftside/crightside -> cside
            'hleftside': 3, 'hrightside': 3,
            # Merge all numbered parts
            **{f'headlight_{i}': 6 for i in range(1, 11)},
            **{f'coach_{i}': 7 for i in range(1, 11)},
            **{f'cfrontside_{i}': 7 for i in range(1, 11)},
            **{f'cleftside_{i}': 7 for i in range(1, 11)},
            **{f'crightside_{i}': 7 for i in range(1, 11)},
            **{f'cbackside_{i}': 7 for i in range(1, 11)},
            **{f'croofside_{i}': 7 for i in range(1, 11)}
        },
        20: {'screen': 1}  # tvmonitor
    }

    def __init__(
            self,
            voc_root: str | Path,
            split: str,
            processor,
            crop_size: Optional[Tuple[float, float]] = None,
            include_class: Optional[int] = None,
            exclude_class: Optional[int] = None,
    ):
        """
        Args:
            voc_root: Root directory of Pascal VOC dataset
            split: Dataset split ('train', 'val', 'trainval')
            processor: SegformerImageProcessor for preprocessing
            crop_size: Tuple of (height_ratio, width_ratio) for random cropping during training
        """
        super().__init__()
        self.voc_root = Path(voc_root)
        self.split = split
        self.processor = processor

        # Copy shared part mappings
        self.PART_INDEX_MAP[7] = self.PART_INDEX_MAP[6].copy()  # car same as bus
        self.PART_INDEX_MAP[12] = self.PART_INDEX_MAP[8].copy()  # dog same as cat
        self.PART_INDEX_MAP[12]['muzzle'] = 10  # dog has additional muzzle (after tail=9)
        self.PART_INDEX_MAP[17] = self.PART_INDEX_MAP[10].copy()  # sheep same as cow

        # Initialize part index mapping
        self._init_part_mapping()

        # Load image IDs for the specified split
        split_file = self.voc_root / "ImageSets" / "Segmentation" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        # # Filter image IDs to only include those with part annotations
        self.image_ids = self._filter_images_with_parts()

        # Set up transforms
        if self.split == "train" and crop_size is not None:
            self.transform = RandomCropAndFlip(crop_size=crop_size)
        else:
            self.transform = None

        assert not (include_class is not None and exclude_class is not None), \
            "Only one of include_class or exclude_class can be non-empty, not both"

        if include_class:
            self.image_ids = [
                img_id for img_id in self.image_ids
                if self.image_has_class(img_id, include_class)
            ]
        elif exclude_class:
            self.image_ids = [
                img_id for img_id in self.image_ids
                if not self.image_has_class(img_id, exclude_class)
            ]

        self.save_first_samples(output_dir="samples")

    def save_first_samples(self, output_dir: str | Path):
        """Save first 2 samples with their segmentation masks for visualization."""
        import matplotlib.pyplot as plt
        from PIL import Image

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        for i in range(5):
            sample = self[i]
            
            # Get original image
            img_path = self.voc_root / "JPEGImages" / f"{self.image_ids[i]}.jpg"
            img = Image.open(img_path).convert('RGB')
            
            # Create figure with 3 subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot original image
            ax1.imshow(img)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Plot class segmentation mask
            ax2.imshow(sample['labels_whole'].squeeze())
            ax2.set_title('Class Segmentation')
            ax2.axis('off')
            
            # Plot part segmentation mask
            ax3.imshow(sample['labels_part'].squeeze())
            ax3.set_title('Part Segmentation')
            ax3.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"sample_{i}.png")
            plt.close()

    def _init_part_mapping(self):
        """Initialize the part index mapping from MATLAB code"""
        # The mapping is already defined in the class variable
        # Calculate offset for each class based on previous classes' part counts
        self.class_part_offsets = {}
        offset = 0
        
        for class_id in sorted(self.PART_INDEX_MAP.keys()):
            self.class_part_offsets[class_id] = offset
            # Find max part ID for this class
            if self.PART_INDEX_MAP[class_id]:
                max_part_id = max(self.PART_INDEX_MAP[class_id].values())
                offset += max_part_id

        # Calculate total number of unique part classes
        self.num_part_classes = offset + 1  # +1 for background

    def _filter_images_with_parts(self) -> List[str]:
        """Filter image IDs to only include those with part annotations"""
        filtered_ids = []
        for image_id in self.image_ids:
            part_file = self.voc_root / "Annotations_Part" / f"{image_id}.mat"
            if part_file.exists():
                filtered_ids.append(image_id)
        return filtered_ids

    def _mat2map(self, anno: dict, img_shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert MATLAB annotation to segmentation masks
        Returns: class_mask, instance_mask, part_mask
        """
        height, width = img_shape[:2]
        cls_mask = np.zeros((height, width), dtype=np.uint8)
        inst_mask = np.zeros((height, width), dtype=np.uint8)
        part_mask = np.zeros((height, width), dtype=np.uint8)

        # Access the nested annotation structure
        annotation_data = anno['anno'][0, 0]
        objects = annotation_data['objects']
        if objects.size == 0:
            return cls_mask, inst_mask, part_mask

        for obj_idx in range(objects.shape[1]):
            obj = objects[0, obj_idx]
            class_ind = int(obj['class_ind'][0, 0])
            silh = obj['mask'].astype(bool)

            # Ensure mask dimensions match image
            if silh.shape != (height, width):
                continue

            inst_mask[silh] = obj_idx + 1
            cls_mask[silh] = class_ind

            # Process parts
            if 'parts' in obj.dtype.names and obj['parts'].size > 0:
                parts = obj['parts'][0]
                if parts.size > 0:
                    for part_idx in range(len(parts)):
                        part = parts[part_idx]
                        part_name = str(part['part_name'][0])
                        part_mask_data = part['mask'].astype(bool)

                        part_id = self.PART_INDEX_MAP[class_ind][part_name]
                        # Apply class offset to make part IDs globally unique
                        global_part_id = part_id + self.class_part_offsets[class_ind]
                        part_mask[part_mask_data] = global_part_id

        return cls_mask, inst_mask, part_mask

    def image_has_class(self, image_id: str, class_id: int) -> bool:
        part_file = self.voc_root / "Annotations_Part" / f"{image_id}.mat"
        anno = loadmat(str(part_file))
        annotation_data = anno['anno'][0, 0]
        objects = annotation_data['objects']
        for obj_idx in range(objects.shape[1]):
            obj = objects[0, obj_idx]
            class_ind = int(obj['class_ind'][0, 0])
            if class_ind == class_id:
                return True
        return False

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Load image
        image_path = self.voc_root / "JPEGImages" / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")

        # Load part annotation
        part_file = self.voc_root / "Annotations_Part" / f"{image_id}.mat"
        anno = loadmat(str(part_file))
        cls_mask, inst_mask, part_mask = self._mat2map(anno, np.array(image).shape)

        # Convert masks to PIL Images
        cls_segmentation_map = Image.fromarray(cls_mask.astype(np.uint8))
        part_segmentation_map = Image.fromarray(part_mask.astype(np.uint8))

        if self.transform:
            segmentation_maps = [cls_segmentation_map, part_segmentation_map]
            image, segmentation_maps = self.transform(image, segmentation_maps)

        # Process with SegformerImageProcessor
        inputs = self.processor(
            images=image,
            segmentation_maps=cls_segmentation_map,
            return_tensors="pt"
        )

        # Process part masks separately
        part_inputs = self.processor(
            images=image,
            segmentation_maps=part_segmentation_map,
            return_tensors="pt"
        )

        # Remove batch dimension
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        part_inputs = {k: v.squeeze() for k, v in part_inputs.items()}

        # Add different granularity labels
        inputs["labels_whole"] = inputs["labels"]  # Class-level labels
        inputs["labels_part"] = part_inputs["labels"]  # Part-level labels

        return inputs

    @classmethod
    def get_class_names(cls) -> List[str]:
        """Return list of class names."""
        return cls.CLASSES.copy()

    @classmethod
    def get_num_classes(cls) -> int:
        """Return number of classes including background."""
        return len(cls.CLASSES)

    def get_num_part_classes(self) -> int:
        """Return number of part classes including background."""
        return self.num_part_classes


class PascalPartDataModule(L.LightningDataModule):
    """
    A LightningDataModule for Pascal Part dataset.
    """

    def __init__(
            self,
            voc_root: str,
            processor,
            batch_size: int = 8,
            crop_size: Tuple[float, float] = (0.8, 0.8),
            num_workers: int = 4,
            zeroshot_class: Optional[str] = None
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
        self.zeroshot_class = zeroshot_class

    def setup(self, stage=None):
        """Set up train/val/test datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = PascalPartDataset(
                voc_root=self.voc_root,
                split="train",
                processor=self.processor,
                crop_size=self.crop_size,
                exclude_class=self.zeroshot_class
            )

        self.val_dataset = PascalPartDataset(
            voc_root=self.voc_root,
            split="val",
            processor=self.processor,
            include_class=self.zeroshot_class
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
        return PascalPartDataset.get_class_names()

    def get_num_classes(self):
        """Return number of classes including background."""
        return PascalPartDataset.get_num_classes()

    def get_num_part_classes(self):
        """Return number of part classes including background."""
        if hasattr(self, 'train_dataset'):
            return self.train_dataset.get_num_part_classes()
        elif hasattr(self, 'val_dataset'):
            return self.val_dataset.get_num_part_classes()
        else:
            # Create a temporary dataset to get the number
            temp_dataset = PascalPartDataset(
                voc_root=self.voc_root,
                split="val",
                processor=self.processor,
            )
            return temp_dataset.get_num_part_classes()

    def get_zeroshot_class_ids(self):
        """Return the class IDs for the zeroshot class."""
        return []



# Example usage
if __name__ == "__main__":
    from transformers import SegformerImageProcessor

    # Initialize processor
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    processor.do_reduce_labels = False

    # Create data module
    data_module = PascalPartDataModule(
        voc_root="/home/misha/data/voc_root",
        processor=processor,
        batch_size=4,
        crop_size=(0.8, 0.8),
        num_workers=4,
    )

    # Setup datasets
    data_module.setup("fit")

    # Get dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Print dataset info
    print(f"Number of training samples: {len(data_module.train_dataset)}")
    print(f"Number of validation samples: {len(data_module.val_dataset)}")
    print(f"Number of classes: {data_module.get_num_classes()}")
    print(f"Number of part classes: {data_module.get_num_part_classes()}")

    # Test a batch
    for batch in train_loader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Input shape: {batch['pixel_values'].shape}")
        print(f"Class labels shape: {batch['labels_whole'].shape}")
        print(f"Part labels shape: {batch['labels_part'].shape}")
        break
