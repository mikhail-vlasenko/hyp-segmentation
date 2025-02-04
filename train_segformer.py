import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
from spin import SPIN
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex

from segformer_head import HyperbolicSegformerDecodeHead

# Configuration
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
TRAIN_ANNOTATION_DIR = "/home/misha/data/PartImageNet/annotations"
TRAIN_IMAGE_DIR = "/home/misha/data/PartImageNet/images"
NUM_CLASSES = 204  # 203 classes + 1 background
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4


class SPINSegmentationDataset(Dataset):
    def __init__(self, annotation_dir, image_dir, split, processor):
        self.spin_api = SPIN(
            annotation_dir=annotation_dir,
            image_dir=image_dir,
            split=split,
            download=False
        )
        self.processor = processor
        self.image_ids = self.spin_api.getImgIds()

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

        # Convert to PIL Image for processing
        segmentation_map = Image.fromarray(segmentation_map.astype('uint8'))

        # Process with Segformer processor
        inputs = self.processor(
            images=image,
            segmentation_maps=segmentation_map,
            return_tensors="pt"
        )

        # Remove batch dimension
        inputs = {k: v.squeeze() for k, v in inputs.items()}

        return inputs

    def num_classes(self):
        return len(self.spin_api.get_categories(granularity="subpart"))


# Initialize processor and model
processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
processor.do_reduce_labels = False  # do not reduce label indices by 1, and let background be 0 (instead of 255)
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)
model.decode_head = HyperbolicSegformerDecodeHead.from_segformer_decode_head(model.decode_head, NUM_CLASSES, 8, hyperbolic=True)

# Create datasets and dataloaders
train_dataset = SPINSegmentationDataset(
    TRAIN_ANNOTATION_DIR,
    TRAIN_IMAGE_DIR,
    "train",
    processor
)
val_dataset = SPINSegmentationDataset(
    TRAIN_ANNOTATION_DIR,
    TRAIN_IMAGE_DIR,
    "val",
    processor
)


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Metric
metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=0).to(device)

# Training loop
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Training loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    metric.reset()
    val_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)  # batch_size, h, w

            outputs = model(pixel_values=pixel_values, labels=labels)
            val_loss += outputs.loss.item()

            # Upsample logits to original image size
            logits = outputs.logits  # shape (batch_size, num_labels, h/4, w/4)
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=labels.shape[-2:],  # (height, width)
                mode='bilinear',
                align_corners=False
            )

            preds = torch.argmax(upsampled_logits, dim=1)
            metric.update(preds, labels)

    miou = metric.compute()
    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation loss: {avg_val_loss:.4f}, mIoU: {miou:.4f}\n")

# Save the fine-tuned model
model.save_pretrained("segformer-finetuned-spin")
processor.save_pretrained("segformer-finetuned-spin")
