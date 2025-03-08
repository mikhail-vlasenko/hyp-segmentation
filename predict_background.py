import os
import argparse
import torch
from torchmetrics.classification import MulticlassJaccardIndex
from transformers import SegformerImageProcessor
from lightning_segformer import background_class_for_granularity, num_labels_for_granularity, SPINDataModule


def main():
    parser = argparse.ArgumentParser(
        description="Predict background baseline on the validation set and compute mIoU."
    )
    parser.add_argument("--dataset_dir", type=str, default="/home/misha/data/PartImageNet/",
                        help="Path to the dataset directory (expects 'annotations' and 'images' subdirectories).")
    parser.add_argument("--granularity", type=str, default="subpart",
                        choices=["whole", "part", "subpart", "all"],
                        help="Segmentation granularity level: whole, part, subpart, or all.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for the validation dataloader.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading.")
    parser.add_argument("--model_name", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512",
                        help="Name of the pretrained model used by the processor.")
    args = parser.parse_args()

    # Determine granularities list
    granularities = ["whole", "part", "subpart"] if args.granularity == "all" else [args.granularity]

    # Create the image processor
    processor = SegformerImageProcessor.from_pretrained(args.model_name)
    processor.do_reduce_labels = False

    dataset_annotation_dir = os.path.join(args.dataset_dir, "annotations")
    dataset_image_dir = os.path.join(args.dataset_dir, "images")

    # Initialize the data module (assumes it creates the validation dataset)
    dm = SPINDataModule(
        annotation_dir=dataset_annotation_dir,
        image_dir=dataset_image_dir,
        granularities=granularities,
        processor=processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup(stage="val")
    val_loader = dm.val_dataloader()

    # Initialize mIoU metrics for each granularity.
    metrics = {}
    for g in granularities:
        metrics[g] = MulticlassJaccardIndex(
            num_classes=num_labels_for_granularity(g),
        )

    # Loop through the validation set and update metrics with dummy background predictions.
    with torch.no_grad():
        for batch in val_loader:
            for g in granularities:
                labels = batch[f"labels_{g}"]
                bg_class = background_class_for_granularity(g)
                # Create dummy predictions: every pixel is set to the background class.
                preds = torch.full_like(labels, fill_value=bg_class)
                metrics[g].update(preds, labels)

    # Compute and print the mIoU for each granularity.
    for g in granularities:
        miou = metrics[g].compute()
        print(f"mIoU for granularity '{g}': {miou.item()}")

if __name__ == "__main__":
    main()
