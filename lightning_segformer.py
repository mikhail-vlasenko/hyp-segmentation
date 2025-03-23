import copy
import os
import argparse
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from dataset import SPINDataModule
from model import SegformerLightningModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train SegFormer for Semantic Segmentation")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--model_name", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512", help="Name of the pre-trained model")
    parser.add_argument("--granularity", type=str, choices=["whole", "part", "subpart", "all"], required=True,
                        help="Level of segmentation granularity: whole, part, or subpart")
    parser.add_argument("--hyperbolic", action="store_true", help="Use hyperbolic decode head if specified")
    parser.add_argument("--curvature", type=float, default=1., help="Hyperbolic curvature")
    parser.add_argument("--max_class_sep", action="store_true", help="Use maximum class separation prototypes pipeline")
    parser.add_argument("--tau", type=float, default=10., help="Temperature parameter for class separation with prototypes in hyperbolic space")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--background_loss_weight", type=float, default=0.01, help="Weight for the background class in the cross-entropy loss")
    parser.add_argument("--focal_loss", type=bool, default=True, help="Use focal loss for training instead of cross-entropy")
    parser.add_argument("--focal_loss_gamma", type=float, default=0.7, help="Gamma parameter for focal loss")
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
        tau=args.tau,
        background_loss_weight=args.background_loss_weight,
        focal_loss=args.focal_loss,
        focal_loss_gamma=args.focal_loss_gamma,
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

    trainer.test(segformer_module, spin_dm)

    save_dir = "segformer-finetuned-spin"
    os.makedirs(save_dir, exist_ok=True)

    segformer_module.model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
