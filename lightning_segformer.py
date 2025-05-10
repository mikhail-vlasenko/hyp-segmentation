import copy
import os
import argparse
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from transformers import SegformerImageProcessor

from dataset import SPINDataModule
from losses import LossParams
from model import SegformerLightningModule
from segformer_head import HeadKwargs
from utils import DoRemapObjects


def parse_args():
    parser = argparse.ArgumentParser(description="Train SegFormer for Semantic Segmentation")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")

    parser.add_argument("--model_name", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512", help="Name of the pre-trained model")
    parser.add_argument("--granularity", type=str, choices=["whole", "part", "subpart", "all"], required=True, help="Level of segmentation granularity: whole, part, or subpart")

    # model configuration
    parser.add_argument("--hyperbolic", action="store_true", help="Use hyperbolic decode head if specified")
    parser.add_argument("--curvature", type=float, default=1., help="Hyperbolic curvature")
    parser.add_argument("--max_class_sep", action="store_true", help="Do prototypical learning")
    parser.add_argument("--tau", type=float, default=10., help="Temperature parameter for class separation with prototypes in hyperbolic space")
    parser.add_argument("--head_dim", type=int, default=None, help="Dimension reduction in the decode head")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    # losses configuration
    parser.add_argument("--background_loss_weight", type=float, default=1.0, help="Weight for the background class in the cross-entropy loss")
    parser.add_argument("--focal_loss", action="store_true", help="Use focal loss for training instead of cross-entropy")
    parser.add_argument("--focal_loss_gamma", type=float, default=0.7, help="Gamma parameter for focal loss")
    parser.add_argument("--ratio_loss_weight", type=float, default=0., help="Weight for the ratio loss. Set to 0 to disable")
    parser.add_argument("--clamp_to", type=float, default=2., help="Clamp the ratio loss to this value on the high side")
    parser.add_argument("--norm_penalty_weight", type=float, default=0., help="Weight of the hinge norm penalty for hyperbolic representations")
    # training configuration
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradient batches before updating weights")

    # evaluation configuration
    parser.add_argument("--whole_embeddings_path", type=str, default=None, help="Path to the embeddings (prototypes) file for the 'whole' granularity")
    parser.add_argument("--part_embeddings_path", type=str, default=None, help="Prototypes of 'part' level")
    parser.add_argument("--subpart_embeddings_path", type=str, default=None, help="Prototypes of 'subpart' level")
    parser.add_argument("--remap_objects", action="store_false", help="Remap object classes to coarse classes")

    parser.add_argument("--crop_size", type=float, nargs=2, default=(0.8, 0.8), help="Crop size as a fraction of image dimensions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--model_file", type=str, default=None, help="Path to the model file to load")
    return parser.parse_args()


def main():
    args = parse_args()
    DoRemapObjects.value = args.remap_objects
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
        remap_objects=args.remap_objects,
    )

    if args.model_file:
        segformer_module = SegformerLightningModule.load_from_checkpoint(args.model_file)
        print(f"Loaded model from {args.model_file}")
    else:
        head_kwargs = HeadKwargs(
            dim=args.head_dim,
            hyperbolic=args.hyperbolic,
            curvature=args.curvature,
            max_class_sep=args.max_class_sep,
            tau=args.tau,
            embeddings_paths={
                "whole": args.whole_embeddings_path,
                "part": args.part_embeddings_path,
                "subpart": args.subpart_embeddings_path,
            }
        )
        loss_params = LossParams(
            background_loss_weight=args.background_loss_weight,
            focal_loss=args.focal_loss,
            focal_loss_gamma=args.focal_loss_gamma,
            ratio_loss_weight=args.ratio_loss_weight,
            clamp_to=args.clamp_to,
            norm_penalty_weight=args.norm_penalty_weight,
        )
        segformer_module = SegformerLightningModule(
            model_name=args.model_name,
            lr=args.learning_rate,
            granularities=granularities,
            head_kwargs=head_kwargs,
            loss_params=loss_params,
        )

    if not args.model_file:
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
    else:
        trainer = L.Trainer(
            accelerator="auto",
            devices="auto",
        )
        trainer.test(segformer_module, spin_dm)


if __name__ == "__main__":
    main()
