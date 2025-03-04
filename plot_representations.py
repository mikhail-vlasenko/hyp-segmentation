#!/usr/bin/env python
"""
inference.py

This script loads a saved SegformerLightningModule checkpoint, performs segmentation on a specified image,
and produces a three-panel plot:
  - Left: the original image.
  - Middle: the segmentation map with random colors for each predicted class.
  - Right: a scatter plot where, for every non-background pixel (sampled every 10th),
           the x-axis is the flattened pixel index and the y-axis is the L2 magnitude of the representation vector.

Usage:
    python inference.py --checkpoint_path <path/to/lightning_checkpoint.ckpt> \
                        --processor_dir <path/to/processor_dir> \
                        --image_path <path/to/image.jpg> \
                        [--granularity subpart]
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

from transformers import SegformerImageProcessor

# Import geoopt and embedding space if required by your head.
import geoopt
import geoopt.manifolds.stereographic.math as pmath
from embedding_space import EmbeddingSpace
from lightning_segformer import SegformerLightningModule, background_class_for_granularity, num_labels_for_granularity


def get_random_colormap(num_classes):
    """Returns a (num_classes, 3) array of random colors."""
    return np.random.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)


def create_segmentation_rgb(segmentation, colormap):
    """Converts a segmentation (with class indices) into an RGB image using the provided colormap."""
    h, w = segmentation.shape
    seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx in np.unique(segmentation):
        seg_rgb[segmentation == class_idx] = colormap[class_idx]
    return seg_rgb


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for lightning checkpoint segmentation and representation visualization"
    )
    parser.add_argument("--checkpoint_path", type=str, default="hyperbolic-segmentation/cwkfylyg/checkpoints/epoch=9-step=11040.ckpt",
                        help="Path to the saved lightning module checkpoint (.ckpt)")
    parser.add_argument("--image_path", type=str, default="/home/misha/data/PartImageNet/images/train",
                        help="Path to the image file to segment")
    parser.add_argument("--granularity", type=str, default="subpart",
                        choices=["whole", "part", "subpart"],
                        help="Segmentation granularity; default is 'subpart' (204 classes)")
    args = parser.parse_args()

    images = ["n04482393_12929.JPEG", "n04612504_5184.JPEG",  "n01614925_1726.JPEG", "n01484850_1581.JPEG"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the saved lightning module checkpoint.
    print("Loading SegformerLightningModule checkpoint ...")
    lightning_module = SegformerLightningModule.load_from_checkpoint(args.checkpoint_path)
    lightning_module.to(device)
    lightning_module.eval()

    # Load the processor.
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    processor.do_reduce_labels = False

    for image_filename in images:
        # Load and preprocess the image.
        image_path = os.path.join(args.image_path, image_filename)
        image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = image.size
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            # Get backbone outputs (with hidden states) from the lightning module's model.
            backbone_outputs = lightning_module.backbone(
                pixel_values,
                output_hidden_states=True,
                return_dict=False
            )
            # The encoder hidden states are the second element.
            encoder_hidden_states = backbone_outputs[1]

            # Select the appropriate decode head based on the granularity.
            decode_head = lightning_module.decode_heads[args.granularity]
            # Run the decode head with the flag to also return the representation.
            logits, rep = decode_head(encoder_hidden_states, return_repr=True)

        # Upsample logits to the original image size.
        upsampled_logits = F.interpolate(
            logits, size=(orig_height, orig_width), mode="bilinear", align_corners=False
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # Process the representation tensor.
        # If returned in channel-last format, convert it to channel-first.
        if rep.ndim == 4 and rep.shape[-1] not in {1, 3}:
            rep = rep.permute(0, 3, 1, 2)
        upsampled_rep = F.interpolate(
            rep, size=(orig_height, orig_width), mode="bilinear", align_corners=False
        )
        rep_magnitude = torch.norm(upsampled_rep, dim=1)[0].cpu().numpy()

        # Normalize representation magnitude for visualization in grayscale.
        rep_min = rep_magnitude.min()
        rep_max = rep_magnitude.max()
        norm_rep = (rep_magnitude - rep_min) / (rep_max - rep_min + 1e-8)

        # Create a random colormap (assume 204 classes for subpart segmentation).
        num_classes = 204
        colormap = get_random_colormap(num_classes)
        seg_rgb = create_segmentation_rgb(pred_seg, colormap)

        # Plot the results.
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Left panel: original image.
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Middle panel: segmentation map with random colors.
        axes[1].imshow(seg_rgb)
        axes[1].set_title("Segmentation")
        axes[1].axis("off")

        # Right panel: Grayscale image based on representation magnitude.
        axes[2].imshow(norm_rep, cmap="gray")
        axes[2].set_title(f"Representation Magnitude (Max is {rep_max:.2f})")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
