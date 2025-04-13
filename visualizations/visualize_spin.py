import numpy as np
from spin import SPIN, InitialRequestError
import matplotlib.pyplot as plt


annotation_dir = "/home/misha/data/PartImageNet/annotations"
image_dir = "/home/misha/data/PartImageNet/images"
split = "train"
spin_api = SPIN(
    annotation_dir=annotation_dir, image_dir=image_dir, split=split, download=False
)

# Get all image IDs in the dataset
image_ids = spin_api.getImgIds()

image_id = image_ids[20]

# # Display the image and its annotations
spin_api.display_annotations(image_id, fig_size=[10, 8], draw_bbox=False)

