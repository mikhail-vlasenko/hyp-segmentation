import numpy as np
from torchvision.transforms import v2 as transforms
import torchvision.transforms.functional as TF

class DoRemapObjects:
    value = False

def background_class_for_granularity(granularity):
    return {"whole": 0 if DoRemapObjects.value else 0, "part": 0, "subpart": 0}[granularity]

def num_labels_for_granularity(granularity):
    return {"whole": 21 if DoRemapObjects.value else 21, "part": 119, "subpart": 0}[granularity]  # classes + 1 background


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
