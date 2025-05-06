# object_to_root_mapping.py
"""
Efficiently build the mapping from the 158 **whole‑object** classes of the
PartImageNet / SPIN dataset to the 11 coarse super‑categories by looking only at
**annotation overlaps**, not at every pixel.

The key idea is:

* Each *part* annotation belongs to a named category such as "Quadruped Tail" or
  "Car Body".  The first token gives the super‑category ("Quadruped", "Car", …).
* When the binary mask of that part overlaps with the binary mask of a *whole*
  annotation, the corresponding whole‑object class label must belong to the
  same super‑category.

We therefore iterate over the dataset *annotation‑by‑annotation*:

1.  For every **part** annotation we compute its RLE mask (via `pycocotools`).
2.  We restrict the search to **whole** annotations whose **bounding box
    contains the part's centroid** – a cheap spatial filter.
3.  For each candidate whole annotation we test the **mask intersection area**
    using `maskUtils.merge(..., intersect=True)`.  One positive overlap is
    enough to assign the whole‑object ID to the super‑category.
4.  We remember the first category we ever see for an object ID; if later we see
    the same object ID overlap with a *different* super‑category we raise an
    explicit error explaining the conflict.

Because we only decode a handful of masks per image (≈ parts × overlapping
objects) the script is orders of magnitude faster and more memory‑friendly than
fully rasterising every pixel.

Usage
-----
```bash
python object_to_root_mapping.py \
  --annotation_dir /path/to/PartImageNet/annotations \
  --image_dir       /path/to/PartImageNet/images \
  --output          mapping.json \
  --splits          train val test  # default
  --max_images      5000            # optional early stop for quick sanity‑check
```

The output *mapping.json* is a JSON list of length 158.  Unobserved object IDs
are left as **‑1** so you can decide how to handle them later.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pycocotools import mask as maskUtils  # type: ignore
from tqdm import tqdm  # type: ignore

try:
    from spin import SPIN  # PartImageNet helper API
except ImportError as e:  # pragma: no cover
    sys.stderr.write("[ERROR] Could not import the `spin` package. Make sure it is on PYTHONPATH.\n")
    raise e

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

COARSE_CLASSES: List[str] = [
    "Quadruped",
    "Biped",
    "Fish",
    "Bird",
    "Snake",
    "Reptile",
    "Car",
    "Bicycle",
    "Boat",
    "Aeroplane",
    "Bottle",
]
COARSE2ID: Dict[str, int] = {name: idx for idx, name in enumerate(COARSE_CLASSES)}

# Background category IDs (defined in PartImageNet utils)
BACKGROUND_WHOLE = 158  # utils.background_class_for_granularity("whole")
BACKGROUND_PART = 40    # utils.background_class_for_granularity("part")


def _ann_to_rle(ann: dict, height: int, width: int) -> dict:
    """Convert a COCO *annotation* to an RLE mask without decoding to a full array."""
    segm = ann["segmentation"]
    if isinstance(segm, list):  # polygon
        rles = maskUtils.frPyObjects(segm, height, width)
        return maskUtils.merge(rles)
    elif isinstance(segm, dict) and "counts" in segm:  # already RLE
        return segm
    else:
        raise ValueError("Unsupported segmentation format")


def _bbox_contains(point: Tuple[float, float], bbox: List[float]) -> bool:
    """Check whether *point* lies inside the axis‑aligned *bbox* (x, y, w, h)."""
    x, y, w, h = bbox
    px, py = point
    return (x <= px <= x + w) and (y <= py <= y + h)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def compute_mapping(
    annotation_dir: str | Path,
    image_dir: str | Path,
    *,
    splits: Tuple[str, ...] = ("train", "val", "test"),
    max_images: int | None = None,
) -> List[int]:
    """Return a list *mapping* where *mapping[obj_id] == coarse_id* (length 158).

    The function stops early after *max_images* images if that parameter is set –
    useful for quick experiments.
    """
    annotation_dir, image_dir = Path(annotation_dir), Path(image_dir)

    object2coarse: Dict[int, str] = {}
    contradictions: defaultdict[int, set[str]] = defaultdict(set)

    images_seen = 0

    for split in splits:
        spin_api = SPIN(annotation_dir=str(annotation_dir), image_dir=str(image_dir), split=split, download=False)
        img_ids = spin_api.getImgIds()

        for img_id in tqdm(img_ids, desc=f"{split} split"):
            if max_images is not None and images_seen >= max_images:
                break
            images_seen += 1

            img_info = spin_api.wholes.imgs[img_id]
            height, width = img_info["height"], img_info["width"]

            # Load annotations for this image -------------------------------------
            whole_ann_ids = spin_api.wholes.getAnnIds(imgIds=[img_id])
            part_ann_ids = spin_api.parts.getAnnIds(imgIds=[img_id])
            if not whole_ann_ids or not part_ann_ids:
                continue
            whole_anns = spin_api.wholes.loadAnns(whole_ann_ids)
            part_anns = spin_api.parts.loadAnns(part_ann_ids)

            # Lazily cache RLE masks for whole annotations ------------------------
            whole_rles: Dict[int, dict] = {}
            for w_ann in whole_anns:
                obj_id = w_ann["category_id"]
                whole_rles[obj_id] = _ann_to_rle(w_ann, height, width)

            # --------------------------------------------------------------------
            for p_ann in part_anns:
                part_cat = spin_api.parts.cats[p_ann["category_id"]]
                coarse_name = part_cat["name"].split(" ", 1)[0]

                # Cheap spatial filter: only consider whole annotations whose
                # bounding box contains the part's centroid -------------------
                x, y, w, h = p_ann["bbox"]
                centroid = (x + w / 2.0, y + h / 2.0)
                candidate_whole_anns = [w_ann for w_ann in whole_anns if _bbox_contains(centroid, w_ann["bbox"])]
                if not candidate_whole_anns:
                    continue

                # Build RLE for the part only once -----------------------------
                part_rle = _ann_to_rle(p_ann, height, width)

                # Test mask intersection with each candidate whole -------------
                for w_ann in candidate_whole_anns:
                    obj_id = w_ann["category_id"]  # 0‑157
                    int_rle = maskUtils.merge([part_rle, whole_rles[obj_id]], intersect=True)  # type: ignore
                    if maskUtils.area(int_rle) == 0:
                        continue  # masks do not actually overlap

                    # Record / validate mapping -------------------------------
                    if obj_id not in object2coarse:
                        object2coarse[obj_id] = coarse_name
                    elif object2coarse[obj_id] != coarse_name:
                        contradictions[obj_id].update({object2coarse[obj_id], coarse_name})
                    break  # one positive overlap is enough for this part

            if max_images is not None and images_seen >= max_images:
                break

        if max_images is not None and images_seen >= max_images:
            break

    # ------------------------------------------------------------------------
    if contradictions:
        lines = ["Contradictory evidence found while building the mapping:"]
        for obj_id, coarse_set in contradictions.items():
            lines.append(f"  object ID {obj_id}: {sorted(coarse_set)}")
        raise RuntimeError("\n".join(lines))

    # Dense list 158 ---------------------------------------------------------
    mapping: List[int] = [-1] * 158
    for obj_id, coarse_name in object2coarse.items():
        mapping[obj_id] = COARSE2ID[coarse_name]

    unmapped = [i for i, x in enumerate(mapping) if x == -1]
    if unmapped:
        sys.stderr.write(
            f"[WARNING] {len(unmapped)} of 158 object classes never overlapped with a part.\n"
            "          They are left as -1 in the mapping list.\n"
        )

    return mapping


# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Compute 158→11 object mapping for PartImageNet (fast mask‑overlap version)")
    parser.add_argument("--dataset_dir", default="/home/misha/data/PartImageNet/")
    parser.add_argument("--output", default="object_supercategory_mapping.json")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"], help="dataset splits to scan")
    parser.add_argument("--max_images", type=int, default=None, help="early stop after this many images (debugging)")
    args = parser.parse_args()

    dataset_annotation_dir = os.path.join(args.dataset_dir, "annotations")
    dataset_image_dir = os.path.join(args.dataset_dir, "images")
    mapping = compute_mapping(
        annotation_dir=dataset_annotation_dir,
        image_dir=dataset_image_dir,
        splits=tuple(args.splits),
        max_images=args.max_images,
    )

    with open(args.output, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"[OK] Saved mapping ({sum(x != -1 for x in mapping)} resolved classes) → {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
