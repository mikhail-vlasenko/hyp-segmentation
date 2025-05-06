from __future__ import annotations

import io
import os
import random
from functools import lru_cache
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from PIL import Image, ImageDraw
from transformers import SegformerImageProcessor

from dataset import SPINSegmentationDataset
from hierarchy_embeddings.class_remapping import COARSE_CLASSES
from hierarchy_embeddings.object_supercategory_mapping import OBJECT_SUPERCATEGORY_MAPPING

ALLOWED_GRANULARITIES: tuple[str, ...] = ("whole", "part", "subpart")

###############################################################################
# Dataset initialisation                                                     #
###############################################################################

@lru_cache(maxsize=1)
def get_dataset() -> SPINSegmentationDataset:
    """Load the *validation* split once and memoise it."""
    dataset_dir = "/home/misha/data/PartImageNet/"

    annotation_dir = os.path.join(dataset_dir, "annotations")
    image_dir = os.path.join(dataset_dir, "images")

    # A processor is needed only because the Dataset __init__ expects it, but we
    # never actually use it for the viewer (we deal with raw PIL masks).
    processor = SegformerImageProcessor(do_reduce_labels=False)

    granularities = list(ALLOWED_GRANULARITIES)
    return SPINSegmentationDataset(
        annotation_dir=annotation_dir,
        image_dir=image_dir,
        split="val",
        granularities=granularities,
        processor=processor,
    )

###############################################################################
# Utility helpers                                                            #
###############################################################################

BACKGROUND_COLOR = (0, 0, 0, 0)  # fully transparent for background label 0


def _stable_color_for_label(label: int) -> tuple[int, int, int]:
    """Deterministically map a numerical label → RGB colour.

    We seed Python's PRNG with the label id so colours stay the same every run
    while remaining visually distinct.
    """
    rng = random.Random(label)
    return tuple(rng.randrange(32, 224) for _ in range(3))  # avoid extremes


def _colorise_mask(mask: Image.Image) -> Image.Image:
    """Convert a single‑channel segmentation mask to an RGB image."""
    arr = np.array(mask, dtype=np.int64)
    h, w = arr.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)

    for label in np.unique(arr):
        if label == 0:  # background
            continue
        colour[arr == label] = _stable_color_for_label(int(label))
    return Image.fromarray(colour)


def _blend(img: Image.Image, mask_rgb: Image.Image, alpha: float = 0.5) -> Image.Image:
    """Alpha‑blend a colour mask on top of the RGB image."""
    if mask_rgb.mode != "RGBA":
        # convert and make background transparent
        mask_rgb = mask_rgb.convert("RGBA")
        datas = mask_rgb.getdata()
        newData = []
        for item in datas:
            if item[:3] == (0, 0, 0):
                newData.append(BACKGROUND_COLOR)
            else:
                newData.append((*item[:3], int(255 * alpha)))
        mask_rgb.putdata(newData)
    base = img.convert("RGBA")
    blended = Image.alpha_composite(base, mask_rgb)
    return blended.convert("RGB")

###############################################################################
# FastAPI app                                                                #
###############################################################################

app = FastAPI(title="PartImageNet Web Viewer", version="0.1.0")

dataset = get_dataset()


@app.get("/ids")
async def list_ids(skip: int = 0, limit: int = Query(100, le=500)) -> JSONResponse:  # noqa: D401
    """Return a slice of available image indices."""
    total = len(dataset)
    ids = list(range(total))[skip : skip + limit]
    return JSONResponse({"total": total, "ids": ids})


@app.get("/image/{idx}")
async def get_image(idx: int) -> StreamingResponse:  # noqa: D401
    """Return raw RGB JPEG image *idx*."""
    if not 0 <= idx < len(dataset):
        raise HTTPException(404, "index out of range")

    image_id = dataset.image_ids[idx]
    img: Image.Image = dataset.spin_api.get_image(image_id)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


@app.get("/mask/{granularity}/{idx}")
async def get_mask(granularity: Literal["whole", "part", "subpart"], idx: int) -> StreamingResponse:  # noqa: E501
    """Return raw label mask as indexed‑colour PNG."""
    if not 0 <= idx < len(dataset):
        raise HTTPException(404)

    image_id = dataset.image_ids[idx]
    mask = dataset.get_segmentation_map(image_id, granularity)

    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/overlay/{granularity}/{idx}")
async def get_overlay(
    granularity: Literal["whole", "part", "subpart"],
    idx: int,
    alpha: float = Query(0.5, ge=0.0, le=1.0),
) -> StreamingResponse:  # noqa: D401
    """Return RGB → colourmask overlay."""
    if not 0 <= idx < len(dataset):
        raise HTTPException(404)

    image_id = dataset.image_ids[idx]
    img = dataset.spin_api.get_image(image_id)
    mask = dataset.get_segmentation_map(image_id, granularity)
    if granularity == "whole":
        whole_lut = np.asarray(OBJECT_SUPERCATEGORY_MAPPING + [11], dtype=np.int16)
        mask = whole_lut[mask]

    mask_rgb = _colorise_mask(mask)
    blended = _blend(img, mask_rgb, alpha=alpha)

    buf = io.BytesIO()
    blended.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


@app.get("/legend/{granularity}")
async def get_legend(granularity: Literal["whole", "part", "subpart"]) -> JSONResponse:
    """Return mapping of label IDs to names and colours for the given granularity."""
    if granularity not in ALLOWED_GRANULARITIES:
        raise HTTPException(404)
    api = dataset.spin_api
    cats = {
        "whole": api.wholes.cats,
        "part": api.parts.cats,
        "subpart": api.subparts.cats,
    }[granularity]
    legend = []
    for entry in sorted(cats.values(), key=lambda e: e['id']):
        lid = entry['id']
        name = entry['name']
        rgb = _stable_color_for_label(lid)
        color_hex = '#{:02x}{:02x}{:02x}'.format(*rgb)
        if granularity == "whole" and lid < len(COARSE_CLASSES):
            # add supercategory name for whole objects
            name = COARSE_CLASSES[lid]
        legend.append({"id": lid, "name": name, "color": color_hex})
    return JSONResponse(legend)

###############################################################################
# Minimal inline HTML viewer                                                 #
###############################################################################

HTML_TEMPLATE = """<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\">
    <title>PartImageNet Viewer</title>
    <style>
        body{font-family:system-ui, sans-serif;margin:0;display:flex;flex-direction:column;align-items:center;background:#111;color:#eee}
        #viewer{display:flex;gap:10px;margin:20px auto;flex-wrap:wrap;justify-content:center}
        img{max-width:32vw;height:auto;border:2px solid #555}
        button{padding:6px 12px;margin:10px;font-size:1rem}
        select{padding:4px;font-size:1rem}
        #legend{max-height:200px;overflow-y:auto;margin:10px;display:flex;flex-wrap:wrap;gap:6px;padding:0 20px}
        .legend-item{display:flex;align-items:center;gap:4px;font-size:0.9rem}
        .legend-color{width:12px;height:12px;border:1px solid #eee;display:inline-block}
    </style>
</head>
<body>
    <h1>PartImageNet Web Viewer</h1>
    <div>
        <button id=\"prev\">⟨ Prev</button>
        <button id=\"next\">Next ⟩</button>
        Granularity:
        <select id=\"gran\">
            <option value=\"whole\">whole</option>
            <option value=\"part\">part</option>
            <option value=\"subpart\">subpart</option>
        </select>
    </div>
    <div id=\"viewer\">
        <img id=\"raw\" src=\"\" alt=\"raw\">
        <img id=\"mask\" src=\"\" alt=\"mask\">
        <img id=\"overlay\" src=\"\" alt=\"overlay\">
    </div>
    <div id=\"legend\"></div>
<script>
let idx = 0;
async function update(){
  const gran = document.getElementById('gran').value;
  document.getElementById('raw').src = `/image/${idx}`;
  document.getElementById('mask').src = `/mask/${gran}/${idx}`;
  document.getElementById('overlay').src = `/overlay/${gran}/${idx}`;
  // fetch and render legend
  const res = await fetch(`/legend/${gran}`);
  const data = await res.json();
  const legend = document.getElementById('legend');
  legend.innerHTML = '';
  data.forEach(item => {
    const container = document.createElement('div');
    container.className = 'legend-item';
    const swatch = document.createElement('span');
    swatch.className = 'legend-color';
    swatch.style.backgroundColor = item.color;
    container.appendChild(swatch);
    const label = document.createElement('span');
    label.textContent = `${item.name} (${item.id})`;
    container.appendChild(label);
    legend.appendChild(container);
  });
}

fetch('/ids').then(r=>r.json()).then(data=>{ window.total=data.total });

document.getElementById('next').onclick = ()=>{ idx = (idx+1)%window.total; update(); };
document.getElementById('prev').onclick = ()=>{ idx = (idx-1+window.total)%window.total; update(); };
document.getElementById('gran').onchange = update;
update();
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root() -> HTMLResponse:  # noqa: D401
    return HTMLResponse(HTML_TEMPLATE)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("dataset_viewer:app", host="127.0.0.1", port=8000, reload=True)
