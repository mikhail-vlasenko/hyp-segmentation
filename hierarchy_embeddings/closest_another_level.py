from pathlib import Path
import json
import re
from typing import Dict, List
import geoopt as gt

import torch

from hyperbolic_layers import fast_dist

# ---------------------------------------------------------------------------
# Configuration – edit paths if your files live elsewhere
# ---------------------------------------------------------------------------

ATTR_CHAIN = "embeddings.weight.tensor"  # attribute to the (N, D) tensor

EMB_A_PATH = Path(
    "hierarchy_embeddings/hierarchies/hierarchy_embeddings/experiments/"
    "spin_dataset/spin_hierarchy/2025-05-11_152400/"
    "HierarchyEmbedding_weights_16_only_level_2.pth"
)
EMB_B_PATH = Path(
    "hierarchy_embeddings/hierarchies/hierarchy_embeddings/experiments/"
    "spin_dataset/spin_hierarchy/2025-05-11_152400/"
    "HierarchyEmbedding_weights_16_only_level_1.pth"
)
CONFIG_PATH = Path("hierarchy_embeddings/spin_dataset/spin_hierarchy.json")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_attr(obj, chain: str):
    """Follow dotted *chain* through attributes **or** dict keys."""
    cur = obj
    for part in chain.split('.'):
        cur = cur[part] if isinstance(cur, dict) else getattr(cur, part)
    return cur


def load_embeddings(path: Path) -> torch.Tensor:
    tensor = _resolve_attr(torch.load(path, map_location="cpu"), ATTR_CHAIN)
    if tensor.ndim != 2:
        raise ValueError(f"{path}: expected 2‑D tensor, got {tuple(tensor.shape)}")
    return tensor.float()


# --- label helpers ---------------------------------------------------------

LEVEL_PATTERN = re.compile(r"level_(\d)")


def infer_level(path: Path) -> int | None:
    """Return 1, 2, 3 or *None* if the level cannot be inferred from *path*."""
    m = LEVEL_PATTERN.search(path.stem)
    return int(m.group(1)) if m else None


def idx_to_label_id(level: int | None, idx: int) -> int:
    """Map an *embedding index* (after splitting) back to the original label id."""
    if level == 3 or level is None:
        return idx
    if level == 2:
        return 0 if idx == 0 else 216 + (idx - 1)
    if level == 1:
        return 0 if idx == 0 else 205 + (idx - 1)
    raise ValueError(f"Unknown level: {level}")


def load_id_to_name(cfg_path: Path) -> Dict[int, str]:
    """Return *id → human‑readable label* mapping from the SPIN hierarchy file."""

    with cfg_path.open() as f:
        data = json.load(f)
    if "nodes" not in data:
        raise KeyError("'nodes' key not found in hierarchy JSON")
    return {n["id"]: n.get("label", str(n["id"]) ) for n in data["nodes"]}


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main() -> None:
    emb_a = load_embeddings(EMB_A_PATH)
    emb_b = load_embeddings(EMB_B_PATH)

    ball = gt.PoincareBall(c=1)
    dists = fast_dist(emb_a, emb_b.unsqueeze(1), ball.k).T
    nn_idx = dists.argmin(dim=1)

    id_to_name = load_id_to_name(CONFIG_PATH)
    level_a = infer_level(EMB_A_PATH)
    level_b = infer_level(EMB_B_PATH)

    # Pretty print
    max_len = max(len(name) for name in id_to_name.values()) if id_to_name else 0
    for idx_a, idx_b in enumerate(nn_idx.tolist()):
        label_id_a = idx_to_label_id(level_a, idx_a)
        label_id_b = idx_to_label_id(level_b, idx_b)
        name_a = id_to_name.get(label_id_a, str(label_id_a))
        name_b = id_to_name.get(label_id_b, str(label_id_b))
        print(f"{name_a:<{max_len}}  ->  {name_b:<{max_len}}  (dist = {dists[idx_a, idx_b]:.4f})")


if __name__ == "__main__":
    main()