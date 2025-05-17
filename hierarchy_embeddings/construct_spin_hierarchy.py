import json
import os
import re
from pprint import pprint

from spin import SPIN, file_to_object_mapping

def rename_part_name(name, join_wheel=False):
    """
    Rename raw part names to standardized format.
    - Replace "<Whole> Body" with "<Whole> Torso" for certain wholes.
    - Replace "Tier" with "Wheel" or "Tire" depending on join_wheel.
    - Replace "Foot" with "Legs" and "Hand" with "Arm".
    """
    # Rename Body to Torso for specified wholes
    body_torso_rename_wholes = [
        "Quadruped", "Biped", "Fish", "Bird", "Snake", "Reptile"
    ]
    pattern = f"({'|'.join(body_torso_rename_wholes)}) Body"
    name = re.sub(re.compile(pattern), r"\1 Torso", name)
    name = name.replace("Tier", "Tire")

    if join_wheel:
        name = name.replace("Tire", "Wheel")

    # Other singular->plural or generic renames
    name = name.replace("Foot", "Legs")
    name = name.replace("Hand", "Arm")
    return name

def build_tree(level2, level3, join_wheel=False):
    # level 1: wholes
    wholes = [
        "Quadruped", "Biped", "Fish", "Bird", "Snake",
        "Reptile", "Car", "Bicycle", "Boat", "Aeroplane", "Bottle"
    ]
    offset = max(entry["id"] for entry in level3) + 1  # 1-based indexing as 0 is background
    root_ids = {cat: offset + 1 + i for i, cat in enumerate(wholes)}

    nodes = [{"id": offset, "name": "Root"}, {"id": 0, "name": "Background"}]
    edges = [{"from": offset, "to": 0}]

    # Add the root nodes.
    for cat, rid in root_ids.items():
        nodes.append({"id": rid, "name": cat})
        edges.append({"from": offset, "to": rid})

    level2_offset = offset + len(nodes) - 1  # -1 for the added background node

    # Build a mapping for level2 nodes
    intermediate_map = {}

    # Process level2 nodes
    for entry in level2:
        level2_id = entry["id"] + level2_offset
        raw_name = entry["name"]
        name = rename_part_name(raw_name, join_wheel)

        tokens = name.split(" ", 1)
        assert len(tokens) == 2, f"Unexpected name format: {name}"
        supercat, part = tokens
        part = part.replace(" ", "_")
        # Add the node using its provided id.
        nodes.append({"id": level2_id, "name": name})
        # Connect it to the corresponding root node.
        if supercat in root_ids:
            edges.append({"from": root_ids[supercat], "to": level2_id})
        # Save in our lookup (using lower-case for matching)
        intermediate_map[(supercat, part)] = level2_id

    # Process the sub dictionary (second set of vertices)
    for entry in level3:
        name = entry["name"]
        parts = name.split("-")
        assert len(parts) == 3, f"Unexpected name format: {name}"
        supercat = parts[0]
        level2_token = parts[1]

        key = (supercat, level2_token)
        assert key in intermediate_map, f"Missing intermediate node: {key}"
        parent_id = intermediate_map[key]

        nodes.append({"id": entry["id"], "name": name})
        edges.append({"from": parent_id, "to": entry["id"]})

    return {"nodes": nodes, "edges": edges}


def convert_dict_to_list(dict_obj):
    return [v for _, v in dict_obj.items()]


def save_tree_to_json(tree, filepath):
    graph_data = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [],
        "links": []
    }

    for node in tree["nodes"]:
        graph_data["nodes"].append({"label": node["name"], "id": node["id"]})
    for edge in tree["edges"]:
        graph_data["links"].append({"source": edge["from"], "target": edge["to"]})

    with open(filepath, "w") as f:
        json.dump(graph_data, f, indent=4)


if __name__ == '__main__':
    dataset_dir = "/home/misha/data/PartImageNet/"
    annotation_dir = os.path.join(dataset_dir, "annotations")
    image_dir = os.path.join(dataset_dir, "images")

    spin_api = SPIN(
        annotation_dir=annotation_dir,
        image_dir=image_dir,
        split="train",  # categories were checked to be the same for all splits
        download=False,
    )
    # fix the format
    whole_cats = convert_dict_to_list(spin_api.wholes.cats)
    part_cats = convert_dict_to_list(spin_api.parts.cats)
    subpart_cats = convert_dict_to_list(spin_api.subparts.cats)

    # Toggle join_wheel=True to collapse Tier into Wheel
    tree = build_tree(part_cats, subpart_cats, join_wheel=False)
    save_tree_to_json(tree, "spin_dataset/spin_hierarchy.json")
