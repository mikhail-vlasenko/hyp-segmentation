import json
import os
import re
from pprint import pprint

from spin import SPIN, file_to_object_mapping

def rename_part_name(name):
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

    # Other singular->plural or generic renames
    name = name.replace("Foot", "Legs")
    name = name.replace("Hand", "Arm")
    return name

def build_tree(level2, level3, part_first=False):
    # level 1: wholes
    wholes = [
        "Quadruped", "Biped", "Fish", "Bird", "Snake",
        "Reptile", "Car", "Bicycle", "Boat", "Aeroplane", "Bottle"
    ]

    offset = 1  # 1-based indexing as 0 is background
    if not part_first:
        offset += max(entry["id"] for entry in level3)

    nodes = [{"id": offset, "name": "Root"}, {"id": 0, "name": "Background"}]
    edges = [{"from": offset, "to": 0}]

    if not part_first:
        # Create whole root nodes
        root_ids = {cat: offset + 1 + i for i, cat in enumerate(wholes)}
        for cat, rid in root_ids.items():
            nodes.append({"id": rid, "name": cat})
            edges.append({"from": offset, "to": rid})
    else:
        # Create part root nodes
        parts = set()
        for entry in level2:
            parts.add(rename_part_name(entry["name"]).split(" ",1)[1].replace(" ","_"))
        part_ids = {p: offset + 1 + i for i, p in enumerate(sorted(parts))}
        for part, pid in part_ids.items():
            nodes.append({"id": pid, "name": part})
            edges.append({"from": offset, "to": pid})

    level2_offset = offset + len(nodes) - 1
    intermediate_map = {}

    # Process level2 nodes
    for entry in level2:
        level2_id = entry["id"] + level2_offset
        name = rename_part_name(entry["name"])
        supercat, part = name.split(" ", 1)
        part_key = part.replace(" ", "_")

        nodes.append({"id": level2_id, "name": name})

        if part_first:
            # connect part -> level2
            pid = part_ids[part_key]
            edges.append({"from": pid, "to": level2_id})
        else:
            # connect whole -> level2
            edges.append({"from": root_ids[supercat], "to": level2_id})

        intermediate_map[(supercat, part_key)] = level2_id

    if not part_first:
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
    part_first = True

    spin_api = SPIN(
        annotation_dir=annotation_dir,
        image_dir=image_dir,
        split="train",
        download=False,
    )

    whole_cats = convert_dict_to_list(spin_api.wholes.cats)
    part_cats = convert_dict_to_list(spin_api.parts.cats)
    subpart_cats = convert_dict_to_list(spin_api.subparts.cats)

    tree = build_tree(part_cats, subpart_cats, part_first=part_first)
    save_tree_to_json(tree, f"spin_dataset/spin_hierarchy{'_part-first' if part_first else ''}.json")
