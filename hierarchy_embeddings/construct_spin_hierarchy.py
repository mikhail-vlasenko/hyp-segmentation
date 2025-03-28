import json
import os
import re
from pprint import pprint

from spin import SPIN, file_to_object_mapping


def build_tree(level2, level3):
    # level 1: wholes
    wholes = [
        "Quadruped", "Biped", "Fish", "Bird", "Snake",
        "Reptile", "Car", "Bicycle", "Boat", "Aeroplane", "Bottle"
    ]
    offset = max([entry["id"] for entry in level3]) + 1  # 1-based indexing as 0 is background
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

    # Process the main dictionary (first set of vertices)
    for entry in level2:
        level2_id = entry["id"] + level2_offset
        name = entry["name"]
        # Rename some categories to match the expected format
        body_torso_rename_wholes = wholes[0:6]
        name = re.sub(re.compile(f"({'|'.join(body_torso_rename_wholes)}) Body"), r"\1 Torso", name)
        name = name.replace("Tier", "Tire")
        name = name.replace("Foot", "Legs")
        name = name.replace("Hand", "Arm")
        # Assume the name is like "Quadruped Head" or "Car Side Mirror"
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
        # Example name: "Quadruped-Head-Eyes"
        parts = name.split("-")
        assert len(parts) == 3, f"Unexpected name format: {name}"
        supercat = parts[0]
        level2_token = parts[1]

        key1 = (supercat, level2_token)

        assert key1 in intermediate_map, f"Missing intermediate node: {key1}"
        parent_id = intermediate_map[key1]

        # Now add the sub node using its provided id.
        nodes.append({"id": entry["id"], "name": name})
        edges.append({"from": parent_id, "to": entry["id"]})

    # Return the tree as a dict with nodes and edges.
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

    # Convert each node to have a label and id
    for node in tree["nodes"]:
        graph_data["nodes"].append({
            "label": node["name"],
            "id": node["id"]
        })

    # Convert each edge to a link with source and target
    for edge in tree["edges"]:
        graph_data["links"].append({
            "source": edge["from"],
            "target": edge["to"]
        })

    # Write out the JSON file
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

    tree = build_tree(part_cats, subpart_cats)
    save_tree_to_json(tree, "spin_dataset/spin_hierarchy.json")
