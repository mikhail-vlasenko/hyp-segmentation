import json
import os
from pascal_part import PascalPartDataset
from transformers import SegformerImageProcessor


def build_pascal_part_tree(part_index_map, part_first=False):
    """
    Build hierarchical tree for Pascal Part dataset.
    
    Args:
        part_index_map: Dictionary mapping class IDs to their part mappings
        part_first: If True, organize by parts first, then classes. If False, organize by classes first.
    
    Returns:
        Dictionary with 'nodes' and 'edges' representing the hierarchy
    """
    # Pascal VOC classes (excluding background at index 0)
    classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # Start with offset 1 for 1-based indexing (0 is background)
    offset = 1
    
    # Initialize with root and background nodes
    nodes = [{"id": offset, "name": "Root"}, {"id": 0, "name": "Background"}]
    edges = [{"from": offset, "to": 0}]
    
    if not part_first:
        # Class-first hierarchy: Root -> Classes -> Parts
        class_ids = {cls: offset + 1 + i for i, cls in enumerate(classes)}
        
        # Add class nodes
        for cls, cls_id in class_ids.items():
            nodes.append({"id": cls_id, "name": cls})
            edges.append({"from": offset, "to": cls_id})
        
        # Add part nodes for each class
        part_offset = offset + len(classes) + 1
        part_counter = 0
        
        for class_idx, cls in enumerate(classes, 1):  # Start from 1 since 0 is background
            if class_idx in part_index_map and part_index_map[class_idx]:
                class_id = class_ids[cls]
                
                # Group parts by their index within this class
                index_to_parts = {}
                for part_name, part_idx in part_index_map[class_idx].items():
                    if part_idx not in index_to_parts:
                        index_to_parts[part_idx] = []
                    index_to_parts[part_idx].append(part_name)
                
                # Create one node per unique part index
                for part_idx, part_names in index_to_parts.items():
                    # Choose the most representative name (shortest, most generic)
                    # Prefer names without numbers, underscores, or l/r prefixes
                    def get_name_priority(name):
                        score = 0
                        # Prefer shorter names
                        score += len(name) * 10
                        # Penalize numbered parts
                        if any(c.isdigit() for c in name):
                            score += 100
                        # Penalize left/right parts
                        if name.startswith(('l', 'r')) and len(name) > 1:
                            score += 50
                        # Penalize parts with underscores
                        if '_' in name:
                            score += 20
                        return score
                    
                    representative_name = min(part_names, key=get_name_priority)
                    part_node_id = part_offset + part_counter
                    part_full_name = f"{cls}_{representative_name}"
                    
                    nodes.append({"id": part_node_id, "name": part_full_name})
                    edges.append({"from": class_id, "to": part_node_id})
                    part_counter += 1
    
    else:
        # Part-first hierarchy: Root -> Parts -> Classes
        # Helper function to get representative part name
        def get_representative_name(part_names):
            def get_name_priority(name):
                score = 0
                # Prefer shorter names
                score += len(name) * 10
                # Penalize numbered parts
                if any(c.isdigit() for c in name):
                    score += 100
                # Penalize left/right parts
                if name.startswith(('l', 'r')) and len(name) > 1:
                    score += 50
                # Penalize parts with underscores
                if '_' in name:
                    score += 20
                return score
            return min(part_names, key=get_name_priority)
        
        # Collect all unique representative part names across all classes
        all_representative_parts = set()
        class_part_representatives = {}  # Maps (class_idx, part_idx) to representative name
        
        for class_idx, part_map in part_index_map.items():
            if part_map:
                # Group parts by their index within this class
                index_to_parts = {}
                for part_name, part_idx in part_map.items():
                    if part_idx not in index_to_parts:
                        index_to_parts[part_idx] = []
                    index_to_parts[part_idx].append(part_name)
                
                # Get representative name for each part index
                for part_idx, part_names in index_to_parts.items():
                    representative_name = get_representative_name(part_names)
                    all_representative_parts.add(representative_name)
                    class_part_representatives[(class_idx, part_idx)] = representative_name
        
        # Create part root nodes
        part_ids = {part: offset + 1 + i for i, part in enumerate(sorted(all_representative_parts))}
        
        # Add part root nodes
        for part, part_id in part_ids.items():
            nodes.append({"id": part_id, "name": part})
            edges.append({"from": offset, "to": part_id})
        
        # Add class-specific part nodes
        class_part_offset = offset + len(all_representative_parts) + 1
        node_counter = 0
        
        for class_idx, cls in enumerate(classes, 1):  # Start from 1 since 0 is background
            if class_idx in part_index_map and part_index_map[class_idx]:
                # Group parts by their index within this class
                index_to_parts = {}
                for part_name, part_idx in part_index_map[class_idx].items():
                    if part_idx not in index_to_parts:
                        index_to_parts[part_idx] = []
                    index_to_parts[part_idx].append(part_name)
                
                # Create one node per unique part index
                for part_idx, part_names in index_to_parts.items():
                    representative_name = get_representative_name(part_names)
                    class_part_id = class_part_offset + node_counter
                    class_part_name = f"{cls}_{representative_name}"
                    
                    nodes.append({"id": class_part_id, "name": class_part_name})
                    # Connect from part root to class-specific part
                    edges.append({"from": part_ids[representative_name], "to": class_part_id})
                    node_counter += 1
    
    return {"nodes": nodes, "edges": edges}


def save_tree_to_json(tree, filepath):
    """Save tree structure to JSON file in graph format."""
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
    # Initialize processor and dataset
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    
    dataset = PascalPartDataset(
        voc_root="/home/misha/data/voc_root",
        split="train",
        processor=processor,
        crop_size=(0.8, 0.8),
    )
    
    # Get part index mapping
    part_index_map = dataset.PART_INDEX_MAP
    
    # Build hierarchies (both class-first and part-first)
    for part_first in [False, True]:
        tree = build_pascal_part_tree(part_index_map, part_first=part_first)
        
        # Save to JSON file
        suffix = "_part-first" if part_first else "_class-first"
        output_path = f"hierarchy_embeddings/pascal_part_dataset/pascal_part_hierarchy{suffix}.json"
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        save_tree_to_json(tree, output_path)
        print(f"Saved Pascal Part hierarchy to: {output_path}")
        print(f"Total nodes: {len(tree['nodes'])}, Total edges: {len(tree['edges'])}")
