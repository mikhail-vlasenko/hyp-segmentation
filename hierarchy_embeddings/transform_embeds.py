import json
import sys
sys.path.append("..")

import torch


BG_CLASS_INDEX = 0
ROOT_CLASS_INDEX = 204
# root_dir = "../h_embeds/"
root_dir = "/home/misha/projects/hyp-segmentation/hierarchy_embeddings/hierarchies/hierarchy_embeddings/experiments/spin_dataset/spin_hierarchy"

is_part_first = True
if is_part_first:
    root_dir += "_part-first"
    ROOT_CLASS_INDEX = 1
embed_dir = root_dir + "/2025-05-17_195428/"
level = 2
renormalize = False
centered_bg = False  # when true, the bg is set to root, making it near the center of the disk

with open(embed_dir + "config.json", "r") as f:
    config = json.load(f)
    dim = config["embedding_dim"]

file_path = embed_dir + f"HierarchyEmbedding_weights_{dim}.pth"
save_path = embed_dir + (f"HierarchyEmbedding_weights_{dim}"
                         f"{'_centered-bg' if centered_bg else ''}"
                         f"_only_level_{level}"
                         f"{'_renormalized' if renormalize else ''}.pth")

embeddings = torch.load(file_path, weights_only=False)

if centered_bg:
    bg = embeddings.embeddings.weight.tensor[ROOT_CLASS_INDEX]
else:
    # use the actual bg class
    bg = embeddings.embeddings.weight.tensor[0]

if level == 3:
    if is_part_first:
        raise ValueError("level 3 is not supported for part-first")
    embeddings.embeddings.weight.tensor = torch.cat(
        (bg.unsqueeze(0), embeddings.embeddings.weight.tensor[1:ROOT_CLASS_INDEX]), dim=0
    )

# level2 is 0 for bg and 216 to 255 (incl)
if level == 2:
    if is_part_first:
        embeddings.embeddings.weight.tensor = embeddings.embeddings.weight.tensor[16:56]
    else:
        embeddings.embeddings.weight.tensor = embeddings.embeddings.weight.tensor[216:256]
    embeddings.embeddings.weight.tensor = torch.cat(
        (embeddings.embeddings.weight.tensor, bg.unsqueeze(0)), dim=0
    )

if level == 1:
    if is_part_first:
        embeddings.embeddings.weight.tensor = embeddings.embeddings.weight.tensor[2:16]
    else:
        embeddings.embeddings.weight.tensor = embeddings.embeddings.weight.tensor[205:216]
    embeddings.embeddings.weight.tensor = torch.cat(
        (embeddings.embeddings.weight.tensor, bg.unsqueeze(0)), dim=0
    )

# print norm
norm = torch.norm(embeddings.embeddings.weight.tensor, dim=1, p=2)
print(norm)
print(norm.max())

# renormalize to 0.95
if renormalize:
    embeddings.embeddings.weight.tensor = embeddings.embeddings.weight.tensor * 0.95 / norm.max()
    print(torch.norm(embeddings.embeddings.weight.tensor, dim=1, p=2))

torch.save(embeddings, save_path)

print(embeddings.embeddings.weight.tensor.shape)
print(f"{level=} {renormalize=} {save_path=} {centered_bg=}")
