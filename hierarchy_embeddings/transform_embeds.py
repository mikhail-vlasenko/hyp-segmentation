import sys
sys.path.append("..")

import torch


BG_CLASS_INDEX = 0
ROOT_CLASS_INDEX = 204
# root_dir = "../h_embeds/"
root_dir = "/home/misha/projects/hyp-segmentation/hierarchy_embeddings/hierarchies/hierarchy_embeddings/experiments/spin_dataset/spin_hierarchy/"
embed_dir = root_dir + "2025-04-14_204854/"
level = 1
renormalize = False
centered_bg = True  # when true, the bg is set to root, making it near the center of the disk
dim = 64

file_path = embed_dir + f"HierarchyEmbedding_weights_{dim}.pth"
save_path = embed_dir + (f"HierarchyEmbedding_weights_{dim}"
                         f"{'_centered-bg' if centered_bg else ''}"
                         f"_only_level_{level}"
                         f"{'_renormalized' if renormalize else ''}.pth")

embeddings = torch.load(file_path)

if centered_bg:
    bg = embeddings.embeddings.weight.tensor[ROOT_CLASS_INDEX]
else:
    # use the actual bg class
    bg = embeddings.embeddings.weight.tensor[0]

if level == 3:
    embeddings.embeddings.weight.tensor = torch.cat(
        (bg.unsqueeze(0), embeddings.embeddings.weight.tensor[1:ROOT_CLASS_INDEX]), dim=0
    )

# level2 is 0 for bg and 216 to 255 (incl)
if level == 2:
    embeddings.embeddings.weight.tensor = embeddings.embeddings.weight.tensor[216:256]
    embeddings.embeddings.weight.tensor = torch.cat(
        (embeddings.embeddings.weight.tensor, bg.unsqueeze(0)), dim=0
    )

if level == 1:
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
