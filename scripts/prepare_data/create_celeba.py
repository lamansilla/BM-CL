import os

import pandas as pd

from src.utils.misc import create_groups

root_dir = "../datasets/celeba"
img_dir = "img_align_celeba"
output_dir = "metadata"

partition_path = os.path.join(root_dir, "list_eval_partition.txt")
attr_path = os.path.join(root_dir, "list_attr_celeba.txt")

with open(partition_path, "r") as f:
    partition_data = f.readlines()

with open(attr_path, "r") as f:
    attribute_data = f.readlines()[2:]

paths, splits, labels, genders = [], [], [], []

for part_info, attr_info in zip(partition_data, attribute_data):
    image_name, split_part = part_info.strip().split()
    attribute_values = attr_info.strip().split()[1:]

    # blond hair: 1, not blond hair: 0
    label = 1 if attribute_values[9] == "1" else 0

    # male: 1, female: 0
    gender = 1 if attribute_values[20] == "1" else 0

    paths.append(os.path.join(img_dir, image_name))
    splits.append(int(split_part))
    labels.append(label)
    genders.append(gender)

df = pd.DataFrame(
    {
        "filepath": paths,
        "y": labels,
        "gender": genders,
        "split": splits,
    }
)

df["y"] = df["y"].astype(int)
df["a"] = df["gender"].astype(int)
df["split"] = df["split"].astype(int)

groups = create_groups(df["a"].tolist(), df["y"].tolist())
df["g"] = groups.astype(int)

df = df[["filepath", "y", "g", "split"]]
df.to_csv(os.path.join(output_dir, "celeba.csv"), index=False)

print("CelebA dataset prepared successfully.")
print(f"Total images: {len(df)}")
print(f"Training images: {len(df[df['split'] == 0])}")
print(f"Validation images: {len(df[df['split'] == 1])}")
print(f"Test images: {len(df[df['split'] == 2])}")
print(f"Unique groups: {df['g'].nunique()}")
print(f"Unique labels: {df['y'].nunique()}")
print(f"Minimum group size (train): {df[df['split'] == 0]['g'].value_counts().min()}")
print(f"Maximum group size (train): {df[df['split'] == 0]['g'].value_counts().max()}")
