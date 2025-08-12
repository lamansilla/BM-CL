import os

import pandas as pd

from src.utils.misc import create_groups

root_dir = "../datasets/waterbirds"
img_dir = os.path.join(root_dir, "waterbird_complete95_forest2water2")
output_dir = os.path.join("metadata")

metadata_path = os.path.join(img_dir, "metadata.csv")
df = pd.read_csv(metadata_path)

df["filepath"] = df["img_filename"].apply(
    lambda x: os.path.join("waterbird_complete95_forest2water2", x)
)
df["y"] = df["y"].astype(int)
df["a"] = df["place"].astype(int)
df["split"] = df["split"].astype(int)

groups = create_groups(df["a"], df["y"])
df["g"] = groups.astype(int)

df = df[["filepath", "y", "g", "split"]]
df.to_csv(os.path.join(output_dir, "waterbirds.csv"), index=False)

print("Waterbirds dataset prepared successfully.")
print(f"Total images: {len(df)}")
print(f"Training images: {len(df[df['split'] == 0])}")
print(f"Validation images: {len(df[df['split'] == 1])}")
print(f"Test images: {len(df[df['split'] == 2])}")
print(f"Unique groups: {len(df['g'].unique())}")
print(f"Unique labels: {len(df['y'].unique())}")
print(f"Minimum group size: {df['g'].value_counts().min()}")
print(f"Maximum group size: {df['g'].value_counts().max()}")
