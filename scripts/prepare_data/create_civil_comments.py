import os
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

root_dir = "../datasets/CivilComments/"
output_dir = "metadata"

df = pd.read_csv(os.path.join(root_dir, "all_data_with_identities.csv"))

identity_attrs = [
    "male",
    "female",
    "LGBTQ",
    "christian",
    "muslim",
    "other_religions",
    "black",
    "white",
]

cols_to_keep = ["comment_text", "split", "toxicity"]
df = df[cols_to_keep + identity_attrs]
df = df.dropna(subset=["comment_text"])

for col in identity_attrs:
    df[col] = (df[col] > 0.5).astype(int)

df.rename(columns={"toxicity": "y"}, inplace=True)
df["y"] = (df["y"] >= 0.5).astype(int)

train_mask = df["split"] == "train"
identity_counts = {attr: df.loc[train_mask, attr].sum() for attr in identity_attrs}
identity_priority = sorted(identity_attrs, key=lambda x: identity_counts[x])

print("Identity counts in train (rarity order, rarest first):")
for attr in identity_priority:
    print(f"  {attr:20s}: {identity_counts[attr]:6d}")

attr_to_idx = {attr: i for i, attr in enumerate(identity_priority)}


def assign_group(row):
    for attr in identity_priority:
        if row[attr] == 1:
            return 2 + attr_to_idx[attr] * 2 + int(row["y"])
    return int(row["y"])


df["g"] = df.apply(assign_group, axis=1)
df["a"] = df["g"] // 2

group_label_map = {0: "no_identity/non-toxic", 1: "no_identity/toxic"}
for i, attr in enumerate(identity_priority):
    group_label_map[2 + 2 * i] = f"{attr}/non-toxic"
    group_label_map[2 + 2 * i + 1] = f"{attr}/toxic"

df["split"] = df["split"].map({"train": 0, "val": 1, "test": 2})

print(f"\nGroup distribution (train, n={train_mask.sum()}):")
train = df[df["split"] == 0]
for g in sorted(train["g"].unique()):
    n = (train["g"] == g).sum()
    print(f"  g={g:2d}  {group_label_map[g]:30s}: {n:6d} ({n / len(train) * 100:.1f}%)")

df = df[["comment_text", "split", "y", "g", "a"]]
df.to_csv(os.path.join(output_dir, "civil_comments.csv"), index=False)
print(f"\nMetadata saved → {output_dir}/civil_comments.csv")
print(f"Total groups: {df['g'].nunique()}")

# Compute and save BERT embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased").to(device)
bert.eval()

split_map = {"train": 0, "val": 1, "test": 2}
batch_size = 64

print("\nComputing BERT embeddings...")
for split_name, split_id in tqdm(split_map.items(), desc="Splits"):
    df_split = df[df["split"] == split_id].reset_index(drop=True)

    all_embeddings = []

    for i in tqdm(range(0, len(df_split), batch_size), desc=split_name, leave=False):
        batch_texts = df_split["comment_text"].iloc[i : i + batch_size].tolist()

        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = bert(**encoded)
            emb_batch = outputs.pooler_output.cpu().numpy()

        all_embeddings.append(emb_batch)

    embeddings_arr = np.concatenate(all_embeddings, axis=0)
    embeddings_path = os.path.join(root_dir, f"embeddings_{split_name}.pt")
    torch.save(torch.from_numpy(embeddings_arr).float(), embeddings_path)

print("Adult dataset prepared successfully.")
print(f"Total samples: {len(df)}")
print(f"Training samples: {len(df[df['split'] == 0])}")
print(f"Validation samples: {len(df[df['split'] == 1])}")
print(f"Test samples: {len(df[df['split'] == 2])}")
print(f"Unique groups: {len(df['g'].unique())}")
print(f"Unique labels: {len(df['y'].unique())}")
print(f"Minimum group size (train): {df[df['split'] == 0]['g'].value_counts().min()}")
print(f"Maximum group size (train): {df[df['split'] == 0]['g'].value_counts().max()}")
