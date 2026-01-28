import os
import sys
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

group_attrs = [
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
df = df[cols_to_keep + group_attrs]
df = df.dropna(subset=["comment_text"])

for col in group_attrs:
    df[col] = (df[col] > 0.5).astype(int)

df.rename(columns={"toxicity": "y"}, inplace=True)
df["y"] = (df["y"] >= 0.5).astype(int)

# Protected groups are those minority groups
protected_groups = ["black", "LGBTQ", "muslim", "other_religions"]
df["protected"] = (
    (df[protected_groups].max(axis=1) > 0.5).astype(int).map({0: "non-protected", 1: "protected"})
)

group_map = {
    "non-protected_0": 0,
    "non-protected_1": 1,
    "protected_0": 2,
    "protected_1": 3,
}

df["group"] = df["protected"] + "_" + df["y"].astype(str)
df["group"] = df["group"].map(group_map)

df.rename(columns={"group": "g"}, inplace=True)
df["filename"] = np.arange(len(df))
df["split"] = df["split"].map({"train": 0, "val": 1, "test": 2})
df = df[["filename", "comment_text", "split", "y", "g"]]

df.to_csv(os.path.join(output_dir, "civil_comments.csv"), index=False)

# Compute and save BERT embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased").to(device)
bert.eval()

split_map = {"train": 0, "val": 1, "test": 2}
batch_size = 64

print("Computing BERT embeddings...")
for split_name, split_id in tqdm(split_map.items(), desc="Splits"):
    df_split = df[df["split"] == split_id].reset_index(drop=True)

    all_embeddings = []

    for i in tqdm(range(0, len(df_split), batch_size)):
        batch_texts = df_split["comment_text"].iloc[i : i + batch_size].tolist()
        batch_ids = df_split["filename"].iloc[i : i + batch_size].tolist()
        batch_labels = df_split["y"].iloc[i : i + batch_size].tolist()
        batch_groups = df_split["g"].iloc[i : i + batch_size].tolist()

        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = bert(**encoded)
            emb_batch = outputs.pooler_output.cpu().numpy()

        for j in range(emb_batch.shape[0]):
            all_embeddings.append(emb_batch[j])

    all_embeddings = np.array(all_embeddings)
    embeddings_arr = np.stack(all_embeddings, axis=0)  # shape (N_split, hidden)

    embeddings_path = os.path.join(root_dir, f"embeddings_{split_name}.pt")
    torch.save(torch.from_numpy(embeddings_arr).float(), embeddings_path)

print("Civil Comments dataset prepared successfully.")
print(f"Total samples: {len(df)}")
print(f"Training samples: {len(df[df['split'] == 0])}")
print(f"Validation samples: {len(df[df['split'] == 1])}")
print(f"Test samples: {len(df[df['split'] == 2])}")
print(f"Unique groups: {len(df['g'].unique())}")
print(f"Unique labels: {len(df['y'].unique())}")
print(f"Minimum group size (train): {df[df['split'] == 0]['g'].value_counts().min()}")
print(f"Maximum group size (train): {df[df['split'] == 0]['g'].value_counts().max()}")
