import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.misc import create_groups

root_dir = "../datasets/CheXpert-v1.0-small"
demo_path = "../datasets/CheXpert-v1.0-small/CHEXPERT DEMO.xlsx"
output_dir = "metadata"

df = pd.concat(
    [
        pd.read_csv(os.path.join(root_dir, "train.csv")),
        pd.read_csv(os.path.join(root_dir, "valid.csv")),
    ],
    ignore_index=True,
)

df["subject_id"] = df["Path"].apply(lambda x: Path(x).parts[-3]).astype(str)

demo = pd.read_excel(demo_path)
demo = demo.rename(columns={"PATIENT": "subject_id", "PRIMARY_RACE": "race_raw"})
demo["subject_id"] = demo["subject_id"].astype(str)

df = df.merge(demo[["subject_id", "race_raw"]], on="subject_id", how="inner")

df = df[df["Frontal/Lateral"].fillna("") == "Frontal"]
df = df[df["Sex"].isin(["Male", "Female"])].reset_index(drop=True)


def map_race(r):
    if isinstance(r, str):
        r_low = r.lower()
        if "white" in r_low:
            return "White"
        if "black" in r_low or "african" in r_low:
            return "Black"
    return "Other"


df["race"] = df["race_raw"].apply(map_race)

race_idx = {"White": 0, "Black": 1, "Other": 2}
sex_idx = {"Female": 0, "Male": 1}
df["a"] = df["race"].map(race_idx) * 2 + df["Sex"].map(sex_idx)

df["y"] = df["No Finding"].fillna(0.0).replace(-1, 0).astype(int)

subject_attr = df.groupby("subject_id")["a"].first()
subject_ids = subject_attr.index.tolist()
subject_groups = subject_attr.tolist()

test_pct, val_pct = 0.2, 0.1
train_val_ids, test_ids = train_test_split(
    subject_ids, test_size=test_pct, stratify=subject_groups, random_state=42
)
train_ids, val_ids = train_test_split(
    train_val_ids,
    test_size=val_pct / (1 - test_pct),
    stratify=[subject_attr[pid] for pid in train_val_ids],
    random_state=42,
)

split_map = {sid: 0 for sid in train_ids}
split_map.update({sid: 1 for sid in val_ids})
split_map.update({sid: 2 for sid in test_ids})
df["split"] = df["subject_id"].map(split_map).astype(int)

df["g"] = create_groups(df["a"], df["y"])

df["filepath"] = df["Path"].apply(lambda x: Path(x).relative_to("CheXpert-v1.0-small").as_posix())

df = df[["filepath", "y", "g", "a", "split"]]

os.makedirs(output_dir, exist_ok=True)
df.to_csv(os.path.join(output_dir, "chexpert.csv"), index=False)

print("CheXpert dataset prepared successfully.")
print(f"Total samples     : {len(df)}")
print(f"Training samples  : {len(df[df['split'] == 0])}")
print(f"Validation samples: {len(df[df['split'] == 1])}")
print(f"Test samples      : {len(df[df['split'] == 2])}")
print(f"Unique groups (g) : {df['g'].nunique()}  (subpopulation: race×sex×label)")
print(f"Unique attrs  (a) : {df['a'].nunique()}  (demographic: race×sex)")

train = df[df["split"] == 0]
print(f"\nGroup distribution (train, n={len(train)}):")
a_labels = {0: "White/F", 1: "White/M", 2: "Black/F", 3: "Black/M", 4: "Other/F", 5: "Other/M"}
g_labels = {2 * a + y: f"{a_labels[a]}/y={y}" for a in range(6) for y in range(2)}
for g in sorted(train["g"].unique()):
    n = (train["g"] == g).sum()
    print(f"  g={g:2d}  {g_labels.get(g, '?'):18s}  {n:6d}  ({n / len(train) * 100:.1f}%)")

print(f"\nMin group size (train): {train['g'].value_counts().min()}")
print(f"Max group size (train): {train['g'].value_counts().max()}")
print(
    f"Imbalance ratio       : {train['g'].value_counts().min() / train['g'].value_counts().max():.3f}"
)
