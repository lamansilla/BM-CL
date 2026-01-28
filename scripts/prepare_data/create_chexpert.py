import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.misc import create_groups

root_dir = "../datasets/CheXpert-v1.0-small"
output_dir = "metadata"

df = pd.concat(
    [
        pd.read_csv(os.path.join(root_dir, "train.csv")),
        pd.read_csv(os.path.join(root_dir, "valid.csv")),
    ],
    ignore_index=True,
)

df["subject_id"] = df["Path"].apply(lambda x: Path(x).parts[-3][7:]).astype(str)

df = df[df["Sex"].isin(["Male", "Female"])]
df = df[df["Age"].notna()]


def categorize_age(age):
    if age < 40:
        return "Young"
    elif age <= 65:
        return "Middle"
    else:
        return "Old"


df["AgeGroup"] = df["Age"].apply(categorize_age)

group_mapping = {
    "Male_Young": 0,
    "Male_Middle": 1,
    "Male_Old": 2,
    "Female_Young": 3,
    "Female_Middle": 4,
    "Female_Old": 5,
}

df["a"] = df["AgeGroup"].map({"Young": 0, "Middle": 1, "Old": 2})

# only frontal images
df = df[df["Frontal/Lateral"].fillna("") == "Frontal"].reset_index(drop=True)

# ensure no patient overlap between splits
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

df["y"] = df["Pleural Effusion"].fillna(0.0)
df["y"] = df["y"].replace(-1, 0)
df["y"] = df["y"].astype(int)

groups = create_groups(df["a"], df["y"])
df["g"] = groups.astype(int)

df["filepath"] = df["Path"].apply(lambda x: Path(x).relative_to("CheXpert-v1.0-small").as_posix())

df = df[["filepath", "y", "g", "a", "split"]]

os.makedirs(output_dir, exist_ok=True)
df.to_csv(os.path.join(output_dir, "chexpert.csv"), index=False)

print("CheXpert dataset prepared successfully.")
print(f"Total samples: {len(df)}")
print(f"Training samples: {len(df[df['split'] == 0])}")
print(f"Validation samples: {len(df[df['split'] == 1])}")
print(f"Test samples: {len(df[df['split'] == 2])}")
print(f"Unique groups: {len(df['g'].unique())}")
print(f"Unique labels: {len(df['y'].unique())}")
print(f"Minimum group size (train): {df[df['split'] == 0]['g'].value_counts().min()}")
print(f"Maximum group size (train): {df[df['split'] == 0]['g'].value_counts().max()}")
