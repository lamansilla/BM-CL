import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

df = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)

# Drop rows with missing values
df = df.dropna()

# Target variable
df["y"] = (df["income"] == ">50K").astype(int)


def group_races(race):
    if race == "White":
        return "White"
    elif race == "Black":
        return "Black"
    else:
        return "Other"


# Group races into three categories: White, Black, Other
df["race_grouped"] = df["race"].apply(group_races)

# Filter out the 'Other' race category
df = df[df["race_grouped"] != "Other"].reset_index(drop=True)

# Group identifier: combine race, sex, and income
group_encoder = LabelEncoder()
df["group_str"] = df["race_grouped"] + "_" + df["sex"] + "_" + df["y"].astype(str)
df["g"] = group_encoder.fit_transform(df["group_str"])

categorical_features = [
    "workclass",
    "education",
    "occupation",
]

numerical_features = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

# Encode categorical features
for col in categorical_features:
    le = LabelEncoder()
    df[col + "_encoded"] = le.fit_transform(df[col])

feature_columns = [col + "_encoded" for col in categorical_features] + numerical_features
features = df[feature_columns].values

# Normalize numerical features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Split the data into train, validation, and test sets (60/20/20)
X_temp, X_test, y_temp, y_test, g_temp, g_test = train_test_split(
    features_normalized,
    df["y"].values,
    df["g"].values,
    test_size=0.2,
    random_state=42,
    stratify=df["y"],
)

X_train, X_val, y_train, y_val, g_train, g_val = train_test_split(
    X_temp,
    y_temp,
    g_temp,
    test_size=0.2 / 0.8,
    random_state=42,
    stratify=y_temp,
)


def create_split_dataframe(X, y, g, split_name):
    split_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    split_df["y"] = y
    split_df["g"] = g
    split_df["split"] = split_name
    return split_df


split_mapping = {"train": 0, "val": 1, "test": 2}
train_df = create_split_dataframe(X_train, y_train, g_train, split_mapping["train"])
val_df = create_split_dataframe(X_val, y_val, g_val, split_mapping["val"])
test_df = create_split_dataframe(X_test, y_test, g_test, split_mapping["test"])

df = pd.concat([train_df, val_df, test_df], ignore_index=True)

output_dir = "metadata"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/adult.csv"
df.to_csv(output_path, index=False)

print("Adult dataset prepared successfully.")
print(f"Total samples: {len(df)}")
print(f"Training samples: {len(df[df['split'] == 0])}")
print(f"Validation samples: {len(df[df['split'] == 1])}")
print(f"Test samples: {len(df[df['split'] == 2])}")
print(f"Unique groups: {len(df['g'].unique())}")
print(f"Unique labels: {len(df['y'].unique())}")
print(f"Minimum group size (train): {df[df['split'] == 0]['g'].value_counts().min()}")
print(f"Maximum group size (train): {df[df['split'] == 0]['g'].value_counts().max()}")
