import os
from abc import ABC, abstractmethod

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, Subset


def get_dataset(
    dataset_name,
    root_dir,
    metadata_dir,
    use_pretrained=True,
):
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not found.")
    datasets = {}
    for split in BaseDataset.SPLITS.keys():
        datasets[split] = DATASETS[dataset_name](
            root_dir,
            metadata_dir,
            split,
            use_pretrained,
        )
    return datasets


class BaseDataset(Dataset, ABC):

    SPLITS = {"train": 0, "val": 1, "test": 2}

    def __init__(self):
        self.prev_outputs = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self._get_features(i)
        y = self.labels[i]
        g = self.groups[i]

        if self.prev_outputs is not None:
            prev = torch.tensor(self.prev_outputs[i], dtype=torch.float32)
            return x, y, g, prev

        return x, y, g

    # Must be implemented by subclasses
    @abstractmethod
    def _get_features(self, i):
        pass

    # Common to all datasets
    def get_num_classes(self):
        return self.num_classes

    def get_num_groups(self):
        return self.num_groups

    def get_group_counts(self):
        counts = [0] * self.num_groups
        for g in self.groups.tolist():
            counts[int(g)] += 1
        return counts

    def get_weights(self):
        counts = self.get_group_counts()
        return [1.0 / counts[g] for g in self.groups.tolist()]

    def set_prev_outputs(self, prev_outputs):
        if len(prev_outputs) != len(self.labels):
            raise ValueError("Size mismatch")
        self.prev_outputs = prev_outputs

    def get_group_subset(self, groups):
        if not isinstance(groups, (list, tuple)):
            groups = [groups]
        indices = [i for i, g in enumerate(self.groups.tolist()) if g in groups]
        return _GroupSubset(self, indices)


class _GroupSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.dataset = dataset

    @property
    def labels(self):
        return [self.dataset.labels[i].item() for i in self.indices]

    @property
    def groups(self):
        return [self.dataset.groups[i].item() for i in self.indices]


class ImageDataset(BaseDataset):

    DATA_TYPE = "images"

    def __init__(
        self,
        root_dir,
        metadata_path,
        split,
        image_root,
        transform,
    ):
        super().__init__()

        df = pd.read_csv(metadata_path)
        df = df[df["split"] == self.SPLITS[split]].reset_index(drop=True)

        self.image_paths = df["filepath"].tolist()
        self.labels = torch.tensor(df["y"].tolist())
        self.groups = torch.tensor(df["g"].tolist())

        self.num_classes = len(torch.unique(self.labels))
        self.num_groups = len(torch.unique(self.groups))

        self.root_dir = image_root
        self.transform = transform

    def _get_features(self, i):
        path = os.path.join(self.root_dir, self.image_paths[i])
        img = Image.open(path).convert("RGB")
        return self.transform(img)


class WaterbirdsDataset(ImageDataset):

    NUM_BATCHES = 150

    def __init__(self, root_dir, metadata_dir, split, use_pretrained=True):
        image_root = os.path.join(root_dir, "Waterbirds")
        metadata_path = f"{metadata_dir}/waterbirds.csv"

        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
        if use_pretrained:
            t.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )

        super().__init__(
            root_dir=root_dir,
            metadata_path=metadata_path,
            split=split,
            image_root=image_root,
            transform=transforms.Compose(t),
        )


class CelebADataset(ImageDataset):

    NUM_BATCHES = 1280

    def __init__(self, root_dir, metadata_dir, split, use_pretrained=True):
        image_root = os.path.join(root_dir, "celeba")
        metadata_path = f"{metadata_dir}/celeba.csv"

        t = [
            transforms.CenterCrop(178),
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
        if use_pretrained:
            t.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )

        super().__init__(
            root_dir=root_dir,
            metadata_path=metadata_path,
            split=split,
            image_root=image_root,
            transform=transforms.Compose(t),
        )


class ChexpertDataset(ImageDataset):

    NUM_BATCHES = 1050

    def __init__(self, root_dir, metadata_dir, split, use_pretrained=True):
        image_root = os.path.join(root_dir, "CheXpert-v1.0-small")
        metadata_path = f"{metadata_dir}/chexpert.csv"

        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
        if use_pretrained:
            t.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )

        super().__init__(
            root_dir=root_dir,
            metadata_path=metadata_path,
            split=split,
            image_root=image_root,
            transform=transforms.Compose(t),
        )


class AdultDataset(BaseDataset):

    DATA_TYPE = "tabular"
    NUM_BATCHES = 500

    def __init__(
        self,
        root_dir,
        metadata_dir,
        split,
        use_pretrained=True,
    ):
        super().__init__()

        df = pd.read_csv(f"{metadata_dir}/adult.csv")
        df = df[df["split"] == self.SPLITS[split]].reset_index(drop=True)

        feature_cols = [c for c in df.columns if c.startswith("feature_")]
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(df["y"].tolist())
        self.groups = torch.tensor(df["g"].tolist())

        self.num_classes = len(torch.unique(self.labels))
        self.num_groups = len(torch.unique(self.groups))

    def _get_features(self, i):
        return self.features[i]


class CivilCommentsDataset(BaseDataset):

    DATA_TYPE = "text"
    NUM_BATCHES = 530

    def __init__(
        self,
        root_dir,
        metadata_dir,
        split,
        use_pretrained=True,
    ):
        super().__init__()

        df = pd.read_csv(f"{metadata_dir}/civil_comments.csv")
        df = df[df["split"] == self.SPLITS[split]].reset_index(drop=True)

        root_dir = os.path.join(root_dir, "CivilComments")
        self.embeddings = torch.load(os.path.join(root_dir, f"embeddings_{split}.pt"))

        self.labels = torch.tensor(df["y"].tolist())
        self.groups = torch.tensor(df["g"].tolist())

        self.num_classes = len(torch.unique(self.labels))
        self.num_groups = len(torch.unique(self.groups))

    def _get_features(self, i):
        return self.embeddings[i].float()


DATASETS = {
    "waterbirds": WaterbirdsDataset,
    "celeba": CelebADataset,
    "chexpert": ChexpertDataset,
    "adult": AdultDataset,
    "civil_comments": CivilCommentsDataset,
}
