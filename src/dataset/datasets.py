import os

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Subset


def get_dataset(dataset_name, root_dir, metadata_dir, use_pretrained=True):
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not found.")
    datasets = {}
    for split in __BaseDataset.SPLITS.keys():
        datasets[split] = DATASETS[dataset_name](
            root_dir,
            metadata_dir,
            split,
            use_pretrained,
        )
    return datasets


class __BaseDataset(torch.utils.data.Dataset):

    SPLITS = {"train": 0, "val": 1, "test": 2}

    def __init__(self, root_dir, metadata, split, transform):
        self.root_dir = root_dir
        df = pd.read_csv(metadata)
        df = df[df["split"] == self.SPLITS[split]].reset_index(drop=True)
        self.transform = transform

        self.image_paths = df["filepath"].tolist()
        self.labels = torch.tensor(df["y"].tolist(), dtype=torch.long)
        self.groups = torch.tensor(df["g"].tolist(), dtype=torch.long)

        self.num_classes = len(torch.unique(self.labels))
        self.num_groups = len(torch.unique(self.groups))

        self.prev_outputs = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        path = os.path.join(self.root_dir, self.image_paths[i])
        label = self.labels[i]
        group = self.groups[i]

        image = self._load_image(path)

        if self.transform is not None:
            image = self.transform(image)

        if self.prev_outputs is not None:
            prev_output = torch.tensor(self.prev_outputs[i], dtype=torch.float32)
            return image, label, group, prev_output

        return image, label, group

    def _load_image(self, path):
        return Image.open(path).convert("RGB")

    def get_num_classes(self):
        return self.num_classes

    def get_num_groups(self):
        return self.num_groups

    def get_group_counts(self):
        counts = [0] * self.num_groups
        for g in self.groups.tolist():
            counts[g] += 1
        return counts

    def get_weights(self):
        group_counts = self.get_group_counts()
        weights = [1.0 / group_counts[g] for g in self.groups.tolist()]
        return weights

    def set_prev_outputs(self, prev_outputs):
        if len(prev_outputs) != len(self.labels):
            raise ValueError("Number of previous outputs must match number of labels.")
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


class WaterbirdsDataset(__BaseDataset):

    DATA_TYPE = "images"
    NUM_BATCHES = 150

    def __init__(self, root_dir, metadata_dir, split, use_pretrained=True):
        root_dir = os.path.join(root_dir, "waterbirds")
        metadata = f"{metadata_dir}/waterbirds.csv"

        transform_list = [
            transforms.Resize(
                (
                    int(224 * (256 / 224)),
                    int(224 * (256 / 224)),
                )
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
        if use_pretrained:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        transform = transforms.Compose(transform_list)

        super().__init__(root_dir, metadata, split, transform)


class CelebADataset(__BaseDataset):

    DATA_TYPE = "images"
    NUM_BATCHES = 1280

    def __init__(self, root_dir, metadata_dir, split, use_pretrained=True):
        root_dir = os.path.join(root_dir, "celeba")
        metadata = f"{metadata_dir}/celeba.csv"

        transform_list = [
            transforms.CenterCrop(178),
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
        if use_pretrained:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        transform = transforms.Compose(transform_list)

        super().__init__(root_dir, metadata, split, transform)


class ChexpertDataset(__BaseDataset):

    DATA_TYPE = "images"
    NUM_BATCHES = 1050

    def __init__(self, root_dir, metadata_dir, split, use_pretrained=True):
        root_dir = os.path.join(root_dir, "CheXpert-v1.0-small")
        metadata = f"{metadata_dir}/chexpert.csv"

        transform_list = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
        if use_pretrained:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        transform = transforms.Compose(transform_list)

        super().__init__(root_dir, metadata, split, transform)


DATASETS = {
    "waterbirds": WaterbirdsDataset,
    "celeba": CelebADataset,
    "chexpert": ChexpertDataset,
}
