import torch
from torch.utils.data import DataLoader


def create_dataloader(dataset, batch_size, weights=None, num_workers=0):
    dataloader = {}
    for split, data in dataset.items():
        if split == "train":
            dataloader[split] = InfiniteDataLoader(
                data,
                batch_size=batch_size,
                weights=weights,
                num_workers=num_workers,
                pin_memory=True,
            )
        elif split in ["val", "test"]:
            dataloader[split] = DataLoader(
                data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
        else:
            raise ValueError(f"Unknown split: {split}")
    return dataloader


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:

    def __init__(self, dataset, weights, batch_size, num_workers, pin_memory):
        super().__init__()

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, replacement=True, num_samples=len(weights)
            )
        else:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=batch_size, drop_last=True
        )

        self._infinite_iterator = iter(
            torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                pin_memory=pin_memory,
                batch_sampler=_InfiniteSampler(batch_sampler),
            )
        )

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError
