import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu=True):
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU.")
    elif not gpu:
        device = torch.device("cpu")
        print("Using CPU.")
    elif not gpu and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Using CPU.")
    return device


def create_groups(attributes, labels):
    num_classes = len(set(labels))
    groups = []
    for i in range(len(labels)):
        group = attributes[i] * num_classes + labels[i]
        groups.append(group)
    return np.array(groups, dtype=int)
