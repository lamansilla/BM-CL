import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

DATA_TYPES = {"images", "tabular", "text"}


def get_network(data_type, num_classes, use_pretrained):
    if data_type not in DATA_TYPES:
        raise ValueError(f"Data type '{data_type}' not supported.")

    if data_type == "images":
        return ResNet(num_classes, use_pretrained)
    elif data_type == "tabular":
        return MLP(8, num_classes)
    elif data_type == "text":
        return MLP(768, num_classes)


def get_optimizer(data_type, network, hparams):
    if data_type not in DATA_TYPES:
        raise ValueError(f"Data type '{data_type}' not supported.")

    if data_type == "images":
        return sgd_optimizer(network, hparams["lr"], hparams["weight_decay"])
    elif data_type == "tabular":
        return sgd_optimizer(network, hparams["lr"], hparams["weight_decay"])
    elif data_type == "text":
        return adam_optimizer(network, hparams["lr"], hparams["weight_decay"])


class ResNet(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super().__init__()

        model = models.resnet50(weights="IMAGENET1K_V1" if use_pretrained else None)

        self.featurizer = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Linear(model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.featurizer(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()

        prev_dim = input_dim
        hidden_dims = [256] * 3
        layers = []

        for hid_dim in hidden_dims:
            layer = [nn.Linear(prev_dim, hid_dim), nn.ReLU()]
            layers.extend(layer)
            prev_dim = hid_dim

        self.featurizer = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        x = self.featurizer(x)
        x = self.classifier(x)
        return x


def sgd_optimizer(network, lr, weight_decay):
    return optim.SGD(
        network.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9,
    )


def adam_optimizer(network, lr, weight_decay):
    return optim.Adam(
        network.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
