import torch
import torch.nn as nn
from torchvision import models


def get_network(network_name, num_classes, use_pretrained):
    if network_name not in NETWORKS:
        raise ValueError(f"Network '{network_name}' not found.")

    if network_name == "resnet":
        return ResNet(num_classes, use_pretrained)


class ResNet(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
        model = models.resnet50(weights=weights)
        if use_pretrained:
            print("Using pretrained weights for ResNet50.")
        else:
            print("Not using pretrained weights for ResNet50.")

        self.featurizer = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Linear(model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.featurizer(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


NETWORKS = {
    "resnet": ResNet,
}
