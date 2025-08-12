import torch.optim as optim
from transformers import AdamW


def get_optimizer(optimizer_name, network, lr, weight_decay):
    if optimizer_name not in OPTIMIZERS:
        raise ValueError(f"Optimizer '{optimizer_name}' not found.")
    return OPTIMIZERS[optimizer_name](network, lr, weight_decay)


def sgd_optimizer(network, lr, weight_decay):
    return optim.SGD(
        network.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9,
    )


OPTIMIZERS = {
    "sgd": sgd_optimizer,
}
