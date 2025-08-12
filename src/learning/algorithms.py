import torch
import torch.nn as nn
from transformers import get_scheduler

from src.models.networks import get_network

from .continual_learning import EWC, LwF
from .optimizers import get_optimizer


def get_algorithm(
    algorithm_name,
    data_type,
    num_classes,
    num_groups,
    hparams,
):
    if algorithm_name not in ALGORITHMS:
        raise NotImplementedError(f"Algorithm '{algorithm_name}' not found.")
    return ALGORITHMS[algorithm_name](
        data_type,
        num_classes,
        num_groups,
        hparams,
    )


class Algorithm(torch.nn.Module):

    def __init__(self, data_type, num_classes, num_groups, hparams):
        super().__init__()
        self.data_type = data_type
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.hparams = hparams

    def _init_model(self):
        raise NotImplementedError

    def _compute_loss(self, x, y, g):
        raise NotImplementedError

    def update(self, minibatch):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def return_groups(self, g):
        """Get indices and samples for each group."""
        idx_g, idx_samples = [], []
        for group_id in g.unique():
            idx_g.append(group_id)
            idx_samples.append(g == group_id)
        return zip(idx_g, idx_samples)


class ERM(Algorithm):
    """Empirical Risk Minimization (ERM)."""

    def __init__(self, data_type, num_classes, num_groups, hparams):
        super().__init__(data_type, num_classes, num_groups, hparams)

        if self.data_type == "images":
            self.network = get_network("resnet", num_classes, hparams["use_pretrained"])
        else:
            raise NotImplementedError(f"Data type '{self.data_type}' not found.")

        self._init_model()

    def _init_model(self):

        if self.data_type in {"images"}:
            self.optimizer = get_optimizer(
                "sgd",
                self.network,
                self.hparams["lr"],
                self.hparams["weight_decay"],
            )

        self.device = self.hparams["device"]
        self.network.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def _compute_loss(self, x, y, g):
        logits = self.predict(x)
        return self.loss_fn(logits, y).mean()

    def update(self, minibatch):
        all_x, all_y, all_g = minibatch
        loss = self._compute_loss(all_x, all_y, all_g)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, x):
        return self.network(x)


class GroupDRO(ERM):
    """Group Distributionally Robust Optimization (GroupDRO)."""

    def __init__(self, data_type, num_classes, num_groups, hparams):
        super().__init__(data_type, num_classes, num_groups, hparams)
        self.q = torch.ones(self.num_groups, device=self.device)

    def _compute_loss(self, x, y, g):
        logits = self.predict(x)
        losses = self.loss_fn(logits, y)

        for idx_g, idx_samples in self.return_groups(g):
            group_loss = losses[idx_samples].mean()
            self.q[idx_g] *= (self.hparams["eta_dro"] * group_loss).exp().item()

        self.q /= self.q.sum()

        loss = 0
        for idx_g, idx_samples in self.return_groups(g):
            loss += self.q[idx_g] * losses[idx_samples].mean()

        return loss


class ReSample(ERM):
    """ERM with resampling."""

    pass


class JTT(ERM):
    """Just Train Twice (JTT)."""

    pass


class GroupDRO_EWC(GroupDRO):
    """GroupDRO with Elastic Weight Consolidation (EWC)."""

    def __init__(self, data_type, num_classes, num_groups, hparams):
        super().__init__(data_type, num_classes, num_groups, hparams)
        self.ewc = EWC(self.network, hparams)

    def init_cl_params(self, dataset, batch_size, num_samples=None):
        self.ewc.init_ewc_params(
            dataset,
            batch_size,
            num_samples,
        )

    def _compute_loss(self, x, y, g):
        loss_dro = super()._compute_loss(x, y, g)
        loss_ewc = self.ewc.penalty()
        return loss_dro + loss_ewc


class GroupDRO_LwF(GroupDRO):
    """GroupDRO with Learning without Forgetting (LwF)."""

    def __init__(self, data_type, num_classes, num_groups, hparams):
        super().__init__(data_type, num_classes, num_groups, hparams)
        self.lwf = LwF(self.network, hparams)

    def set_best_groups(self, best_groups_ids):
        self.lwf.set_best_groups(best_groups_ids)

    def _compute_loss(self, x, y, g, prev_logits):
        loss_dro = super()._compute_loss(x, y, g)
        logits = self.predict(x)
        loss_lwf = self.lwf.penalty(logits, g, prev_logits)
        return loss_dro + loss_lwf

    def update(self, minibatch):
        all_x, all_y, all_g, all_prev_logits = minibatch
        loss = self._compute_loss(all_x, all_y, all_g, all_prev_logits)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.lr_scheduler:
            self.lr_scheduler.step()

        if self.clip_grad:
            self.network.zero_grad()

        return loss.item()


class ReSample_EWC(ReSample):
    """ReSample with Elastic Weight Consolidation (EWC)."""

    def __init__(self, data_type, num_classes, num_groups, hparams):
        super().__init__(data_type, num_classes, num_groups, hparams)
        self.ewc = EWC(self.network, hparams)

    def init_cl_params(self, dataset, batch_size, num_samples=None):
        self.ewc.init_ewc_params(
            dataset,
            batch_size,
            num_samples,
        )

    def _compute_loss(self, x, y, g):
        loss_resample = super()._compute_loss(x, y, g)
        loss_ewc = self.ewc.penalty()
        return loss_resample + loss_ewc


class ReSample_LwF(ReSample):
    """ReSample with Learning without Forgetting (LwF)."""

    def __init__(self, data_type, num_classes, num_groups, hparams):
        super().__init__(data_type, num_classes, num_groups, hparams)
        self.lwf = LwF(self.network, hparams)

    def set_best_groups(self, best_groups_ids):
        self.lwf.set_best_groups(best_groups_ids)

    def _compute_loss(self, x, y, g, prev_logits):
        loss_resample = super()._compute_loss(x, y, g)
        logits = self.predict(x)
        loss_lwf = self.lwf.penalty(logits, g, prev_logits)
        return loss_resample + loss_lwf

    def update(self, minibatch):
        all_x, all_y, all_g, all_prev_logits = minibatch
        loss = self._compute_loss(all_x, all_y, all_g, all_prev_logits)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.lr_scheduler:
            self.lr_scheduler.step()

        if self.clip_grad:
            self.network.zero_grad()

        return loss.item()


ALGORITHMS = {
    "ERM": ERM,
    "GroupDRO": GroupDRO,
    "ReSample": ReSample,
    "JTT": JTT,
    "GroupDRO-EWC": GroupDRO_EWC,
    "GroupDRO-LwF": GroupDRO_LwF,
    "ReSample-EWC": ReSample_EWC,
    "ReSample-LwF": ReSample_LwF,
}
