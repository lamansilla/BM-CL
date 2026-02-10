import torch
import torch.nn as nn
from torch import autograd

from .continual_learning import EWC, LwF
from .models import get_network, get_optimizer


def get_algorithm(algorithm_name, data_type, num_classes, num_groups, hparams):
    if algorithm_name not in ALGORITHMS:
        raise NotImplementedError(f"Algorithm '{algorithm_name}' not found.")

    return ALGORITHMS[algorithm_name](data_type, num_classes, num_groups, hparams)


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

        self.device = self.hparams["device"]
        self.network = get_network(self.data_type, num_classes, hparams["use_pretrained"])
        self.network.to(self.device)
        self.optimizer = get_optimizer(self.data_type, self.network, self.hparams)
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


class IRM(ERM):
    """Invariant Risk Minimization (IRM)."""

    def __init__(self, data_type, num_classes, num_groups, hparams):
        super(IRM, self).__init__(data_type, num_classes, num_groups, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    def _irm_penalty(self, logits, y):
        scale = torch.tensor(1.0).to(self.device).requires_grad_()

        loss_1 = self.loss_fn(logits[::2] * scale, y[::2]).mean()
        loss_2 = self.loss_fn(logits[1::2] * scale, y[1::2]).mean()

        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]

        return torch.sum(grad_1 * grad_2)

    def _compute_loss(self, x, y, g):
        if self.update_count.item() >= self.hparams["irm_warmup_iters"]:
            penalty_weight = self.hparams["irm_lambda"]
        else:
            penalty_weight = 1.0

        logits = self.predict(x)
        nll, penalty = 0.0, 0.0
        num_groups = len(g.unique())

        for idx_g, idx_samples in self.return_groups(g):
            group_logits = logits[idx_samples]
            group_y = y[idx_samples]

            nll += self.loss_fn(group_logits, group_y).mean()
            penalty += self._irm_penalty(group_logits, group_y)

        nll /= num_groups
        penalty /= num_groups

        loss = nll + penalty_weight * penalty
        self.update_count += 1

        return loss


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
    """ERM with resampling by group weight.
    
    During training, samples are resampled such that each group has
    equal probability of being sampled, regardless of group size.
    """

    pass


class JTT(ERM):
    """Just Train Twice (JTT).
    
    First stage: train ERM to identify hard examples.
    Second stage: retrain with upweighted hard examples.
    """


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

        return loss.item()


ALGORITHMS = {
    "ERM": ERM,
    "IRM": IRM,
    "GroupDRO": GroupDRO,
    "ReSample": ReSample,
    "JTT": JTT,
    "GroupDRO-EWC": GroupDRO_EWC,
    "GroupDRO-LwF": GroupDRO_LwF,
    "ReSample-EWC": ReSample_EWC,
    "ReSample-LwF": ReSample_LwF,
}
