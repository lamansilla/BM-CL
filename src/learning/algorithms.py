import torch
import torch.nn as nn
from torch import autograd

from .continual_learning import EWC, LwF
from .models import get_head_optimizer, get_network, get_optimizer


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
        idx_g, idx_samples = [], []
        for group_id in g.unique():
            idx_g.append(group_id)
            idx_samples.append(g == group_id)
        return zip(idx_g, idx_samples)

    def _add_cl_loss(self, bm_loss, cl_loss):
        return bm_loss + self.hparams["lambda_cl"] * cl_loss


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
        loss = self.loss_fn(logits * scale, y).mean()
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    def _compute_loss(self, x, y, g):
        logits = self.predict(x)
        num_groups = len(g.unique())
        nll = 0.0

        for _, idx_samples in self.return_groups(g):
            nll += self.loss_fn(logits[idx_samples], y[idx_samples]).mean()
        nll /= num_groups

        self.update_count += 1

        if self.update_count.item() <= self.hparams["irm_warmup_iters"]:
            return nll

        penalty = 0.0
        for _, idx_samples in self.return_groups(g):
            penalty += self._irm_penalty(logits[idx_samples], y[idx_samples])
        penalty /= num_groups

        loss = nll + self.hparams["irm_lambda"] * penalty
        if self.hparams["irm_lambda"] > 1.0:
            loss /= self.hparams["irm_lambda"]
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


class ReWeighting(ERM):
    """ERM with group-inverse-frequency loss reweighting."""

    def _compute_loss(self, x, y, g):
        logits = self.predict(x)
        losses = self.loss_fn(logits, y)
        group_losses = [losses[idx_samples].mean() for _, idx_samples in self.return_groups(g)]
        return torch.stack(group_losses).mean()


class DFR(ERM):
    """Deep Feature Reweighting (DFR)."""

    def freeze_featurizer(self):
        for p in self.network.featurizer.parameters():
            p.requires_grad = False
        trainable = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = get_head_optimizer(self.data_type, trainable, self.hparams)

    def _compute_loss(self, x, y, g):
        logits = self.predict(x)
        losses = self.loss_fn(logits, y)
        group_losses = [losses[idx_samples].mean() for _, idx_samples in self.return_groups(g)]
        return torch.stack(group_losses).mean()


class JTT(ERM):
    """Just Train Twice (JTT)."""


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
        return self._add_cl_loss(loss_dro, loss_ewc)


class GroupDRO_LwF(GroupDRO):
    """GroupDRO with Learning without Forgetting (LwF)."""

    def __init__(self, data_type, num_classes, num_groups, hparams):
        super().__init__(data_type, num_classes, num_groups, hparams)
        self.lwf = LwF(self.network, hparams)

    def set_best_groups(self, best_groups_ids):
        self.lwf.set_best_groups(best_groups_ids)

    def _compute_loss(self, x, y, g, prev_logits):
        logits = self.predict(x)
        losses = self.loss_fn(logits, y)

        for idx_g, idx_samples in self.return_groups(g):
            self.q[idx_g] *= (self.hparams["eta_dro"] * losses[idx_samples].mean()).exp().item()
        self.q /= self.q.sum()

        loss_dro = sum(
            self.q[idx_g] * losses[idx_samples].mean()
            for idx_g, idx_samples in self.return_groups(g)
        )
        loss_lwf = self.lwf.penalty(logits, g, prev_logits)
        return self._add_cl_loss(loss_dro, loss_lwf)

    def update(self, minibatch):
        all_x, all_y, all_g, all_prev_logits = minibatch
        loss = self._compute_loss(all_x, all_y, all_g, all_prev_logits)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class ReWeighting_EWC(ReWeighting):
    """ReWeighting with Elastic Weight Consolidation (EWC)."""

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
        loss_rw = super()._compute_loss(x, y, g)
        loss_ewc = self.ewc.penalty()
        return self._add_cl_loss(loss_rw, loss_ewc)


class ReWeighting_LwF(ReWeighting):
    """ReWeighting with Learning without Forgetting (LwF)."""

    def __init__(self, data_type, num_classes, num_groups, hparams):
        super().__init__(data_type, num_classes, num_groups, hparams)
        self.lwf = LwF(self.network, hparams)

    def set_best_groups(self, best_groups_ids):
        self.lwf.set_best_groups(best_groups_ids)

    def _compute_loss(self, x, y, g, prev_logits):
        logits = self.predict(x)
        losses = self.loss_fn(logits, y)
        group_losses = [losses[idx_samples].mean() for _, idx_samples in self.return_groups(g)]
        loss_rw = torch.stack(group_losses).mean()
        loss_lwf = self.lwf.penalty(logits, g, prev_logits)
        return self._add_cl_loss(loss_rw, loss_lwf)

    def update(self, minibatch):
        all_x, all_y, all_g, all_prev_logits = minibatch
        loss = self._compute_loss(all_x, all_y, all_g, all_prev_logits)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


ALGORITHMS = {
    "ERM": ERM,
    "DFR": DFR,
    "IRM": IRM,
    "GroupDRO": GroupDRO,
    "ReWeighting": ReWeighting,
    "JTT": JTT,
    "GroupDRO-EWC": GroupDRO_EWC,
    "GroupDRO-LwF": GroupDRO_LwF,
    "ReWeighting-EWC": ReWeighting_EWC,
    "ReWeighting-LwF": ReWeighting_LwF,
}
