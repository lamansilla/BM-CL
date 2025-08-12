import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class EWC:
    """Elastic Weight Consolidation (EWC)."""

    def __init__(self, network, hparams):
        self.network = network
        self.hparams = hparams
        self.device = hparams["device"]

    def init_ewc_params(self, dataset, batch_size, num_samples=None):

        self.params = {}
        self.importance = {}

        for n, p in self.network.named_parameters():
            self.params[n] = p.clone().detach().to(self.device)
            self.importance[n] = torch.zeros_like(p).to(self.device)

        self._compute_importance(dataset, batch_size, num_samples)

    def _compute_importance(self, dataset, batch_size, num_samples=None):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self.network.eval()
        total_batches = 0

        for batch in dataloader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)

            logits = self.network(x)
            loss = F.cross_entropy(logits, y)

            self.network.zero_grad()
            loss.backward()

            for n, p in self.network.named_parameters():
                if p.grad is not None:
                    self.importance[n] += p.grad.detach().pow(2)

            total_batches += 1

            if num_samples and total_batches * batch_size >= num_samples:
                break

        for n in self.importance.keys():
            self.importance[n] /= total_batches

        self.network.train()

    def penalty(self):
        loss = 0
        for n, p in self.network.named_parameters():
            loss += (self.importance[n] * (p - self.params[n]).pow(2)).sum()
        return 0.5 * self.hparams["lambda_ewc"] * loss * 1e3


class LwF:
    """Learning without Forgetting (LwF)."""

    def __init__(self, network, hparams):
        self.network = network
        self.hparams = hparams
        self.best_group_ids = None
        self.device = hparams["device"]

    def set_best_groups(self, best_group_ids):
        self.best_group_ids = best_group_ids

    def _distillation_loss(self, logits, prev_logits):
        log_p = torch.log_softmax(logits / self.hparams["tau_lwf"], dim=1)
        q = torch.softmax(prev_logits / self.hparams["tau_lwf"], dim=1)
        return F.kl_div(log_p, q, reduction="batchmean")

    def penalty(self, logits, g, prev_logits):
        g = torch.as_tensor(g, device=self.device)
        mask_best = torch.isin(g, torch.tensor(self.best_group_ids, device=self.device))

        if mask_best.any():
            logits_best = logits[mask_best]
            prev_logits_best = prev_logits[mask_best]
            loss_best = self._distillation_loss(logits_best, prev_logits_best)
        else:
            loss_best = torch.tensor(0.0, device=self.device)

        return self.hparams["alpha_lwf"] * loss_best
