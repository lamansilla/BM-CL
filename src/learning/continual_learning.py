import math

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

        fisher_batch_size = min(batch_size, 32)
        dataloader = DataLoader(
            dataset,
            batch_size=fisher_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self.network.eval()
        seen = 0

        for batch in dataloader:
            x = batch[0].to(self.device)

            self.network.zero_grad()
            logits = self.network(x)
            log_probs = F.log_softmax(logits, dim=1)

            sampled_y = torch.multinomial(log_probs.exp().detach(), 1).squeeze(1)
            loss = F.nll_loss(log_probs, sampled_y)
            loss.backward()

            for n, p in self.network.named_parameters():
                if p.grad is not None:
                    self.importance[n] += p.grad.detach().pow(2) * len(x)

            seen += len(x)

            if num_samples and seen >= num_samples:
                break

        for n in self.importance.keys():
            self.importance[n] /= max(seen, 1)

        self.network.train()

    def penalty(self):
        numerator = 0
        denominator = 0
        for n, p in self.network.named_parameters():
            numerator += (self.importance[n] * (p - self.params[n]).pow(2)).sum()
            denominator += self.importance[n].sum()
        return 0.5 * numerator / (denominator + 1e-8)


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
        T = self.hparams["tau_lwf"]
        num_classes = logits.size(1)
        log_p = torch.log_softmax(logits / T, dim=1)
        q = torch.softmax(prev_logits / T, dim=1)
        kl = F.kl_div(log_p, q, reduction="batchmean") * T**2
        return kl / (T**2 * math.log(num_classes) + 1e-8)

    def penalty(self, logits, g, prev_logits):
        g = torch.as_tensor(g, device=self.device)
        mask_best = torch.isin(g, torch.tensor(self.best_group_ids, device=self.device))

        if mask_best.any():
            logits_best = logits[mask_best]
            prev_logits_best = prev_logits[mask_best]
            loss_best = self._distillation_loss(logits_best, prev_logits_best)
        else:
            loss_best = torch.tensor(0.0, device=self.device)

        return loss_best
