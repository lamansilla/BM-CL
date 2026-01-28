import torch


class EarlyStopping:
    def __init__(self, save_path, patience, delta=0.0, lower_is_better=False):
        self.save_path = save_path
        self.patience = patience
        self.delta = delta
        self.lower_is_better = lower_is_better
        self.best_score = None
        self.epochs_without_improvement = 0
        self.early_stop = False

    def step(self, metric, state_dict):
        score = -metric if self.lower_is_better else metric

        if self.best_score is None:
            self.best_score = score
            torch.save(state_dict, self.save_path)
            self.epochs_without_improvement = 0
        elif score > self.best_score - self.delta:
            self.best_score = score
            torch.save(state_dict, self.save_path)
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.early_stop = True
            return True

        return False
