import os
import time
from collections import defaultdict

import pandas as pd
import torch
from src.learning.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

METHODS = {"worst_acc", "balanced_acc"}


def train(
    model,
    num_epochs,
    num_batches,
    train_loader,
    val_loader,
    save_path,
    log_dir,
    early_stopping=None,
    method="worst_acc",
    device="cuda",
):

    assert method in METHODS, f"Unknown method '{method}'."

    model.to(device)

    if early_stopping is None:
        early_stopping = EarlyStopping(
            save_path,
            patience=10,
            delta=0.0,
            lower_is_better=False,
        )

    writer = SummaryWriter(log_dir)
    history = []
    print(f"Training {model.__class__.__name__} on {device}.")

    total_time = 0.0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()
        run_loss = 0.0

        for i, batch in enumerate(train_loader, 1):
            batch = [item.to(device) for item in batch]
            loss = model.update(batch)
            run_loss += loss

            if i == num_batches:
                break

        run_loss /= num_batches

        group_scores = evaluate_by_group(model, val_loader, device)
        group_values = list(group_scores.values())
        group_str = " ".join([f"{val:.3f}" for val in group_values])

        if method == "worst_acc":
            val_score = min(group_values)
        elif method == "balanced_acc":
            val_score = sum(group_values) / len(group_values)

        epoch_time = (time.time() - start_time) / 60
        total_time += epoch_time

        print(
            f"[Epoch {epoch}] train_loss: {run_loss:.3f}, val_group_acc: {group_str}, "
            f"val_{method}: {val_score:.3f}, epoch_time: {epoch_time:.2f} min, total_time: {total_time:.2f} min"
        )

        writer.add_scalar("loss/train", run_loss, epoch)
        writer.add_scalar("score/validation", val_score, epoch)

        for g_id, val in group_scores.items():
            writer.add_scalar(f"acc/group_{g_id}", val, epoch)

        record = {"epoch": epoch, "loss": run_loss, "score": val_score}

        for g_id, val in group_scores.items():
            record[f"group_{g_id}"] = val

        history.append(record)

        if early_stopping.step(val_score, model.state_dict()):
            print(f"Early stopping at epoch {epoch}")
            break

    writer.close()

    df = pd.DataFrame(history)
    df.to_csv(os.path.join(log_dir, "history.csv"), index=False)

    print(f"Training finished.")
    print(f"Best model saved to {save_path}.")
    print(f"Total training time: {total_time:.2f} min.")


def evaluate_by_group(model, val_loader, device="cuda"):
    correct_by_group = defaultdict(int)
    total_by_group = defaultdict(int)

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            g = batch[2].to(device)

            logits = model.predict(x)
            preds = logits.argmax(dim=1)

            for g_id in torch.unique(g):
                mask = g == g_id
                y_true = y[mask]
                y_pred = preds[mask]

                correct = (y_pred == y_true).sum().item()
                total = mask.sum().item()
                correct_by_group[g_id.item()] += correct
                total_by_group[g_id.item()] += total

    return {g: correct_by_group[g] / total_by_group[g] for g in correct_by_group}


def get_predictions(
    model,
    dataset,
    batch_size,
    num_workers=0,
    device="cuda",
    return_logits=False,
):
    model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    results = {"y_pred": [], "y_true": [], "group": []}

    if return_logits:
        results["logits"] = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            g = batch[2].to(device)

            logits = model.predict(x)
            preds = logits.argmax(dim=1)

            results["y_pred"].append(preds.cpu())
            results["y_true"].append(y.cpu())
            results["group"].append(g.cpu())

            if return_logits:
                results["logits"].append(logits.cpu())

    final_results = {
        "y_pred": torch.cat(results["y_pred"]).numpy(),
        "y_true": torch.cat(results["y_true"]).numpy(),
        "group": torch.cat(results["group"]).numpy(),
    }

    if return_logits:
        final_results["logits"] = torch.cat(results["logits"]).numpy()

    return final_results


def partition_groups(results, metric="acc"):
    labels = results["y_true"]
    preds = results["y_pred"]
    groups = results["group"]

    group_ids = list(set(groups))
    group_metrics = {}

    for g_id in group_ids:
        mask = groups == g_id
        y_true = labels[mask]
        y_pred = preds[mask]

        val = (y_pred == y_true).sum() / mask.sum()
        group_metrics[g_id] = val

    balanced_val = sum(group_metrics.values()) / len(group_metrics)
    best_groups = [g_id for g_id, val in group_metrics.items() if val > balanced_val]
    worst_groups = [g_id for g_id in group_metrics if g_id not in best_groups]

    group_partition = {
        "best_groups": best_groups,
        "worst_groups": worst_groups,
        "group_ids": [g_id for g_id in group_ids],
        "group_metrics": [group_metrics[g_id] for g_id in group_ids],
        "balanced_" + metric: balanced_val,
    }

    return group_partition
