import argparse
import math
import os

import pandas as pd
import torch

from src.dataset.dataloader import create_dataloader
from src.dataset.datasets import get_dataset
from src.learning.algorithms import get_algorithm
from src.learning.early_stopping import EarlyStopping
from src.utils.misc import get_device, set_seed
from src.utils.training import get_predictions, partition_groups, train


def main(args):
    set_seed(args.seed)
    device = get_device(gpu=args.use_gpu)

    os.makedirs(args.output_dir, exist_ok=True)
    dataset = get_dataset(
        args.dataset,
        args.root_dir,
        args.metadata_dir,
        use_pretrained=args.use_pretrained,
    )

    dataloader = create_dataloader(dataset, args.batch_size, num_workers=4)

    hparams = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "use_pretrained": args.use_pretrained,
        "device": device,
        "eta_dro": args.eta_dro,
        "pretrain_ratio_erm": args.pretrain_ratio_erm,
        "lambda_ewc": args.lambda_ewc,
        "tau_lwf": args.tau_lwf,
        "alpha_lwf": args.alpha_lwf,
    }

    # Stage 1: ERM pretraining
    model = get_algorithm(
        "ERM",
        data_type=dataset["train"].DATA_TYPE,
        num_classes=dataset["train"].get_num_classes(),
        num_groups=dataset["train"].get_num_groups(),
        hparams=hparams,
    )

    save_path_stage1 = os.path.join(args.output_dir, "best_model1.pth")
    log_dir1 = os.path.join(args.output_dir, "logs1")
    os.makedirs(log_dir1, exist_ok=True)

    pretrain_epochs = int(args.num_epochs * args.pretrain_ratio_erm)
    print(f"\n=== Pretraining (ERM) for {pretrain_epochs} epochs")

    train(
        model=model,
        num_epochs=pretrain_epochs,
        num_batches=dataset["train"].NUM_BATCHES,
        train_loader=dataloader["train"],
        val_loader=dataloader["val"],
        save_path=save_path_stage1,
        log_dir=log_dir1,
        method=args.method,
        device=device,
    )

    # Reload the best ERM model
    model.load_state_dict(torch.load(save_path_stage1))

    # Compute predictions on train (used for partition)
    print("\nEvaluating trained ERM model on training set...")
    results_val = get_predictions(
        model,
        dataset["val"],
        batch_size=args.batch_size,
        num_workers=4,
        device=device,
        return_logits=True,
    )

    print("\nPerforming initial group partition after ERM...")
    group_partition = partition_groups(results_val)

    with open(os.path.join(args.output_dir, "group_partition_initial.txt"), "w") as f:
        for key, value in group_partition.items():
            f.write(f"{key}: {value}\n")

    print("Initial Best groups:", group_partition["best_groups"])
    print("Initial Worst groups:", group_partition["worst_groups"])

    bm_method = args.model.split("-")[0]
    cl_method = args.model.split("-")[1]

    if cl_method in ["EWC", "LwF"]:
        hparams["weight_decay"] = 0.0  # No weight decay for CL stage

    # LwF collects teacher logits from ERM model
    if cl_method == "LwF":
        results_train_ERM = get_predictions(
            model,
            dataset["train"],
            batch_size=args.batch_size,
            num_workers=4,
            device=device,
            return_logits=True,
        )

    # Stage 2: BM-CL retraining

    # New BM-CL model initialized from ERM weights
    model = get_algorithm(
        args.model,
        data_type=dataset["train"].DATA_TYPE,
        num_classes=dataset["train"].get_num_classes(),
        num_groups=dataset["train"].get_num_groups(),
        hparams=hparams,
    )
    model.load_state_dict(torch.load(save_path_stage1))

    print("\nPreparing Continual-Learning model for Stage 2...")

    best_groups = group_partition["best_groups"]

    if cl_method == "EWC":
        train_dataset_best = dataset["train"].get_group_subset(best_groups)
        model.init_cl_params(train_dataset_best, args.batch_size)

    if cl_method == "LwF":
        model.set_best_groups(best_groups)
        dataset["train"].set_prev_outputs(results_train_ERM["logits"])

    # ReSample balancing
    weights = dataset["train"].get_weights() if bm_method == "ReSample" else None
    dataloader = create_dataloader(dataset, args.batch_size, weights, num_workers=4)

    save_path_stage2 = os.path.join(args.output_dir, "best_model2.pth")
    log_dir2 = os.path.join(args.output_dir, "logs2")
    os.makedirs(log_dir2, exist_ok=True)

    retrain_epochs = args.num_epochs - pretrain_epochs

    print(f"\n=== Stage 2: Retraining {retrain_epochs} epochs")

    # Initialize EMA
    ema_alpha = args.ema_alpha
    ema_acc = {}
    initial_acc = group_partition["group_metrics"]
    for i in range(len(initial_acc)):
        ema_acc[i] = initial_acc[i]

    # Dynamic partitioning loop
    epochs_left = retrain_epochs
    block_size = epochs_left  # No repartitioning, int(args.repartition_every) if needed
    # block_size = int(args.repartition_every)
    block_id = 0

    early_stopping_stage2 = EarlyStopping(
        save_path_stage2,
        patience=10,
        delta=0.0,
        lower_is_better=False,
    )

    while True:
        block_id += 1
        chunk = block_size if epochs_left >= block_size else epochs_left

        print(f"\n--- Block {block_id}: train {chunk} epochs ---")

        # Train block
        train(
            model=model,
            num_epochs=chunk,
            num_batches=dataset["train"].NUM_BATCHES,
            train_loader=dataloader["train"],
            val_loader=dataloader["val"],
            save_path=save_path_stage2,
            log_dir=log_dir2,
            early_stopping=early_stopping_stage2,
            method=args.method,
            device=device,
        )

        epochs_left -= chunk
        if epochs_left <= 0 or early_stopping_stage2.early_stop:
            break

        # Load updated model
        model.load_state_dict(torch.load(save_path_stage2))

        # Recompute raw accuracies from validation
        print(f"Re-evaluating for dynamic partitioning (Block {block_id})...")
        results_val_dynamic = get_predictions(
            model,
            dataset["val"],
            batch_size=args.batch_size,
            num_workers=4,
            device=device,
            return_logits=True,
        )

        raw_partition = partition_groups(results_val_dynamic)
        current_acc = raw_partition["group_metrics"]

        # EMA update
        print("Updating EMA accuracies...")
        for g in range(len(current_acc)):
            ema_acc[g] = ema_alpha * current_acc[g] + (1 - ema_alpha) * ema_acc[g]

        # EMA-based partitioning
        tau = sum(ema_acc.values()) / len(ema_acc)
        new_best = [g for g, v in ema_acc.items() if v > tau]
        new_worst = [g for g, v in ema_acc.items() if v <= tau]

        print(f"[Block {block_id}] EMA best groups:  {new_best}")
        print(f"[Block {block_id}] EMA worst groups: {new_worst}")

        # Save partition snapshot
        with open(os.path.join(args.output_dir, f"group_partition_block{block_id}.txt"), "w") as f:
            f.write(f"EMA-based partition (tau={tau})\n")
            for g, v in ema_acc.items():
                f.write(f"group {g}: ema_acc = {v:.4f}\n")

        # Update CL state if best groups changed
        best_groups = new_best

        if cl_method == "EWC":
            print("Updating EWC fisher for new best groups...")
            train_dataset_best = dataset["train"].get_group_subset(best_groups)
            model.init_cl_params(train_dataset_best, args.batch_size)

        if cl_method == "LwF":
            print("Updating LwF teacher logits...")
            updated_teacher = get_predictions(
                model,
                dataset["train"],
                batch_size=args.batch_size,
                num_workers=4,
                device=device,
                return_logits=True,
            )
            dataset["train"].set_prev_outputs(updated_teacher["logits"])
            model.set_best_groups(best_groups)

        # Recreate dataloader
        weights = dataset["train"].get_weights() if bm_method == "ReSample" else None
        dataloader = create_dataloader(dataset, args.batch_size, weights, num_workers=4)

    # Final evaluation
    model.load_state_dict(torch.load(save_path_stage2))

    print("\nEvaluating final model on test set...")
    results = get_predictions(
        model,
        dataset["test"],
        batch_size=args.batch_size,
        num_workers=4,
        device=device,
    )
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, "preds_test.csv"), index=False)

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train two-stage model.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--metadata_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--method", type=str, default="worst_acc")
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--eta_dro", type=float, default=0.01)
    parser.add_argument("--pretrain_ratio_erm", type=float, default=0.1)
    parser.add_argument("--lambda_ewc", type=float, default=1)
    parser.add_argument("--tau_lwf", type=float, default=2)
    parser.add_argument("--alpha_lwf", type=float, default=1)
    parser.add_argument("--repartition_every", type=int, default=5)
    parser.add_argument("--ema_alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
