import argparse
import json
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
        demographic=args.demographic,
    )

    hparams = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "use_pretrained": args.use_pretrained,
        "device": device,
        "eta_dro": args.eta_dro,
        "pretrain_ratio_erm": args.pretrain_ratio_erm,
        "lambda_cl": args.lambda_cl,
        "tau_lwf": args.tau_lwf,
    }

    bm_method = args.model.split("-")[0]
    cl_method = args.model.split("-")[1]

    save_path_stage1 = os.path.join(args.output_dir, "model_stage1.pth")
    save_path_stage2 = os.path.join(args.output_dir, "model_stage2.pth")

    if not args.eval_only:
        dataloader = create_dataloader(dataset, args.batch_size, num_workers=4)

        num_batches = (
            args.steps_per_epoch
            if args.steps_per_epoch > 0
            else len(dataset["train"]) // args.batch_size
        )

        # Stage 1: ERM pretraining
        model = get_algorithm(
            "ERM",
            data_type=dataset["train"].DATA_TYPE,
            num_classes=dataset["train"].get_num_classes(),
            num_groups=dataset["train"].get_num_groups(),
            hparams=hparams,
        )

        tb_dir1 = os.path.join(args.output_dir, "tb_stage1")
        os.makedirs(tb_dir1, exist_ok=True)

        pretrain_epochs = int(args.num_epochs * args.pretrain_ratio_erm)
        print(f"\n=== Pretraining (ERM) for {pretrain_epochs} epochs")

        train(
            model=model,
            num_epochs=pretrain_epochs,
            num_batches=num_batches,
            train_loader=dataloader["train"],
            val_loader=dataloader["val"],
            save_path=save_path_stage1,
            log_dir=tb_dir1,
            history_path=os.path.join(args.output_dir, "history_stage1.csv"),
            method=args.method,
            device=device,
        )

        # Reload the best ERM model
        model.load_state_dict(torch.load(save_path_stage1))

        print("\nEvaluating trained ERM model on val set...")
        results_val = get_predictions(
            model,
            dataset["val"],
            batch_size=args.batch_size,
            num_workers=4,
            device=device,
            return_logits=True,
        )

        print("\nPerforming initial group partition after ERM...")
        partition_metric = "tpr" if "tpr" in args.method else "acc"
        group_partition = partition_groups(results_val, metric=partition_metric)

        partition_json = {
            "best_groups": [int(g) for g in group_partition["best_groups"]],
            "worst_groups": [int(g) for g in group_partition["worst_groups"]],
            "group_ids": [int(g) for g in group_partition["group_ids"]],
            "group_metrics": [float(v) for v in group_partition["group_metrics"]],
            "balanced_score": float(group_partition[f"balanced_{partition_metric}"]),
        }
        with open(os.path.join(args.output_dir, "group_partition.json"), "w") as f:
            json.dump(partition_json, f, indent=2)

        print("Initial Best groups:", group_partition["best_groups"])
        print("Initial Worst groups:", group_partition["worst_groups"])

        if cl_method == "LwF":
            results_train_ERM = get_predictions(
                model,
                dataset["train"],
                batch_size=args.batch_size,
                num_workers=4,
                device=device,
                return_logits=True,
            )

        if cl_method == "EWC":
            hparams["weight_decay"] = 0.0

        # Stage 2: BM-CL retraining
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

        dataloader = create_dataloader(dataset, args.batch_size, num_workers=4)

        tb_dir2 = os.path.join(args.output_dir, "tb_stage2")
        os.makedirs(tb_dir2, exist_ok=True)

        retrain_epochs = args.num_epochs - pretrain_epochs
        print(f"\n=== Stage 2: Retraining {retrain_epochs} epochs")

        early_stopping_stage2 = EarlyStopping(
            save_path_stage2,
            patience=15,
            delta=0.0,
            lower_is_better=False,
        )

        train(
            model=model,
            num_epochs=retrain_epochs,
            num_batches=num_batches,
            train_loader=dataloader["train"],
            val_loader=dataloader["val"],
            save_path=save_path_stage2,
            log_dir=tb_dir2,
            history_path=os.path.join(args.output_dir, "history_stage2.csv"),
            early_stopping=early_stopping_stage2,
            method=args.method,
            device=device,
        )

    # Final evaluation
    model = get_algorithm(
        args.model,
        data_type=dataset["train"].DATA_TYPE,
        num_classes=dataset["train"].get_num_classes(),
        num_groups=dataset["train"].get_num_groups(),
        hparams=hparams,
    )
    model.load_state_dict(torch.load(save_path_stage2))

    for split in ["val", "test"]:
        print(f"\nEvaluating final model on {split} set...")
        results = get_predictions(
            model,
            dataset[split],
            batch_size=args.batch_size,
            num_workers=4,
            device=device,
        )
        pd.DataFrame(results).to_csv(
            os.path.join(args.output_dir, f"eval_{split}.csv"), index=False
        )

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
    parser.add_argument("--demographic", action="store_true")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--eta_dro", type=float, default=0.01)
    parser.add_argument("--pretrain_ratio_erm", type=float, default=0.1)
    parser.add_argument("--lambda_cl", type=float, default=1)
    parser.add_argument("--tau_lwf", type=float, default=2)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()
    main(args)
