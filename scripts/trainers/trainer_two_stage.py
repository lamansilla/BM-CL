import argparse
import os

import pandas as pd
import torch

from src.dataset.dataloader import create_dataloader
from src.dataset.datasets import get_dataset
from src.learning.algorithms import get_algorithm
from src.utils.misc import get_device, set_seed
from src.utils.training import get_group_partition, get_predictions, train


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

    model = get_algorithm(
        "ERM",
        data_type=dataset["train"].DATA_TYPE,
        num_classes=dataset["train"].get_num_classes(),
        num_groups=dataset["train"].get_num_groups(),
        hparams=hparams,
    )

    save_path = os.path.join(args.output_dir, "best_model1.pth")
    log_dir = os.path.join(args.output_dir, "logs1")
    os.makedirs(log_dir, exist_ok=True)

    pretrain_epochs = int(args.num_epochs * args.pretrain_ratio_erm)
    print(f"Pretraining for {pretrain_epochs} epochs.")

    # Pretrain the model
    train(
        model=model,
        num_epochs=pretrain_epochs,
        num_batches=dataset["train"].NUM_BATCHES,
        train_loader=dataloader["train"],
        val_loader=dataloader["val"],
        save_path=save_path,
        log_dir=log_dir,
        method=args.method,
        metric=args.metric,
        device=device,
    )

    model.load_state_dict(torch.load(save_path))

    # Get predictions for the validation set
    print("Evaluating trained model on training set.")
    results_val = get_predictions(
        model,
        dataset["train"],
        batch_size=args.batch_size,
        num_workers=4,
        device=device,
        return_logits=True,
    )

    # Perform group partitioning
    print("Performing group partitioning.")
    group_partition = get_group_partition(results_val, metric=args.metric)
    with open(os.path.join(args.output_dir, "group_partition.txt"), "w") as f:
        for key, value in group_partition.items():
            f.write(f"{key}: {value}\n")

    print("Best groups:", group_partition["best_groups"])
    print("Worst groups:", group_partition["worst_groups"])

    bm_method = args.model.split("-")[0]
    cl_method = args.model.split("-")[1]

    if cl_method == "EWC":
        hparams["weight_decay"] = 1e-5

    # Get predictions for the training set if using LwF
    if cl_method == "LwF":
        results_train = get_predictions(
            model,
            dataset["train"],
            batch_size=args.batch_size,
            num_workers=4,
            device=device,
            return_logits=True,
        )

    # Prepare model for retraining
    model = get_algorithm(
        args.model,
        data_type=dataset["train"].DATA_TYPE,
        num_classes=dataset["train"].get_num_classes(),
        num_groups=dataset["train"].get_num_groups(),
        hparams=hparams,
    )

    model.load_state_dict(torch.load(save_path))

    print("Preparing model for retraining.")
    best_groups = group_partition["best_groups"]

    if cl_method == "EWC":
        train_dataset_best = dataset["train"].get_group_subset(best_groups)
        model.init_cl_params(train_dataset_best, args.batch_size)
    elif cl_method == "LwF":
        model.set_best_groups(best_groups)

    weights = dataset["train"].get_weights() if bm_method == "ReSample" else None
    if cl_method == "LwF":
        dataset["train"].set_prev_outputs(results_train["logits"])
    dataloader = create_dataloader(dataset, args.batch_size, weights, num_workers=4)

    save_path = os.path.join(args.output_dir, "best_model2.pth")
    log_dir = os.path.join(args.output_dir, "logs2")
    os.makedirs(log_dir, exist_ok=True)

    retrain_epochs = args.num_epochs - pretrain_epochs
    print(f"Retraining for {retrain_epochs} epochs.")

    # Retrain the model
    train(
        model=model,
        num_epochs=retrain_epochs,
        num_batches=dataset["train"].NUM_BATCHES,
        train_loader=dataloader["train"],
        val_loader=dataloader["val"],
        save_path=save_path,
        log_dir=log_dir,
        method=args.method,
        metric=args.metric,
        device=device,
    )

    model.load_state_dict(torch.load(save_path))

    # Get predictions for the test set
    print("Evaluating trained model on test set.")
    results = get_predictions(
        model,
        dataset["test"],
        batch_size=args.batch_size,
        num_workers=4,
        device=device,
    )
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, "preds_test.csv"), index=False)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a two-stage model.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--metadata_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--method", type=str, default="max_worst")
    parser.add_argument("--metric", type=str, default="acc")
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--eta_dro", type=float, default=0.01)
    parser.add_argument("--pretrain_ratio_erm", type=float, default=0.5)
    parser.add_argument("--lambda_ewc", type=float, default=1)
    parser.add_argument("--tau_lwf", type=float, default=2)
    parser.add_argument("--alpha_lwf", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
