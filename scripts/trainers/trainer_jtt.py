import argparse
import os

import pandas as pd
import torch

from src.dataset.dataloader import create_dataloader
from src.dataset.datasets import get_dataset
from src.learning.algorithms import get_algorithm
from src.utils.misc import get_device, set_seed
from src.utils.training import get_predictions, train


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
        "pretrain_ratio_erm": args.pretrain_ratio_erm,
        "lambda_jtt": args.lambda_jtt,
    }

    model = get_algorithm(
        args.model,
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

    # Get predictions for the training set
    print("Evaluating trained model on training set.")
    results = get_predictions(
        model,
        dataset["train"],
        batch_size=args.batch_size,
        num_workers=4,
        device=device,
    )

    # Upweight samples with wrong predictions and recreate dataloader
    weights = torch.ones(len(dataset["train"]))
    wrong_preds_idx = results["y_pred"] != results["y_true"]
    weights[wrong_preds_idx] = hparams["lambda_jtt"]

    dataloader = create_dataloader(dataset, args.batch_size, weights, num_workers=4)
    print(f"Upweighting {wrong_preds_idx.sum()} samples.")

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
    parser.add_argument("--pretrain_ratio_erm", type=float, default=0.5)
    parser.add_argument("--lambda_jtt", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
