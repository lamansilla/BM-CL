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

    # Only used in case of ReSample
    weights = dataset["train"].get_weights() if args.model == "ReSample" else None

    dataloader = create_dataloader(dataset, args.batch_size, weights, num_workers=4)

    hparams = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "use_pretrained": args.use_pretrained,
        "device": device,
        "eta_dro": args.eta_dro,
        "irm_lambda": args.irm_lambda,
        "irm_warmup_iters": args.irm_warmup_iters,
    }

    model = get_algorithm(
        args.model,
        data_type=dataset["train"].DATA_TYPE,
        num_classes=dataset["train"].get_num_classes(),
        num_groups=dataset["train"].get_num_groups(),
        hparams=hparams,
    )

    save_path = os.path.join(args.output_dir, "best_model.pth")
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    train(
        model=model,
        num_epochs=args.num_epochs,
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
    parser = argparse.ArgumentParser(description="Train a single-stage model.")
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
    parser.add_argument("--irm_lambda", type=float, default=1e4)
    parser.add_argument("--irm_warmup_iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
