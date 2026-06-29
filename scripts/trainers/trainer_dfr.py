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

    dataset = get_dataset(args.dataset, args.root_dir, args.metadata_dir, args.use_pretrained,
                          demographic=args.demographic)
    dataloader = create_dataloader(dataset, args.batch_size, num_workers=4)

    num_batches = (
        args.steps_per_epoch
        if args.steps_per_epoch > 0
        else len(dataset["train"]) // args.batch_size
    )

    hparams = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "use_pretrained": args.use_pretrained,
        "device": device,
        "pretrain_ratio_erm": args.pretrain_ratio_erm,
    }

    save_path_stage2 = os.path.join(args.output_dir, "model_stage2.pth")

    if not args.eval_only:
        save_path_stage1 = os.path.join(args.output_dir, "model_stage1.pth")
        tb_dir1 = os.path.join(args.output_dir, "tb_stage1")
        os.makedirs(tb_dir1, exist_ok=True)

        pretrain_epochs = int(args.num_epochs * args.pretrain_ratio_erm)
        print(f"\nPretraining ERM for {pretrain_epochs} epochs...")

        # Stage 1: full ERM training
        erm_model = get_algorithm(
            "ERM",
            data_type=dataset["train"].DATA_TYPE,
            num_classes=dataset["train"].get_num_classes(),
            num_groups=dataset["train"].get_num_groups(),
            hparams=hparams,
        )

        train(
            model=erm_model,
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

        # Stage 2: load ERM weights, freeze featurizer, retrain head with group-balanced loss
        retrain_epochs = args.num_epochs - pretrain_epochs
        print(f"\nDFR — freezing featurizer, retraining head for {retrain_epochs} epochs...")

        model = get_algorithm(
            "DFR",
            data_type=dataset["train"].DATA_TYPE,
            num_classes=dataset["train"].get_num_classes(),
            num_groups=dataset["train"].get_num_groups(),
            hparams=hparams,
        )
        model.load_state_dict(torch.load(save_path_stage1))
        model.freeze_featurizer()

        tb_dir2 = os.path.join(args.output_dir, "tb_stage2")
        os.makedirs(tb_dir2, exist_ok=True)

        train(
            model=model,
            num_epochs=retrain_epochs,
            num_batches=num_batches,
            train_loader=dataloader["train"],
            val_loader=dataloader["val"],
            save_path=save_path_stage2,
            log_dir=tb_dir2,
            history_path=os.path.join(args.output_dir, "history_stage2.csv"),
            method=args.method,
            device=device,
        )

    model = get_algorithm(
        "DFR",
        data_type=dataset["train"].DATA_TYPE,
        num_classes=dataset["train"].get_num_classes(),
        num_groups=dataset["train"].get_num_groups(),
        hparams=hparams,
    )
    model.load_state_dict(torch.load(save_path_stage2))

    for split in ["val", "test"]:
        print(f"\nEvaluating final model on {split} set.")
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
    parser = argparse.ArgumentParser(description="Train DFR.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--metadata_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="DFR")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--method", type=str, default="worst_acc")
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--demographic", action="store_true")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--pretrain_ratio_erm", type=float, default=0.7)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true")

    args = parser.parse_args()
    main(args)
