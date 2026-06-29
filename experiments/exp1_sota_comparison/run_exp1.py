import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.baselines import BASELINE_CONFIG
from config.bmcl import BMCL_CONFIG
from config.datasets import DATASET_CONFIG

SEEDS = [42, 137, 256, 512, 1024]
NUM_RUNS = len(SEEDS)

dataset_name = "chexpert"
dataset_cfg = DATASET_CONFIG[dataset_name]
model_cfg = {**BASELINE_CONFIG[dataset_name], **BMCL_CONFIG[dataset_name]}

base_results_dir = "./experiments/exp1_sota_comparison/results"
model_list = [
    "ERM",
    "ReWeighting",
    "GroupDRO",
    "JTT",
    "DFR",
    "GroupDRO-EWC",
    "GroupDRO-LwF",
    "ReWeighting-EWC",
    "ReWeighting-LwF",
]


def dict_to_cli_args(arg_dict):
    cli_args = []
    for key, value in arg_dict.items():
        cli_args.append(f"--{key}")
        cli_args.append(str(value))
    return cli_args


def run_experiment(model_name, extra_args_dict, results_dir):

    if model_name == "JTT":
        script = "scripts.trainers.trainer_jtt"
    elif model_name == "DFR":
        script = "scripts.trainers.trainer_dfr"
    elif "EWC" in model_name or "LwF" in model_name:
        script = "scripts.trainers.trainer_two_stage"
    else:
        script = "scripts.trainers.trainer_single_stage"

    for run_id, seed in enumerate(SEEDS, start=1):

        run_output_dir = os.path.join(results_dir, f"run_{run_id}")

        cmd = [
            "python",
            "-m",
            script,
            "--dataset",
            dataset_name,
            "--root_dir",
            dataset_cfg["root_dir"],
            "--metadata_dir",
            dataset_cfg["metadata_dir"],
            "--model",
            model_name,
            "--output_dir",
            str(run_output_dir),
            "--num_epochs",
            str(dataset_cfg["num_epochs"]),
            "--batch_size",
            str(dataset_cfg["batch_size"]),
            "--lr",
            str(dataset_cfg["lr"]),
            "--weight_decay",
            str(dataset_cfg["weight_decay"]),
            "--method",
            dataset_cfg["method"],
            "--steps_per_epoch",
            str(dataset_cfg.get("steps_per_epoch") or 0),
            "--use_pretrained",
            "--use_gpu",
            "--seed",
            str(seed),
        ]

        if dataset_cfg.get("demographic", False):
            cmd.append("--demographic")

        if extra_args_dict:
            cmd += dict_to_cli_args(extra_args_dict)

        print(f"\nRunning: {' '.join(cmd)}")

        subprocess.run(cmd, check=True)


if __name__ == "__main__":

    print("\nPython version :", sys.version.split("|")[0].strip())
    print("PyTorch version:", torch.__version__)
    print("CUDA version   :", torch.version.cuda)
    print("CUDA available :", torch.cuda.is_available())

    for model_name in model_list:
        results_dir = os.path.join(base_results_dir, dataset_name, model_name)
        extra_args = model_cfg[model_name].copy()

        run_experiment(model_name, extra_args, results_dir)
