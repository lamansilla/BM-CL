import argparse
import os
import subprocess
import sys
from itertools import product

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.datasets import DATASET_CONFIG
from grid import GRID_CONFIG

SEEDS = [42, 137, 256]
NUM_RUNS = len(SEEDS)

dataset_name = "waterbirds"
dataset_cfg = DATASET_CONFIG[dataset_name]
model_cfg = GRID_CONFIG

base_results_dir = "./experiments/exp2_ablation_study/results"

model_list = [
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


def run_experiment(model_name, extra_args_dict, results_dir, eval_only=False):
    if model_name == "JTT":
        script = "scripts.trainers.trainer_jtt"
    elif "EWC" in model_name or "LwF" in model_name:
        script = "scripts.trainers.trainer_two_stage"
    else:
        script = "scripts.trainers.trainer_single_stage"

    for run_id, seed in enumerate(SEEDS, start=1):

        run_output_dir = os.path.join(results_dir, f"run_{run_id}")

        # Skip completed runs
        if not eval_only and os.path.exists(os.path.join(run_output_dir, "eval_test.csv")):
            print(f"\nSkipping (already done): {run_output_dir}")
            continue

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

        if eval_only:
            cmd.append("--eval_only")

        print(f"\nRunning: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Run failed (exit code {e.returncode}): {run_output_dir}")
            print("Continuing with next run...")
            continue


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_only", action="store_true")
    script_args = parser.parse_args()

    print("\nPython version :", sys.version.split("|")[0].strip())
    print("PyTorch version:", torch.__version__)
    print("CUDA version   :", torch.version.cuda)
    print("CUDA available :", torch.cuda.is_available())

    for model_name in model_list:
        hparam_values = model_cfg[model_name]
        hparam_names = list(hparam_values.keys())
        hparam_combinations = list(product(*[hparam_values[h] for h in hparam_names]))

        for hparam_values_tuple in hparam_combinations:
            extra_args = {}
            config_parts = []
            for name, value in zip(hparam_names, hparam_values_tuple):
                extra_args[name] = value
                config_parts.append(f"{name}_{value}")

            config_name = "_".join(config_parts)
            results_dir = os.path.join(base_results_dir, dataset_name, model_name, config_name)
            run_experiment(model_name, extra_args, results_dir, eval_only=script_args.eval_only)
