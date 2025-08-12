DATASET_CONFIG = {
    "waterbirds": {
        "root_dir": "../datasets",
        "metadata_dir": "./metadata",
        "num_epochs": 30,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "method": "max_worst",
        "metric": "acc",
    },
    "celeba": {
        "root_dir": "../datasets",
        "metadata_dir": "./metadata",
        "num_epochs": 50,
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "method": "max_worst",
        "metric": "acc",
    },
    "chexpert": {
        "root_dir": "../datasets",
        "metadata_dir": "./metadata",
        "num_epochs": 50,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "method": "max_worst",
        "metric": "acc",
    },
}

# Grid configuration for each model
MODEL_CONFIG = {
    "GroupDRO-EWC": {
        "pretrain_ratio_erm": [0.1, 0.2, 0.3],
        "lambda_ewc": [0.1, 1],
        "eta_dro": [0.001],
    },
    "GroupDRO-LwF": {
        "pretrain_ratio_erm": [0.1, 0.2, 0.3],
        "tau_lwf": [2],
        "alpha_lwf": [0.1, 1, 10],
        "eta_dro": [0.001],
    },
    "ReSample-EWC": {
        "pretrain_ratio_erm": [0.1, 0.2, 0.3],
        "lambda_ewc": [0.1, 1],
    },
    "ReSample-LwF": {
        "pretrain_ratio_erm": [0.1, 0.2, 0.3],
        "tau_lwf": [2],
        "alpha_lwf": [0.1, 1, 10],
    },
    "JTT": {
        # "lr": [1e-4],  # waterbirds
        # "lr": [1e-5],  # celeba
        # "lr": [1e-3],  # chexpert
        "pretrain_ratio_erm": [0.1, 0.2, 0.3],
        "lambda_jtt": [5, 20, 50],
    },
}
