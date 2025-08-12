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


MODEL_CONFIG = {
    "waterbirds": {
        "ERM": {},
        "ReSample": {},
        "GroupDRO": {
            "eta_dro": 0.001,
        },
        "JTT": {
            "lr": 1e-4,
            "pretrain_ratio_erm": 0.2,
            "lambda_jtt": 50,
        },
        "GroupDRO-EWC": {
            "pretrain_ratio_erm": 0.2,
            "eta_dro": 0.001,
            "lambda_ewc": 0.1,
        },
        "GroupDRO-LwF": {
            "pretrain_ratio_erm": 0.2,
            "eta_dro": 0.001,
            "tau_lwf": 2,
            "alpha_lwf": 0.1,
        },
        "ReSample-EWC": {
            "pretrain_ratio_erm": 0.1,
            "lambda_ewc": 1,
        },
        "ReSample-LwF": {
            "pretrain_ratio_erm": 0.3,
            "tau_lwf": 2,
            "alpha_lwf": 0.1,
        },
    },
    "celeba": {
        "ERM": {},
        "ReSample": {},
        "GroupDRO": {
            "eta_dro": 0.001,
        },
        "JTT": {
            "lr": 1e-5,
            "pretrain_ratio_erm": 0.2,
            "lambda_jtt": 20,
        },
        "GroupDRO-EWC": {
            "pretrain_ratio_erm": 0.3,
            "eta_dro": 0.001,
            "lambda_ewc": 10,
        },
        "GroupDRO-LwF": {
            "pretrain_ratio_erm": 0.2,
            "eta_dro": 0.001,
            "tau_lwf": 2,
            "alpha_lwf": 1,
        },
        "ReSample-EWC": {
            "pretrain_ratio_erm": 0.3,
            "lambda_ewc": 10,
        },
        "ReSample-LwF": {
            "pretrain_ratio_erm": 0.3,
            "tau_lwf": 2,
            "alpha_lwf": 1,
        },
    },
    "chexpert": {
        "ERM": {},
        "ReSample": {},
        "GroupDRO": {
            "eta_dro": 0.001,
        },
        "JTT": {
            "pretrain_ratio_erm": 0.1,
            "lambda_jtt": 5,
        },
        "GroupDRO-EWC": {
            "pretrain_ratio_erm": 0.2,
            "eta_dro": 0.001,
            "lambda_ewc": 1,
        },
        "GroupDRO-LwF": {
            "pretrain_ratio_erm": 0.3,
            "eta_dro": 0.001,
            "tau_lwf": 2,
            "alpha_lwf": 10,
        },
        "ReSample-EWC": {
            "pretrain_ratio_erm": 0.2,
            "lambda_ewc": 0.1,
        },
        "ReSample-LwF": {
            "pretrain_ratio_erm": 0.3,
            "tau_lwf": 2,
            "alpha_lwf": 10,
        },
    },
}
