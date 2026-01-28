DATASET_CONFIG = {
    "waterbirds": {
        "root_dir": "../datasets",
        "metadata_dir": "./metadata",
        "num_epochs": 30,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "method": "worst_acc",
        "metric": "acc",
    },
    "celeba": {
        "root_dir": "../datasets",
        "metadata_dir": "./metadata",
        "num_epochs": 50,
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "method": "worst_acc",
        "metric": "acc",
    },
    "chexpert": {
        "root_dir": "../datasets",
        "metadata_dir": "./metadata",
        "num_epochs": 50,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "method": "worst_acc",
        "metric": "acc",
    },
    "adult": {
        "root_dir": "",
        "metadata_dir": "./metadata",
        "num_epochs": 50,
        "batch_size": 128,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "method": "worst_acc",
        "metric": "acc",
    },
    "civil_comments": {
        "root_dir": "../datasets",
        "metadata_dir": "./metadata",
        "num_epochs": 50,
        "batch_size": 128,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "method": "worst_acc",
        "metric": "acc",
    },
}


MODEL_CONFIG = {
    "waterbirds": {
        "ERM": {},
        "IRM": {
            "irm_lambda": 0.1,
            "irm_warmup_iters": 500,
        },
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
        "IRM": {
            "irm_lambda": 0.01,
            "irm_warmup_iters": 5000,
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
        "IRM": {
            "irm_lambda": 0.01,
            "irm_warmup_iters": 5000,
        },
        "JTT": {
            "pretrain_ratio_erm": 0.1,
            "lambda_jtt": 5,
        },
        "GroupDRO-EWC": {
            "pretrain_ratio_erm": 0.2,
            "eta_dro": 0.001,
            "lambda_ewc": 0.1,
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
    "adult": {
        "ERM": {},
        "IRM": {
            "irm_lambda": 5,
            "irm_warmup_iters": 1000,
        },
        "ReSample": {},
        "GroupDRO": {
            "eta_dro": 0.001,
        },
        "JTT": {
            "lr": 1e-4,
            "pretrain_ratio_erm": 0.2,
            "lambda_jtt": 5,
        },
        "GroupDRO-EWC": {
            "pretrain_ratio_erm": 0.2,
            "eta_dro": 0.001,
            "lambda_ewc": 1,
        },
        "GroupDRO-LwF": {
            "pretrain_ratio_erm": 0.2,
            "eta_dro": 0.001,
            "tau_lwf": 2,
            "alpha_lwf": 2,
        },
        "ReSample-EWC": {
            "pretrain_ratio_erm": 0.2,
            "lambda_ewc": 1.7,
        },
        "ReSample-LwF": {
            "pretrain_ratio_erm": 0.2,
            "tau_lwf": 2,
            "alpha_lwf": 1.5,
        },
    },
    "civil_comments": {
        "ERM": {},
        "IRM": {
            "irm_lambda": 1,
            "irm_warmup_iters": 2500,
        },
        "ReSample": {},
        "GroupDRO": {
            "eta_dro": 0.001,
        },
        "JTT": {
            "lr": 1e-4,
            "pretrain_ratio_erm": 0.2,
            "lambda_jtt": 10,
        },
        "GroupDRO-EWC": {
            "pretrain_ratio_erm": 0.2,
            "eta_dro": 0.0001,
            "lambda_ewc": 1,
        },
        "GroupDRO-LwF": {
            "pretrain_ratio_erm": 0.2,
            "eta_dro": 0.0001,
            "tau_lwf": 2,
            "alpha_lwf": 1,
        },
        "ReSample-EWC": {
            "pretrain_ratio_erm": 0.2,
            "lambda_ewc": 1,
        },
        "ReSample-LwF": {
            "pretrain_ratio_erm": 0.2,
            "tau_lwf": 2,
            "alpha_lwf": 1,
        },
    },
}
