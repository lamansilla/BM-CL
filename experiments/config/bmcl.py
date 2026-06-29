BMCL_CONFIG = {
    "waterbirds": {
        "GroupDRO-EWC": {
            "pretrain_ratio_erm": 0.05,
            "eta_dro": 0.01,
            "lambda_cl": 0.5,
        },
        "GroupDRO-LwF": {
            "pretrain_ratio_erm": 0.02,
            "eta_dro": 0.01,
            "tau_lwf": 1,
            "lambda_cl": 1.0,
        },
        "ReWeighting-EWC": {
            "pretrain_ratio_erm": 0.05,
            "lambda_cl": 0.5,
        },
        "ReWeighting-LwF": {
            "pretrain_ratio_erm": 0.02,
            "tau_lwf": 1,
            "lambda_cl": 1.0,
        },
    },
    "celeba": {
        "GroupDRO-EWC": {
            "pretrain_ratio_erm": 0.1,
            "eta_dro": 0.001,
            "lambda_cl": 0.5,
        },
        "GroupDRO-LwF": {
            "pretrain_ratio_erm": 0.1,
            "eta_dro": 0.001,
            "tau_lwf": 1,
            "lambda_cl": 0.5,
        },
        "ReWeighting-EWC": {
            "pretrain_ratio_erm": 0.1,
            "lambda_cl": 0.5,
        },
        "ReWeighting-LwF": {
            "pretrain_ratio_erm": 0.1,
            "tau_lwf": 1,
            "lambda_cl": 0.5,
        },
    },
    "chexpert": {
        "GroupDRO-EWC": {
            "pretrain_ratio_erm": 0.2,
            "eta_dro": 0.001,
            "lambda_cl": 1.0,
        },
        "GroupDRO-LwF": {
            "pretrain_ratio_erm": 0.2,
            "eta_dro": 0.001,
            "tau_lwf": 1,
            "lambda_cl": 1.0,
        },
        "ReWeighting-EWC": {
            "pretrain_ratio_erm": 0.2,
            "lambda_cl": 1.0,
        },
        "ReWeighting-LwF": {
            "pretrain_ratio_erm": 0.2,
            "tau_lwf": 1,
            "lambda_cl": 0.5,
        },
    },
    "adult": {
        "GroupDRO-EWC": {
            "pretrain_ratio_erm": 0.2,
            "eta_dro": 0.01,
            "lambda_cl": 1.0,
        },
        "GroupDRO-LwF": {
            "pretrain_ratio_erm": 0.2,
            "eta_dro": 0.01,
            "tau_lwf": 1,
            "lambda_cl": 1.0,
        },
        "ReWeighting-EWC": {
            "pretrain_ratio_erm": 0.2,
            "lambda_cl": 1.0,
        },
        "ReWeighting-LwF": {
            "pretrain_ratio_erm": 0.2,
            "tau_lwf": 1,
            "lambda_cl": 0.01,
        },
    },
    "civil_comments": {
        "GroupDRO-EWC": {
            "pretrain_ratio_erm": 0.5,
            "eta_dro": 0.001,
            "lambda_cl": 1.0,
        },
        "GroupDRO-LwF": {
            "pretrain_ratio_erm": 0.4,
            "eta_dro": 0.001,
            "tau_lwf": 1,
            "lambda_cl": 0.1,
        },
        "ReWeighting-EWC": {
            "pretrain_ratio_erm": 0.5,
            "lambda_cl": 1.0,
        },
        "ReWeighting-LwF": {
            "pretrain_ratio_erm": 0.4,
            "tau_lwf": 1,
            "lambda_cl": 0.1,
        },
    },
}
