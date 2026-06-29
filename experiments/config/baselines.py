BASELINE_CONFIG = {
    "waterbirds": {
        "ERM": {},
        "DFR": {"pretrain_ratio_erm": 0.1},
        "ReWeighting": {},
        "GroupDRO": {
            "eta_dro": 0.01,
        },
        "JTT": {
            "lr": 1e-4,
            "pretrain_ratio_erm": 0.02,
            "lambda_jtt": 100,
        },
    },
    "celeba": {
        "ERM": {},
        "DFR": {"pretrain_ratio_erm": 0.1},
        "ReWeighting": {},
        "GroupDRO": {
            "eta_dro": 0.01,
        },
        "JTT": {
            "lr": 1e-5,
            "pretrain_ratio_erm": 0.1,
            "lambda_jtt": 50,
        },
    },
    "chexpert": {
        "ERM": {},
        "DFR": {"pretrain_ratio_erm": 0.2},
        "ReWeighting": {},
        "GroupDRO": {
            "eta_dro": 0.001,
        },
        "JTT": {
            "lr": 1e-4,
            "pretrain_ratio_erm": 0.1,
            "lambda_jtt": 50,
        },
    },
    "adult": {
        "ERM": {},
        "DFR": {"pretrain_ratio_erm": 0.2},
        "ReWeighting": {},
        "GroupDRO": {
            "eta_dro": 0.01,
        },
        "JTT": {
            "lr": 1e-4,
            "pretrain_ratio_erm": 0.05,
            "lambda_jtt": 5,
        },
    },
    "civil_comments": {
        "ERM": {},
        "DFR": {"pretrain_ratio_erm": 0.2},
        "ReWeighting": {},
        "GroupDRO": {
            "eta_dro": 0.001,
        },
        "JTT": {
            "lr": 1e-5,
            "pretrain_ratio_erm": 0.05,
            "lambda_jtt": 10,
        },
    },
}
