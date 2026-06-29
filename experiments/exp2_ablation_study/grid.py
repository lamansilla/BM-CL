pretrain_ratio_erm = [0.05, 0.1, 0.2, 0.5]
lambda_cl = [0, 0.1, 0.5, 1.0, 5.0]
tau_lwf = 1

GRID_CONFIG = {
    "GroupDRO-EWC": {
        "pretrain_ratio_erm": pretrain_ratio_erm,
        "lambda_cl": lambda_cl,
        "eta_dro": [0.01],
    },
    "GroupDRO-LwF": {
        "pretrain_ratio_erm": pretrain_ratio_erm,
        "lambda_cl": lambda_cl,
        "tau_lwf": [tau_lwf],
        "eta_dro": [0.01],
    },
    "ReWeighting-EWC": {
        "pretrain_ratio_erm": pretrain_ratio_erm,
        "lambda_cl": lambda_cl,
    },
    "ReWeighting-LwF": {
        "pretrain_ratio_erm": pretrain_ratio_erm,
        "lambda_cl": lambda_cl,
        "tau_lwf": [tau_lwf],
    },
}
