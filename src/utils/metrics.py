import numpy as np


def compute_metrics(
    y_true,
    y_pred,
    groups=None,
    max_group=None,
    min_group=None,
    metric="acc",
):
    if groups is None:
        groups = np.zeros_like(y_true)

    if metric == "acc":
        overall = average_acc(y_true, y_pred)
        group_scores = group_acc(y_true, y_pred, groups)
    elif metric == "tpr":
        overall = average_tpr(y_true, y_pred)
        group_scores = group_tpr(y_true, y_pred, groups)
    else:
        raise ValueError("Unsupported metric. Use 'acc' or 'tpr'.")

    balanced = sum(group_scores.values()) / len(group_scores)

    if min_group is None:
        min_group = min(group_scores, key=group_scores.get)
    if max_group is None:
        max_group = max(group_scores, key=group_scores.get)

    min_score = group_scores[min_group]
    max_score = group_scores[max_group]
    disparity = max_score - min_score

    return {
        f"global_{metric}": overall,
        f"group_{metric}": group_scores,
        f"balanced_{metric}": balanced,
        f"min_{metric}": min_score,
        f"max_{metric}": max_score,
        "disparity": disparity,
        "min_group": min_group,
        "max_group": max_group,
    }


def average_acc(y_true, y_pred):
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))

    num = tp + tn
    denom = num + fp + fn

    if denom > 0:
        return num / denom
    else:
        return 0.0


def group_acc(y_true, y_pred, groups):
    group_accs = {}
    for g in set(groups):
        mask = groups == g
        group_accs[g] = average_acc(y_true[mask], y_pred[mask])
    return group_accs


def average_tpr(y_true, y_pred):
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def group_tpr(y_true, y_pred, groups):
    group_tprs = {}
    for g in set(groups):
        mask = groups == g
        group_tprs[g] = average_tpr(y_true[mask], y_pred[mask])
    return group_tprs
