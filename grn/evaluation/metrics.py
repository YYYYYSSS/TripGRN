import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    confusion_matrix,
)


def calculate_metrics(y_true, y_pred, y_prob):
    """计算所有评估指标（含每类准确率、OvR/OvO AUC）。

    说明：多分类下 ``recall_score(..., average="weighted")`` 与 ``accuracy_score``
    恒等（各类召回按支持度加权后等于整体正确率），因此 ``recall`` 会与 ``accuracy``
    相同；若需与准确率可区分的召回，请看 ``recall_macro`` / ``precision_macro`` / ``f1_macro``。
    """
    metrics = {}
    n_classes = y_prob.shape[1]

    # 1. 多分类整体AUC (OvO)
    try:
        metrics["auc_ovo"] = roc_auc_score(y_true, y_prob, multi_class="ovo")
    except ValueError:
        metrics["auc_ovo"] = float("nan")

    # 2. 每类OvR AUC（one-vs-rest）
    metrics["auc_ovr"] = []
    metrics["auprc"] = []
    for i in range(n_classes):
        try:
            ovr_auc = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
        except ValueError:
            ovr_auc = float("nan")
        metrics["auc_ovr"].append(ovr_auc)

        try:
            auprc = average_precision_score((y_true == i).astype(int), y_prob[:, i])
        except ValueError:
            auprc = float("nan")
        metrics["auprc"].append(auprc)

    # 3. 宏平均AUC
    metrics["macro_auc"] = np.nanmean(metrics["auc_ovr"])

    # 4. 每类OvO AUC（类别i vs 其他每个类别j的AUC平均）
    metrics["auc_ovo_per_class"] = []
    for i in range(n_classes):
        class_pair_aucs = []
        for j in range(n_classes):
            if i == j:
                continue
            pair_mask = (y_true == i) | (y_true == j)
            if pair_mask.sum() > 0:
                binary_labels = (y_true[pair_mask] == i).astype(int)
                prob_i = y_prob[pair_mask][:, i]
                if len(np.unique(binary_labels)) > 1:
                    try:
                        pair_auc = roc_auc_score(binary_labels, prob_i)
                        class_pair_aucs.append(pair_auc)
                    except ValueError:
                        pass
        metrics["auc_ovo_per_class"].append(np.mean(class_pair_aucs) if class_pair_aucs else float("nan"))

    # 5. 基础指标
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    except ValueError:
        metrics["mcc"] = float("nan")

    # 固定 n_classes×n_classes，避免某折测试集缺类时 sklearn 返回 2×2 等导致下游 IndexError
    labels_all = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels_all)
    metrics["confusion_matrix"] = cm

    # 5b. 每类 specificity（OvR）
    metrics["specificity_ovr"] = []
    for cls in range(n_classes):
        tp = float(cm[cls, cls])
        fn = float(cm[cls, :].sum() - tp)
        fp = float(cm[:, cls].sum() - tp)
        tn = float(cm.sum() - tp - fn - fp)
        denom = tn + fp
        metrics["specificity_ovr"].append(float(tn / denom) if denom > 0 else float("nan"))

    # 6. 每类准确率（原始类别顺序为 [无调控, 抑制, 激活]）
    class_names = ["无调控_accuracy", "抑制_accuracy", "激活_accuracy"]
    for cls in range(n_classes):
        mask = y_true == cls
        if np.sum(mask) == 0:
            acc = float("nan")
        else:
            acc = accuracy_score(y_true[mask], y_pred[mask])
        metrics[class_names[cls]] = acc

    # 7. 重排到展示顺序 [NoReg, Act, Rep]（与 per-class 图 / 报告 / auc_ovr 一致）
    _ord = (0, 2, 1)
    metrics["auc_ovr"] = [metrics["auc_ovr"][i] for i in _ord]
    metrics["auprc"] = [metrics["auprc"][i] for i in _ord]
    metrics["auc_ovo_per_class"] = [
        metrics["auc_ovo_per_class"][0],
        metrics["auc_ovo_per_class"][2],
        metrics["auc_ovo_per_class"][1],
    ]
    metrics["specificity_ovr"] = [
        metrics["specificity_ovr"][0],
        metrics["specificity_ovr"][2],
        metrics["specificity_ovr"][1],
    ]

    # 重排并重新命名类别准确率
    acc_noreg = metrics.pop("无调控_accuracy")
    acc_repress = metrics.pop("抑制_accuracy")
    acc_activate = metrics.pop("激活_accuracy")

    metrics["无调控_accuracy"] = acc_noreg
    metrics["抑制_accuracy"] = acc_repress
    metrics["激活_accuracy"] = acc_activate

    # auc_ovr / auprc / auc_ovo_per_class 已统一为展示顺序：无调控、激活、抑制
    metrics["per_class_order"] = "no_regulation_activation_repression"

    return metrics

