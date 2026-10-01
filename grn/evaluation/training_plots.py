"""
训练/评估可视化：写入 run_dir/plots/，供 benchmark 结果章节使用。
使用非交互后端 Agg，适合服务器与 bat 调用。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

CLASS_NAMES_ZH: Tuple[str, ...] = ("No regulation", "Repression", "Activation")

_plot_font_configured = False


def _use_agg() -> None:
    import matplotlib

    matplotlib.use("Agg")


def _english_only_plots_requested() -> bool:
    v = os.environ.get("GRN_PLOTS_ENGLISH_ONLY", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _pyplot():
    """Agg backend + pyplot; optional English-only fonts via GRN_PLOTS_ENGLISH_ONLY=1."""
    _use_agg()
    import matplotlib.pyplot as plt

    global _plot_font_configured
    if not _plot_font_configured:
        if _english_only_plots_requested():
            plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
        else:
            plt.rcParams["font.sans-serif"] = [
                "Microsoft YaHei",
                "SimHei",
                "DengXian",
                "Arial Unicode MS",
                "DejaVu Sans",
            ]
        plt.rcParams["axes.unicode_minus"] = False
        _plot_font_configured = True
    return plt


def _savefig_png_pdf(fig, path_png: str, dpi: int = 150) -> str:
    """Write matplotlib figure as PNG and sibling PDF (step03/step05 convention)."""
    fig.savefig(path_png, dpi=dpi)
    base, ext = os.path.splitext(path_png)
    if ext.lower() != ".png":
        pdf_path = path_png + ".pdf"
    else:
        pdf_path = base + ".pdf"
    fig.savefig(pdf_path, format="pdf")
    return path_png


def save_learning_curve(
    plot_dir: str,
    epochs: Sequence[int],
    train_loss: Sequence[float],
    val_acc: Sequence[float],
    val_f1: Optional[Sequence[float]] = None,
    fname: str = "learning_curve.png",
) -> str:
    """左轴：train loss；右轴：val accuracy（及可选 val weighted F1）。"""
    plt = _pyplot()

    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, fname)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="tab:blue")
    ax1.plot(epochs, train_loss, color="tab:blue", marker="o", markersize=3, label="Train loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Val accuracy / F1", color="tab:orange")
    ax2.plot(epochs, val_acc, color="tab:orange", marker="s", markersize=3, label="Val acc")
    if val_f1 is not None and len(val_f1) == len(epochs):
        ax2.plot(epochs, val_f1, color="tab:green", marker="^", markersize=3, label="Val weighted F1")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0.0, 1.05)

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="center right", fontsize=8)
    fig.suptitle("Learning curve")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_confusion_matrix_fig(
    plot_dir: str,
    cm: np.ndarray,
    title: str,
    fname: str,
    class_names: Sequence[str] = CLASS_NAMES_ZH,
) -> str:
    plt = _pyplot()

    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, fname)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )
    ax.set_title(title)
    fig.tight_layout()
    _savefig_png_pdf(fig, path, dpi=150)
    plt.close(fig)
    return path


def save_accuracy_bar_val_test(
    plot_dir: str,
    val_acc: float,
    test_acc: float,
    fname: str = "accuracy_val_vs_test.png",
    title: str = "Accuracy: Validation vs Test",
) -> str:
    plt = _pyplot()

    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, fname)
    fig, ax = plt.subplots(figsize=(4, 4))
    x = np.arange(2)
    ax.bar(x, [val_acc, test_acc], color=["steelblue", "darkorange"], width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Val", "Test"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    for i, v in enumerate([val_acc, test_acc]):
        ax.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=10)
    fig.tight_layout()
    _savefig_png_pdf(fig, path, dpi=150)
    plt.close(fig)
    return path


def per_class_ovr_accuracy_precision_recall_from_cm(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从混淆矩阵计算每类 one-vs-rest 准确率、precision、recall（三者一般互不相同）。

    对类别 k：TP=cm[k,k]，FN=行和-TP，FP=列和-TP，TN=N-TP-FN-FP；
    OvR 准确率_k = (TP+TN)/N；precision_k = TP/(TP+FP)；recall_k = TP/(TP+FN)。
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"confusion_matrix must be square 2D, got shape={cm.shape}")
    n = float(cm.sum())
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = n - tp - fn - fp
    if n <= 0:
        z = np.zeros_like(tp)
        return z, z, z
    ovr_acc = (tp + tn) / n
    precision = np.divide(tp, tp + fp, out=np.full_like(tp, np.nan), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan), where=(tp + fn) > 0)
    return ovr_acc, precision, recall


def per_class_rpf_from_cm(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从混淆矩阵（行=真实类，列=预测类）计算每类的 recall、precision、F1。"""
    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"confusion_matrix must be square 2D, got shape={cm.shape}")
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    denom_r = tp + fn
    denom_p = tp + fp
    recall = np.divide(tp, denom_r, out=np.full_like(tp, np.nan), where=denom_r > 0)
    precision = np.divide(tp, denom_p, out=np.full_like(tp, np.nan), where=denom_p > 0)
    r = np.asarray(recall, dtype=np.float64)
    p = np.asarray(precision, dtype=np.float64)
    denom_f1 = r + p
    mask = np.logical_and(np.logical_and(np.isfinite(r), np.isfinite(p)), denom_f1 > 0.0)
    f1 = np.divide(2.0 * r * p, denom_f1, out=np.full_like(r, np.nan, dtype=np.float64), where=mask)
    return recall, precision, f1


def _safe_float_metric(m: Mapping[str, Any], key: str) -> float:
    v = m.get(key)
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def metrics_dict_with_macro_fallback(m: Mapping[str, Any]) -> Dict[str, Any]:
    """旧版 metrics JSON 无 macro 键时，用 confusion_matrix 回填 recall/precision/f1 的 macro 值（供重画图）。"""
    out = dict(m)
    if np.isfinite(_safe_float_metric(out, "f1_macro")):
        return out
    cm = out.get("confusion_matrix")
    if cm is None:
        return out
    recall, prec, f1 = per_class_rpf_from_cm(np.asarray(cm))
    out["recall_macro"] = float(np.nanmean(recall))
    out["precision_macro"] = float(np.nanmean(prec))
    out["f1_macro"] = float(np.nanmean(f1))
    return out


def save_grouped_metrics_val_test(
    plot_dir: str,
    val_m: Mapping,
    test_m: Mapping,
    fname: str = "metrics_val_vs_test.png",
    title: str = "Core metrics: Validation vs Test",
) -> None:
    """上图：Accuracy、Weighted F1，及（若 metrics 中含键）Macro F1；下图：MCC。"""
    plt = _pyplot()

    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, fname)

    val_eff = metrics_dict_with_macro_fallback(val_m)
    test_eff = metrics_dict_with_macro_fallback(test_m)

    acc_v, acc_t = float(val_eff.get("accuracy", 0.0)), float(test_eff.get("accuracy", 0.0))
    f1_v, f1_t = float(val_eff.get("f1", 0.0)), float(test_eff.get("f1", 0.0))
    mf1_v, mf1_t = _safe_float_metric(val_eff, "f1_macro"), _safe_float_metric(test_eff, "f1_macro")
    has_macro_f1 = bool(np.isfinite(mf1_v) and np.isfinite(mf1_t))
    mcc_v = val_eff.get("mcc", float("nan"))
    mcc_t = test_eff.get("mcc", float("nan"))
    mcc_v = float(mcc_v) if mcc_v is not None else float("nan")
    mcc_t = float(mcc_t) if mcc_t is not None else float("nan")
    has_mcc = not (np.isnan(mcc_v) or np.isnan(mcc_t))

    if has_mcc:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2 if has_macro_f1 else 5.5, 6.5), gridspec_kw={"height_ratios": [1, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=(6.2 if has_macro_f1 else 5.5, 3.8))
        ax2 = None

    w = 0.35
    if has_macro_f1:
        x1 = np.arange(3)
        ax1.bar(x1 - w / 2, [acc_v, f1_v, mf1_v], w, label="Val", color="steelblue")
        ax1.bar(x1 + w / 2, [acc_t, f1_t, mf1_t], w, label="Test", color="darkorange")
        ax1.set_xticks(x1)
        ax1.set_xticklabels(["Accuracy", "Weighted F1", "Macro F1"])
    else:
        x1 = np.arange(2)
        ax1.bar(x1 - w / 2, [acc_v, f1_v], w, label="Val", color="steelblue")
        ax1.bar(x1 + w / 2, [acc_t, f1_t], w, label="Test", color="darkorange")
        ax1.set_xticks(x1)
        ax1.set_xticklabels(["Accuracy", "Weighted F1"])
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.08)
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_title(title)

    if ax2 is not None:
        x2 = np.arange(1)
        ax2.bar(x2 - w / 2, [mcc_v], w, label="Val", color="steelblue")
        ax2.bar(x2 + w / 2, [mcc_t], w, label="Test", color="darkorange")
        ax2.set_xticks(x2)
        ax2.set_xticklabels(["MCC"])
        ax2.set_ylabel("MCC")
        ax2.axhline(0, color="gray", linewidth=0.8)
        ax2.set_ylim(-1.05, 1.05)
        ax2.legend()
        ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    _savefig_png_pdf(fig, path, dpi=150)
    plt.close(fig)


def save_multi_accuracy_bar(
    plot_dir: str,
    labels: Sequence[str],
    accs: Sequence[float],
    fname: str,
    title: str,
) -> str:
    plt = _pyplot()

    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, fname)
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.2), 4))
    x = np.arange(len(labels))
    ax.bar(x, list(accs), color=plt.cm.viridis(np.linspace(0.2, 0.8, len(labels))))
    ax.set_xticks(x)
    ax.set_xticklabels(list(labels), rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    for i, v in enumerate(accs):
        ax.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_per_class_metric_bars(
    plot_dir: str,
    class_names: Sequence[str],
    per_class_accuracy_ovr: Sequence[float],
    per_class_precision: Sequence[float],
    per_class_recall: Sequence[float],
    fname: str = "per_class_accuracy_precision_recall.png",
    title: str = "Per-class Accuracy (OvR) / Precision / Recall",
) -> str:
    plt = _pyplot()
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, fname)
    x = np.arange(len(class_names))
    w = 0.24
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    b1 = ax.bar(x - w, list(per_class_accuracy_ovr), width=w, label="Accuracy (OvR)", color="#4C78A8")
    b2 = ax.bar(x, list(per_class_precision), width=w, label="Precision", color="#F58518")
    b3 = ax.bar(x + w, list(per_class_recall), width=w, label="Recall", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(list(class_names))
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    fig.text(
        0.5,
        0.02,
        "OvR accuracy per class k: (TP_k+TN_k)/N on the k-vs-rest binary partition (differs from recall).",
        ha="center",
        fontsize=7,
        color="dimgray",
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    for bars in (b1, b2, b3):
        for bar in bars:
            v = float(bar.get_height())
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(bottom=0.18, top=0.92)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_per_class_auprc_bar(
    plot_dir: str,
    class_names: Sequence[str],
    auprc_values: Sequence[float],
    fname: str = "per_class_auprc.png",
    title: str = "Per-class AUPRC",
) -> str:
    plt = _pyplot()
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, fname)
    x = np.arange(len(class_names))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(x, list(auprc_values), color=["#4C78A8", "#F58518", "#54A24B"], width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(list(class_names))
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("AUPRC")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(auprc_values):
        if np.isfinite(v):
            ax.text(i, float(v) + 0.02, f"{float(v):.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_ovr_roc_pr_curves(
    plot_dir: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    class_indices: Sequence[int],
    roc_fname: str = "per_class_roc_curves_ovr.png",
    pr_fname: str = "per_class_pr_curves_ovr.png",
    legend_roc_ovr: Optional[Sequence[float]] = None,
    legend_auprc: Optional[Sequence[float]] = None,
) -> Tuple[str, str]:
    plt = _pyplot()
    os.makedirs(plot_dir, exist_ok=True)
    roc_path = os.path.join(plot_dir, roc_fname)
    pr_path = os.path.join(plot_dir, pr_fname)

    colors = ["#4C78A8", "#F58518", "#54A24B"]
    fig1, ax1 = plt.subplots(figsize=(6.6, 5.2))
    fig2, ax2 = plt.subplots(figsize=(6.6, 5.2))
    for k, (name, idx, color) in enumerate(zip(class_names, class_indices, colors)):
        y_bin = (y_true == idx).astype(int)
        score = y_prob[:, idx]
        fpr, tpr, _ = roc_curve(y_bin, score)
        if legend_roc_ovr is not None and k < len(legend_roc_ovr):
            auc_roc = float(legend_roc_ovr[k])
        else:
            try:
                auc_roc = float(roc_auc_score(y_bin, score))
            except ValueError:
                auc_roc = float("nan")
        ax1.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={auc_roc:.3f})")

        precision, recall, _ = precision_recall_curve(y_bin, score)
        if legend_auprc is not None and k < len(legend_auprc):
            auprc = float(legend_auprc[k])
        else:
            try:
                auprc = float(average_precision_score(y_bin, score))
            except ValueError:
                auprc = float("nan")
        ax2.plot(recall, precision, color=color, linewidth=2, label=f"{name} (AUPRC={auprc:.3f})")

    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("One-vs-Rest ROC curves")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)
    fig1.tight_layout()
    fig1.savefig(roc_path, dpi=150)
    plt.close(fig1)

    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("One-vs-Rest PR curves")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(pr_path, dpi=150)
    plt.close(fig2)
    return roc_path, pr_path
