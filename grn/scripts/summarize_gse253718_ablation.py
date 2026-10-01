"""
Aggregate GSE253718 Step1 modality ablation runs (train/val/test split already inside each run).

Writes:
  - ablation_test_metrics_summary.csv (overall + per-class Accuracy / Recall / Precision for three classes)
  - plots/ablation_test_accuracy_f1.png (English labels only)

CSV 说明：``test_recall`` / ``test_precision`` 来自 sklearn 的 *weighted* 平均；多分类下
weighted recall 与 ``test_accuracy`` 数学上恒等，因此二者会完全相同。若要与准确率区分，
请看 ``test_recall_macro`` / ``test_precision_macro`` / ``test_f1_macro``（旧结果无这些键时
由 ``confusion_matrix`` 回算）。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _load_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _float(d: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    v = d.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _macro_prf1_from_cm(cm_raw: Any) -> Tuple[float, float, float]:
    """由固定形状混淆矩阵得到 macro recall / precision / F1（与 sklearn macro 一致，缺支持类为 0）。"""
    cm = np.asarray(cm_raw, dtype=np.float64)
    if cm.size == 0 or cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        return float("nan"), float("nan"), float("nan")
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    sup = tp + fn
    pred_sup = tp + fp
    rec = np.divide(tp, sup, out=np.zeros_like(tp, dtype=np.float64), where=sup > 0)
    prec = np.divide(tp, pred_sup, out=np.zeros_like(tp, dtype=np.float64), where=pred_sup > 0)
    f1 = np.divide(
        2.0 * prec * rec,
        prec + rec,
        out=np.zeros_like(tp, dtype=np.float64),
        where=(prec + rec) > 0,
    )
    return float(np.mean(rec)), float(np.mean(prec)), float(np.mean(f1))


def _macro_metrics(mj: Dict[str, Any]) -> Tuple[float, float, float]:
    rm = _float(mj, "recall_macro")
    if not np.isnan(rm):
        return rm, _float(mj, "precision_macro"), _float(mj, "f1_macro")
    cm = mj.get("confusion_matrix")
    if cm is not None:
        return _macro_prf1_from_cm(cm)
    return float("nan"), float("nan"), float("nan")


def _per_class_pr_from_cm(cm_raw: Any) -> Tuple[np.ndarray, np.ndarray]:
    """从混淆矩阵计算每类 recall / precision（原顺序: NoReg, Repress, Activate）。"""
    cm = np.asarray(cm_raw, dtype=np.float64)
    if cm.size == 0 or cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        return np.full((3,), np.nan, dtype=np.float64), np.full((3,), np.nan, dtype=np.float64)
    tp = np.diag(cm)
    sup = cm.sum(axis=1)
    pred_sup = cm.sum(axis=0)
    rec = np.divide(tp, sup, out=np.full_like(tp, np.nan, dtype=np.float64), where=sup > 0)
    prec = np.divide(tp, pred_sup, out=np.full_like(tp, np.nan, dtype=np.float64), where=pred_sup > 0)
    return rec, prec


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize GSE253718 ablation runs into CSV + English plots.")
    p.add_argument(
        "--parent-dir",
        type=Path,
        default=Path("experiments/gse253718_ablation_step1_seed42"),
        help="Directory containing 01_expr_only/, 02_expr_cds/, …, 07_cds_ppi/",
    )
    args = p.parse_args()
    root = args.parent_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Missing parent dir: {root}")

    # 指定展示顺序：expr, cds, ppi, expr+cds, expr+ppi, cds+ppi, all
    runs: List[Tuple[str, str, Path]] = [
        ("01_expr_only", "expr", root / "01_expr_only"),
        ("05_cds_only", "cds", root / "05_cds_only"),
        ("06_ppi_only", "ppi", root / "06_ppi_only"),
        ("02_expr_cds", "expr+cds", root / "02_expr_cds"),
        ("03_expr_ppi", "expr+ppi", root / "03_expr_ppi"),
        ("07_cds_ppi", "cds+ppi", root / "07_cds_ppi"),
        ("04_expr_cds_ppi", "all", root / "04_expr_cds_ppi"),
    ]

    rows: List[Dict[str, Any]] = []
    labels: List[str] = []
    accs: List[float] = []
    f1s: List[float] = []
    macro_f1s: List[float] = []
    macro_aucs: List[float] = []
    per_cls_acc_no: List[float] = []
    per_cls_acc_act: List[float] = []
    per_cls_acc_rep: List[float] = []
    per_cls_rec_no: List[float] = []
    per_cls_rec_act: List[float] = []
    per_cls_rec_rep: List[float] = []
    per_cls_pre_no: List[float] = []
    per_cls_pre_act: List[float] = []
    per_cls_pre_rep: List[float] = []
    per_cls_auc_no: List[float] = []
    per_cls_auc_act: List[float] = []
    per_cls_auc_rep: List[float] = []
    per_cls_auprc_no: List[float] = []
    per_cls_auprc_act: List[float] = []
    per_cls_auprc_rep: List[float] = []

    def _vec3(mj: Dict[str, Any], key: str) -> Tuple[float, float, float]:
        v = mj.get(key)
        if not isinstance(v, (list, tuple)) or len(v) < 3:
            return float("nan"), float("nan"), float("nan")
        try:
            return float(v[0]), float(v[1]), float(v[2])
        except (TypeError, ValueError):
            return float("nan"), float("nan"), float("nan")

    for run_id, label, run_dir in runs:
        mj = _load_json(run_dir / "unseen_samples_test_metrics.json")
        if mj is None:
            print(f"[warn] skip (no test metrics): {run_dir}")
            continue
        labels.append(label)
        accs.append(_float(mj, "accuracy"))
        f1s.append(_float(mj, "f1"))
        macro_aucs.append(_float(mj, "macro_auc"))
        r_macro, p_macro, f1_macro = _macro_metrics(mj)
        macro_f1s.append(f1_macro)
        # per-class: 输出顺序统一为 NoReg, Activate, Repress
        cm = mj.get("confusion_matrix")
        rec_raw, pre_raw = _per_class_pr_from_cm(cm)
        per_cls_acc_no.append(_float(mj, "无调控_accuracy"))
        per_cls_acc_act.append(_float(mj, "激活_accuracy"))
        per_cls_acc_rep.append(_float(mj, "抑制_accuracy"))
        per_cls_rec_no.append(float(rec_raw[0]))
        per_cls_rec_act.append(float(rec_raw[2]))
        per_cls_rec_rep.append(float(rec_raw[1]))
        per_cls_pre_no.append(float(pre_raw[0]))
        per_cls_pre_act.append(float(pre_raw[2]))
        per_cls_pre_rep.append(float(pre_raw[1]))
        # per-class OvR AUC / AUPRC: JSON order NoReg, Activation, Repression
        auc_no, auc_act, auc_rep = _vec3(mj, "auc_ovr")
        if np.isnan(auc_no):
            auc_no, auc_act, auc_rep = _vec3(mj, "auc_ovo_per_class")
        auprc_no, auprc_act, auprc_rep = _vec3(mj, "auprc")
        per_cls_auc_no.append(auc_no)
        per_cls_auc_act.append(auc_act)
        per_cls_auc_rep.append(auc_rep)
        per_cls_auprc_no.append(auprc_no)
        per_cls_auprc_act.append(auprc_act)
        per_cls_auprc_rep.append(auprc_rep)
        rows.append(
            {
                "run_id": run_id,
                "modalities_label": label,
                "test_accuracy": accs[-1],
                "test_weighted_f1": f1s[-1],
                "test_macro_auc": macro_aucs[-1],
                "test_mcc": _float(mj, "mcc"),
                "test_precision": _float(mj, "precision"),
                "test_recall": _float(mj, "recall"),
                "test_precision_macro": p_macro,
                "test_recall_macro": r_macro,
                "test_f1_macro": f1_macro,
                # Per-class (test unseen): order No regulation, Activation, Repression (与 JSON / CM 一致)
                "test_pc_noreg_accuracy": per_cls_acc_no[-1],
                "test_pc_activation_accuracy": per_cls_acc_act[-1],
                "test_pc_repression_accuracy": per_cls_acc_rep[-1],
                "test_pc_noreg_recall": per_cls_rec_no[-1],
                "test_pc_activation_recall": per_cls_rec_act[-1],
                "test_pc_repression_recall": per_cls_rec_rep[-1],
                "test_pc_noreg_precision": per_cls_pre_no[-1],
                "test_pc_activation_precision": per_cls_pre_act[-1],
                "test_pc_repression_precision": per_cls_pre_rep[-1],
                "test_pc_noreg_auc_ovr": per_cls_auc_no[-1],
                "test_pc_activation_auc_ovr": per_cls_auc_act[-1],
                "test_pc_repression_auc_ovr": per_cls_auc_rep[-1],
                "test_pc_noreg_auprc": per_cls_auprc_no[-1],
                "test_pc_activation_auprc": per_cls_auprc_act[-1],
                "test_pc_repression_auprc": per_cls_auprc_rep[-1],
            }
        )

    if not rows:
        raise SystemExit("No run metrics found; train ablation stages first.")

    csv_path = root / "ablation_test_metrics_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ablation-summary] wrote {csv_path}")
    print(
        "[ablation-summary] note: multiclass weighted recall == accuracy; "
        "use test_recall_macro / test_precision_macro / test_f1_macro to compare runs."
    )

    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_arr = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bw = 0.25
    ax.bar(x_arr - bw, accs, width=bw, label="Test accuracy", color="steelblue")
    ax.bar(x_arr, f1s, width=bw, label="Test weighted F1", color="darkorange")
    ax.bar(x_arr + bw, macro_f1s, width=bw, label="Test macro F1", color="#2ca02c")
    ax.set_xticks(x_arr)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("GSE253718 Step1 ablation (fixed seed, column split 5:2:3)")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = plot_dir / "ablation_test_accuracy_f1.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"[ablation-summary] wrote {p1}")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    xpos = np.arange(len(labels))
    ax2.bar(xpos, macro_aucs, color="seagreen", width=0.55)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("Test macro-averaged OvO AUC")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_title("GSE253718 Step1 ablation: test macro AUC")
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    p2 = plot_dir / "ablation_test_macro_auc.png"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    print(f"[ablation-summary] wrote {p2}")

    # 一张图展示三类（No/Act/Rep）在各实验下的 acc / recall / precision
    fig3, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)
    x3 = np.arange(len(labels))
    bw3 = 0.24
    panels = [
        ("No regulation", per_cls_acc_no, per_cls_rec_no, per_cls_pre_no),
        ("Activation", per_cls_acc_act, per_cls_rec_act, per_cls_pre_act),
        ("Repression", per_cls_acc_rep, per_cls_rec_rep, per_cls_pre_rep),
    ]
    for ax, (ttl, acc_v, rec_v, pre_v) in zip(axes, panels):
        ax.bar(x3 - bw3, acc_v, width=bw3, label="Accuracy", color="#4C78A8")
        ax.bar(x3, rec_v, width=bw3, label="Recall", color="#F58518")
        ax.bar(x3 + bw3, pre_v, width=bw3, label="Precision", color="#54A24B")
        ax.set_xticks(x3)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(ttl)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Score")
    handles, lgd_labels = axes[0].get_legend_handles_labels()
    fig3.legend(handles, lgd_labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig3.suptitle("GSE253718 ablation: per-class Accuracy / Recall / Precision (No, Activation, Repression)", y=1.08)
    fig3.tight_layout()
    p3 = plot_dir / "ablation_per_class_acc_recall_precision.png"
    fig3.savefig(p3, dpi=160, bbox_inches="tight")
    plt.close(fig3)
    print(f"[ablation-summary] wrote {p3}")


if __name__ == "__main__":
    main()
