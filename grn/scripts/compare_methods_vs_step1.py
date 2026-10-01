from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.container import BarContainer


def _mean_std(x: pd.Series) -> Tuple[float, float]:
    v = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def _load_deepfgrn_metrics(path: Path) -> Dict[str, Tuple[float, float]]:
    df = pd.read_excel(path, sheet_name=0)
    # DeepFGRN 字段: AUC / Macro_AUC_OVR / ACC_... / Recall / Precision / F1 / MCC
    out: Dict[str, Tuple[float, float]] = {}
    if "Macro_AUC_OVR" in df.columns:
        out["AUC"] = _mean_std(df["Macro_AUC_OVR"])
    elif "AUC" in df.columns:
        out["AUC"] = _mean_std(df["AUC"])
    else:
        out["AUC"] = (np.nan, np.nan)
    out["Accuracy"] = _mean_std(df["Recall"])  # DeepFGRN 表里未显式给 overall acc；沿用其主表 recall 字段
    out["Recall"] = _mean_std(df["Recall"])
    out["Precision"] = _mean_std(df["Precision"])
    out["F1"] = _mean_std(df["F1"])
    out["MCC"] = _mean_std(df["MCC"])
    return out


def _load_deepfgrn_per_class(path: Path) -> Dict[str, Tuple[float, float]]:
    df = pd.read_excel(path, sheet_name=0)
    acc_cols = [c for c in df.columns if str(c).startswith("ACC_")]
    if len(acc_cols) < 3:
        raise ValueError(f"DeepFGRN ACC_* columns not found, got: {df.columns.tolist()}")
    # 历史导出顺序为 NoReg, Repress, Activate；展示顺序改为 No, Active, Repress
    m_no, s_no = _mean_std(df[acc_cols[0]])
    m_rep, s_rep = _mean_std(df[acc_cols[1]])
    m_act, s_act = _mean_std(df[acc_cols[2]])
    return {
        "acc_no": (m_no, s_no),
        "acc_active": (m_act, s_act),
        "acc_repress": (m_rep, s_rep),
    }


def _load_dgcgrn_metrics(path: Path) -> Dict[str, Tuple[float, float]]:
    df = pd.read_excel(path, sheet_name=0)
    out: Dict[str, Tuple[float, float]] = {
        "AUC": _mean_std(df["test_auc"]),
        "Accuracy": _mean_std(df["test_acc"]),
        "Recall": _mean_std(df["test_recall"]),
        "Precision": _mean_std(df["test_precision"]),
        "F1": _mean_std(df["test_f1"]),
        "MCC": _mean_std(df["test_mcc"]),
    }
    return out


def _load_step1_metrics(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        j = json.load(f)
    return {
        "AUC": float(j.get("macro_auc", np.nan)),
        "Accuracy": float(j.get("accuracy", np.nan)),
        "Recall": float(j.get("recall", np.nan)),
        "Precision": float(j.get("precision", np.nan)),
        "F1": float(j.get("f1", np.nan)),
        "MCC": float(j.get("mcc", np.nan)),
    }


def _load_step1_per_class(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        j = json.load(f)
    cm = np.asarray(j.get("confusion_matrix"), dtype=float)
    if cm.shape != (3, 3):
        raise ValueError(f"unexpected confusion matrix shape: {cm.shape}")
    # 0=NoReg, 1=Repress, 2=Activate; display order: No, Active, Repress
    tp = np.diag(cm)
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    recall = np.divide(tp, row_sum, out=np.full_like(tp, np.nan), where=row_sum > 0)
    precision = np.divide(tp, col_sum, out=np.full_like(tp, np.nan), where=col_sum > 0)
    return {
        "acc_no": float(j.get("无调控_accuracy", np.nan)),
        "acc_active": float(j.get("激活_accuracy", np.nan)),
        "acc_repress": float(j.get("抑制_accuracy", np.nan)),
        "recall_no": float(recall[0]),
        "recall_active": float(recall[2]),
        "recall_repress": float(recall[1]),
        "precision_no": float(precision[0]),
        "precision_active": float(precision[2]),
        "precision_repress": float(precision[1]),
    }


def _build_summary_df(
    deep: Dict[str, Tuple[float, float]],
    dgc: Dict[str, Tuple[float, float]],
    step1: Dict[str, float],
) -> pd.DataFrame:
    metrics = ["AUC", "Accuracy", "Recall", "Precision", "F1", "MCC"]
    rows = []
    for m in metrics:
        rows.append(
            {
                "metric": m,
                "DeepFGRN_mean": deep[m][0],
                "DeepFGRN_std": deep[m][1],
                "DGCGRN_mean": dgc[m][0],
                "DGCGRN_std": dgc[m][1],
                "Step1_single_run": step1[m],
            }
        )
    return pd.DataFrame(rows)


def _plot_grouped(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = df["metric"].tolist()
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(
        x - width,
        df["DeepFGRN_mean"],
        width=width,
        yerr=df["DeepFGRN_std"],
        capsize=3,
        label="DeepFGRN (5-fold mean±std)",
    )
    ax.bar(
        x,
        df["DGCGRN_mean"],
        width=width,
        yerr=df["DGCGRN_std"],
        capsize=3,
        label="DGCGRN (5-fold mean±std)",
    )
    ax.bar(
        x + width,
        df["Step1_single_run"],
        width=width,
        label="Our (single run)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Method comparison on GSE253718")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    for bars in ax.containers:
        if isinstance(bars, BarContainer):
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "method_compare_grouped_metrics.png", dpi=220)
    plt.close(fig)


def _plot_per_metric(df: pd.DataFrame, out_dir: Path) -> None:
    for _, r in df.iterrows():
        m = r["metric"]
        labels = ["DeepFGRN", "DGCGRN", "Our"]
        means = [r["DeepFGRN_mean"], r["DGCGRN_mean"], r["Step1_single_run"]]
        errs = [r["DeepFGRN_std"], r["DGCGRN_std"], 0.0]
        x = np.arange(3)
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.bar(x, means, yerr=errs, capsize=3, color=["#4C78A8", "#F58518", "#54A24B"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{m} comparison")
        ax.grid(axis="y", alpha=0.3)
        for bars in ax.containers:
            if isinstance(bars, BarContainer):
                ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"method_compare_{m.lower()}.png", dpi=220)
        plt.close(fig)


def _plot_our_perclass_three_figs(per_cls: Dict[str, float], out_dir: Path) -> None:
    pass


def _plot_method_compare_perclass_triplets(
    deep: Dict[str, Tuple[float, float]],
    deep_pc: Dict[str, Tuple[float, float]],
    dgc: Dict[str, Tuple[float, float]],
    step1_pc: Dict[str, float],
    out_dir: Path,
) -> None:
    methods = ["DeepFGRN", "DGCGRN", "Our"]
    x = np.arange(len(methods))
    width = 0.24

    panels = [
        (
            "no_regulation",
            "No regulation",
            [deep_pc["acc_no"][0], dgc["Accuracy"][0], step1_pc["acc_no"]],
            [deep["Recall"][0], dgc["Recall"][0], step1_pc["recall_no"]],
            [deep["Precision"][0], dgc["Precision"][0], step1_pc["precision_no"]],
        ),
        (
            "activation",
            "Activation",
            [deep_pc["acc_active"][0], dgc["Accuracy"][0], step1_pc["acc_active"]],
            [deep["Recall"][0], dgc["Recall"][0], step1_pc["recall_active"]],
            [deep["Precision"][0], dgc["Precision"][0], step1_pc["precision_active"]],
        ),
        (
            "repression",
            "Repression",
            [deep_pc["acc_repress"][0], dgc["Accuracy"][0], step1_pc["acc_repress"]],
            [deep["Recall"][0], dgc["Recall"][0], step1_pc["recall_repress"]],
            [deep["Precision"][0], dgc["Precision"][0], step1_pc["precision_repress"]],
        ),
    ]

    for slug, title, acc_vals, rec_vals, pre_vals in panels:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.bar(x - width, acc_vals, width=width, label="Accuracy", color="#4C78A8")
        ax.bar(x, rec_vals, width=width, label="Recall", color="#F58518")
        ax.bar(x + width, pre_vals, width=width, label="Precision", color="#54A24B")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"Method comparison ({title}): Accuracy / Recall / Precision")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        for bars in ax.containers:
            if isinstance(bars, BarContainer):
                ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"our_perclass_acc_recall_precision_{slug}.png", dpi=220)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare DeepFGRN / DGCGRN against step1 benchmark and draw figures.")
    ap.add_argument("--method-dir", default="experiments/method_compare")
    ap.add_argument("--step1-test-metrics", default="experiments/benchmark_luad_gse253718_step1/unseen_samples_test_metrics.json")
    ap.add_argument("--out-dir", default=None, help="Default: <method-dir>/summary")
    args = ap.parse_args()

    method_dir = Path(args.method_dir).resolve()
    if not method_dir.is_dir():
        raise SystemExit(f"method dir not found: {method_dir}")

    deep_path = method_dir / "DeepFGRN_GSE253718_cellUnseen_5FCV_2026-04-14-160718_metrics.xlsx"
    dgc_path = method_dir / "DGCGRN_gse253718_20260414_164952.test_metrics.xlsx"
    step1_path = Path(args.step1_test_metrics).resolve()
    for p in [deep_path, dgc_path, step1_path]:
        if not p.is_file():
            raise SystemExit(f"missing file: {p}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (method_dir / "summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    deep = _load_deepfgrn_metrics(deep_path)
    deep_pc = _load_deepfgrn_per_class(deep_path)
    dgc = _load_dgcgrn_metrics(dgc_path)
    step1 = _load_step1_metrics(step1_path)
    step1_per_class = _load_step1_per_class(step1_path)

    df = _build_summary_df(deep, dgc, step1)
    df.to_csv(out_dir / "method_compare_vs_step1_summary.csv", index=False, encoding="utf-8-sig")
    _plot_grouped(df, out_dir)
    _plot_per_metric(df, out_dir)
    _plot_method_compare_perclass_triplets(deep, deep_pc, dgc, step1_per_class, out_dir)

    print(f"[method-compare] done. out_dir={out_dir}")


if __name__ == "__main__":
    main()
