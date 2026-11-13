#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig.2 ablation singles (manuscript Figure 3 style panels) as PDF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grn.scripts.summarize_gse253718_ablation_10cv import (  # noqa: E402
    _acc_precision_recall_boxplot,
    _configure_matplotlib,
    _per_class_metric_boxplot,
    _save_figure,
    _stage_boxplot,
)

STAGE_ORDER = [
    "01_expr_only",
    "05_cds_only",
    "06_ppi_only",
    "02_expr_cds",
    "03_expr_ppi",
    "07_cds_ppi",
    "04_expr_cds_ppi",
]
LABEL_MAP = {
    "01_expr_only": "expr",
    "05_cds_only": "cds",
    "06_ppi_only": "ppi",
    "02_expr_cds": "expr+cds",
    "03_expr_ppi": "expr+ppi",
    "07_cds_ppi": "cds+ppi",
    "04_expr_cds_ppi": "all",
}


def _heatmap(mat: pd.DataFrame, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Score"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Modality configuration")
    fig.tight_layout()
    _save_figure(fig, out)
    plt.close(fig)


def _stage_means_from_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid in STAGE_ORDER:
        sub = fold_df[fold_df["stage_id"] == sid]
        if sub.empty:
            continue
        label = LABEL_MAP.get(sid, sid)
        row = {"stage_id": sid, "modalities_label": label}
        for col in (
            "macro_auc",
            "accuracy",
            "mcc",
            "f1",
            "f1_macro",
            "precision_macro",
            "recall_macro",
            "pc_noreg_accuracy",
            "pc_noreg_recall",
            "pc_noreg_precision",
            "pc_activation_accuracy",
            "pc_activation_recall",
            "pc_activation_precision",
            "pc_repression_accuracy",
            "pc_repression_recall",
            "pc_repression_precision",
            "auc_ovr_NoReg",
            "auc_ovr_Activation",
            "auc_ovr_Repression",
            "auprc_NoReg",
            "auprc_Activation",
            "auprc_Repression",
        ):
            if col in sub.columns:
                row[col] = float(pd.to_numeric(sub[col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp-dir",
        type=Path,
        default=ROOT
        / "fig02_modality_ablation"
        / "experiments"
        / "gse253718_ablation_epithelial_10cv_seed42",
    )
    args = ap.parse_args()
    _configure_matplotlib()

    fold_csv = args.exp_dir / "summary" / "ablation_10cv_fold_metrics.csv"
    if not fold_csv.is_file():
        raise FileNotFoundError(fold_csv)
    fold_df = pd.read_csv(fold_csv)
    means = _stage_means_from_folds(fold_df)

    out = ROOT / "fig02_modality_ablation" / "figures" / "main"
    out.mkdir(parents=True, exist_ok=True)

    overall = means.set_index("modalities_label")[
        ["macro_auc", "accuracy", "mcc", "f1", "f1_macro", "precision_macro", "recall_macro"]
    ].rename(
        columns={
            "macro_auc": "Macro-AUC",
            "accuracy": "Accuracy",
            "mcc": "MCC",
            "f1": "Weighted F1",
            "f1_macro": "Macro F1",
            "precision_macro": "Macro precision",
            "recall_macro": "Macro recall",
        }
    )
    _heatmap(overall, "Overall test metrics — modality ablation (10CV mean)", out / "Fig3C_ablation_overall_heatmap.png")

    for title, cols, stem in (
        (
            "Per-class test metrics — No regulation",
            {
                "pc_noreg_accuracy": "Accuracy (OvR)",
                "pc_noreg_recall": "Recall",
                "pc_noreg_precision": "Precision",
                "auc_ovr_NoReg": "OvR AUC",
                "auprc_NoReg": "AUPRC",
            },
            "Fig3_perclass_noreg_heatmap",
        ),
        (
            "Per-class test metrics — Activation",
            {
                "pc_activation_accuracy": "Accuracy (OvR)",
                "pc_activation_recall": "Recall",
                "pc_activation_precision": "Precision",
                "auc_ovr_Activation": "OvR AUC",
                "auprc_Activation": "AUPRC",
            },
            "Fig3_perclass_activation_heatmap",
        ),
        (
            "Per-class test metrics — Repression",
            {
                "pc_repression_accuracy": "Accuracy (OvR)",
                "pc_repression_recall": "Recall",
                "pc_repression_precision": "Precision",
                "auc_ovr_Repression": "OvR AUC",
                "auprc_Repression": "AUPRC",
            },
            "Fig3_perclass_repression_heatmap",
        ),
    ):
        use = {k: v for k, v in cols.items() if k in means.columns}
        mat = means.set_index("modalities_label")[list(use.keys())].rename(columns=use)
        _heatmap(mat, title, out / f"{stem}.png")

    _stage_boxplot(
        fold_df,
        value_col="macro_auc",
        ylabel="Macro-AUC",
        title="Ablation 10CV: Macro-AUC by modality",
        out_path=out / "ablation_10cv_macro_auc_boxplot.png",
    )
    _stage_boxplot(
        fold_df,
        value_col="accuracy",
        ylabel="Accuracy",
        title="Ablation 10CV: Accuracy by modality",
        out_path=out / "ablation_10cv_accuracy_boxplot.png",
    )
    _stage_boxplot(
        fold_df,
        value_col="f1_macro",
        ylabel="Macro F1",
        title="Ablation 10CV: Macro-F1 by modality",
        out_path=out / "ablation_10cv_f1_macro_boxplot.png",
    )
    # per-class boxplots use pc_*_{metric} naming in summarize script for accuracy etc.
    # For auc_ovr the fold file has auc_ovr_NoReg columns — adapt via rename for helper
    auc_long = fold_df.copy()
    # helper expects pc_noreg_auc_ovr style; map
    if "auc_ovr_NoReg" in auc_long.columns:
        auc_long["pc_noreg_auc_ovr"] = auc_long["auc_ovr_NoReg"]
        auc_long["pc_activation_auc_ovr"] = auc_long["auc_ovr_Activation"]
        auc_long["pc_repression_auc_ovr"] = auc_long["auc_ovr_Repression"]
        _per_class_metric_boxplot(auc_long, "auc_ovr", "OvR AUC", out / "ablation_10cv_auc_ovr_boxplot.png")
    if "auprc_NoReg" in fold_df.columns:
        ap = fold_df.copy()
        ap["pc_noreg_auprc"] = ap["auprc_NoReg"]
        ap["pc_activation_auprc"] = ap["auprc_Activation"]
        ap["pc_repression_auprc"] = ap["auprc_Repression"]
        _per_class_metric_boxplot(ap, "auprc", "AUPRC", out / "ablation_10cv_auprc_boxplot.png")
    _acc_precision_recall_boxplot(fold_df, out / "ablation_10cv_acc_precision_recall_boxplot.png")

    print(f"[fig02] done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
