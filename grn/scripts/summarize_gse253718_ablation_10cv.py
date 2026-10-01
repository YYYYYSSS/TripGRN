#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate GSE253718 modality ablation 10-fold CV metrics + step01-style boxplots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from grn.data.gse253718_paths import (
    ABLATION_STAGES,
    DEFAULT_ABLATION_EXPERIMENT,
    ablation_stage_dir,
    gse253718_layout,
)
from grn.evaluation.training_plots import per_class_ovr_accuracy_precision_recall_from_cm

METRIC_KEYS = [
    "accuracy",
    "precision",
    "precision_macro",
    "recall",
    "recall_macro",
    "f1",
    "f1_macro",
    "mcc",
    "macro_auc",
    "macro_auprc",
    "auc_ovo",
]


def _macro_auprc(m: Dict[str, Any]) -> float:
    """Macro AUPRC = mean of per-class one-vs-rest AUPRC."""
    vals = m.get("auprc")
    if not isinstance(vals, list) or not vals:
        return float("nan")
    return float(np.nanmean(np.asarray(vals, dtype=float)))

CLASS_LABELS = ("No regulation", "Activation", "Repression")
CM_IDX = {"No regulation": 0, "Activation": 2, "Repression": 1}
AUC_IDX = {"No regulation": 0, "Activation": 1, "Repression": 2}

STAGE_COLORS = {
    "01_expr_only": "#4C78A8",
    "02_expr_cds": "#72B7B2",
    "03_expr_ppi": "#54A24B",
    "04_expr_cds_ppi": "#EECA3B",
    "05_cds_only": "#F58518",
    "06_ppi_only": "#B279A2",
    "07_cds_ppi": "#9D755D",
}

FIG_W = 14.0
FIG_H = 7.0
FIG_DPI = 180


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False


def _load_fold_metrics(run_dir: Path) -> Dict[str, Any] | None:
    p = run_dir / "unseen_samples_test_metrics.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _vec3(mj: Dict[str, Any], key: str) -> Tuple[float, float, float]:
    v = mj.get(key)
    if not isinstance(v, (list, tuple)) or len(v) < 3:
        return float("nan"), float("nan"), float("nan")
    try:
        return float(v[0]), float(v[1]), float(v[2])
    except (TypeError, ValueError):
        return float("nan"), float("nan"), float("nan")


def _flatten_fold_row(stage_id: str, label: str, fold_idx: int, m: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "stage_id": stage_id,
        "modalities_label": label,
        "fold": fold_idx,
    }
    for k in METRIC_KEYS:
        if k == "macro_auprc":
            row[k] = _macro_auprc(m)
        else:
            row[k] = m.get(k, np.nan)
    auc_no, auc_act, auc_rep = _vec3(m, "auc_ovr")
    if np.isnan(auc_no):
        auc_no, auc_act, auc_rep = _vec3(m, "auc_ovo_per_class")
    auprc_no, auprc_act, auprc_rep = _vec3(m, "auprc")
    row["auc_ovr_NoReg"] = auc_no
    row["auc_ovr_Activation"] = auc_act
    row["auc_ovr_Repression"] = auc_rep
    row["auprc_NoReg"] = auprc_no
    row["auprc_Activation"] = auprc_act
    row["auprc_Repression"] = auprc_rep
    cm = m.get("confusion_matrix")
    if cm is not None:
        acc, prec, rec = per_class_ovr_accuracy_precision_recall_from_cm(np.asarray(cm, dtype=float))
        row["pc_noreg_accuracy"] = float(acc[CM_IDX["No regulation"]])
        row["pc_activation_accuracy"] = float(acc[CM_IDX["Activation"]])
        row["pc_repression_accuracy"] = float(acc[CM_IDX["Repression"]])
        row["pc_noreg_precision"] = float(prec[CM_IDX["No regulation"]])
        row["pc_activation_precision"] = float(prec[CM_IDX["Activation"]])
        row["pc_repression_precision"] = float(prec[CM_IDX["Repression"]])
        row["pc_noreg_recall"] = float(rec[CM_IDX["No regulation"]])
        row["pc_activation_recall"] = float(rec[CM_IDX["Activation"]])
        row["pc_repression_recall"] = float(rec[CM_IDX["Repression"]])
    row["confusion_matrix"] = json.dumps(cm)
    return row


def _metric_ylim(values: np.ndarray, *, bottom_pad: float = 0.1, top_pad: float = 0.02, ymax_cap: float = 1.05) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, ymax_cap
    ymin = float(np.min(vals) - bottom_pad)
    ymax = min(float(np.max(vals) + top_pad), ymax_cap)
    if ymax <= ymin:
        ymax = ymin + 0.05
    return ymin, ymax


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, facecolor="white", edgecolor="none")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white", edgecolor="none")


def _stage_boxplot(
    df: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    stage_ids = [s[0] for s in ABLATION_STAGES]
    labels = [s[1] for s in ABLATION_STAGES]
    centers = np.arange(len(stage_ids), dtype=float)
    data: List[np.ndarray] = []
    colors: List[str] = []

    for sid in stage_ids:
        vals = df.loc[df["stage_id"] == sid, value_col].dropna().values
        data.append(vals)
        colors.append(STAGE_COLORS.get(sid, "#888888"))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bp = ax.boxplot(
        data,
        positions=centers,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
    )
    for box, color in zip(bp["boxes"], colors):
        box.set(facecolor=color, alpha=0.78, edgecolor="#333333", linewidth=0.8)
    for whisker in bp["whiskers"]:
        whisker.set(color="#555555", linewidth=0.8)
    for cap in bp["caps"]:
        cap.set(color="#555555", linewidth=0.8)

    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plot_vals = df[value_col].to_numpy(dtype=float)
    ax.set_ylim(*_metric_ylim(plot_vals))
    handles = [Patch(facecolor=STAGE_COLORS.get(sid, "#888"), alpha=0.78, label=lab) for sid, lab in zip(stage_ids, labels)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def _per_class_metric_boxplot(df: pd.DataFrame, metric_prefix: str, metric_title: str, out_path: Path) -> None:
    stage_ids = [s[0] for s in ABLATION_STAGES]
    labels = [s[1] for s in ABLATION_STAGES]
    centers = np.arange(len(stage_ids), dtype=float)
    col_map = {
        "No regulation": f"pc_noreg_{metric_prefix}",
        "Activation": f"pc_activation_{metric_prefix}",
        "Repression": f"pc_repression_{metric_prefix}",
    }

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H), sharey=True)
    all_vals: List[float] = []
    for cls in CLASS_LABELS:
        col = col_map[cls]
        all_vals.extend(df[col].dropna().tolist())

    ylo, yhi = _metric_ylim(np.asarray(all_vals, dtype=float))
    for ax, cls in zip(axes, CLASS_LABELS):
        col = col_map[cls]
        data = [df.loc[df["stage_id"] == sid, col].dropna().values for sid in stage_ids]
        colors = [STAGE_COLORS.get(sid, "#888") for sid in stage_ids]
        bp = ax.boxplot(
            data,
            positions=centers,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.1},
        )
        for box, color in zip(bp["boxes"], colors):
            box.set(facecolor=color, alpha=0.78, edgecolor="#333333", linewidth=0.8)
        ax.set_title(cls)
        ax.set_xticks(centers)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_ylim(ylo, yhi)

    axes[0].set_ylabel(metric_title)
    fig.suptitle(f"GSE253718 ablation 10-fold CV (epithelial cells only): per-class {metric_title} (10 folds per modality)", y=1.02)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def _acc_precision_recall_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    metric_specs = (
        ("accuracy", "pc_noreg_accuracy", "pc_activation_accuracy", "pc_repression_accuracy", "Accuracy (OvR)"),
        ("precision", "pc_noreg_precision", "pc_activation_precision", "pc_repression_precision", "Precision"),
        ("recall", "pc_noreg_recall", "pc_activation_recall", "pc_repression_recall", "Recall"),
    )
    stage_ids = [s[0] for s in ABLATION_STAGES]
    labels = [s[1] for s in ABLATION_STAGES]
    centers = np.arange(len(stage_ids), dtype=float)

    fig, axes = plt.subplots(3, 3, figsize=(FIG_W, FIG_H), sharey="row")
    for row, (_, col_no, col_act, col_rep, row_title) in enumerate(metric_specs):
        row_vals = np.concatenate(
            [
                df[col_no].dropna().values,
                df[col_act].dropna().values,
                df[col_rep].dropna().values,
            ]
        )
        ylo, yhi = _metric_ylim(row_vals)
        class_cols = (col_no, col_act, col_rep)
        for col_i, cls in enumerate(CLASS_LABELS):
            ax = axes[row, col_i]
            col = class_cols[col_i]
            data = [df.loc[df["stage_id"] == sid, col].dropna().values for sid in stage_ids]
            colors = [STAGE_COLORS.get(sid, "#888") for sid in stage_ids]
            bp = ax.boxplot(
                data,
                positions=centers,
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.0},
            )
            for box, color in zip(bp["boxes"], colors):
                box.set(facecolor=color, alpha=0.78, edgecolor="#333333", linewidth=0.8)
            if row == 0:
                ax.set_title(cls)
            if col_i == 0:
                ax.set_ylabel(row_title)
            ax.set_xticks(centers)
            ax.set_xticklabels(labels if row == 2 else [], rotation=30, ha="right", fontsize=7)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.set_ylim(ylo, yhi)

    fig.suptitle("GSE253718 ablation 10-fold CV (epithelial cells only): per-class Accuracy / Precision / Recall", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize GSE253718 ablation 10-fold CV (epithelial cells only)")
    ap.add_argument("--experiment-name", default=DEFAULT_ABLATION_EXPERIMENT)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    layout = gse253718_layout()
    grn_root = layout["grn_root"]
    summary_dir = layout["summary_dir"]
    figures_dir = layout["figures_dir"]
    summary_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: List[Dict[str, Any]] = []
    stage_summaries: List[Dict[str, Any]] = []

    for stage_id, label, _flags in ABLATION_STAGES:
        stage_metrics: List[Dict[str, Any]] = []
        for fold_idx in range(1, 11):
            run_dir = ablation_stage_dir(stage_id, fold_idx, grn_root)
            m = _load_fold_metrics(run_dir)
            if m is None:
                print(f"[ablation-10cv-summary] missing {stage_id} fold {fold_idx:02d}")
                continue
            row = _flatten_fold_row(stage_id, label, fold_idx, m)
            fold_rows.append(row)
            stage_metrics.append(m)

        if not stage_metrics:
            continue
        agg: Dict[str, Any] = {"stage_id": stage_id, "modalities_label": label, "n_folds": len(stage_metrics)}
        for k in METRIC_KEYS:
            if k == "macro_auprc":
                vals = [_macro_auprc(x) for x in stage_metrics]
            else:
                vals = [float(x.get(k, np.nan)) for x in stage_metrics]
            agg[f"{k}_mean"] = float(np.nanmean(vals))
            agg[f"{k}_std"] = float(np.nanstd(vals))
        stage_summaries.append(agg)

    if not fold_rows:
        raise SystemExit("No fold metrics found; run run_gse253718_ablation_10cv first.")

    fold_df = pd.DataFrame(fold_rows)
    stage_df = pd.DataFrame(stage_summaries)
    fold_csv = summary_dir / "ablation_10cv_fold_metrics.csv"
    stage_csv = summary_dir / "ablation_10cv_stage_mean_std.csv"
    fold_df.to_csv(fold_csv, index=False, encoding="utf-8-sig")
    stage_df.to_csv(stage_csv, index=False, encoding="utf-8-sig")

    manifest = {
        "experiment": args.experiment_name,
        "n_fold_rows": len(fold_rows),
        "n_stages": len(stage_summaries),
        "fold_metrics_csv": str(fold_csv.resolve()),
        "stage_mean_std_csv": str(stage_csv.resolve()),
    }
    (summary_dir / "ablation_10cv_summary_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ablation-10cv-summary] fold rows -> {fold_csv}")
    print(f"[ablation-10cv-summary] stage mean±sd -> {stage_csv}")

    if args.no_plots:
        return

    _configure_matplotlib()
    figures_dir.mkdir(parents=True, exist_ok=True)

    _stage_boxplot(
        fold_df,
        value_col="macro_auc",
        title="GSE253718 epithelial ablation: test macro AUC (10-fold CV, unseen cells)",
        ylabel="Macro AUC",
        out_path=figures_dir / "ablation_10cv_macro_auc_boxplot.png",
    )
    _stage_boxplot(
        fold_df,
        value_col="accuracy",
        title="GSE253718 epithelial ablation: test accuracy (10-fold CV, unseen cells)",
        ylabel="Accuracy",
        out_path=figures_dir / "ablation_10cv_accuracy_boxplot.png",
    )
    _stage_boxplot(
        fold_df,
        value_col="f1_macro",
        title="GSE253718 epithelial ablation: test macro F1 (10-fold CV, unseen cells)",
        ylabel="Macro F1",
        out_path=figures_dir / "ablation_10cv_f1_macro_boxplot.png",
    )
    class_value_cols = {
        "auc_ovr": {
            "No regulation": "auc_ovr_NoReg",
            "Activation": "auc_ovr_Activation",
            "Repression": "auc_ovr_Repression",
        },
        "auprc": {
            "No regulation": "auprc_NoReg",
            "Activation": "auprc_Activation",
            "Repression": "auprc_Repression",
        },
    }
    for metric, fname in (
        ("auc_ovr", "ablation_10cv_auc_ovr_boxplot.png"),
        ("auprc", "ablation_10cv_auprc_boxplot.png"),
    ):
        long_rows: List[Dict[str, Any]] = []
        col_map = class_value_cols[metric]
        for _, r in fold_df.iterrows():
            for cls in CLASS_LABELS:
                long_rows.append(
                    {
                        "stage_id": r["stage_id"],
                        "modalities_label": r["modalities_label"],
                        "fold": r["fold"],
                        "class": cls,
                        "value": r.get(col_map[cls], np.nan),
                    }
                )
        long_df = pd.DataFrame(long_rows)
        stage_ids = [s[0] for s in ABLATION_STAGES]
        labels = [s[1] for s in ABLATION_STAGES]
        centers = np.arange(len(stage_ids), dtype=float)
        fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H), sharey=True)
        ylo, yhi = _metric_ylim(long_df["value"].to_numpy(dtype=float))
        for ax, cls in zip(axes, CLASS_LABELS):
            sub = long_df[long_df["class"] == cls]
            data = [sub.loc[sub["stage_id"] == sid, "value"].dropna().values for sid in stage_ids]
            colors = [STAGE_COLORS.get(sid, "#888") for sid in stage_ids]
            bp = ax.boxplot(
                data,
                positions=centers,
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.1},
            )
            for box, color in zip(bp["boxes"], colors):
                box.set(facecolor=color, alpha=0.78, edgecolor="#333333", linewidth=0.8)
            ax.set_title(cls)
            ax.set_xticks(centers)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.set_ylim(ylo, yhi)
        metric_title = "OvR AUC" if metric == "auc_ovr" else "AUPRC"
        axes[0].set_ylabel(metric_title)
        fig.suptitle(f"GSE253718 ablation 10-fold CV (epithelial cells only): per-class {metric_title}", y=1.02)
        fig.tight_layout()
        _save_figure(fig, figures_dir / fname)
        plt.close(fig)

    _acc_precision_recall_boxplot(fold_df, figures_dir / "ablation_10cv_acc_precision_recall_boxplot.png")
    print(f"[ablation-10cv-summary] figures -> {figures_dir}")


if __name__ == "__main__":
    main()
