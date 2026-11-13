#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step01 pancancer epithelial 10CV: method comparison boxplots (binary + 3-class)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from grn.data.pancancer_paths import PANCANCER_GSE_IDS, grn_root_from_here, scRNA_layout
from grn.evaluation.training_plots import per_class_ovr_accuracy_precision_recall_from_cm

CLASS_LABELS = ("No regulation", "Activation", "Repression")

METHODS_BINARY = ("TripGRN", "DeepFGRN", "DeepTGI", "scMGATGRN", "GENELink")
METHODS_3CLASS = ("TripGRN", "DeepFGRN")

# Display palette for boxplots (TripGRN / GENELink unchanged; others rotated per spec)
METHOD_COLORS_BINARY = {
    "TripGRN": "#C44E52",    # red (unchanged)
    "DeepFGRN": "#4C72B0",   # blue (was scMGATGRN)
    "DeepTGI": "#DD8452",    # orange (was DeepFGRN)
    "scMGATGRN": "#55A868",  # green (was DeepTGI)
    "GENELink": "#8172B3",   # purple (unchanged)
}

METHOD_COLORS_3CLASS = {
    "TripGRN": "#DD8452",    # orange (acc/precision/recall plot)
    "DeepFGRN": "#4C72B0",   # blue
}

# AUC / AUPRC 3-class plots: TripGRN & DeepFGRN match binary_auc_boxplot
METHOD_COLORS_3CLASS_AUC_AUPRC = {
    "TripGRN": METHOD_COLORS_BINARY["TripGRN"],
    "DeepFGRN": METHOD_COLORS_BINARY["DeepFGRN"],
}

# Unified figure size (width x height, inches) for all exported boxplots
FIG_W = 14.0
FIG_H = 7.0
FIG_DPI = 180

TRIP_AUC_IDX = {"No regulation": 0, "Activation": 1, "Repression": 2}
TRIP_CM_IDX = {"No regulation": 0, "Activation": 2, "Repression": 1}

FGRN_AUC_IDX = {"No regulation": 0, "Activation": 1, "Repression": 2}
FGRN_CM_IDX = {"No regulation": 0, "Activation": 2, "Repression": 1}


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False


def _parse_cm(raw: Any) -> np.ndarray:
    if isinstance(raw, str):
        return np.asarray(json.loads(raw), dtype=np.float64)
    return np.asarray(raw, dtype=np.float64)


def _collapse_cm_to_binary(cm: np.ndarray) -> np.ndarray:
    cm = np.asarray(cm, dtype=np.float64)
    tn = cm[0, 0]
    fp = cm[0, 1:].sum()
    fn = cm[1:, 0].sum()
    tp = cm[1:, 1:].sum()
    return np.array([[tn, fp], [fn, tp]], dtype=np.float64)


def _regulated_ovr_mean(values: Sequence[float], pos_indices: Sequence[int]) -> float:
    vals = [float(values[i]) for i in pos_indices if i < len(values)]
    return float(np.nanmean(vals)) if vals else float("nan")


def _binary_auc_auprc_from_3class_json(j: Dict[str, Any], method: str) -> Tuple[float, float]:
    """Strict binary ranking for Trip/Deep: regulated vs not.

    Score = 1 - P(NoReg). ROC-AUC equals NoReg OvR AUC. AUPRC is not recoverable
    from Act/Rep OvR AUPRC alone; return NaN for AUPRC here (filled via checkpoint cache).
    """
    auc_ovr = list(j.get("auc_ovr") or [])
    # Display order: NoReg, Activation, Repression
    auc = float(auc_ovr[0]) if len(auc_ovr) >= 1 else float("nan")
    return auc, float("nan")


def _per_class_from_3class_json(j: Dict[str, Any], method: str) -> Dict[str, Dict[str, float]]:
    cm = _parse_cm(j["confusion_matrix"])
    acc, prec, rec = per_class_ovr_accuracy_precision_recall_from_cm(cm)
    auc_ovr = list(j.get("auc_ovr") or [])
    auprc = list(j.get("auprc") or [])
    idx_map = TRIP_CM_IDX if method == "TripGRN" else FGRN_CM_IDX
    auc_map = TRIP_AUC_IDX if method == "TripGRN" else FGRN_AUC_IDX
    out: Dict[str, Dict[str, float]] = {}
    for cls in CLASS_LABELS:
        ci = idx_map[cls]
        ai = auc_map[cls]
        out[cls] = {
            "accuracy": float(acc[ci]),
            "precision": float(prec[ci]),
            "recall": float(rec[ci]),
            "auc": float(auc_ovr[ai]) if ai < len(auc_ovr) else float("nan"),
            "auprc": float(auprc[ai]) if ai < len(auprc) else float("nan"),
        }
    return out


def _load_binary_fold_table(summary_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    binary_files = {
        "DeepTGI": "deeptgi_10cv_fold_metrics.csv",
        "scMGATGRN": "scmgatgrn_10cv_fold_metrics.csv",
        "GENELink": "genelink_10cv_fold_metrics.csv",
    }
    for method, fname in binary_files.items():
        path = summary_dir / fname
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append(
                {
                    "gse_id": r["gse_id"],
                    "method": method,
                    "fold": int(r["fold"]),
                    "auc": float(r["macro_auc"]),
                    "auprc": float(r["auprc"]),
                }
            )

    for method, fname in (("TripGRN", "tripgrn_10cv_fold_metrics.csv"), ("DeepFGRN", "deepfgrn_10cv_fold_metrics.csv")):
        path = summary_dir / fname
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            cm = _parse_cm(r["confusion_matrix"])
            _collapse_cm_to_binary(cm)
            j = {
                "auc_ovr": [
                    r.get("auc_ovr_无调控", np.nan),
                    r.get("auc_ovr_激活", np.nan),
                    r.get("auc_ovr_抑制", np.nan),
                ],
                "auprc": [
                    r.get("auprc_无调控", np.nan),
                    r.get("auprc_激活", np.nan),
                    r.get("auprc_抑制", np.nan),
                ],
                "confusion_matrix": cm.tolist(),
            }
            auc, auprc = _binary_auc_auprc_from_3class_json(j, method)
            rows.append(
                {
                    "gse_id": r["gse_id"],
                    "method": method,
                    "fold": int(r["fold"]),
                    "auc": auc,
                    "auprc": auprc,
                }
            )

    if not rows:
        raise FileNotFoundError(f"No fold metrics found under {summary_dir}")
    return pd.DataFrame(rows)


def _load_3class_fold_table(summary_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for method, fname in (("TripGRN", "tripgrn_10cv_fold_metrics.csv"), ("DeepFGRN", "deepfgrn_10cv_fold_metrics.csv")):
        path = summary_dir / fname
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            j = {
                "confusion_matrix": _parse_cm(r["confusion_matrix"]).tolist(),
                "auc_ovr": [
                    r.get("auc_ovr_无调控", np.nan),
                    r.get("auc_ovr_激活", np.nan),
                    r.get("auc_ovr_抑制", np.nan),
                ],
                "auprc": [
                    r.get("auprc_无调控", np.nan),
                    r.get("auprc_激活", np.nan),
                    r.get("auprc_抑制", np.nan),
                ],
            }
            per_cls = _per_class_from_3class_json(j, method)
            for cls in CLASS_LABELS:
                rows.append(
                    {
                        "gse_id": r["gse_id"],
                        "method": method,
                        "fold": int(r["fold"]),
                        "class": cls,
                        "accuracy": per_cls[cls]["accuracy"],
                        "precision": per_cls[cls]["precision"],
                        "recall": per_cls[cls]["recall"],
                        "auc": per_cls[cls]["auc"],
                        "auprc": per_cls[cls]["auprc"],
                    }
                )
    if not rows:
        raise FileNotFoundError("Missing TripGRN/DeepFGRN fold metrics for 3-class plots")
    return pd.DataFrame(rows)


def _metric_ylim(
    values: np.ndarray,
    *,
    bottom_pad: float = 0.1,
    top_pad: float = 0.02,
    ymax_cap: float = 1.05,
) -> Tuple[float, float]:
    """Y-axis from (min - bottom_pad) to min(max + top_pad, ymax_cap)."""
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
    fig.savefig(out_path, dpi=FIG_DPI, facecolor="white", edgecolor="none", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white", edgecolor="none", bbox_inches="tight")


def _safe_layout(fig: plt.Figure, *, bottom: float = 0.18, top: float = 0.92, left: float = 0.07, right: float = 0.99) -> None:
    """Avoid fig.tight_layout(): it can hard-crash on some Windows/matplotlib builds."""
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


def _grouped_boxplot(
    df: pd.DataFrame,
    *,
    value_col: str,
    methods: Sequence[str],
    gse_ids: Sequence[str],
    title: str,
    ylabel: str,
    out_path: Path,
    method_colors: Dict[str, str],
) -> None:
    n_gse = len(gse_ids)
    n_methods = len(methods)
    group_width = 0.78
    box_width = group_width / max(n_methods, 1)
    centers = np.arange(n_gse, dtype=float)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    legend_handles: List[Patch] = []

    for mi, method in enumerate(methods):
        offsets = centers - group_width / 2 + box_width / 2 + mi * box_width
        data: List[np.ndarray] = []
        positions: List[float] = []
        for gi, gse in enumerate(gse_ids):
            vals = df.loc[(df["gse_id"] == gse) & (df["method"] == method), value_col].dropna().values
            if vals.size == 0:
                continue
            data.append(vals)
            positions.append(float(offsets[gi]))
        if data:
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=box_width * 0.82,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.2},
            )
            color = method_colors.get(method, "#888888")
            for box in bp["boxes"]:
                box.set(facecolor=color, alpha=0.75, edgecolor="#333333", linewidth=0.8)
            for whisker in bp["whiskers"]:
                whisker.set(color="#555555", linewidth=0.8)
            for cap in bp["caps"]:
                cap.set(color="#555555", linewidth=0.8)
        legend_handles.append(Patch(facecolor=method_colors.get(method, "#888888"), alpha=0.75, label=method))

    ax.set_xticks(centers)
    ax.set_xticklabels(gse_ids, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plot_vals = df.loc[df["method"].isin(methods) & df["gse_id"].isin(gse_ids), value_col].to_numpy(dtype=float)
    ax.set_ylim(*_metric_ylim(plot_vals))
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=min(len(methods), 5), frameon=False)
    _safe_layout(fig, bottom=0.22)
    _save_figure(fig, out_path)
    plt.close(fig)


def _faceted_class_method_boxplot(
    df: pd.DataFrame,
    *,
    value_col: str,
    title_prefix: str,
    ylabel: str,
    out_path: Path,
    method_colors: Dict[str, str],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H), sharey=True)
    gse_ids = list(PANCANCER_GSE_IDS)
    n_gse = len(gse_ids)
    group_width = 0.55
    box_width = group_width / 2
    centers = np.arange(n_gse, dtype=float)
    ylo, yhi = _metric_ylim(df[value_col].to_numpy(dtype=float))

    for ax, cls in zip(axes, CLASS_LABELS):
        sub = df[df["class"] == cls]
        for mi, method in enumerate(METHODS_3CLASS):
            offsets = centers - group_width / 2 + box_width / 2 + mi * box_width
            data: List[np.ndarray] = []
            positions: List[float] = []
            for gi, gse in enumerate(gse_ids):
                vals = sub.loc[(sub["gse_id"] == gse) & (sub["method"] == method), value_col].dropna().values
                if vals.size == 0:
                    continue
                data.append(vals)
                positions.append(float(offsets[gi]))
            if not data:
                continue
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=box_width * 0.82,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.1},
            )
            color = method_colors[method]
            for box in bp["boxes"]:
                box.set(facecolor=color, alpha=0.75, edgecolor="#333333", linewidth=0.8)
        ax.set_title(cls)
        ax.set_xticks(centers)
        ax.set_xticklabels(gse_ids, rotation=40, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_ylim(ylo, yhi)

    axes[0].set_ylabel(ylabel)
    handles = [Patch(facecolor=method_colors[m], alpha=0.75, label=m) for m in METHODS_3CLASS]
    fig.suptitle(title_prefix, y=0.98, fontsize=13)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    _safe_layout(fig, bottom=0.20, top=0.88)
    _save_figure(fig, out_path)
    plt.close(fig)


def _faceted_prf_by_class_boxplot(df: pd.DataFrame, out_path: Path, method_colors: Dict[str, str]) -> None:
    metric_specs = (("accuracy", "Accuracy (OvR)"), ("precision", "Precision"), ("recall", "Recall"))
    fig, axes = plt.subplots(3, 3, figsize=(FIG_W, FIG_H), sharey="row")
    gse_ids = list(PANCANCER_GSE_IDS)
    centers = np.arange(len(gse_ids), dtype=float)
    group_width = 0.55
    box_width = group_width / 2

    for row, (col, row_title) in enumerate(metric_specs):
        ylo, yhi = _metric_ylim(df[col].to_numpy(dtype=float))
        for col_i, cls in enumerate(CLASS_LABELS):
            ax = axes[row, col_i]
            sub = df[df["class"] == cls]
            for mi, method in enumerate(METHODS_3CLASS):
                offsets = centers - group_width / 2 + box_width / 2 + mi * box_width
                data: List[np.ndarray] = []
                positions: List[float] = []
                for gi, gse in enumerate(gse_ids):
                    vals = sub.loc[(sub["gse_id"] == gse) & (sub["method"] == method), col].dropna().values
                    if vals.size == 0:
                        continue
                    data.append(vals)
                    positions.append(float(offsets[gi]))
                if not data:
                    continue
                bp = ax.boxplot(
                    data,
                    positions=positions,
                    widths=box_width * 0.82,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.0},
                )
                for box in bp["boxes"]:
                    box.set(facecolor=method_colors[method], alpha=0.75, edgecolor="#333333", linewidth=0.8)
            if row == 0:
                ax.set_title(cls)
            if col_i == 0:
                ax.set_ylabel(row_title)
            ax.set_xticks(centers)
            ax.set_xticklabels(gse_ids if row == 2 else [], rotation=40, ha="right", fontsize=7)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.set_ylim(ylo, yhi)

    handles = [Patch(facecolor=method_colors[m], alpha=0.75, label=m) for m in METHODS_3CLASS]
    fig.suptitle("3-class per-class comparison: Accuracy / Precision / Recall (10 folds per GSE)", y=0.995)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)
    _safe_layout(fig, bottom=0.08, top=0.92)
    _save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot step01 pancancer method comparison boxplots")
    ap.add_argument("--grn-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    _configure_matplotlib()
    grn_root = Path(args.grn_root or grn_root_from_here()).resolve()
    summary_dir = scRNA_layout(grn_root)["experiments_summary"]
    out_dir = Path(args.out_dir or (grn_root / "experiments" / "step01_pancancer_epithelial_10cv" / "figures"))
    out_dir.mkdir(parents=True, exist_ok=True)

    binary_df = _load_binary_fold_table(summary_dir)
    three_df = _load_3class_fold_table(summary_dir)

    binary_df.to_csv(out_dir / "plot_binary_fold_long.csv", index=False, encoding="utf-8-sig")
    three_df.to_csv(out_dir / "plot_3class_fold_long.csv", index=False, encoding="utf-8-sig")

    gse_ids = list(PANCANCER_GSE_IDS)

    _grouped_boxplot(
        binary_df,
        value_col="auc",
        methods=METHODS_BINARY,
        gse_ids=gse_ids,
        title="Binary comparison: AUC (10-fold CV, unseen cells)\n"
        "TripGRN/DeepFGRN: regulated = activation + repression (OvR mean)",
        ylabel="AUC",
        out_path=out_dir / "binary_auc_boxplot.png",
        method_colors=METHOD_COLORS_BINARY,
    )
    _grouped_boxplot(
        binary_df,
        value_col="auprc",
        methods=METHODS_BINARY,
        gse_ids=gse_ids,
        title="Binary comparison: AUPRC (10-fold CV, unseen cells)\n"
        "TripGRN/DeepFGRN: regulated = activation + repression (OvR mean)",
        ylabel="AUPRC",
        out_path=out_dir / "binary_auprc_boxplot.png",
        method_colors=METHOD_COLORS_BINARY,
    )

    _faceted_class_method_boxplot(
        three_df,
        value_col="auc",
        title_prefix="3-class comparison: OvR AUC by class (TripGRN vs DeepFGRN)",
        ylabel="AUC",
        out_path=out_dir / "threeclass_auc_boxplot.png",
        method_colors=METHOD_COLORS_3CLASS_AUC_AUPRC,
    )
    _faceted_class_method_boxplot(
        three_df,
        value_col="auprc",
        title_prefix="3-class comparison: OvR AUPRC by class (TripGRN vs DeepFGRN)",
        ylabel="AUPRC",
        out_path=out_dir / "threeclass_auprc_boxplot.png",
        method_colors=METHOD_COLORS_3CLASS_AUC_AUPRC,
    )

    _faceted_prf_by_class_boxplot(three_df, out_dir / "threeclass_acc_precision_recall_boxplot.png", METHOD_COLORS_3CLASS)

    print(f"[plot-step01] wrote figures -> {out_dir}")


if __name__ == "__main__":
    main()
