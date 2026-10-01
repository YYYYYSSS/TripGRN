#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate GSE131907 step06 TripGRN 10CV metrics + boxplots; export val CMs if missing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from grn.data.gse131907_celltype_paths import (
    CELL_TYPES,
    DEFAULT_STEP06_EXPERIMENT,
    gse131907_celltype_layout,
    tripgrn_10cv_fold_dir,
)
from grn.evaluation.confusion_matrix_io import export_confusion_matrix_artifacts
from grn.evaluation.training_plots import per_class_ovr_accuracy_precision_recall_from_cm

METRIC_KEYS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
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

# Distinct palette for per-Cell_type boxplots (fallback gray if unknown).
_PALETTE = (
    "#C44E52",
    "#4C72B0",
    "#55A868",
    "#8172B3",
    "#CCB974",
    "#64B5CD",
    "#E17C71",
    "#8C8C8C",
    "#DA8BC3",
)
TYPE_COLORS = {tid: _PALETTE[i % len(_PALETTE)] for i, (tid, _) in enumerate(CELL_TYPES)}

FIG_W, FIG_H, FIG_DPI = 14.0, 6.5, 180


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False


def _ensure_val_cm(run_dir: Path, *, type_id: str, fold_idx: int) -> None:
    if (run_dir / "confusion_matrix_val.json").is_file():
        return
    p = run_dir / "unseen_samples_val_metrics.json"
    if not p.is_file():
        return
    mj = json.loads(p.read_text(encoding="utf-8"))
    cm = mj.get("confusion_matrix")
    if cm is None:
        return
    export_confusion_matrix_artifacts(
        run_dir,
        np.asarray(cm, dtype=np.int64),
        stem="confusion_matrix_val",
        title=f"TripGRN val confusion matrix ({type_id} fold {fold_idx})",
        class_names=["No regulation", "Activation", "Repression"],
        extra={"method": "TripGRN", "cell_type": type_id, "fold_idx": fold_idx, "split": "val"},
    )


def _load_fold_metrics(run_dir: Path) -> Dict[str, Any] | None:
    p = run_dir / "unseen_samples_test_metrics.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _flatten_row(type_id: str, label: str, fold_idx: int, m: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"type_id": type_id, "type_label": label, "fold": fold_idx}
    for k in METRIC_KEYS:
        if k == "macro_auprc":
            row[k] = _macro_auprc(m)
        else:
            v = m.get(k)
            row[k] = float(v) if v is not None else float("nan")
    cm = m.get("confusion_matrix")
    if isinstance(cm, list) and len(cm) == 3:
        acc, prec, rec = per_class_ovr_accuracy_precision_recall_from_cm(np.asarray(cm, dtype=np.int64))
        for i, cls_name in enumerate(("NoReg", "Activate", "Repress")):
            row[f"{cls_name}_accuracy"] = float(acc[i])
            row[f"{cls_name}_precision"] = float(prec[i])
            row[f"{cls_name}_recall"] = float(rec[i])
    return row


def _boxplot_metric(df: pd.DataFrame, metric: str, out_path: Path, *, ylabel: str) -> None:
    _configure_matplotlib()
    types = [t[0] for t in CELL_TYPES]
    labels = [t[1] for t in CELL_TYPES]
    data = [df.loc[df["type_id"] == tid, metric].dropna().values for tid in types]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
    for patch, tid in zip(bp["boxes"], types):
        patch.set_facecolor(TYPE_COLORS.get(tid, "#999999"))
        patch.set_alpha(0.85)
    ax.set_ylabel(ylabel)
    ax.set_title(f"TripGRN 10CV — {ylabel}")
    ax.tick_params(axis="x", labelrotation=25)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    ax.grid(axis="y", alpha=0.25)
    legend_patches = [
        Patch(facecolor=TYPE_COLORS.get(tid, "#999999"), label=lab, alpha=0.85)
        for tid, lab in zip(types, labels)
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize GSE131907 step06 TripGRN 10CV")
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--experiment-name", default=DEFAULT_STEP06_EXPERIMENT)
    args = ap.parse_args()

    layout = gse131907_celltype_layout()
    summary_dir = layout["summary_dir"]
    figures_dir = layout["figures_dir"]
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for tid, label in CELL_TYPES:
        for fold_idx in range(1, int(args.n_folds) + 1):
            run_dir = tripgrn_10cv_fold_dir(tid, fold_idx, layout["grn_root"])
            _ensure_val_cm(run_dir, type_id=tid, fold_idx=fold_idx)
            m = _load_fold_metrics(run_dir)
            if m is None:
                continue
            rows.append(_flatten_row(tid, label, fold_idx, m))

    if not rows:
        print("[celltype-summary] no completed folds found")
        return

    df = pd.DataFrame(rows)
    per_fold_csv = summary_dir / "tripgrn_10cv_per_fold_metrics.csv"
    df.to_csv(per_fold_csv, index=False, encoding="utf-8-sig")

    agg_rows: List[Dict[str, Any]] = []
    for tid, label in CELL_TYPES:
        sub = df[df["type_id"] == tid]
        if sub.empty:
            continue
        row: Dict[str, Any] = {"type_id": tid, "type_label": label, "n_folds": int(len(sub))}
        for k in METRIC_KEYS:
            row[f"{k}_mean"] = float(sub[k].mean())
            row[f"{k}_std"] = float(sub[k].std(ddof=0))
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)
    agg_csv = summary_dir / "tripgrn_10cv_type_summary.csv"
    agg_df.to_csv(agg_csv, index=False, encoding="utf-8-sig")

    for metric, ylabel in (
        ("accuracy", "Accuracy"),
        ("macro_auc", "Macro AUC"),
        ("f1_macro", "Macro F1"),
        ("mcc", "MCC"),
    ):
        out_png = figures_dir / f"tripgrn_10cv_{metric}_boxplot.png"
        _boxplot_metric(df, metric, out_png, ylabel=ylabel)
        print(f"[celltype-summary] figure -> {out_png}")

    print(f"[celltype-summary] per-fold -> {per_fold_csv}")
    print(f"[celltype-summary] aggregate -> {agg_csv}")


if __name__ == "__main__":
    main()
