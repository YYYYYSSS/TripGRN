#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate GSE253718 cell-type TripGRN 10CV metrics + boxplots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from grn.data.gse253718_celltype_paths import (
    DEFAULT_CELLTYPE_EXPERIMENT,
    gse253718_celltype_layout,
    resolve_type_ids,
)
from grn.evaluation.training_plots import per_class_ovr_accuracy_precision_recall_from_cm

METRIC_KEYS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "mcc",
    "macro_auc",
    "auc_ovo",
    "macro_auprc",
]

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
    "#4C78A8",
)


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False


def _load_fold_metrics(run_dir: Path) -> Dict[str, Any] | None:
    p = run_dir / "unseen_samples_test_metrics.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _flatten_row(type_id: str, label: str, fold_idx: int, m: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"type_id": type_id, "type_label": label, "fold": fold_idx}
    for k in METRIC_KEYS:
        if k == "macro_auprc":
            vals = m.get("auprc")
            row[k] = (
                float(np.nanmean(np.asarray(vals, dtype=float)))
                if isinstance(vals, list) and vals
                else float("nan")
            )
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


def _boxplot_metric(df: pd.DataFrame, types: List[tuple], metric: str, out_path: Path, *, ylabel: str) -> None:
    _configure_matplotlib()
    type_ids = [t[0] for t in types]
    labels = [t[1] for t in types]
    colors = {tid: _PALETTE[i % len(_PALETTE)] for i, tid in enumerate(type_ids)}
    data = [df.loc[df["type_id"] == tid, metric].dropna().values for tid in type_ids]
    fig, ax = plt.subplots(figsize=(max(10.0, 1.2 * len(types)), 6.5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
    for patch, tid in zip(bp["boxes"], type_ids):
        patch.set_facecolor(colors[tid])
        patch.set_alpha(0.85)
    ax.set_ylabel(ylabel)
    ax.set_title(f"GSE253718 cell-type TripGRN 10CV — {ylabel}")
    ax.tick_params(axis="x", labelrotation=25)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    ax.grid(axis="y", alpha=0.25)
    legend_patches = [Patch(facecolor=colors[tid], label=lab, alpha=0.85) for tid, lab in zip(type_ids, labels)]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize GSE253718 cell-type TripGRN 10CV")
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--experiment-name", default=DEFAULT_CELLTYPE_EXPERIMENT)
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    layout = gse253718_celltype_layout()
    if args.experiment_name != DEFAULT_CELLTYPE_EXPERIMENT:
        layout["summary_dir"] = layout["grn_root"] / "experiments" / args.experiment_name / "summary"
        layout["figures_dir"] = layout["grn_root"] / "experiments" / args.experiment_name / "figures"
        layout["cv10_base"] = layout["grn_root"] / "experiments" / args.experiment_name / "tripgrn_10cv"

    summary_dir = layout["summary_dir"]
    figures_dir = layout["figures_dir"]
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    types = resolve_type_ids("ALL")
    rows: List[Dict[str, Any]] = []
    for tid, label in types:
        for fold_idx in range(1, int(args.n_folds) + 1):
            run_dir = layout["cv10_base"] / tid / f"fold_{fold_idx:02d}"
            m = _load_fold_metrics(run_dir)
            if m is None:
                continue
            rows.append(_flatten_row(tid, label, fold_idx, m))

    if not rows:
        print("[gse253718-celltype-summary] no completed folds found")
        return

    df = pd.DataFrame(rows)
    per_fold_csv = summary_dir / "tripgrn_10cv_per_fold_metrics.csv"
    df.to_csv(per_fold_csv, index=False, encoding="utf-8-sig")
    print(f"[gse253718-celltype-summary] per-fold -> {per_fold_csv} n={len(df)}", flush=True)

    agg_rows: List[Dict[str, Any]] = []
    for tid, label in types:
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
    print(f"[gse253718-celltype-summary] aggregate -> {agg_csv}", flush=True)

    if not args.skip_plots:
        for metric, ylabel in (
            ("accuracy", "Accuracy"),
            ("macro_auc", "Macro AUC"),
            ("f1_macro", "Macro F1"),
            ("mcc", "MCC"),
        ):
            out_png = figures_dir / f"tripgrn_10cv_{metric}_boxplot.png"
            try:
                _boxplot_metric(df, types, metric, out_png, ylabel=ylabel)
                print(f"[gse253718-celltype-summary] figure -> {out_png}", flush=True)
            except Exception as ex:  # noqa: BLE001
                print(f"[gse253718-celltype-summary] plot failed {metric}: {ex}", flush=True)


if __name__ == "__main__":
    main()
