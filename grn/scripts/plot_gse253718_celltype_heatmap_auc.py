#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSE253718 cell-type TripGRN figures (PDF), ablation-style heatmaps.

Outputs under experiments/gse253718_celltype_bytype_seed42/figures/
ROC optional (--folds); default --skip-roc for speed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from grn.data.gse253718_celltype_paths import (
    DEFAULT_CELLTYPE_EXPERIMENT,
    gse253718_celltype_layout,
    resolve_type_ids,
)
from grn.scripts.plot_gse131907_celltype_heatmap_auc import (
    METRIC_COLS,
    METRIC_SRC,
    draw_heatmap_pdf,
    _write_heatmap_pair,
)
from grn.scripts.plot_gse253718_ablation_heatmap_auc import _zscore_column, _zscore_global


def _load_core_matrix(summary_csv: Path, types: Sequence[Tuple[str, str]]) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    df = pd.read_csv(summary_csv)
    rows: List[List[float]] = []
    labels: List[str] = []
    for tid, lab in types:
        sub = df.loc[df["type_id"] == tid]
        if sub.empty:
            raise SystemExit(f"Missing type {tid!r} in {summary_csv}")
        r = sub.iloc[0]
        vals = []
        for c in METRIC_SRC:
            col = f"{c}_mean"
            if col not in r.index:
                # macro_auprc may be absent in older summaries
                vals.append(float("nan"))
            else:
                vals.append(float(r[col]))
        rows.append(vals)
        labels.append(lab)
    mat = np.asarray(rows, dtype=float)
    # Drop all-NaN columns
    keep = [j for j in range(mat.shape[1]) if np.isfinite(mat[:, j]).any()]
    cols = [METRIC_COLS[j] for j in keep]
    mat = mat[:, keep]
    table = pd.DataFrame(mat, index=labels, columns=cols)
    return table, mat, labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-name", default=DEFAULT_CELLTYPE_EXPERIMENT)
    ap.add_argument("--skip-roc", action="store_true", default=True)
    args = ap.parse_args()

    layout = gse253718_celltype_layout()
    exp_base = layout["grn_root"] / "experiments" / args.experiment_name
    summary_csv = exp_base / "summary" / "tripgrn_10cv_type_summary.csv"
    fold_csv = exp_base / "summary" / "tripgrn_10cv_per_fold_metrics.csv"
    figures_dir = exp_base / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not summary_csv.is_file():
        raise SystemExit(f"Missing {summary_csv}; run summarize first")

    types = resolve_type_ids("ALL")
    short_rows = [lab if len(lab) <= 14 else lab[:12] + "…" for _, lab in types]

    table, raw, full_labels = _load_core_matrix(summary_csv, types)
    table.to_csv(figures_dir / "celltype_heatmap_raw_metrics.csv", encoding="utf-8-sig")
    z_col = _zscore_column(raw)
    z_glb = _zscore_global(raw)
    pd.DataFrame(z_col, index=full_labels, columns=list(table.columns)).to_csv(
        figures_dir / "celltype_heatmap_zscore_column.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(z_glb, index=full_labels, columns=list(table.columns)).to_csv(
        figures_dir / "celltype_heatmap_zscore_global.csv", encoding="utf-8-sig"
    )
    draw_heatmap_pdf(
        z_col,
        raw,
        title="GSE253718 cell-type metrics (column z-score)",
        out_pdf=figures_dir / "celltype_metrics_heatmap_zscore_column.pdf",
        cbar_label="Column z-score",
        row_labels=short_rows,
        col_labels=list(table.columns),
    )
    draw_heatmap_pdf(
        z_glb,
        raw,
        title="GSE253718 cell-type metrics (global z-score)",
        out_pdf=figures_dir / "celltype_metrics_heatmap_zscore_global.pdf",
        cbar_label="Global z-score",
        row_labels=short_rows,
        col_labels=list(table.columns),
    )

    if fold_csv.is_file():
        df = pd.read_csv(fold_csv)
        extra_cols = tuple(
            c
            for c in (
                "auc_ovo",
                "NoReg_accuracy",
                "Activate_accuracy",
                "Repress_accuracy",
                "NoReg_precision",
                "Activate_precision",
                "Repress_precision",
                "NoReg_recall",
                "Activate_recall",
                "Repress_recall",
            )
            if c in df.columns
        )
        if extra_cols:
            rows = []
            for tid, lab in types:
                sub = df.loc[df["type_id"] == tid]
                rows.append([float(sub[c].mean()) for c in extra_cols])
            raw_extra = np.asarray(rows, dtype=float)
            _write_heatmap_pair(
                raw_extra,
                row_labels=short_rows,
                col_labels=extra_cols,
                figures_dir=figures_dir,
                stem="celltype_extra_metrics",
                title_prefix="GSE253718 cell-type extra metrics",
            )

    print(f"[plot-gse253718-celltype] done -> {figures_dir} (ROC skipped by default)")


if __name__ == "__main__":
    main()
