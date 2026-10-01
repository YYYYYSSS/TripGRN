#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSE131907 step06 by-cell-type TripGRN figures (PDF via reportlab),
matching ablation plot style:

1) Metrics heatmap (rows: cell types; cols: macro_auc, macro_auprc..mcc)
   - column-wise z-score
   - global z-score
2) Extra-metrics heatmap
3) Macro-average OvR ROC curves (mean±band over folds)

Outputs under experiments/gse131907_celltype_step06_bytype_seed42/figures/
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from reportlab.lib.colors import HexColor, black, white
from reportlab.pdfgen import canvas

from grn.data.gse131907_celltype_paths import (
    CELL_TYPES,
    DEFAULT_STEP06_EXPERIMENT,
    gse131907_celltype_layout,
    tripgrn_10cv_fold_dir,
)
from grn.data.pancancer_cv import get_fold_split, load_cv_splits
from grn.reproducibility import set_seed
from grn.scripts.plot_gse253718_ablation_heatmap_auc import (
    FPR_GRID,
    _color_limits_global,
    _draw_colorbar_ylorrd,
    _hex_alpha,
    _macro_roc_from_probs,
    _ylorrd_color,
    _zscore_column,
    _zscore_global,
)
from grn.scripts.regenerate_unseen_run_plots_from_checkpoint import _build_pipeline
from grn.scripts.train_unseen_samples_no_cv import _predict_arrays

# Core metrics (align ablation column names for display)
# X-axis order: AUC, PRC, MCC, accuracy, precision, recall, f1
METRIC_COLS = ("macro_auc", "macro_auprc", "mcc", "accuracy", "precision", "recall", "f1")
# Source columns in tripgrn_10cv_* CSVs (same order as METRIC_COLS)
METRIC_SRC = (
    "macro_auc",
    "macro_auprc",
    "mcc",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
)

METRIC_COLS_EXTRA = (
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

# Short row labels for heatmap (full names used in ROC legend)
ROW_LABELS_SHORT: Tuple[str, ...] = (
    "Epithelial",
    "T lymph.",
    "B lymph.",
    "NK",
    "Myeloid",
    "Fibroblast",
    "Endothelial",
    "MAST",
    "Oligodend.",
)

TYPE_COLORS: Dict[str, str] = {
    "epithelial_cells": "#C44E52",
    "t_lymphocytes": "#4C72B0",
    "b_lymphocytes": "#55A868",
    "nk_cells": "#8172B3",
    "myeloid_cells": "#CCB974",
    "fibroblasts": "#64B5CD",
    "endothelial_cells": "#E17C71",
    "mast_cells": "#8C8C8C",
    "oligodendrocytes": "#DA8BC3",
}


def _type_order() -> List[Tuple[str, str]]:
    return list(CELL_TYPES)


def _load_core_matrix(summary_csv: Path) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    df = pd.read_csv(summary_csv)
    rows: List[List[float]] = []
    labels: List[str] = []
    for tid, lab in _type_order():
        sub = df.loc[df["type_id"] == tid]
        if sub.empty:
            raise SystemExit(f"Missing type {tid!r} in {summary_csv}")
        r = sub.iloc[0]
        rows.append([float(r[f"{c}_mean"]) for c in METRIC_SRC])
        labels.append(lab)
    mat = np.asarray(rows, dtype=float)
    table = pd.DataFrame(mat, index=labels, columns=list(METRIC_COLS))
    return table, mat, labels


def _load_extra_matrix(fold_csv: Path) -> Tuple[pd.DataFrame, np.ndarray, Tuple[str, ...], List[str]]:
    df = pd.read_csv(fold_csv)
    cols = tuple(c for c in METRIC_COLS_EXTRA if c in df.columns)
    if not cols:
        raise SystemExit(f"No extra metric columns found in {fold_csv}")
    rows: List[List[float]] = []
    labels: List[str] = []
    for tid, lab in _type_order():
        sub = df.loc[df["type_id"] == tid]
        if sub.empty:
            raise SystemExit(f"Missing type {tid!r} in {fold_csv}")
        rows.append([float(sub[c].mean()) for c in cols])
        labels.append(lab)
    mat = np.asarray(rows, dtype=float)
    table = pd.DataFrame(mat, index=labels, columns=list(cols))
    return table, mat, cols, labels


def draw_heatmap_pdf(
    mat_z: np.ndarray,
    raw: np.ndarray,
    *,
    title: str,
    out_pdf: Path,
    cbar_label: str,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    cell_w: float | None = None,
) -> None:
    n_r, n_c = mat_z.shape
    if len(row_labels) != n_r:
        raise ValueError(f"row_labels length {len(row_labels)} != n_rows {n_r}")
    if len(col_labels) != n_c:
        raise ValueError(f"col_labels length {len(col_labels)} != n_cols {n_c}")

    cw = float(cell_w) if cell_w is not None else (56.0 if n_c > 8 else 78.0)
    cell_h = 42.0 if n_r > 6 else 52.0
    ml = 100.0
    mr, mt = 90.0, 50.0
    mb = 95.0 if n_c > 8 else 70.0
    plot_w = n_c * cw
    plot_h = n_r * cell_h
    page_w = ml + plot_w + mr
    page_h = mt + plot_h + mb
    vmin, vmax = _color_limits_global(mat_z)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_pdf), pagesize=(page_w, page_h))
    c.setFillColor(white)
    c.rect(0, 0, page_w, page_h, fill=1, stroke=0)

    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(page_w / 2, page_h - 28, title)

    font_val = 7 if n_c > 8 else 8
    font_z = 6 if n_c > 8 else 7
    for i in range(n_r):
        for j in range(n_c):
            x = ml + j * cw
            y = mb + (n_r - 1 - i) * cell_h
            c.setFillColor(_ylorrd_color(float(mat_z[i, j]), vmin, vmax))
            c.setStrokeColor(HexColor("#DDDDDD"))
            c.rect(x, y, cw, cell_h, fill=1, stroke=1)
            c.setFillColor(black)
            c.setFont("Helvetica", font_val)
            c.drawCentredString(x + cw / 2, y + cell_h / 2 + 4, f"{raw[i, j]:.3f}")
            c.setFont("Helvetica", font_z)
            c.drawCentredString(x + cw / 2, y + cell_h / 2 - 8, f"z={mat_z[i, j]:+.2f}")

    c.setFont("Helvetica", 7 if n_c > 8 else 9)
    for j, name in enumerate(col_labels):
        x = ml + j * cw + cw / 2
        c.saveState()
        c.translate(x, mb - 12)
        c.rotate(35 if n_c > 8 else 30)
        c.drawRightString(0, 0, name)
        c.restoreState()

    c.setFont("Helvetica", 8 if n_r > 6 else 9)
    for i, name in enumerate(row_labels):
        y = mb + (n_r - 1 - i) * cell_h + cell_h / 2 - 3
        c.drawRightString(ml - 8, y, name)

    _draw_colorbar_ylorrd(c, ml + plot_w + 12, mb, 12, plot_h, vmin, vmax, cbar_label)
    c.save()
    print(f"[plot-celltype] heatmap -> {out_pdf}")


def draw_heatmap_pdf_raw(
    raw: np.ndarray,
    *,
    title: str,
    out_pdf: Path,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    cbar_label: str = "Raw value (yellow→red)",
    cell_w: float | None = None,
) -> None:
    """Raw-value heatmap; same yellow→red scale as ablation raw heatmaps."""
    n_r, n_c = raw.shape
    if len(row_labels) != n_r:
        raise ValueError(f"row_labels length {len(row_labels)} != n_rows {n_r}")
    if len(col_labels) != n_c:
        raise ValueError(f"col_labels length {len(col_labels)} != n_cols {n_c}")

    cw = float(cell_w) if cell_w is not None else (56.0 if n_c > 8 else 78.0)
    cell_h = 42.0 if n_r > 6 else 52.0
    ml = 100.0
    mr, mt = 90.0, 50.0
    mb = 95.0 if n_c > 8 else 70.0
    plot_w = n_c * cw
    plot_h = n_r * cell_h
    page_w = ml + plot_w + mr
    page_h = mt + plot_h + mb
    vmin, vmax = _color_limits_global(raw)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_pdf), pagesize=(page_w, page_h))
    c.setFillColor(white)
    c.rect(0, 0, page_w, page_h, fill=1, stroke=0)

    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(page_w / 2, page_h - 28, title)

    font_val = 7 if n_c > 8 else 8
    for i in range(n_r):
        for j in range(n_c):
            x = ml + j * cw
            y = mb + (n_r - 1 - i) * cell_h
            val = float(raw[i, j])
            c.setFillColor(_ylorrd_color(val, vmin, vmax))
            c.setStrokeColor(HexColor("#DDDDDD"))
            c.rect(x, y, cw, cell_h, fill=1, stroke=1)
            c.setFillColor(black)
            c.setFont("Helvetica", font_val)
            c.drawCentredString(x + cw / 2, y + cell_h / 2 - 3, f"{val:.3f}")

    c.setFont("Helvetica", 7 if n_c > 8 else 9)
    for j, name in enumerate(col_labels):
        x = ml + j * cw + cw / 2
        c.saveState()
        c.translate(x, mb - 12)
        c.rotate(35 if n_c > 8 else 30)
        c.drawRightString(0, 0, name)
        c.restoreState()

    c.setFont("Helvetica", 8 if n_r > 6 else 9)
    for i, name in enumerate(row_labels):
        y = mb + (n_r - 1 - i) * cell_h + cell_h / 2 - 3
        c.drawRightString(ml - 8, y, name)

    _draw_colorbar_ylorrd(c, ml + plot_w + 12, mb, 12, plot_h, vmin, vmax, cbar_label)
    c.save()
    print(f"[plot-celltype] heatmap -> {out_pdf}")


def _write_heatmap_pair(
    raw: np.ndarray,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    figures_dir: Path,
    stem: str,
    title_prefix: str,
) -> None:
    z_col = _zscore_column(raw)
    z_glb = _zscore_global(raw)
    pd.DataFrame(raw, index=list(row_labels), columns=list(col_labels)).to_csv(
        figures_dir / f"{stem}_raw_metrics.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(z_col, index=list(row_labels), columns=list(col_labels)).to_csv(
        figures_dir / f"{stem}_zscore_column.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(z_glb, index=list(row_labels), columns=list(col_labels)).to_csv(
        figures_dir / f"{stem}_zscore_global.csv", encoding="utf-8-sig"
    )
    draw_heatmap_pdf(
        z_col,
        raw,
        title=f"{title_prefix} (column z-score)",
        out_pdf=figures_dir / f"{stem}_heatmap_zscore_column.pdf",
        cbar_label="Column z-score",
        row_labels=row_labels,
        col_labels=col_labels,
    )
    draw_heatmap_pdf(
        z_glb,
        raw,
        title=f"{title_prefix} (global z-score)",
        out_pdf=figures_dir / f"{stem}_heatmap_zscore_global.pdf",
        cbar_label="Global z-score",
        row_labels=row_labels,
        col_labels=col_labels,
    )


def _load_column_splits(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cv_path = cfg.get("cv_splits_json")
    fold_idx = cfg.get("fold_idx")
    if not cv_path or fold_idx is None:
        raise ValueError("run_config missing cv_splits_json / fold_idx")
    payload = load_cv_splits(cv_path)
    fold = get_fold_split(payload, int(fold_idx))
    return (
        np.asarray(fold["train_cols"], dtype=np.int64),
        np.asarray(fold["val_cols"], dtype=np.int64),
        np.asarray(fold["test_cols"], dtype=np.int64),
    )


def _predict_fold_tpr(run_dir: Path) -> Tuple[np.ndarray, float]:
    cfg_path = run_dir / "run_config.json"
    ckpt = run_dir / "best_model_fold0.pth"
    if not cfg_path.is_file() or not ckpt.is_file():
        raise FileNotFoundError(f"Missing config/ckpt in {run_dir}")
    cfg: Dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
    args = Namespace(**cfg)
    set_seed(int(args.seed))
    col_splits = _load_column_splits(cfg)
    model, _, _val_loader, test_loader, device, use_amp_bf16 = _build_pipeline(
        args, column_splits=col_splits
    )
    state = torch.load(str(ckpt), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    y_true, y_prob = _predict_arrays(model, test_loader, device, use_amp_bf16=use_amp_bf16)
    return _macro_roc_from_probs(y_true, y_prob)


def _collect_type_rocs(
    grn_root: Path,
    fold_indices: Sequence[int],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for tid, lab in _type_order():
        tpr_list: List[np.ndarray] = []
        auc_list: List[float] = []
        for fold_idx in fold_indices:
            run_dir = tripgrn_10cv_fold_dir(tid, fold_idx, grn_root)
            print(f"[plot-celltype] ROC infer {lab} fold {fold_idx:02d} ...", flush=True)
            try:
                tpr, mac = _predict_fold_tpr(run_dir)
            except Exception as exc:  # noqa: BLE001 — continue other folds
                print(f"[plot-celltype] skip {run_dir}: {exc}", flush=True)
                continue
            if np.isnan(mac):
                print(f"[plot-celltype] skip empty ROC: {run_dir}", flush=True)
                continue
            tpr_list.append(tpr)
            auc_list.append(mac)
            print(f"[plot-celltype]   macro-AUC={mac:.4f}", flush=True)
        if not tpr_list:
            raise SystemExit(f"No ROC curves for cell type {lab}")
        stack = np.vstack(tpr_list)
        out[tid] = {
            "label": lab,
            "tpr_mean": np.mean(stack, axis=0),
            "tpr_std": np.std(stack, axis=0, ddof=0),
            "auc_mean": float(np.mean(auc_list)),
            "auc_std": float(np.std(auc_list, ddof=0)),
            "n_folds": len(auc_list),
        }
    return out


def draw_roc_pdf(roc_data: Dict[str, Dict[str, Any]], out_pdf: Path) -> None:
    w, h = 520.0, 480.0
    ml, mr, mt, mb = 55.0, 20.0, 45.0, 50.0
    plot_w = w - ml - mr
    plot_h = h - mt - mb

    def x_pt(fpr: float) -> float:
        return ml + float(fpr) * plot_w

    def y_pt(tpr: float) -> float:
        return mb + float(tpr) * plot_h

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_pdf), pagesize=(w, h))
    c.setFillColor(white)
    c.rect(0, 0, w, h, fill=1, stroke=0)

    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(w / 2, h - 26, "Macro-average ROC (GSE131907 cell types)")

    c.setFont("Helvetica", 8)
    for t in np.linspace(0, 1, 6):
        xx, yy = x_pt(float(t)), y_pt(float(t))
        c.setStrokeColor(HexColor("#DDDDDD"))
        c.setDash(2, 2)
        c.line(ml, yy, ml + plot_w, yy)
        c.line(xx, mb, xx, mb + plot_h)
        c.setDash()
        c.setFillColor(black)
        c.drawCentredString(xx, mb - 14, f"{t:.1f}")
        c.drawRightString(ml - 6, yy - 3, f"{t:.1f}")

    c.setStrokeColor(HexColor("#333333"))
    c.setLineWidth(1)
    c.rect(ml, mb, plot_w, plot_h, fill=0, stroke=1)
    c.setFont("Helvetica", 10)
    c.drawCentredString(ml + plot_w / 2, 16, "False Positive Rate")
    c.saveState()
    c.translate(16, mb + plot_h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "True Positive Rate")
    c.restoreState()

    c.setStrokeColor(HexColor("#888888"))
    c.setDash(4, 3)
    c.setLineWidth(1)
    c.line(x_pt(0), y_pt(0), x_pt(1), y_pt(1))
    c.setDash()

    for tid, _lab in _type_order():
        d = roc_data[tid]
        color = TYPE_COLORS.get(tid, "#333333")
        mean_tpr = np.asarray(d["tpr_mean"], dtype=float)
        std_tpr = np.asarray(d["tpr_std"], dtype=float)

        if d["n_folds"] > 1:
            xs = list(FPR_GRID) + list(FPR_GRID[::-1])
            ys_hi = np.clip(mean_tpr + std_tpr, 0.0, 1.0)
            ys_lo = np.clip(mean_tpr - std_tpr, 0.0, 1.0)
            ys = list(ys_hi) + list(ys_lo[::-1])
            path = c.beginPath()
            path.moveTo(x_pt(xs[0]), y_pt(ys[0]))
            for xf, yf in zip(xs[1:], ys[1:]):
                path.lineTo(x_pt(xf), y_pt(yf))
            path.close()
            c.setFillColor(_hex_alpha(color, 0.12))
            c.setStrokeColor(_hex_alpha(color, 0.0))
            c.drawPath(path, fill=1, stroke=0)

        c.setStrokeColor(HexColor(color))
        c.setLineWidth(0.9)
        path = c.beginPath()
        path.moveTo(x_pt(float(FPR_GRID[0])), y_pt(float(mean_tpr[0])))
        for xf, yf in zip(FPR_GRID[1:], mean_tpr[1:]):
            path.lineTo(x_pt(float(xf)), y_pt(float(yf)))
        c.drawPath(path, fill=0, stroke=1)

    # legend at bottom-right of plot area
    type_items = list(_type_order())
    n_legend = len(type_items) + 1  # + Chance
    row_h = 11.0
    legend_x = ml + plot_w - 145
    legend_y = mb + 10 + (n_legend - 1) * row_h
    for tid, _lab in type_items:
        d = roc_data[tid]
        color = TYPE_COLORS.get(tid, "#333333")
        c.setStrokeColor(HexColor(color))
        c.setLineWidth(1.1)
        c.line(legend_x, legend_y, legend_x + 16, legend_y)
        c.setFillColor(black)
        c.setFont("Helvetica", 7)
        short = d["label"]
        if len(short) > 18:
            short = short[:16] + "…"
        c.drawString(
            legend_x + 20,
            legend_y - 2,
            f"{short} ({d['auc_mean']:.3f}\u00b1{d['auc_std']:.3f})",
        )
        legend_y -= row_h

    c.setStrokeColor(HexColor("#888888"))
    c.setDash(4, 3)
    c.setLineWidth(1)
    c.line(legend_x, legend_y, legend_x + 16, legend_y)
    c.setDash()
    c.setFillColor(black)
    c.setFont("Helvetica", 7)
    c.drawString(legend_x + 20, legend_y - 2, "Chance")

    c.save()
    print(f"[plot-celltype] AUC curves -> {out_pdf}")


def _resolve_folds(fold_arg: str) -> List[int]:
    s = str(fold_arg).strip().upper()
    if s in {"ALL", "*"}:
        return list(range(1, 11))
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.replace(";", ",").split(",") if str(x).strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-name", default=DEFAULT_STEP06_EXPERIMENT)
    ap.add_argument("--folds", default="ALL", help="Folds for ROC (ALL, 1, 1-3, or 1,2,5).")
    ap.add_argument("--skip-roc", action="store_true", help="Only write heatmaps.")
    ap.add_argument(
        "--heatmap-set",
        choices=("core", "extra", "both"),
        default="both",
        help="core=macro_auc/macro_auprc+classification metrics; extra=per-class etc.; both=default.",
    )
    args = ap.parse_args()

    layout = gse131907_celltype_layout()
    grn_root = layout["grn_root"]
    exp_base = grn_root / "experiments" / args.experiment_name
    summary_csv = exp_base / "summary" / "tripgrn_10cv_type_summary.csv"
    fold_csv = exp_base / "summary" / "tripgrn_10cv_per_fold_metrics.csv"
    figures_dir = exp_base / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    short_rows = list(ROW_LABELS_SHORT)

    if args.heatmap_set in ("core", "both"):
        if not summary_csv.is_file():
            raise SystemExit(f"Missing summary CSV: {summary_csv}")
        table, raw, full_labels = _load_core_matrix(summary_csv)
        # Prefer short labels for PDF rows; keep full names in CSV index
        table.to_csv(figures_dir / "celltype_heatmap_raw_metrics.csv", encoding="utf-8-sig")
        z_col = _zscore_column(raw)
        z_glb = _zscore_global(raw)
        pd.DataFrame(z_col, index=full_labels, columns=list(METRIC_COLS)).to_csv(
            figures_dir / "celltype_heatmap_zscore_column.csv", encoding="utf-8-sig"
        )
        pd.DataFrame(z_glb, index=full_labels, columns=list(METRIC_COLS)).to_csv(
            figures_dir / "celltype_heatmap_zscore_global.csv", encoding="utf-8-sig"
        )
        draw_heatmap_pdf(
            z_col,
            raw,
            title="GSE131907 cell-type metrics (column z-score)",
            out_pdf=figures_dir / "celltype_metrics_heatmap_zscore_column.pdf",
            cbar_label="Column z-score",
            row_labels=short_rows,
            col_labels=METRIC_COLS,
        )
        draw_heatmap_pdf(
            z_glb,
            raw,
            title="GSE131907 cell-type metrics (global z-score)",
            out_pdf=figures_dir / "celltype_metrics_heatmap_zscore_global.pdf",
            cbar_label="Global z-score",
            row_labels=short_rows,
            col_labels=METRIC_COLS,
        )
        draw_heatmap_pdf_raw(
            raw,
            title="GSE131907 cell-type metrics (raw values)",
            out_pdf=figures_dir / "celltype_metrics_heatmap_raw.pdf",
            row_labels=short_rows,
            col_labels=METRIC_COLS,
        )

    if args.heatmap_set in ("extra", "both"):
        if not fold_csv.is_file():
            raise SystemExit(f"Missing fold metrics CSV: {fold_csv}")
        _extra_table, raw_extra, extra_cols, full_labels = _load_extra_matrix(fold_csv)
        _write_heatmap_pair(
            raw_extra,
            row_labels=short_rows,
            col_labels=extra_cols,
            figures_dir=figures_dir,
            stem="celltype_extra_metrics",
            title_prefix="GSE131907 cell-type extra metrics",
        )
        draw_heatmap_pdf_raw(
            raw_extra,
            title="GSE131907 cell-type extra metrics (raw values)",
            out_pdf=figures_dir / "celltype_extra_metrics_heatmap_raw.pdf",
            row_labels=short_rows,
            col_labels=extra_cols,
        )
        # overwrite CSVs with full type labels as index
        _extra_table.to_csv(figures_dir / "celltype_extra_metrics_raw_metrics.csv", encoding="utf-8-sig")
        pd.DataFrame(_zscore_column(raw_extra), index=full_labels, columns=list(extra_cols)).to_csv(
            figures_dir / "celltype_extra_metrics_zscore_column.csv", encoding="utf-8-sig"
        )
        pd.DataFrame(_zscore_global(raw_extra), index=full_labels, columns=list(extra_cols)).to_csv(
            figures_dir / "celltype_extra_metrics_zscore_global.csv", encoding="utf-8-sig"
        )

    if args.skip_roc or args.heatmap_set == "extra":
        if args.heatmap_set == "extra" and not args.skip_roc:
            print("[plot-celltype] heatmap-set=extra: skip ROC (use core/both to redraw ROC)")
        else:
            print("[plot-celltype] skip ROC")
        print(f"[plot-celltype] done -> {figures_dir}")
        return

    folds = _resolve_folds(args.folds)
    roc_data = _collect_type_rocs(grn_root, folds)
    draw_roc_pdf(roc_data, figures_dir / "celltype_macro_roc_bytype.pdf")

    rows = []
    for tid, _lab in _type_order():
        d = roc_data[tid]
        for fpr, tpr, tstd in zip(FPR_GRID, d["tpr_mean"], d["tpr_std"]):
            rows.append(
                {
                    "type_id": tid,
                    "type_label": d["label"],
                    "fpr": float(fpr),
                    "tpr_mean": float(tpr),
                    "tpr_std": float(tstd),
                    "auc_mean": d["auc_mean"],
                    "auc_std": d["auc_std"],
                    "n_folds": d["n_folds"],
                }
            )
    pd.DataFrame(rows).to_csv(
        figures_dir / "celltype_macro_roc_curve_points.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"[plot-celltype] done -> {figures_dir}")


if __name__ == "__main__":
    main()
