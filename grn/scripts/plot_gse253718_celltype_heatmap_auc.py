#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSE253718 cell-type TripGRN figures (PDF), ablation-style heatmaps + optional ROC.

Outputs under experiments/gse253718_celltype_bytype_seed42/figures/
ROC: pass --no-skip-roc (re-infers each fold; slow).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from grn.data.gse253718_celltype_paths import (
    DEFAULT_CELLTYPE_EXPERIMENT,
    gse253718_celltype_layout,
    resolve_type_ids,
    tripgrn_10cv_fold_dir,
)
from reportlab.lib.colors import HexColor, black, white
from reportlab.pdfgen import canvas

from grn.scripts.plot_gse131907_celltype_heatmap_auc import (
    METRIC_COLS,
    METRIC_SRC,
    _predict_fold_tpr,
    _write_heatmap_pair,
    draw_heatmap_pdf,
)
from grn.scripts.plot_gse253718_ablation_heatmap_auc import (
    FPR_GRID,
    _hex_alpha,
    _zscore_column,
    _zscore_global,
)

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
                vals.append(float("nan"))
            else:
                vals.append(float(r[col]))
        rows.append(vals)
        labels.append(lab)
    mat = np.asarray(rows, dtype=float)
    keep = [j for j in range(mat.shape[1]) if np.isfinite(mat[:, j]).any()]
    cols = [METRIC_COLS[j] for j in keep]
    mat = mat[:, keep]
    table = pd.DataFrame(mat, index=labels, columns=cols)
    return table, mat, labels


def _resolve_folds(s: str) -> List[int]:
    s = str(s).strip().upper()
    if s == "ALL":
        return list(range(1, 11))
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.replace(";", ",").split(",") if str(x).strip()]


def _collect_type_rocs(
    grn_root: Path,
    types: Sequence[Tuple[str, str]],
    fold_indices: Sequence[int],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for tid, lab in types:
        tpr_list: List[np.ndarray] = []
        auc_list: List[float] = []
        for fold_idx in fold_indices:
            run_dir = tripgrn_10cv_fold_dir(tid, fold_idx, grn_root)
            print(f"[plot-celltype] ROC infer {lab} fold {fold_idx:02d} ...", flush=True)
            try:
                tpr, mac = _predict_fold_tpr(run_dir)
            except Exception as exc:  # noqa: BLE001
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


def _draw_celltype_roc_pdf(
    roc_data: Dict[str, Dict[str, Any]],
    types: Sequence[Tuple[str, str]],
    out_pdf: Path,
) -> None:
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
    c.drawCentredString(w / 2, h - 26, "Macro-average ROC (GSE253718 cell types)")

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
    c.line(x_pt(0), y_pt(0), x_pt(1), y_pt(1))
    c.setDash()

    for i, (tid, _lab) in enumerate(types):
        d = roc_data[tid]
        color = _PALETTE[i % len(_PALETTE)]
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

    lx, ly = ml + 8, mb + plot_h - 12
    c.setFont("Helvetica", 7)
    for i, (tid, lab) in enumerate(types):
        d = roc_data[tid]
        color = _PALETTE[i % len(_PALETTE)]
        c.setStrokeColor(HexColor(color))
        c.setLineWidth(1.2)
        c.line(lx, ly - i * 11, lx + 14, ly - i * 11)
        c.setFillColor(black)
        c.drawString(
            lx + 18,
            ly - i * 11 - 2,
            f"{lab}  AUC={d['auc_mean']:.3f}±{d['auc_std']:.3f}",
        )
    c.save()
    print(f"[plot-celltype] ROC -> {out_pdf}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-name", default=DEFAULT_CELLTYPE_EXPERIMENT)
    ap.add_argument("--skip-roc", action="store_true", default=True)
    ap.add_argument(
        "--no-skip-roc",
        dest="skip_roc",
        action="store_false",
        help="Re-infer per-fold probs and draw celltype_macro_roc_bytype.pdf",
    )
    ap.add_argument("--folds", default="ALL", help="Folds for ROC (ALL, 1, 1-3, or 1,2,5).")
    ap.add_argument("--skip-heatmap", action="store_true")
    args = ap.parse_args()

    layout = gse253718_celltype_layout()
    grn_root = layout["grn_root"]
    exp_base = grn_root / "experiments" / args.experiment_name
    summary_csv = exp_base / "summary" / "tripgrn_10cv_type_summary.csv"
    fold_csv = exp_base / "summary" / "tripgrn_10cv_per_fold_metrics.csv"
    figures_dir = exp_base / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not summary_csv.is_file():
        raise SystemExit(f"Missing {summary_csv}; run summarize first")

    types = resolve_type_ids("ALL")
    short_rows = [lab if len(lab) <= 14 else lab[:12] + "…" for _, lab in types]

    if not args.skip_heatmap:
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
                for tid, _lab in types:
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

    if args.skip_roc:
        print(f"[plot-gse253718-celltype] done -> {figures_dir} (ROC skipped)")
        return

    folds = _resolve_folds(args.folds)
    roc_data = _collect_type_rocs(grn_root, types, folds)
    _draw_celltype_roc_pdf(roc_data, types, figures_dir / "celltype_macro_roc_bytype.pdf")
    rows = []
    for tid, _lab in types:
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
    print(f"[plot-gse253718-celltype] done -> {figures_dir} (ROC written)")


if __name__ == "__main__":
    main()
