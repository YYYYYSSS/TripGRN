#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSE253718 epithelial modality ablation figures (PDF via reportlab):

1) Metrics heatmap (rows: expr / expr+cds / all; cols: macro_auc, macro_auprc..mcc)
   - column-wise z-score
   - global z-score
2) Macro-average OvR ROC curves for the three modalities (mean±band over folds)
3) Per-class OvR ROC curves: 3 modalities × 3 classes = 9 curves (mean±band over folds)

Outputs under experiments/gse253718_ablation_epithelial_10cv_seed42/figures/
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
from reportlab.lib.colors import Color, HexColor, black, white
from reportlab.pdfgen import canvas
from sklearn.metrics import auc, roc_auc_score, roc_curve

from grn.data.gse253718_paths import DEFAULT_ABLATION_EXPERIMENT, ablation_stage_dir, gse253718_layout
from grn.data.pancancer_cv import get_fold_split, load_cv_splits
from grn.reproducibility import set_seed
from grn.scripts.regenerate_unseen_run_plots_from_checkpoint import _build_pipeline
from grn.scripts.train_unseen_samples_no_cv import _predict_arrays

TARGET_LABELS = ("expr", "expr+cds", "all")
TARGET_STAGE_IDS = ("01_expr_only", "02_expr_cds", "04_expr_cds_ppi")
# Display order (matches saved auc_ovr / confusion_matrix display):
# NoReg, Activation, Repression. Model/y_prob column order is NoReg, Repression, Activation.
CLASS_LABELS = ("NoReg", "Activation", "Repression")
CLASS_FULL_NAMES = ("No regulation", "Activation", "Repression")
CLASS_MODEL_IDX = (0, 2, 1)
# X-axis order: AUC, PRC, MCC, accuracy, precision, recall, f1
METRIC_COLS = ("macro_auc", "macro_auprc", "mcc", "accuracy", "precision", "recall", "f1")

# Remaining metrics (exclude the core columns above); fold-level means.
METRIC_COLS_EXTRA = (
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "auc_ovo",
    "auc_ovr_NoReg",
    "auc_ovr_Activation",
    "auc_ovr_Repression",
    "auprc_NoReg",
    "auprc_Activation",
    "auprc_Repression",
    "pc_noreg_accuracy",
    "pc_activation_accuracy",
    "pc_repression_accuracy",
    "pc_noreg_precision",
    "pc_activation_precision",
    "pc_repression_precision",
    "pc_noreg_recall",
    "pc_activation_recall",
    "pc_repression_recall",
)

MODALITY_COLORS = {
    "expr": "#4C78A8",
    "expr+cds": "#F58518",
    "all": "#54A24B",
}

# Line dash patterns for the three OvR classes (reportlab setDash args)
CLASS_DASH = {
    "NoReg": None,           # solid
    "Activation": (5, 3),    # dashed
    "Repression": (1.5, 2),  # dotted
}

# 9 distinct solid colors for modality × class (display order)
PERCLASS_NINE_COLORS: Dict[Tuple[str, str], str] = {
    ("expr", "NoReg"): "#4C78A8",
    ("expr", "Activation"): "#72B7B2",
    ("expr", "Repression"): "#54A24B",
    ("expr+cds", "NoReg"): "#EECA3B",
    ("expr+cds", "Activation"): "#F58518",
    ("expr+cds", "Repression"): "#E45756",
    ("all", "NoReg"): "#B279A2",
    ("all", "Activation"): "#FF9DA6",
    ("all", "Repression"): "#9D755D",
}

FPR_GRID = np.linspace(0.0, 1.0, 201)


def _zscore_column(mat: np.ndarray) -> np.ndarray:
    out = np.empty_like(mat, dtype=float)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        sd = float(np.std(col, ddof=0))
        out[:, j] = 0.0 if sd < 1e-12 else (col - float(np.mean(col))) / sd
    return out


def _zscore_global(mat: np.ndarray) -> np.ndarray:
    flat = mat.astype(float).ravel()
    sd = float(np.std(flat, ddof=0))
    if sd < 1e-12:
        return np.zeros_like(mat, dtype=float)
    return (mat - float(np.mean(flat))) / sd


def _load_metric_matrix(stage_csv: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(stage_csv)
    rows: List[List[float]] = []
    for sid, lab in zip(TARGET_STAGE_IDS, TARGET_LABELS):
        sub = df.loc[df["stage_id"] == sid]
        if sub.empty:
            sub = df.loc[df["modalities_label"] == lab]
        if sub.empty:
            raise SystemExit(f"Missing stage {sid!r} / label {lab!r} in {stage_csv}")
        r = sub.iloc[0]
        rows.append([float(r[f"{m}_mean"]) for m in METRIC_COLS])
    mat = np.asarray(rows, dtype=float)
    table = pd.DataFrame(mat, index=list(TARGET_LABELS), columns=list(METRIC_COLS))
    return table, mat


def _load_extra_metric_matrix(fold_csv: Path) -> Tuple[pd.DataFrame, np.ndarray, Tuple[str, ...]]:
    """10-fold mean matrix for metrics other than the core six."""
    df = pd.read_csv(fold_csv)
    cols = tuple(c for c in METRIC_COLS_EXTRA if c in df.columns)
    if not cols:
        raise SystemExit(f"No extra metric columns found in {fold_csv}")
    rows: List[List[float]] = []
    for sid, lab in zip(TARGET_STAGE_IDS, TARGET_LABELS):
        sub = df.loc[df["stage_id"] == sid]
        if sub.empty:
            sub = df.loc[df["modalities_label"] == lab]
        if sub.empty:
            raise SystemExit(f"Missing stage {sid!r} / label {lab!r} in {fold_csv}")
        rows.append([float(sub[c].mean()) for c in cols])
    mat = np.asarray(rows, dtype=float)
    table = pd.DataFrame(mat, index=list(TARGET_LABELS), columns=list(cols))
    return table, mat, cols


def _ylorrd_color(v: float, vmin: float, vmax: float) -> Color:
    """Map v in [vmin, vmax] to yellow (low) -> orange -> red (high)."""
    if not np.isfinite(v):
        return Color(0.92, 0.92, 0.92)
    span = float(vmax - vmin)
    t = 0.5 if span < 1e-12 else float(np.clip((v - vmin) / span, 0.0, 1.0))
    # #FFFFB2 -> #FECC5C -> #FD8D3C -> #F03B20
    stops = (
        (1.00, 1.00, 0.70),
        (0.996, 0.800, 0.361),
        (0.992, 0.553, 0.235),
        (0.941, 0.231, 0.125),
    )
    x = t * (len(stops) - 1)
    i = int(np.floor(x))
    if i >= len(stops) - 1:
        r, g, b = stops[-1]
    else:
        u = x - i
        r0, g0, b0 = stops[i]
        r1, g1, b1 = stops[i + 1]
        r = r0 + (r1 - r0) * u
        g = g0 + (g1 - g0) * u
        b = b0 + (b1 - b0) * u
    return Color(r, g, b)


def _color_limits_global(raw: np.ndarray) -> Tuple[float, float]:
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6
    return vmin, vmax


def _draw_colorbar_ylorrd(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    vmin: float,
    vmax: float,
    label: str,
) -> None:
    n = 80
    for i in range(n):
        v = vmin + (vmax - vmin) * (i + 0.5) / n
        c.setFillColor(_ylorrd_color(v, vmin, vmax))
        c.rect(x, y + h * i / n, w, h / n + 0.3, fill=1, stroke=0)
    c.setStrokeColor(HexColor("#333333"))
    c.setLineWidth(0.8)
    c.rect(x, y, w, h, fill=0, stroke=1)
    c.setFillColor(black)
    c.setFont("Helvetica", 8)
    c.drawString(x + w + 4, y - 2, f"{vmin:.2f}")
    c.drawString(x + w + 4, y + h / 2 - 3, f"{0.5 * (vmin + vmax):.2f}")
    c.drawString(x + w + 4, y + h - 2, f"{vmax:.2f}")
    c.setFont("Helvetica", 9)
    c.saveState()
    c.translate(x + w + 28, y + h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, label)
    c.restoreState()


def draw_heatmap_pdf(
    mat_z: np.ndarray,
    raw: np.ndarray,
    *,
    title: str,
    out_pdf: Path,
    cbar_label: str,
    col_labels: Sequence[str] | None = None,
    cell_w: float | None = None,
) -> None:
    """Z-score heatmap; same yellow→red color scale as raw heatmaps."""
    n_r, n_c = mat_z.shape
    labels = list(col_labels) if col_labels is not None else list(METRIC_COLS)
    if len(labels) != n_c:
        raise ValueError(f"col_labels length {len(labels)} != n_cols {n_c}")
    # Shrink cells for wide extra-metric heatmaps
    cw = float(cell_w) if cell_w is not None else (56.0 if n_c > 8 else 78.0)
    cell_h = 52.0
    ml, mr, mt, mb = 90.0, 90.0, 50.0, 95.0 if n_c > 8 else 70.0
    plot_w = n_c * cw
    plot_h = n_r * cell_h
    page_w = ml + plot_w + mr
    page_h = mt + plot_h + mb
    # Same scale rule as draw_heatmap_pdf_raw / test raw PDFs
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
    for j, name in enumerate(labels):
        x = ml + j * cw + cw / 2
        c.saveState()
        c.translate(x, mb - 12)
        c.rotate(35 if n_c > 8 else 30)
        c.drawRightString(0, 0, name)
        c.restoreState()

    c.setFont("Helvetica", 9)
    for i, name in enumerate(TARGET_LABELS):
        y = mb + (n_r - 1 - i) * cell_h + cell_h / 2 - 3
        c.drawRightString(ml - 8, y, name)

    _draw_colorbar_ylorrd(c, ml + plot_w + 12, mb, 12, plot_h, vmin, vmax, cbar_label)
    c.save()
    print(f"[plot-ablation] heatmap -> {out_pdf}")


def draw_heatmap_pdf_raw(
    raw: np.ndarray,
    *,
    title: str,
    out_pdf: Path,
    cbar_label: str = "Raw value",
    col_labels: Sequence[str] | None = None,
    cell_w: float | None = None,
) -> None:
    """Heatmap of raw metric values; yellow (low) -> red (high). No z-score."""
    n_r, n_c = raw.shape
    labels = list(col_labels) if col_labels is not None else list(METRIC_COLS)
    if len(labels) != n_c:
        raise ValueError(f"col_labels length {len(labels)} != n_cols {n_c}")
    cw = float(cell_w) if cell_w is not None else (56.0 if n_c > 8 else 78.0)
    cell_h = 52.0
    ml, mr, mt, mb = 90.0, 90.0, 50.0, 95.0 if n_c > 8 else 70.0
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
    for j, name in enumerate(labels):
        x = ml + j * cw + cw / 2
        c.saveState()
        c.translate(x, mb - 12)
        c.rotate(35 if n_c > 8 else 30)
        c.drawRightString(0, 0, name)
        c.restoreState()

    c.setFont("Helvetica", 9)
    for i, name in enumerate(TARGET_LABELS):
        y = mb + (n_r - 1 - i) * cell_h + cell_h / 2 - 3
        c.drawRightString(ml - 8, y, name)

    _draw_colorbar_ylorrd(c, ml + plot_w + 12, mb, 12, plot_h, vmin, vmax, cbar_label)
    c.save()
    print(f"[plot-ablation] heatmap -> {out_pdf}")


def _write_test_raw_heatmaps(
    *,
    raw_core: np.ndarray,
    raw_extra: np.ndarray,
    extra_cols: Sequence[str],
    test_dir: Path,
) -> None:
    """Write 4 raw (no z-score) yellow->red heatmaps under figures/test/."""
    test_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw_core, index=TARGET_LABELS, columns=list(METRIC_COLS)).to_csv(
        test_dir / "ablation_heatmap_raw_metrics.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(raw_extra, index=TARGET_LABELS, columns=list(extra_cols)).to_csv(
        test_dir / "ablation_extra_metrics_raw_metrics.csv", encoding="utf-8-sig"
    )
    # Mirror the four original heatmap slots; without z-score column/global share the same raw view.
    specs = (
        (
            raw_core,
            METRIC_COLS,
            "GSE253718 ablation metrics (raw values)",
            "ablation_metrics_heatmap_raw_column.pdf",
        ),
        (
            raw_core,
            METRIC_COLS,
            "GSE253718 ablation metrics (raw values)",
            "ablation_metrics_heatmap_raw_global.pdf",
        ),
        (
            raw_extra,
            extra_cols,
            "GSE253718 ablation extra metrics (raw values)",
            "ablation_extra_metrics_heatmap_raw_column.pdf",
        ),
        (
            raw_extra,
            extra_cols,
            "GSE253718 ablation extra metrics (raw values)",
            "ablation_extra_metrics_heatmap_raw_global.pdf",
        ),
    )
    for mat, cols, title, fname in specs:
        draw_heatmap_pdf_raw(
            mat,
            title=title,
            out_pdf=test_dir / fname,
            cbar_label="Raw value (yellow→red)",
            col_labels=cols,
        )


def _write_heatmap_pair(
    raw: np.ndarray,
    col_labels: Sequence[str],
    *,
    figures_dir: Path,
    stem: str,
    title_prefix: str,
) -> None:
    """Write column/global z-score PDFs + CSVs without clobbering other stems."""
    z_col = _zscore_column(raw)
    z_glb = _zscore_global(raw)
    pd.DataFrame(raw, index=TARGET_LABELS, columns=list(col_labels)).to_csv(
        figures_dir / f"{stem}_raw_metrics.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(z_col, index=TARGET_LABELS, columns=list(col_labels)).to_csv(
        figures_dir / f"{stem}_zscore_column.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(z_glb, index=TARGET_LABELS, columns=list(col_labels)).to_csv(
        figures_dir / f"{stem}_zscore_global.csv", encoding="utf-8-sig"
    )
    draw_heatmap_pdf(
        z_col,
        raw,
        title=f"{title_prefix} (column z-score)",
        out_pdf=figures_dir / f"{stem}_heatmap_zscore_column.pdf",
        cbar_label="Column z-score",
        col_labels=col_labels,
    )
    draw_heatmap_pdf(
        z_glb,
        raw,
        title=f"{title_prefix} (global z-score)",
        out_pdf=figures_dir / f"{stem}_heatmap_zscore_global.pdf",
        cbar_label="Global z-score",
        col_labels=col_labels,
    )


def _macro_roc_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, float]:
    per = _per_class_roc_from_probs(y_true, y_prob)
    tprs = [per[c]["tpr"] for c in CLASS_LABELS if np.isfinite(per[c]["auc"])]
    aucs = [per[c]["auc"] for c in CLASS_LABELS if np.isfinite(per[c]["auc"])]
    if not tprs:
        return np.full_like(FPR_GRID, np.nan), float("nan")
    return np.mean(np.vstack(tprs), axis=0), float(np.mean(aucs))


def _per_class_roc_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """One-vs-rest ROC for each class in display order, interpolated onto FPR_GRID."""
    n_cls = int(y_prob.shape[1])
    out: Dict[str, Dict[str, Any]] = {}
    for cls_name, cls_idx in zip(CLASS_LABELS, CLASS_MODEL_IDX):
        if cls_idx >= n_cls:
            out[cls_name] = {
                "tpr": np.full_like(FPR_GRID, np.nan),
                "auc": float("nan"),
            }
            continue
        y_bin = (y_true == cls_idx).astype(int)
        if len(np.unique(y_bin)) < 2:
            out[cls_name] = {
                "tpr": np.full_like(FPR_GRID, np.nan),
                "auc": float("nan"),
            }
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, cls_idx])
        interp = np.interp(FPR_GRID, fpr, tpr)
        interp[0] = 0.0
        try:
            auc_v = float(roc_auc_score(y_bin, y_prob[:, cls_idx]))
        except ValueError:
            auc_v = float(auc(fpr, tpr))
        out[cls_name] = {"tpr": interp, "auc": auc_v}
    return out


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


def _predict_fold_rocs(run_dir: Path) -> Tuple[np.ndarray, float, Dict[str, Dict[str, Any]]]:
    """Return macro TPR/AUC and per-class OvR curves for one fold."""
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
    per = _per_class_roc_from_probs(y_true, y_prob)
    tprs = [per[c]["tpr"] for c in CLASS_LABELS if np.isfinite(per[c]["auc"])]
    aucs = [per[c]["auc"] for c in CLASS_LABELS if np.isfinite(per[c]["auc"])]
    if not tprs:
        return np.full_like(FPR_GRID, np.nan), float("nan"), per
    return np.mean(np.vstack(tprs), axis=0), float(np.mean(aucs)), per


def _predict_fold_tpr(run_dir: Path) -> Tuple[np.ndarray, float]:
    tpr, mac, _per = _predict_fold_rocs(run_dir)
    return tpr, mac


def _collect_modality_and_class_rocs(
    grn_root: Path,
    fold_indices: Sequence[int],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
    """Collect 10-fold mean±std macro ROC and per-class OvR ROC for each modality."""
    macro_out: Dict[str, Dict[str, Any]] = {}
    class_out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for sid, lab in zip(TARGET_STAGE_IDS, TARGET_LABELS):
        tpr_list: List[np.ndarray] = []
        auc_list: List[float] = []
        per_tprs: Dict[str, List[np.ndarray]] = {c: [] for c in CLASS_LABELS}
        per_aucs: Dict[str, List[float]] = {c: [] for c in CLASS_LABELS}
        for fold_idx in fold_indices:
            run_dir = ablation_stage_dir(sid, fold_idx, grn_root)
            print(f"[plot-ablation] ROC infer {lab} fold {fold_idx:02d} ...", flush=True)
            tpr, mac, per = _predict_fold_rocs(run_dir)
            if np.isnan(mac):
                print(f"[plot-ablation] skip empty ROC: {run_dir}")
                continue
            tpr_list.append(tpr)
            auc_list.append(mac)
            cls_msg = []
            for c in CLASS_LABELS:
                if not np.isfinite(per[c]["auc"]):
                    continue
                per_tprs[c].append(np.asarray(per[c]["tpr"], dtype=float))
                per_aucs[c].append(float(per[c]["auc"]))
                cls_msg.append(f"{c}={per[c]['auc']:.4f}")
            print(
                f"[plot-ablation]   macro-AUC={mac:.4f} | " + " ".join(cls_msg),
                flush=True,
            )
        if not tpr_list:
            raise SystemExit(f"No ROC curves for modality {lab}")
        stack = np.vstack(tpr_list)
        macro_out[lab] = {
            "tpr_mean": np.mean(stack, axis=0),
            "tpr_std": np.std(stack, axis=0, ddof=0),
            "auc_mean": float(np.mean(auc_list)),
            "auc_std": float(np.std(auc_list, ddof=0)),
            "n_folds": len(auc_list),
        }
        class_out[lab] = {}
        for c in CLASS_LABELS:
            if not per_tprs[c]:
                raise SystemExit(f"No per-class ROC for {lab}/{c}")
            cstack = np.vstack(per_tprs[c])
            class_out[lab][c] = {
                "tpr_mean": np.mean(cstack, axis=0),
                "tpr_std": np.std(cstack, axis=0, ddof=0),
                "auc_mean": float(np.mean(per_aucs[c])),
                "auc_std": float(np.std(per_aucs[c], ddof=0)),
                "n_folds": len(per_aucs[c]),
            }
    return macro_out, class_out


def _collect_modality_rocs(
    grn_root: Path,
    fold_indices: Sequence[int],
) -> Dict[str, Dict[str, Any]]:
    macro, _cls = _collect_modality_and_class_rocs(grn_root, fold_indices)
    return macro


def _hex_alpha(hex_color: str, alpha: float) -> Color:
    base = HexColor(hex_color)
    return Color(base.red, base.green, base.blue, alpha=alpha)


def draw_roc_pdf(roc_data: Dict[str, Dict[str, Any]], out_pdf: Path) -> None:
    w, h = 480.0, 420.0
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
    c.drawCentredString(w / 2, h - 26, "Macro-average ROC (expr / expr+cds / all)")

    # grid
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

    # chance line
    c.setStrokeColor(HexColor("#888888"))
    c.setDash(4, 3)
    c.setLineWidth(1)
    c.line(x_pt(0), y_pt(0), x_pt(1), y_pt(1))
    c.setDash()

    for lab in TARGET_LABELS:
        d = roc_data[lab]
        color = MODALITY_COLORS[lab]
        mean_tpr = np.asarray(d["tpr_mean"], dtype=float)
        std_tpr = np.asarray(d["tpr_std"], dtype=float)

        if d["n_folds"] > 1:
            # band as filled polygon
            xs = list(FPR_GRID) + list(FPR_GRID[::-1])
            ys_hi = np.clip(mean_tpr + std_tpr, 0.0, 1.0)
            ys_lo = np.clip(mean_tpr - std_tpr, 0.0, 1.0)
            ys = list(ys_hi) + list(ys_lo[::-1])
            path = c.beginPath()
            path.moveTo(x_pt(xs[0]), y_pt(ys[0]))
            for xf, yf in zip(xs[1:], ys[1:]):
                path.lineTo(x_pt(xf), y_pt(yf))
            path.close()
            c.setFillColor(_hex_alpha(color, 0.18))
            c.setStrokeColor(_hex_alpha(color, 0.0))
            c.drawPath(path, fill=1, stroke=0)

        c.setStrokeColor(HexColor(color))
        c.setLineWidth(2.0)
        path = c.beginPath()
        path.moveTo(x_pt(float(FPR_GRID[0])), y_pt(float(mean_tpr[0])))
        for xf, yf in zip(FPR_GRID[1:], mean_tpr[1:]):
            path.lineTo(x_pt(float(xf)), y_pt(float(yf)))
        c.drawPath(path, fill=0, stroke=1)

    # legend at bottom-right of plot area
    n_legend = len(TARGET_LABELS) + 1  # + Chance
    row_h = 14.0
    legend_x = ml + plot_w - 158
    legend_y = mb + 10 + (n_legend - 1) * row_h
    for lab in TARGET_LABELS:
        d = roc_data[lab]
        color = MODALITY_COLORS[lab]
        c.setStrokeColor(HexColor(color))
        c.setLineWidth(2.5)
        c.line(legend_x, legend_y, legend_x + 24, legend_y)
        c.setFillColor(black)
        c.setFont("Helvetica", 8)
        c.drawString(
            legend_x + 30,
            legend_y - 3,
            f"{lab} (AUC={d['auc_mean']:.3f}\u00b1{d['auc_std']:.3f})",
        )
        legend_y -= row_h

    c.setStrokeColor(HexColor("#888888"))
    c.setDash(4, 3)
    c.setLineWidth(1)
    c.line(legend_x, legend_y, legend_x + 24, legend_y)
    c.setDash()
    c.setFillColor(black)
    c.setFont("Helvetica", 8)
    c.drawString(legend_x + 30, legend_y - 3, "Chance")

    c.save()
    print(f"[plot-ablation] AUC curves -> {out_pdf}")


def draw_per_class_roc_pdf(
    class_roc: Dict[str, Dict[str, Dict[str, Any]]],
    out_pdf: Path,
    *,
    style: str = "modality_dash",
) -> None:
    """9 OvR ROC curves: 3 modalities × 3 classes, 10-fold mean±band.

    style:
      - modality_dash: modality color + class dash pattern (default)
      - nine_solid: nine distinct solid colors
    """
    if style not in {"modality_dash", "nine_solid"}:
        raise ValueError(f"Unknown per-class ROC style: {style!r}")

    w, h = 520.0, 460.0
    ml, mr, mt, mb = 55.0, 20.0, 45.0, 50.0
    plot_w = w - ml - mr
    plot_h = h - mt - mb

    def x_pt(fpr: float) -> float:
        return ml + float(fpr) * plot_w

    def y_pt(tpr: float) -> float:
        return mb + float(tpr) * plot_h

    def curve_color(lab: str, cls: str) -> str:
        if style == "nine_solid":
            return PERCLASS_NINE_COLORS[(lab, cls)]
        return MODALITY_COLORS[lab]

    def curve_dash(cls: str):
        if style == "nine_solid":
            return None
        return CLASS_DASH[cls]

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_pdf), pagesize=(w, h))
    c.setFillColor(white)
    c.rect(0, 0, w, h, fill=1, stroke=0)

    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(
        ml + plot_w / 2,
        h - 26,
        "Per-class OvR ROC (expr / expr+cds / all)",
    )

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

    # chance line
    c.setStrokeColor(HexColor("#888888"))
    c.setDash(4, 3)
    c.setLineWidth(1)
    c.line(x_pt(0), y_pt(0), x_pt(1), y_pt(1))
    c.setDash()

    # Draw bands first (under lines), then mean curves
    for lab in TARGET_LABELS:
        for cls in CLASS_LABELS:
            d = class_roc[lab][cls]
            color = curve_color(lab, cls)
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
                c.setFillColor(_hex_alpha(color, 0.08))
                c.setStrokeColor(_hex_alpha(color, 0.0))
                c.drawPath(path, fill=1, stroke=0)

    for lab in TARGET_LABELS:
        for cls in CLASS_LABELS:
            d = class_roc[lab][cls]
            color = curve_color(lab, cls)
            mean_tpr = np.asarray(d["tpr_mean"], dtype=float)
            dash = curve_dash(cls)
            c.setStrokeColor(HexColor(color))
            c.setLineWidth(1.8)
            if dash is None:
                c.setDash()
            else:
                c.setDash(*dash)
            path = c.beginPath()
            path.moveTo(x_pt(float(FPR_GRID[0])), y_pt(float(mean_tpr[0])))
            for xf, yf in zip(FPR_GRID[1:], mean_tpr[1:]):
                path.lineTo(x_pt(float(xf)), y_pt(float(yf)))
            c.drawPath(path, fill=0, stroke=1)
            c.setDash()

    # legend at bottom-right of plot area
    n_legend = len(TARGET_LABELS) * len(CLASS_LABELS) + 1  # + Chance
    row_h = 11.0
    legend_x = ml + plot_w - 168
    legend_y = mb + 10 + (n_legend - 1) * row_h
    for lab in TARGET_LABELS:
        for cls, full in zip(CLASS_LABELS, CLASS_FULL_NAMES):
            d = class_roc[lab][cls]
            color = curve_color(lab, cls)
            dash = curve_dash(cls)
            c.setStrokeColor(HexColor(color))
            c.setLineWidth(2.0)
            if dash is None:
                c.setDash()
            else:
                c.setDash(*dash)
            c.line(legend_x, legend_y, legend_x + 18, legend_y)
            c.setDash()
            c.setFillColor(black)
            c.setFont("Helvetica", 6.5)
            c.drawString(
                legend_x + 22,
                legend_y - 2,
                f"{lab} · {full} ({d['auc_mean']:.3f}\u00b1{d['auc_std']:.3f})",
            )
            legend_y -= row_h

    c.setStrokeColor(HexColor("#888888"))
    c.setDash(4, 3)
    c.setLineWidth(1)
    c.line(legend_x, legend_y, legend_x + 18, legend_y)
    c.setDash()
    c.setFillColor(black)
    c.setFont("Helvetica", 6.5)
    c.drawString(legend_x + 22, legend_y - 2, "Chance")

    c.save()
    print(f"[plot-ablation] per-class AUC curves -> {out_pdf}")


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
    ap.add_argument("--experiment-name", default=DEFAULT_ABLATION_EXPERIMENT)
    ap.add_argument("--folds", default="ALL", help="Folds for ROC (ALL, 1, 1-3, or 1,2,5).")
    ap.add_argument("--skip-roc", action="store_true", help="Only write heatmaps.")
    ap.add_argument(
        "--skip-heatmap",
        action="store_true",
        help="Skip heatmap writing; only (re)draw ROC curves.",
    )
    ap.add_argument(
        "--heatmap-set",
        choices=("core", "extra", "both"),
        default="both",
        help="core=macro_auc/macro_auprc+classification metrics; extra=remaining metrics; both=default.",
    )
    ap.add_argument(
        "--test-raw",
        action="store_true",
        help="Write raw yellow→red heatmaps under figures/test/ only (do not overwrite originals).",
    )
    args = ap.parse_args()

    layout = gse253718_layout()
    grn_root = layout["grn_root"]
    # Prefer fig02_modality_ablation/experiments/... (layout["experiment_base"])
    exp_base = layout["experiment_base"]
    if args.experiment_name and args.experiment_name != DEFAULT_ABLATION_EXPERIMENT:
        # Allow override while still resolving under fig02 first, then repo experiments/
        cand = layout["fig02"] / "experiments" / args.experiment_name
        exp_base = cand if cand.is_dir() else (grn_root / "experiments" / args.experiment_name)
    summary_csv = exp_base / "summary" / "ablation_10cv_stage_mean_std.csv"
    fold_csv = exp_base / "summary" / "ablation_10cv_fold_metrics.csv"
    figures_dir = exp_base / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plot-ablation] exp_base={exp_base}", flush=True)

    if args.test_raw:
        if not summary_csv.is_file():
            raise SystemExit(f"Missing summary CSV: {summary_csv}")
        if not fold_csv.is_file():
            raise SystemExit(f"Missing fold metrics CSV: {fold_csv}")
        _table, raw = _load_metric_matrix(summary_csv)
        _extra_table, raw_extra, extra_cols = _load_extra_metric_matrix(fold_csv)
        test_dir = figures_dir / "test"
        _write_test_raw_heatmaps(
            raw_core=raw,
            raw_extra=raw_extra,
            extra_cols=extra_cols,
            test_dir=test_dir,
        )
        print(f"[plot-ablation] test raw heatmaps done -> {test_dir}")
        return

    if not args.skip_heatmap and args.heatmap_set in ("core", "both"):
        if not summary_csv.is_file():
            raise SystemExit(f"Missing summary CSV: {summary_csv}")
        _table, raw = _load_metric_matrix(summary_csv)
        # Keep original core filenames (do not rename).
        _table.to_csv(figures_dir / "ablation_heatmap_raw_metrics.csv", encoding="utf-8-sig")
        z_col = _zscore_column(raw)
        z_glb = _zscore_global(raw)
        pd.DataFrame(z_col, index=TARGET_LABELS, columns=METRIC_COLS).to_csv(
            figures_dir / "ablation_heatmap_zscore_column.csv", encoding="utf-8-sig"
        )
        pd.DataFrame(z_glb, index=TARGET_LABELS, columns=METRIC_COLS).to_csv(
            figures_dir / "ablation_heatmap_zscore_global.csv", encoding="utf-8-sig"
        )
        draw_heatmap_pdf(
            z_col,
            raw,
            title="GSE253718 ablation metrics (column z-score)",
            out_pdf=figures_dir / "ablation_metrics_heatmap_zscore_column.pdf",
            cbar_label="Column z-score",
            col_labels=METRIC_COLS,
        )
        draw_heatmap_pdf(
            z_glb,
            raw,
            title="GSE253718 ablation metrics (global z-score)",
            out_pdf=figures_dir / "ablation_metrics_heatmap_zscore_global.pdf",
            cbar_label="Global z-score",
            col_labels=METRIC_COLS,
        )

    if not args.skip_heatmap and args.heatmap_set in ("extra", "both"):
        if not fold_csv.is_file():
            raise SystemExit(f"Missing fold metrics CSV: {fold_csv}")
        _extra_table, raw_extra, extra_cols = _load_extra_metric_matrix(fold_csv)
        _write_heatmap_pair(
            raw_extra,
            extra_cols,
            figures_dir=figures_dir,
            stem="ablation_extra_metrics",
            title_prefix="GSE253718 ablation extra metrics",
        )

    if args.skip_roc or (args.heatmap_set == "extra" and not args.skip_heatmap):
        # Extra-only runs should never rewrite ROC by default.
        if args.heatmap_set == "extra" and not args.skip_roc and not args.skip_heatmap:
            print("[plot-ablation] heatmap-set=extra: skip ROC (use core/both to redraw ROC)")
        else:
            print("[plot-ablation] skip ROC")
        print(f"[plot-ablation] done -> {figures_dir}")
        return

    folds = _resolve_folds(args.folds)
    roc_data, class_roc = _collect_modality_and_class_rocs(grn_root, folds)
    draw_roc_pdf(roc_data, figures_dir / "ablation_macro_roc_expr_exprcds_all.pdf")
    draw_per_class_roc_pdf(
        class_roc,
        figures_dir / "ablation_perclass_roc_expr_exprcds_all.pdf",
    )

    rows = []
    for lab in TARGET_LABELS:
        d = roc_data[lab]
        for fpr, tpr, tstd in zip(FPR_GRID, d["tpr_mean"], d["tpr_std"]):
            rows.append(
                {
                    "modality": lab,
                    "fpr": float(fpr),
                    "tpr_mean": float(tpr),
                    "tpr_std": float(tstd),
                    "auc_mean": d["auc_mean"],
                    "auc_std": d["auc_std"],
                    "n_folds": d["n_folds"],
                }
            )
    pd.DataFrame(rows).to_csv(
        figures_dir / "ablation_macro_roc_curve_points.csv",
        index=False,
        encoding="utf-8-sig",
    )

    class_rows = []
    for lab in TARGET_LABELS:
        for cls, full in zip(CLASS_LABELS, CLASS_FULL_NAMES):
            d = class_roc[lab][cls]
            for fpr, tpr, tstd in zip(FPR_GRID, d["tpr_mean"], d["tpr_std"]):
                class_rows.append(
                    {
                        "modality": lab,
                        "class": cls,
                        "class_full": full,
                        "fpr": float(fpr),
                        "tpr_mean": float(tpr),
                        "tpr_std": float(tstd),
                        "auc_mean": d["auc_mean"],
                        "auc_std": d["auc_std"],
                        "n_folds": d["n_folds"],
                    }
                )
    pd.DataFrame(class_rows).to_csv(
        figures_dir / "ablation_perclass_roc_curve_points.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"[plot-ablation] done -> {figures_dir}")


if __name__ == "__main__":
    main()
