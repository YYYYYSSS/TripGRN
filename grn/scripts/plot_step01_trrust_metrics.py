#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot step01 pancancer 10CV metrics as PDF (KEGG or TRRUST):

1) macro_auc grouped boxplot (5 methods)
2) 3-class per-class OvR AUC boxplot (TripGRN vs DeepFGRN)
3) one radar per metric (axes = GSE; values = 10-fold mean per GSE; lines = methods)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from reportlab.lib.colors import Color, HexColor, black, white
from reportlab.pdfgen import canvas

from grn.data.pancancer_paths import PANCANCER_GSE_IDS, grn_root_from_here

DEFAULT_EXP = "step01_pancancer_epithelial_10cv_trrust"

METHODS_BINARY = ("TripGRN", "DeepFGRN", "DeepTGI", "scMGATGRN", "GENELink")
METHOD_COLORS_BINARY = {
    "TripGRN": "#C44E52",
    "DeepFGRN": "#4C72B0",
    "DeepTGI": "#DD8452",
    "scMGATGRN": "#55A868",
    "GENELink": "#8172B3",
}

FOLD_FILES = {
    "TripGRN": "tripgrn_10cv_fold_metrics.csv",
    "DeepFGRN": "deepfgrn_10cv_fold_metrics.csv",
    "DeepTGI": "deeptgi_10cv_fold_metrics.csv",
    "scMGATGRN": "scmgatgrn_10cv_fold_metrics.csv",
    "GENELink": "genelink_10cv_fold_metrics.csv",
}

# One radar PDF per metric (macro_auc stays as boxplot only)
RADAR_SPECS: List[Tuple[str, str]] = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("mcc", "MCC"),
    ("auprc", "AUPRC"),
    ("auc_ovo", "AUC-OvO"),
]

CLASS_LABELS = ("No regulation", "Activation", "Repression")
METHODS_3CLASS = ("TripGRN", "DeepFGRN")
METHOD_COLORS_3CLASS_AUC = {
    "TripGRN": METHOD_COLORS_BINARY["TripGRN"],
    "DeepFGRN": METHOD_COLORS_BINARY["DeepFGRN"],
}

# PDF page size in points (1 pt = 1/72 in); match KEGG ~14x7 in boxplot
BOX_W, BOX_H = 14.0 * 72, 7.0 * 72
RADAR_W, RADAR_H = 10.0 * 72, 10.0 * 72


def _pick_col(df: pd.DataFrame, *candidates: str) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _mean_auprc_row(r: pd.Series) -> float:
    if "auprc" in r.index and pd.notna(r.get("auprc")):
        return float(r["auprc"])
    cols = [c for c in r.index if str(c).startswith("auprc_")]
    vals = [float(r[c]) for c in cols if pd.notna(r[c])]
    return float(np.nanmean(vals)) if vals else float("nan")


def _load_fold_long(summary_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for method, fname in FOLD_FILES.items():
        path = summary_dir / fname
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        prec_c = _pick_col(df, "precision_macro", "precision")
        rec_c = _pick_col(df, "recall_macro", "recall")
        f1_c = _pick_col(df, "f1_macro", "f1")
        for _, r in df.iterrows():
            rows.append(
                {
                    "gse_id": str(r["gse_id"]),
                    "method": method,
                    "fold": int(r["fold"]),
                    "macro_auc": float(r["macro_auc"]),
                    "accuracy": float(r["accuracy"]),
                    "precision": float(r[prec_c]) if prec_c else float("nan"),
                    "recall": float(r[rec_c]) if rec_c else float("nan"),
                    "f1": float(r[f1_c]) if f1_c else float("nan"),
                    "mcc": float(r["mcc"]),
                    "auc_ovo": float(r["auc_ovo"]) if "auc_ovo" in r.index else float("nan"),
                    "auprc": _mean_auprc_row(r),
                }
            )
    return pd.DataFrame(rows)


def _resolve_auc_ovr_cols(df: pd.DataFrame) -> Dict[str, str]:
    """Map English class labels -> auc_ovr_* column names (Chinese suffixes in CSV)."""
    mapping: Dict[str, str] = {}
    needles = {
        "No regulation": ("无调控", "NoReg", "noreg", "no_reg"),
        "Activation": ("激活", "Activate", "activation", "act"),
        "Repression": ("抑制", "Repress", "repression", "inh"),
    }
    for lab, keys in needles.items():
        for c in df.columns:
            cs = str(c)
            if not cs.startswith("auc_ovr_"):
                continue
            if any(k in cs for k in keys):
                mapping[lab] = cs
                break
    if len(mapping) < 3:
        cols = [c for c in df.columns if str(c).startswith("auc_ovr_")]
        for i, lab in enumerate(CLASS_LABELS):
            if lab not in mapping and i < len(cols):
                mapping[lab] = cols[i]
    missing = [lab for lab in CLASS_LABELS if lab not in mapping]
    if missing:
        raise KeyError(f"Cannot resolve auc_ovr columns for {missing}; have={list(df.columns)}")
    return mapping


def _load_3class_auc_fold_table(summary_dir: Path) -> pd.DataFrame:
    """Per-class OvR AUC for TripGRN / DeepFGRN (matches KEGG threeclass_auc_boxplot)."""
    rows: List[Dict[str, Any]] = []
    for method, fname in (
        ("TripGRN", "tripgrn_10cv_fold_metrics.csv"),
        ("DeepFGRN", "deepfgrn_10cv_fold_metrics.csv"),
    ):
        path = summary_dir / fname
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        colmap = _resolve_auc_ovr_cols(df)
        for _, r in df.iterrows():
            for cls in CLASS_LABELS:
                rows.append(
                    {
                        "gse_id": str(r["gse_id"]),
                        "method": method,
                        "fold": int(r["fold"]),
                        "class": cls,
                        "auc": float(r[colmap[cls]]),
                    }
                )
    return pd.DataFrame(rows)


def _gse_method_means(df: pd.DataFrame, metric: str, methods: Sequence[str], gse_ids: Sequence[str]) -> pd.DataFrame:
    """Mean over folds for each (gse_id, method)."""
    rows = []
    for gse in gse_ids:
        for method in methods:
            sub = df.loc[(df["gse_id"] == gse) & (df["method"] == method), metric]
            rows.append(
                {
                    "gse_id": gse,
                    "method": method,
                    "mean": float(sub.mean()) if len(sub) else float("nan"),
                    "n_folds": int(len(sub)),
                }
            )
    return pd.DataFrame(rows)


def _box_stats(vals: np.ndarray) -> Dict[str, float]:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo = float(np.min(v[v >= q1 - 1.5 * iqr])) if np.any(v >= q1 - 1.5 * iqr) else float(np.min(v))
    hi = float(np.max(v[v <= q3 + 1.5 * iqr])) if np.any(v <= q3 + 1.5 * iqr) else float(np.max(v))
    return {"q1": float(q1), "med": float(med), "q3": float(q3), "whislo": lo, "whishi": hi}


def _ylim_step(
    values: np.ndarray,
    *,
    step: float = 0.1,
    ymax_ceil: float | None = 1.0,
) -> Tuple[float, float]:
    """Snap y-limits to a fixed step grid (default 0.1). Shared by KEGG/TRRUST boxplots."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0 if ymax_ceil is None else float(ymax_ceil)
    lo = math.floor(float(np.min(vals)) / step - 1e-12) * step
    hi = math.ceil(float(np.max(vals)) / step + 1e-12) * step
    if ymax_ceil is not None and float(np.max(vals)) <= float(ymax_ceil):
        hi = min(hi, float(ymax_ceil))
    if hi <= lo:
        hi = lo + step
    lo = round(lo / step) * step
    hi = round(hi / step) * step
    return float(lo), float(hi)


def _yticks(ymin: float, ymax: float, step: float = 0.1) -> np.ndarray:
    n = int(round((ymax - ymin) / step))
    return np.array([round(ymin + i * step, 10) for i in range(n + 1)], dtype=float)


def _paired_experiment_names() -> Tuple[str, str]:
    return "step01_pancancer_epithelial_10cv", "step01_pancancer_epithelial_10cv_trrust"


def _load_macro_auc_values_for_ylim(grn_root: Path) -> np.ndarray:
    vals: List[np.ndarray] = []
    for exp in _paired_experiment_names():
        summary = grn_root / "experiments" / exp / "summary"
        if not summary.is_dir():
            continue
        try:
            df = _load_fold_long(summary)
            vals.append(df["macro_auc"].to_numpy(dtype=float))
        except (FileNotFoundError, KeyError, ValueError):
            continue
    return np.concatenate(vals) if vals else np.asarray([], dtype=float)


def _load_3class_auc_values_for_ylim(grn_root: Path) -> np.ndarray:
    vals: List[np.ndarray] = []
    for exp in _paired_experiment_names():
        summary = grn_root / "experiments" / exp / "summary"
        if not summary.is_dir():
            continue
        try:
            df = _load_3class_auc_fold_table(summary)
            vals.append(df["auc"].to_numpy(dtype=float))
        except (FileNotFoundError, KeyError, ValueError):
            continue
    return np.concatenate(vals) if vals else np.asarray([], dtype=float)


def _hex(c: str) -> Color:
    return HexColor(c)


def _fill_alpha(hex_color: str, alpha: float) -> Color:
    base = HexColor(hex_color)
    return Color(base.red, base.green, base.blue, alpha=alpha)


def draw_macro_auc_boxplot_pdf(
    df: pd.DataFrame,
    *,
    methods: Sequence[str],
    gse_ids: Sequence[str],
    method_colors: Dict[str, str],
    out_path: Path,
    network_label: str,
    ymin: float,
    ymax: float,
    y_step: float = 0.1,
    value_col: str = "macro_auc",
    title: str | None = None,
    ylabel: str | None = None,
    subtitle: str | None = None,
) -> None:
    w, h = BOX_W, BOX_H
    ml, mr, mt, mb = 70, 24, 56, 100
    plot_w = w - ml - mr
    plot_h = h - mt - mb

    if ymax <= ymin:
        ymax = ymin + y_step

    n_gse = len(gse_ids)
    n_methods = len(methods)
    group_width = 0.78
    box_width = group_width / max(n_methods, 1)
    centers = np.arange(n_gse, dtype=float)
    title_text = title if title is not None else f"{network_label}: Macro-AUC (10-fold CV, unseen cells)"
    ylabel_text = ylabel or "Macro-AUC"

    def x_pt(x: float) -> float:
        return ml + (x + 0.5) / n_gse * plot_w

    def y_pt(y: float) -> float:
        return mb + (y - ymin) / (ymax - ymin) * plot_h

    c = canvas.Canvas(str(out_path), pagesize=(w, h))
    c.setFillColor(white)
    c.rect(0, 0, w, h, fill=1, stroke=0)

    if title_text:
        c.setFillColor(black)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(w / 2, h - 28, title_text)
    if subtitle:
        c.setFont("Helvetica", 10)
        c.drawCentredString(w / 2, h - 44, subtitle)

    # grid
    c.setFont("Helvetica", 9)
    for t in _yticks(ymin, ymax, y_step):
        yy = y_pt(float(t))
        c.setStrokeColor(HexColor("#CCCCCC"))
        c.setDash(3, 3)
        c.line(ml, yy, ml + plot_w, yy)
        c.setDash()
        c.setFillColor(black)
        c.drawRightString(ml - 6, yy - 3, f"{t:.1f}")

    c.setStrokeColor(HexColor("#333333"))
    c.setLineWidth(1)
    c.rect(ml, mb, plot_w, plot_h, fill=0, stroke=1)

    c.saveState()
    c.translate(18, mb + plot_h / 2)
    c.rotate(90)
    c.setFont("Helvetica", 11)
    c.setFillColor(black)
    c.drawCentredString(0, 0, ylabel_text)
    c.restoreState()

    for mi, method in enumerate(methods):
        color = method_colors.get(method, "#888888")
        offsets = centers - group_width / 2 + box_width / 2 + mi * box_width
        bw = box_width * 0.82
        for gi, gse in enumerate(gse_ids):
            vals = df.loc[(df["gse_id"] == gse) & (df["method"] == method), value_col].dropna().to_numpy(dtype=float)
            st = _box_stats(vals)
            if not st:
                continue
            cx = float(offsets[gi])
            x0, x1 = x_pt(cx - bw / 2), x_pt(cx + bw / 2)
            xc = 0.5 * (x0 + x1)
            y_lo, y_hi = y_pt(st["whislo"]), y_pt(st["whishi"])
            y_q1, y_q3 = y_pt(st["q1"]), y_pt(st["q3"])
            y_med = y_pt(st["med"])

            c.setStrokeColor(HexColor("#555555"))
            c.setLineWidth(0.8)
            c.line(xc, y_lo, xc, y_hi)
            c.line(x0, y_lo, x1, y_lo)
            c.line(x0, y_hi, x1, y_hi)

            bot, top = min(y_q1, y_q3), max(y_q1, y_q3)
            c.setFillColor(_fill_alpha(color, 0.75))
            c.setStrokeColor(HexColor("#333333"))
            c.rect(x0, bot, x1 - x0, max(top - bot, 0.5), fill=1, stroke=1)

            c.setStrokeColor(black)
            c.setLineWidth(1.2)
            c.line(x0, y_med, x1, y_med)

    c.setFont("Helvetica", 9)
    c.setFillColor(black)
    for gi, gse in enumerate(gse_ids):
        xx = x_pt(float(centers[gi]))
        c.saveState()
        c.translate(xx, mb - 8)
        c.rotate(35)
        c.drawRightString(0, 0, gse)
        c.restoreState()

    # legend
    lx = ml
    ly = 22
    for method in methods:
        color = method_colors.get(method, "#888888")
        c.setFillColor(_fill_alpha(color, 0.75))
        c.setStrokeColor(black)
        c.rect(lx, ly, 12, 12, fill=1, stroke=1)
        c.setFillColor(black)
        c.setFont("Helvetica", 10)
        c.drawString(lx + 16, ly + 2, method)
        lx += 130

    c.showPage()
    c.save()


def draw_threeclass_auc_boxplot_pdf(
    df: pd.DataFrame,
    *,
    gse_ids: Sequence[str],
    method_colors: Dict[str, str],
    out_path: Path,
    network_label: str,
    ymin: float,
    ymax: float,
    y_step: float = 0.1,
    value_col: str = "auc",
    ylabel: str = "AUC",
    title: str | None = None,
) -> None:
    """1x3 faceted per-class boxplot (TripGRN vs DeepFGRN), Fig.2B/D / S3–S4 layout."""
    w, h = BOX_W, BOX_H
    ml0, mr0, mt, mb = 54, 16, 52, 88
    gap = 14
    usable_w = w - ml0 - mr0 - 2 * gap
    panel_w = usable_w / 3.0
    plot_h = h - mt - mb

    ylo, yhi = float(ymin), float(ymax)
    if yhi <= ylo:
        yhi = ylo + y_step

    n_gse = len(gse_ids)
    n_methods = len(METHODS_3CLASS)
    group_width = 0.55
    box_width = group_width / max(n_methods, 1)
    centers = np.arange(n_gse, dtype=float)
    title_text = title if title is not None else f"{network_label}: 3-class OvR AUC by class (TripGRN vs DeepFGRN)"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=(w, h))
    c.setFillColor(white)
    c.rect(0, 0, w, h, fill=1, stroke=0)

    if title_text:
        c.setFillColor(black)
        c.setFont("Helvetica-Bold", 13)
        c.drawCentredString(w / 2, h - 26, title_text)

    for pi, cls in enumerate(CLASS_LABELS):
        left = ml0 + pi * (panel_w + gap)
        plot_w = panel_w

        def x_pt(x: float, _left=left, _pw=plot_w) -> float:
            return _left + (x + 0.5) / n_gse * _pw

        def y_pt(y: float) -> float:
            return mb + (y - ylo) / (yhi - ylo) * plot_h

        # title
        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(black)
        c.drawCentredString(left + plot_w / 2, mb + plot_h + 10, cls)

        # grid
        c.setFont("Helvetica", 8)
        for tick in _yticks(ylo, yhi, y_step):
            yy = y_pt(float(tick))
            c.setStrokeColor(HexColor("#CCCCCC"))
            c.setDash(3, 3)
            c.line(left, yy, left + plot_w, yy)
            c.setDash()
            if pi == 0:
                c.setFillColor(black)
                c.drawRightString(left - 5, yy - 2, f"{tick:.1f}")

        c.setStrokeColor(HexColor("#333333"))
        c.setLineWidth(1)
        c.rect(left, mb, plot_w, plot_h, fill=0, stroke=1)

        if pi == 0:
            c.saveState()
            c.translate(16, mb + plot_h / 2)
            c.rotate(90)
            c.setFont("Helvetica", 11)
            c.setFillColor(black)
            c.drawCentredString(0, 0, ylabel)
            c.restoreState()

        sub = df[df["class"] == cls]
        for mi, method in enumerate(METHODS_3CLASS):
            color = method_colors.get(method, "#888888")
            offsets = centers - group_width / 2 + box_width / 2 + mi * box_width
            bw = box_width * 0.82
            for gi, gse in enumerate(gse_ids):
                vals = sub.loc[(sub["gse_id"] == gse) & (sub["method"] == method), value_col].dropna().to_numpy(dtype=float)
                st = _box_stats(vals)
                if not st:
                    continue
                cx = float(offsets[gi])
                x0, x1 = x_pt(cx - bw / 2), x_pt(cx + bw / 2)
                xc = 0.5 * (x0 + x1)
                y_lo, y_hi = y_pt(st["whislo"]), y_pt(st["whishi"])
                y_q1, y_q3 = y_pt(st["q1"]), y_pt(st["q3"])
                y_med = y_pt(st["med"])

                c.setStrokeColor(HexColor("#555555"))
                c.setLineWidth(0.7)
                c.line(xc, y_lo, xc, y_hi)
                c.line(x0, y_lo, x1, y_lo)
                c.line(x0, y_hi, x1, y_hi)

                bot, top = min(y_q1, y_q3), max(y_q1, y_q3)
                c.setFillColor(_fill_alpha(color, 0.75))
                c.setStrokeColor(HexColor("#333333"))
                c.rect(x0, bot, x1 - x0, max(top - bot, 0.5), fill=1, stroke=1)

                c.setStrokeColor(black)
                c.setLineWidth(1.1)
                c.line(x0, y_med, x1, y_med)

        c.setFont("Helvetica", 7)
        c.setFillColor(black)
        for gi, gse in enumerate(gse_ids):
            xx = x_pt(float(centers[gi]))
            c.saveState()
            c.translate(xx, mb - 6)
            c.rotate(40)
            c.drawRightString(0, 0, gse)
            c.restoreState()

    # legend
    lx = w / 2 - 120
    ly = 20
    for method in METHODS_3CLASS:
        color = method_colors.get(method, "#888888")
        c.setFillColor(_fill_alpha(color, 0.75))
        c.setStrokeColor(black)
        c.setLineWidth(0.8)
        c.rect(lx, ly, 12, 12, fill=1, stroke=1)
        c.setFillColor(black)
        c.setFont("Helvetica", 10)
        c.drawString(lx + 16, ly + 2, method)
        lx += 140

    c.showPage()
    c.save()


def draw_metric_radar_pdf(
    mean_df: pd.DataFrame,
    *,
    metric_key: str,
    metric_label: str,
    methods: Sequence[str],
    gse_ids: Sequence[str],
    method_colors: Dict[str, str],
    out_path: Path,
    network_label: str,
    ymin: float = 0.0,
    ymax: float = 1.0,
    title: Optional[str] = None,
) -> None:
    """Radar: each axis = one GSE; value = fold-mean; one polygon per method.

    Radial scale is fixed to [ymin, ymax] (default 0–1) so all metric radars share
    the same numeric rings.
    """
    w, h = RADAR_W, RADAR_H
    cx, cy = w / 2, h / 2 + 10
    radius = min(w, h) * 0.32
    n = len(gse_ids)
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]
    if ymax <= ymin:
        ymax = ymin + 1.0

    def polar(i: int, v: float) -> Tuple[float, float]:
        t = (v - ymin) / (ymax - ymin)
        t = max(0.0, min(1.0, t))
        a = angles[i]
        return cx + radius * t * math.cos(a), cy + radius * t * math.sin(a)

    c = canvas.Canvas(str(out_path), pagesize=(w, h))
    c.setFillColor(white)
    c.rect(0, 0, w, h, fill=1, stroke=0)

    title_text = title if title is not None else f"{network_label}: {metric_label} radar"
    if title_text:
        c.setFillColor(black)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(w / 2, h - 36, title_text)
        c.setFont("Helvetica", 10)
        c.drawCentredString(
            w / 2,
            h - 52,
            f"Axes = datasets; values = 10-fold CV mean per GSE; scale [{ymin:.2f}, {ymax:.2f}]",
        )

    # rings + numeric scale labels (shared across all radar PDFs)
    ring_vals = [ymin + f * (ymax - ymin) for f in (0.25, 0.5, 0.75, 1.0)]
    c.setFont("Helvetica", 8)
    for v in ring_vals:
        pts = [polar(i, v) for i in range(n)]
        p = c.beginPath()
        p.moveTo(pts[0][0], pts[0][1])
        for x, y in pts[1:]:
            p.lineTo(x, y)
        p.close()
        c.setStrokeColor(HexColor("#DDDDDD"))
        c.setLineWidth(0.8)
        c.drawPath(p, fill=0, stroke=1)
        tx, ty = pts[0]
        c.setFillColor(HexColor("#555555"))
        c.drawRightString(tx - 5, ty - 3, f"{v:.2f}")

    # spokes + GSE labels
    c.setFont("Helvetica", 9)
    for i, gse in enumerate(gse_ids):
        x, y = polar(i, ymax)
        c.setStrokeColor(HexColor("#CCCCCC"))
        c.line(cx, cy, x, y)
        lx = cx + (radius + 28) * math.cos(angles[i])
        ly = cy + (radius + 28) * math.sin(angles[i])
        c.setFillColor(black)
        c.drawCentredString(lx, ly - 3, gse)

    # method polygons (skip methods with no finite values — do not collapse NaN→ymin)
    for method in methods:
        sub = mean_df[mean_df["method"] == method]
        if sub.empty:
            continue
        gse_to_v = {str(r["gse_id"]): float(r["mean"]) for _, r in sub.iterrows()}
        vals = [gse_to_v.get(gse, float("nan")) for gse in gse_ids]
        if not any(np.isfinite(v) for v in vals):
            print(f"[radar] skip {method} for {metric_key}: all NaN (missing binary metric cache?)")
            continue
        if not all(np.isfinite(v) for v in vals):
            n_nan = sum(1 for v in vals if not np.isfinite(v))
            print(f"[radar] skip {method} for {metric_key}: {n_nan}/{len(vals)} NaN GSE means")
            continue
        pts = [polar(i, float(v)) for i, v in enumerate(vals)]
        color = method_colors.get(method, "#888888")
        p = c.beginPath()
        p.moveTo(pts[0][0], pts[0][1])
        for x, y in pts[1:]:
            p.lineTo(x, y)
        p.close()
        c.setFillColor(_fill_alpha(color, 0.12))
        c.setStrokeColor(_hex(color))
        c.setLineWidth(2.0)
        c.drawPath(p, fill=1, stroke=1)

    # legend
    lx = 40
    ly = 28
    for method in methods:
        color = method_colors.get(method, "#888888")
        c.setFillColor(_fill_alpha(color, 0.75))
        c.setStrokeColor(black)
        c.setLineWidth(0.8)
        c.rect(lx, ly, 12, 12, fill=1, stroke=1)
        c.setFillColor(black)
        c.setFont("Helvetica", 10)
        c.drawString(lx + 16, ly + 2, method)
        lx += 130

    c.showPage()
    c.save()


def _default_network_label(experiment_name: str) -> str:
    name = experiment_name.lower()
    if "trrust" in name:
        return "TRRUST TF-target"
    return "KEGG PPrel"


def _default_out_dir(grn_root: Path, experiment_name: str) -> Path:
    exp_dir = grn_root / "experiments" / experiment_name
    # Keep historical folder names: KEGG -> figures/, TRRUST -> figure/
    if "trrust" in experiment_name.lower():
        return exp_dir / "figure"
    return exp_dir / "figures"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step01 pancancer PDF figures: macro_auc boxplot + 3-class AUC + per-metric GSE radars"
    )
    ap.add_argument("--grn-root", type=Path, default=None)
    ap.add_argument("--experiment-name", default=DEFAULT_EXP)
    ap.add_argument("--network-label", default=None, help='Title prefix, e.g. "KEGG PPrel" or "TRRUST TF-target"')
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    grn_root = Path(args.grn_root or grn_root_from_here()).resolve()
    exp = str(args.experiment_name).strip()
    network_label = str(args.network_label or _default_network_label(exp)).strip()
    summary_dir = grn_root / "experiments" / exp / "summary"
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(grn_root, exp)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = _load_fold_long(summary_dir)
    long_df.to_csv(out_dir / "plot_fold_long.csv", index=False, encoding="utf-8-sig")

    gse_ids = [g for g in PANCANCER_GSE_IDS if g in set(long_df["gse_id"])]
    if not gse_ids:
        gse_ids = sorted(long_df["gse_id"].unique().tolist())

    # Shared y-axis across KEGG + TRRUST boxplots; tick step = 0.1
    y_step = 0.1
    macro_ymin, macro_ymax = _ylim_step(_load_macro_auc_values_for_ylim(grn_root), step=y_step)
    three_ymin, three_ymax = _ylim_step(_load_3class_auc_values_for_ylim(grn_root), step=y_step)
    print(f"[plot-step01] shared macro_auc ylim=[{macro_ymin:.1f}, {macro_ymax:.1f}] step={y_step}")
    print(f"[plot-step01] shared 3class auc ylim=[{three_ymin:.1f}, {three_ymax:.1f}] step={y_step}")

    # 1) macro_auc boxplot PDF
    draw_macro_auc_boxplot_pdf(
        long_df,
        methods=METHODS_BINARY,
        gse_ids=gse_ids,
        method_colors=METHOD_COLORS_BINARY,
        out_path=out_dir / "macro_auc_boxplot.pdf",
        network_label=network_label,
        ymin=macro_ymin,
        ymax=macro_ymax,
        y_step=y_step,
    )

    # 1b) 3-class per-class OvR AUC boxplot (TripGRN vs DeepFGRN)
    three_df = _load_3class_auc_fold_table(summary_dir)
    three_df.to_csv(out_dir / "plot_3class_auc_fold_long.csv", index=False, encoding="utf-8-sig")
    draw_threeclass_auc_boxplot_pdf(
        three_df,
        gse_ids=gse_ids,
        method_colors=METHOD_COLORS_3CLASS_AUC,
        out_path=out_dir / "threeclass_auc_boxplot.pdf",
        network_label=network_label,
        ymin=three_ymin,
        ymax=three_ymax,
        y_step=y_step,
    )

    # 2) per-metric radar PDFs (axes = GSE)
    mean_rows: List[Dict[str, Any]] = []
    for metric_key, metric_label in RADAR_SPECS:
        mean_df = _gse_method_means(long_df, metric_key, METHODS_BINARY, gse_ids)
        mean_df.to_csv(out_dir / f"radar_{metric_key}_gse_means.csv", index=False, encoding="utf-8-sig")
        for _, r in mean_df.iterrows():
            mean_rows.append(
                {
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "gse_id": r["gse_id"],
                    "method": r["method"],
                    "mean": r["mean"],
                    "n_folds": r["n_folds"],
                }
            )
        draw_metric_radar_pdf(
            mean_df,
            metric_key=metric_key,
            metric_label=metric_label,
            methods=METHODS_BINARY,
            gse_ids=gse_ids,
            method_colors=METHOD_COLORS_BINARY,
            out_path=out_dir / f"radar_{metric_key}.pdf",
            network_label=network_label,
            ymin=0.0,
            ymax=1.0,
        )

    pd.DataFrame(mean_rows).to_csv(out_dir / "radar_all_metrics_gse_means.csv", index=False, encoding="utf-8-sig")

    print(f"[plot-step01] network={network_label} wrote PDFs -> {out_dir}")
    for p in sorted(out_dir.glob("*.pdf")):
        print(f"  {p.name} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
