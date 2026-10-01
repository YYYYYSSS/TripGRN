#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot TCGA bulk 10-fold CV metrics as one bar chart PDF per metric.

X = cancer dataset, Y = metric mean (± std error bar).
Output: experiments/bulk_tcga_10cv_seed42/figure/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from reportlab.lib.colors import Color, HexColor, black, white
from reportlab.pdfgen import canvas

from grn.data.bulk_paths import DEFAULT_BULK_EXPERIMENT, TCGA_CANCER_IDS, bulk_layout

# Core evaluation metrics (one PDF each)
METRIC_SPECS: List[Tuple[str, str]] = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("precision_macro", "Precision (macro)"),
    ("recall", "Recall"),
    ("recall_macro", "Recall (macro)"),
    ("f1", "F1"),
    ("f1_macro", "F1 (macro)"),
    ("mcc", "MCC"),
    ("macro_auc", "Macro-AUC"),
    ("auc_ovo", "AUC-OvO"),
    ("macro_auprc", "Macro-AUPRC"),
]

BAR_COLOR = "#4C78A8"
LINE_COLOR = "#E45756"
ERR_COLOR = "#333333"


def _short_cancer(cancer: str) -> str:
    s = str(cancer).strip()
    if s.upper().startswith("TCGA-"):
        return s.split("-", 1)[1]
    return s


def _shared_ylim(
    all_means: Sequence[np.ndarray],
    all_stds: Sequence[np.ndarray],
    *,
    step: float = 0.02,
) -> Tuple[float, float]:
    """One y-axis for all metric PDFs (snap to step grid)."""
    lows: List[float] = []
    highs: List[float] = []
    for means, stds in zip(all_means, all_stds):
        m = np.asarray(means, dtype=float)
        s = np.asarray(stds, dtype=float)
        ok = np.isfinite(m)
        if not np.any(ok):
            continue
        s = np.where(np.isfinite(s), s, 0.0)
        lows.append(float(np.nanmin(m[ok] - s[ok])))
        highs.append(float(np.nanmax(m[ok] + s[ok])))
    if not lows:
        return 0.0, 1.0
    lo, hi = min(lows), max(highs)
    pad = max(step, 0.05 * (hi - lo if hi > lo else 1.0))
    ymin = np.floor((lo - pad) / step) * step
    ymax = np.ceil((hi + pad) / step) * step
    if lo >= 0:
        ymin = max(0.0, float(ymin))
    if hi <= 1.0:
        ymax = min(1.0, float(ymax))
        # Keep headroom if data hug the ceiling (e.g. AUC ~0.99)
        if ymax <= hi + 1e-9:
            ymax = min(1.0, float(np.ceil((hi + step) / step) * step))
    if ymax <= ymin:
        ymax = ymin + step * 5
    return float(ymin), float(ymax)


def _yticks(ymin: float, ymax: float, step: float = 0.02) -> np.ndarray:
    # Inclusive ticks on a fixed step so all panels share the same labels.
    start = np.ceil(ymin / step - 1e-12) * step
    stop = np.floor(ymax / step + 1e-12) * step
    if stop < start:
        return np.asarray([ymin, ymax], dtype=float)
    ticks = np.arange(start, stop + step * 0.5, step)
    if abs(ticks[0] - ymin) > 1e-9:
        ticks = np.concatenate([[ymin], ticks])
    if abs(ticks[-1] - ymax) > 1e-9:
        ticks = np.concatenate([ticks, [ymax]])
    return ticks


def draw_metric_bar_pdf(
    cancers: Sequence[str],
    means: np.ndarray,
    stds: np.ndarray,
    *,
    metric_key: str,
    metric_title: str,
    out_pdf: Path,
    ylim: Tuple[float, float],
    y_step: float = 0.02,
) -> None:
    n = len(cancers)
    w, h = 720.0, 360.0
    ml, mr, mt, mb = 55.0, 20.0, 42.0, 70.0
    plot_w = w - ml - mr
    plot_h = h - mt - mb
    ymin, ymax = float(ylim[0]), float(ylim[1])

    def x_center(i: int) -> float:
        return ml + (i + 0.5) / n * plot_w

    def y_pt(v: float) -> float:
        return mb + (float(v) - ymin) / (ymax - ymin) * plot_h

    bar_w = (plot_w / n) * 0.62

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_pdf), pagesize=(w, h))
    c.setFillColor(white)
    c.rect(0, 0, w, h, fill=1, stroke=0)

    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(w / 2, h - 24, f"TCGA bulk 10-fold CV: {metric_title}")

    # grid + y ticks (shared scale across metric PDFs)
    c.setFont("Helvetica", 8)
    for t in _yticks(ymin, ymax, step=y_step):
        yy = y_pt(float(t))
        c.setStrokeColor(HexColor("#DDDDDD"))
        c.setDash(2, 2)
        c.line(ml, yy, ml + plot_w, yy)
        c.setDash()
        c.setFillColor(black)
        c.drawRightString(ml - 6, yy - 3, f"{t:.2f}")

    c.setStrokeColor(HexColor("#333333"))
    c.setLineWidth(1)
    c.rect(ml, mb, plot_w, plot_h, fill=0, stroke=1)

    c.setFont("Helvetica", 10)
    c.saveState()
    c.translate(16, mb + plot_h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, metric_title)
    c.restoreState()

    # zero line if ylim spans negative (MCC)
    if ymin < 0 < ymax:
        c.setStrokeColor(HexColor("#888888"))
        c.setDash(3, 2)
        c.line(ml, y_pt(0.0), ml + plot_w, y_pt(0.0))
        c.setDash()

    fill = HexColor(BAR_COLOR)
    for i, cancer in enumerate(cancers):
        m = float(means[i])
        s = float(stds[i]) if np.isfinite(stds[i]) else 0.0
        if not np.isfinite(m):
            continue
        xc = x_center(i)
        x0 = xc - bar_w / 2
        # From 0 when ylim crosses 0 (MCC); otherwise from axis floor (ymin).
        y_base = y_pt(0.0) if ymin < 0.0 < ymax else mb
        y_val = y_pt(m)
        bot, top = min(y_base, y_val), max(y_base, y_val)
        c.setFillColor(Color(fill.red, fill.green, fill.blue, alpha=0.85))
        c.setStrokeColor(HexColor("#2F5478"))
        c.setLineWidth(0.7)
        c.rect(x0, bot, bar_w, max(top - bot, 0.5), fill=1, stroke=1)

        # error bar
        if s > 0:
            y_lo = y_pt(m - s)
            y_hi = y_pt(m + s)
            c.setStrokeColor(HexColor(ERR_COLOR))
            c.setLineWidth(1.0)
            c.line(xc, y_lo, xc, y_hi)
            tick = bar_w * 0.28
            c.line(xc - tick, y_lo, xc + tick, y_lo)
            c.line(xc - tick, y_hi, xc + tick, y_hi)

        # x label
        c.setFillColor(black)
        c.setFont("Helvetica", 8)
        label = _short_cancer(cancer)
        c.saveState()
        c.translate(xc, mb - 10)
        c.rotate(40)
        c.drawRightString(0, 0, label)
        c.restoreState()

    c.setFont("Helvetica", 9)
    c.drawCentredString(ml + plot_w / 2, 14, "Dataset (TCGA cancer)")
    c.setFont("Helvetica", 7)
    c.setFillColor(HexColor("#555555"))
    c.drawString(ml, h - 38, "Bars = 10-fold mean; whiskers = SD")

    c.save()
    print(f"[bulk-bar] {metric_key} -> {out_pdf}")


def draw_recall_precision_combo_pdf(
    cancers: Sequence[str],
    recall_means: np.ndarray,
    recall_stds: np.ndarray,
    precision_means: np.ndarray,
    precision_stds: np.ndarray,
    *,
    out_pdf: Path,
    ylim_recall: Tuple[float, float],
    ylim_precision: Tuple[float, float],
    y_step: float = 0.02,
) -> None:
    """Recall as bars (left y), Precision as line (right y)."""
    n = len(cancers)
    w, h = 760.0, 380.0
    ml, mr, mt, mb = 58.0, 58.0, 48.0, 78.0
    plot_w = w - ml - mr
    plot_h = h - mt - mb
    y0_r, y1_r = float(ylim_recall[0]), float(ylim_recall[1])
    y0_p, y1_p = float(ylim_precision[0]), float(ylim_precision[1])

    def x_center(i: int) -> float:
        return ml + (i + 0.5) / n * plot_w

    def y_recall(v: float) -> float:
        return mb + (float(v) - y0_r) / (y1_r - y0_r) * plot_h

    def y_prec(v: float) -> float:
        return mb + (float(v) - y0_p) / (y1_p - y0_p) * plot_h

    bar_w = (plot_w / n) * 0.55

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_pdf), pagesize=(w, h))
    c.setFillColor(white)
    c.rect(0, 0, w, h, fill=1, stroke=0)

    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(w / 2, h - 22, "TCGA bulk 10-fold CV: Recall (bars) & Precision (line)")

    # horizontal grid from left (recall) ticks
    c.setFont("Helvetica", 8)
    for t in _yticks(y0_r, y1_r, step=y_step):
        yy = y_recall(float(t))
        c.setStrokeColor(HexColor("#DDDDDD"))
        c.setDash(2, 2)
        c.line(ml, yy, ml + plot_w, yy)
        c.setDash()
        c.setFillColor(HexColor(BAR_COLOR))
        c.drawRightString(ml - 6, yy - 3, f"{t:.2f}")

    # right-axis ticks (precision)
    for t in _yticks(y0_p, y1_p, step=y_step):
        yy = y_prec(float(t))
        c.setFillColor(HexColor(LINE_COLOR))
        c.drawString(ml + plot_w + 6, yy - 3, f"{t:.2f}")

    c.setStrokeColor(HexColor("#333333"))
    c.setLineWidth(1)
    c.rect(ml, mb, plot_w, plot_h, fill=0, stroke=1)

    # left / right axis titles
    c.setFillColor(HexColor(BAR_COLOR))
    c.setFont("Helvetica", 10)
    c.saveState()
    c.translate(14, mb + plot_h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Recall")
    c.restoreState()

    c.setFillColor(HexColor(LINE_COLOR))
    c.saveState()
    c.translate(w - 14, mb + plot_h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Precision")
    c.restoreState()

    # recall bars
    fill = HexColor(BAR_COLOR)
    for i, cancer in enumerate(cancers):
        m = float(recall_means[i])
        s = float(recall_stds[i]) if np.isfinite(recall_stds[i]) else 0.0
        if not np.isfinite(m):
            continue
        xc = x_center(i)
        x0 = xc - bar_w / 2
        y_base = mb
        y_val = y_recall(m)
        bot, top = min(y_base, y_val), max(y_base, y_val)
        c.setFillColor(Color(fill.red, fill.green, fill.blue, alpha=0.80))
        c.setStrokeColor(HexColor("#2F5478"))
        c.setLineWidth(0.7)
        c.rect(x0, bot, bar_w, max(top - bot, 0.5), fill=1, stroke=1)
        if s > 0:
            y_lo = y_recall(m - s)
            y_hi = y_recall(m + s)
            c.setStrokeColor(HexColor(ERR_COLOR))
            c.setLineWidth(1.0)
            c.line(xc, y_lo, xc, y_hi)
            tick = bar_w * 0.28
            c.line(xc - tick, y_lo, xc + tick, y_lo)
            c.line(xc - tick, y_hi, xc + tick, y_hi)

        c.setFillColor(black)
        c.setFont("Helvetica", 8)
        c.saveState()
        c.translate(xc, mb - 10)
        c.rotate(40)
        c.drawRightString(0, 0, _short_cancer(cancer))
        c.restoreState()

    # precision line + markers + error bars
    pts = []
    for i in range(n):
        m = float(precision_means[i])
        if np.isfinite(m):
            pts.append((x_center(i), y_prec(m), i, m))

    if len(pts) >= 2:
        c.setStrokeColor(HexColor(LINE_COLOR))
        c.setLineWidth(2.0)
        path = c.beginPath()
        path.moveTo(pts[0][0], pts[0][1])
        for x, y, _, _ in pts[1:]:
            path.lineTo(x, y)
        c.drawPath(path, fill=0, stroke=1)

    for x, y, i, m in pts:
        s = float(precision_stds[i]) if np.isfinite(precision_stds[i]) else 0.0
        if s > 0:
            y_lo = y_prec(m - s)
            y_hi = y_prec(m + s)
            c.setStrokeColor(HexColor(LINE_COLOR))
            c.setLineWidth(1.0)
            c.line(x, y_lo, x, y_hi)
            tick = bar_w * 0.22
            c.line(x - tick, y_lo, x + tick, y_lo)
            c.line(x - tick, y_hi, x + tick, y_hi)
        # marker
        c.setFillColor(HexColor(LINE_COLOR))
        c.setStrokeColor(HexColor("#8B1E1E"))
        c.setLineWidth(0.6)
        c.circle(x, y, 3.2, fill=1, stroke=1)

    # legend
    lx, ly = ml + 8, h - 40
    c.setFillColor(Color(fill.red, fill.green, fill.blue, alpha=0.80))
    c.setStrokeColor(HexColor("#2F5478"))
    c.rect(lx, ly - 4, 14, 10, fill=1, stroke=1)
    c.setFillColor(black)
    c.setFont("Helvetica", 8)
    c.drawString(lx + 18, ly - 2, "Recall (mean±SD)")
    c.setStrokeColor(HexColor(LINE_COLOR))
    c.setLineWidth(2.0)
    c.line(lx + 120, ly + 1, lx + 145, ly + 1)
    c.setFillColor(HexColor(LINE_COLOR))
    c.circle(lx + 132.5, ly + 1, 2.8, fill=1, stroke=0)
    c.setFillColor(black)
    c.drawString(lx + 150, ly - 2, "Precision (mean±SD)")

    c.setFont("Helvetica", 9)
    c.setFillColor(black)
    c.drawCentredString(ml + plot_w / 2, 14, "Dataset (TCGA cancer)")

    c.save()
    print(f"[bulk-bar] recall+precision combo -> {out_pdf}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-name", default=DEFAULT_BULK_EXPERIMENT)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: experiments/<exp>/figure",
    )
    args = ap.parse_args()

    layout = bulk_layout()
    grn_root = layout["grn_root"]
    exp_base = grn_root / "experiments" / args.experiment_name
    summary_csv = exp_base / "summary" / "bulk_10cv_cancer_mean_std.csv"
    out_dir = Path(args.out_dir) if args.out_dir else (exp_base / "figure")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not summary_csv.is_file():
        raise SystemExit(f"Missing summary CSV: {summary_csv}")

    df = pd.read_csv(summary_csv)
    # Preserve canonical cancer order
    order = [c for c in TCGA_CANCER_IDS if c in set(df["cancer"].astype(str))]
    extras = [c for c in df["cancer"].astype(str).tolist() if c not in order]
    cancers = order + extras
    df = df.set_index("cancer").loc[cancers].reset_index()

    y_step = 0.02
    series_means: List[np.ndarray] = []
    series_stds: List[np.ndarray] = []
    plot_jobs: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
    for key, title in METRIC_SPECS:
        mean_col, std_col = f"{key}_mean", f"{key}_std"
        if mean_col not in df.columns:
            print(f"[bulk-bar] skip missing column {mean_col}")
            continue
        means = df[mean_col].to_numpy(dtype=float)
        stds = (
            df[std_col].to_numpy(dtype=float)
            if std_col in df.columns
            else np.zeros_like(means)
        )
        series_means.append(means)
        series_stds.append(stds)
        plot_jobs.append((key, title, means, stds))

    ylim = _shared_ylim(series_means, series_stds, step=y_step)
    print(f"[bulk-bar] shared ylim=[{ylim[0]:.2f}, {ylim[1]:.2f}] step={y_step}")

    for key, title, means, stds in plot_jobs:
        draw_metric_bar_pdf(
            cancers,
            means,
            stds,
            metric_key=key,
            metric_title=title,
            out_pdf=out_dir / f"bulk_10cv_bar_{key}.pdf",
            ylim=ylim,
            y_step=y_step,
        )

    # Dual-axis: Recall bars (left) + Precision line (right)
    if {"recall_mean", "precision_mean"}.issubset(df.columns):
        rec_m = df["recall_mean"].to_numpy(dtype=float)
        rec_s = (
            df["recall_std"].to_numpy(dtype=float)
            if "recall_std" in df.columns
            else np.zeros_like(rec_m)
        )
        pre_m = df["precision_mean"].to_numpy(dtype=float)
        pre_s = (
            df["precision_std"].to_numpy(dtype=float)
            if "precision_std" in df.columns
            else np.zeros_like(pre_m)
        )
        ylim_rec = _shared_ylim([rec_m], [rec_s], step=y_step)
        ylim_pre = _shared_ylim([pre_m], [pre_s], step=y_step)
        draw_recall_precision_combo_pdf(
            cancers,
            rec_m,
            rec_s,
            pre_m,
            pre_s,
            out_pdf=out_dir / "bulk_10cv_bar_recall_line_precision.pdf",
            ylim_recall=ylim_rec,
            ylim_precision=ylim_pre,
            y_step=y_step,
        )
    else:
        print("[bulk-bar] skip recall/precision combo: missing columns")

    print(f"[bulk-bar] done -> {out_dir}")


if __name__ == "__main__":
    main()
