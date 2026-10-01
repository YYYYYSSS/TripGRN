#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot DoRothEA A-B-C TripGRN vs DeepFGRN figures with reportlab (PDF)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from reportlab.lib.colors import HexColor, black, white
from reportlab.pdfgen import canvas

from grn.data.pancancer_paths import PANCANCER_GSE_IDS, grn_root_from_here
from grn.scripts.plot_step01_trrust_metrics import (
    METHOD_COLORS_3CLASS_AUC,
    _box_stats,
    _fill_alpha,
    _load_3class_auc_fold_table,
    _ylim_step,
    _yticks,
    draw_macro_auc_boxplot_pdf,
    draw_threeclass_auc_boxplot_pdf,
)

EXP_DEFAULT = "step01_pancancer_epithelial_10cv_dorothea_abc"
METHODS = ("TripGRN", "DeepFGRN")
COLORS = {"TripGRN": "#C44E52", "DeepFGRN": "#4C72B0"}
NETWORK = "DoRothEA A-C (TF-target)"


def _load_folds(summary: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for method, fname in (
        ("TripGRN", "tripgrn_10cv_fold_metrics.csv"),
        ("DeepFGRN", "deepfgrn_10cv_fold_metrics.csv"),
    ):
        df = pd.read_csv(summary / fname)
        for _, r in df.iterrows():
            rows.append(
                {
                    "gse_id": str(r["gse_id"]),
                    "method": method,
                    "fold": int(r["fold"]),
                    "accuracy": float(r["accuracy"]),
                    "macro_auc": float(r["macro_auc"]),
                    "mcc": float(r["mcc"]),
                }
            )
    return pd.DataFrame(rows)


def _gse_means(folds: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    for (gse, method), sub in folds.groupby(["gse_id", "method"]):
        row = {"gse_id": gse, "method": method}
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce")
            row[m] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def _draw_grouped_metric_bars(
    means: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    gse_ids: Sequence[str],
    out_path: Path,
    title: str,
    ymin: float,
    ymax: float,
) -> None:
    w, h = 14.0 * 72, 5.2 * 72
    ml, mr, mt, mb = 70, 24, 50, 90
    plot_w = w - ml - mr
    plot_h = h - mt - mb
    n_gse = len(gse_ids)
    centers = np.arange(n_gse, dtype=float)
    group_w = 0.72
    bar_w = group_w / 2.0

    def x_pt(x: float) -> float:
        return ml + (x + 0.5) / n_gse * plot_w

    def y_pt(y: float) -> float:
        return mb + (y - ymin) / (ymax - ymin) * plot_h

    c = canvas.Canvas(str(out_path), pagesize=(w, h))
    c.setFillColor(white)
    c.rect(0, 0, w, h, fill=1, stroke=0)
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(w / 2, h - 28, title)

    for t in _yticks(ymin, ymax, 0.1):
        yy = y_pt(float(t))
        c.setStrokeColor(HexColor("#CCCCCC"))
        c.setDash(3, 3)
        c.line(ml, yy, ml + plot_w, yy)
        c.setDash()
        c.setFillColor(black)
        c.setFont("Helvetica", 9)
        c.drawRightString(ml - 6, yy - 3, f"{t:.1f}")

    c.setStrokeColor(HexColor("#333333"))
    c.rect(ml, mb, plot_w, plot_h, fill=0, stroke=1)

    c.saveState()
    c.translate(18, mb + plot_h / 2)
    c.rotate(90)
    c.setFont("Helvetica", 11)
    c.setFillColor(black)
    c.drawCentredString(0, 0, ylabel)
    c.restoreState()

    for mi, method in enumerate(METHODS):
        color = COLORS[method]
        for gi, gse in enumerate(gse_ids):
            row = means[(means.gse_id == gse) & (means.method == method)]
            if not len(row):
                continue
            val = float(row.iloc[0][metric])
            err = float(row.iloc[0][f"{metric}_std"])
            cx = centers[gi] - group_w / 2 + bar_w / 2 + mi * bar_w
            x0 = x_pt(cx - bar_w * 0.4)
            x1 = x_pt(cx + bar_w * 0.4)
            y0 = y_pt(0.0 if ymin <= 0 else ymin)
            y1 = y_pt(val)
            bot, top = min(y0, y1), max(y0, y1)
            c.setFillColor(_fill_alpha(color, 0.85))
            c.setStrokeColor(HexColor("#333333"))
            c.rect(x0, bot, max(x1 - x0, 0.5), max(top - bot, 0.5), fill=1, stroke=1)
            # error bar
            yc = y_pt(val)
            ye0 = y_pt(max(ymin, val - err))
            ye1 = y_pt(min(ymax, val + err))
            xc = 0.5 * (x0 + x1)
            c.setStrokeColor(HexColor("#444444"))
            c.setLineWidth(0.8)
            c.line(xc, ye0, xc, ye1)
            c.line(xc - 3, ye0, xc + 3, ye0)
            c.line(xc - 3, ye1, xc + 3, ye1)

    c.setFont("Helvetica", 8)
    c.setFillColor(black)
    for gi, gse in enumerate(gse_ids):
        xx = x_pt(float(centers[gi]))
        c.saveState()
        c.translate(xx, mb - 8)
        c.rotate(35)
        c.drawRightString(0, 0, gse)
        c.restoreState()

    lx, ly = ml, 18
    for method in METHODS:
        c.setFillColor(_fill_alpha(COLORS[method], 0.85))
        c.setStrokeColor(black)
        c.rect(lx, ly, 12, 12, fill=1, stroke=1)
        c.setFillColor(black)
        c.setFont("Helvetica", 10)
        c.drawString(lx + 16, ly + 2, method)
        lx += 120

    c.showPage()
    c.save()


def _draw_pan_mean_bars(folds: pd.DataFrame, out_path: Path, title: str) -> None:
    metrics = [("accuracy", "Accuracy"), ("macro_auc", "Macro-AUC"), ("mcc", "MCC")]
    w, h = 8.5 * 72, 5.0 * 72
    ml, mr, mt, mb = 60, 24, 48, 50
    plot_w = w - ml - mr
    plot_h = h - mt - mb
    ymin, ymax = 0.0, 1.15
    n = len(metrics)
    centers = np.arange(n, dtype=float)
    group_w = 0.62
    bar_w = group_w / 2.0

    def x_pt(x: float) -> float:
        return ml + (x + 0.5) / n * plot_w

    def y_pt(y: float) -> float:
        return mb + (y - ymin) / (ymax - ymin) * plot_h

    c = canvas.Canvas(str(out_path), pagesize=(w, h))
    c.setFillColor(white)
    c.rect(0, 0, w, h, fill=1, stroke=0)
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(w / 2, h - 26, title)

    for t in _yticks(0.0, 1.0, 0.2):
        yy = y_pt(float(t))
        c.setStrokeColor(HexColor("#CCCCCC"))
        c.setDash(3, 3)
        c.line(ml, yy, ml + plot_w, yy)
        c.setDash()
        c.setFillColor(black)
        c.setFont("Helvetica", 9)
        c.drawRightString(ml - 6, yy - 3, f"{t:.1f}")
    c.setStrokeColor(HexColor("#333333"))
    c.rect(ml, mb, plot_w, plot_h, fill=0, stroke=1)

    for mi, method in enumerate(METHODS):
        color = COLORS[method]
        for gi, (key, lab) in enumerate(metrics):
            vals = folds.loc[folds.method == method, key].astype(float)
            mean = float(vals.mean())
            std = float(vals.std(ddof=0))
            cx = centers[gi] - group_w / 2 + bar_w / 2 + mi * bar_w
            x0 = x_pt(cx - bar_w * 0.4)
            x1 = x_pt(cx + bar_w * 0.4)
            y0 = y_pt(0.0)
            y1 = y_pt(mean)
            c.setFillColor(_fill_alpha(color, 0.85))
            c.setStrokeColor(HexColor("#333333"))
            c.rect(x0, y0, max(x1 - x0, 0.5), max(y1 - y0, 0.5), fill=1, stroke=1)
            xc = 0.5 * (x0 + x1)
            c.setStrokeColor(HexColor("#444444"))
            c.line(xc, y_pt(max(0, mean - std)), xc, y_pt(min(1.1, mean + std)))
            c.setFillColor(black)
            c.setFont("Helvetica", 8)
            c.drawCentredString(xc, y_pt(mean + std) + 6, f"{mean:.3f}")

    c.setFont("Helvetica", 10)
    c.setFillColor(black)
    for gi, (_, lab) in enumerate(metrics):
        c.drawCentredString(x_pt(float(centers[gi])), mb - 18, lab)

    lx = ml
    for method in METHODS:
        c.setFillColor(_fill_alpha(COLORS[method], 0.85))
        c.setStrokeColor(black)
        c.rect(lx, 14, 12, 12, fill=1, stroke=1)
        c.setFillColor(black)
        c.setFont("Helvetica", 10)
        c.drawString(lx + 16, 16, method)
        lx += 120

    c.showPage()
    c.save()


def _draw_delta_bars(means: pd.DataFrame, gse_ids: Sequence[str], out_path: Path, title: str) -> None:
    metrics = [("accuracy", "Accuracy"), ("macro_auc", "Macro-AUC"), ("mcc", "MCC")]
    w, h = 14.0 * 72, 4.6 * 72
    ml0, mr0, mt, mb = 50, 18, 46, 78
    gap = 16
    usable = w - ml0 - mr0 - 2 * gap
    panel_w = usable / 3.0
    plot_h = h - mt - mb
    n_gse = len(gse_ids)
    trip = means[means.method == "TripGRN"].set_index("gse_id")
    deep = means[means.method == "DeepFGRN"].set_index("gse_id")

    c = canvas.Canvas(str(out_path), pagesize=(w, h))
    c.setFillColor(white)
    c.rect(0, 0, w, h, fill=1, stroke=0)
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(w / 2, h - 24, title)

    for pi, (key, lab) in enumerate(metrics):
        left = ml0 + pi * (panel_w + gap)
        deltas = np.array([float(trip.loc[g, key] - deep.loc[g, key]) for g in gse_ids], dtype=float)
        lo = float(np.min(deltas))
        hi = float(np.max(deltas))
        pad = 0.05
        ymin = min(0.0, math.floor((lo - pad) * 10) / 10)
        ymax = max(0.0, math.ceil((hi + pad) * 10) / 10)
        if ymax <= ymin:
            ymax = ymin + 0.1

        def x_pt(i: float, _left=left, _pw=panel_w) -> float:
            return _left + (i + 0.5) / n_gse * _pw

        def y_pt(y: float) -> float:
            return mb + (y - ymin) / (ymax - ymin) * plot_h

        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(black)
        c.drawCentredString(left + panel_w / 2, mb + plot_h + 8, f"Delta {lab}")

        for t in _yticks(ymin, ymax, 0.1):
            yy = y_pt(float(t))
            c.setStrokeColor(HexColor("#DDDDDD"))
            c.setDash(2, 2)
            c.line(left, yy, left + panel_w, yy)
            c.setDash()
            if pi == 0:
                c.setFillColor(black)
                c.setFont("Helvetica", 7)
                c.drawRightString(left - 4, yy - 2, f"{t:.1f}")

        c.setStrokeColor(HexColor("#333333"))
        c.rect(left, mb, panel_w, plot_h, fill=0, stroke=1)
        # zero line
        c.setStrokeColor(HexColor("#333333"))
        c.setLineWidth(0.9)
        c.line(left, y_pt(0.0), left + panel_w, y_pt(0.0))

        for gi, d in enumerate(deltas):
            x0 = x_pt(gi - 0.35)
            x1 = x_pt(gi + 0.35)
            y0 = y_pt(0.0)
            y1 = y_pt(float(d))
            bot, top = min(y0, y1), max(y0, y1)
            color = "#2E7D32" if d >= 0 else "#C62828"
            c.setFillColor(_fill_alpha(color, 0.85))
            c.setStrokeColor(HexColor("#333333"))
            c.rect(x0, bot, max(x1 - x0, 0.5), max(top - bot, 0.5), fill=1, stroke=1)

        c.setFont("Helvetica", 6.5)
        c.setFillColor(black)
        for gi, gse in enumerate(gse_ids):
            xx = x_pt(float(gi))
            c.saveState()
            c.translate(xx, mb - 5)
            c.rotate(40)
            c.drawRightString(0, 0, gse)
            c.restoreState()

    c.showPage()
    c.save()


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot DoRothEA ABC Trip vs Deep PDFs")
    ap.add_argument("--experiment-name", default=EXP_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    root = grn_root_from_here()
    exp = root / "experiments" / args.experiment_name
    summary = exp / "summary"
    out_dir = Path(args.out_dir) if args.out_dir else exp / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = _load_folds(summary)
    gse_ids = [g for g in PANCANCER_GSE_IDS if g in set(folds["gse_id"])]
    means = _gse_means(folds, ["accuracy", "macro_auc", "mcc"])
    means.to_csv(out_dir / "plot_gse_means.csv", index=False)
    folds.to_csv(out_dir / "plot_fold_long.csv", index=False)

    # 1) pan-cancer mean bars
    _draw_pan_mean_bars(
        folds,
        out_dir / "pan_mean_trip_vs_deep.pdf",
        f"{NETWORK}: pan-cancer mean +/- SD (100 folds)",
    )
    print("[plot] pan_mean_trip_vs_deep.pdf")

    # 2) per-GSE bars for each metric
    for key, lab in (("accuracy", "Accuracy"), ("macro_auc", "Macro-AUC"), ("mcc", "MCC")):
        vals = means[key].to_numpy(dtype=float)
        ymin, ymax = _ylim_step(vals, step=0.1, ymax_ceil=1.0)
        ymin = min(ymin, 0.3)
        _draw_grouped_metric_bars(
            means,
            metric=key,
            ylabel=lab,
            gse_ids=gse_ids,
            out_path=out_dir / f"per_gse_bars_{key}.pdf",
            title=f"{NETWORK}: {lab} by cohort (mean +/- SD)",
            ymin=ymin,
            ymax=max(ymax, 1.0),
        )
        print(f"[plot] per_gse_bars_{key}.pdf")

    # 3) macro-AUC fold boxplot (reuse KEGG/TRRUST drawer)
    auc_ymin, auc_ymax = _ylim_step(folds["macro_auc"].to_numpy(dtype=float), step=0.1)
    draw_macro_auc_boxplot_pdf(
        folds,
        methods=METHODS,
        gse_ids=gse_ids,
        method_colors=COLORS,
        out_path=out_dir / "macro_auc_boxplot.pdf",
        network_label=NETWORK,
        ymin=auc_ymin,
        ymax=auc_ymax,
        y_step=0.1,
    )
    print("[plot] macro_auc_boxplot.pdf")

    # 4) 3-class OvR AUC
    three = _load_3class_auc_fold_table(summary)
    three.to_csv(out_dir / "plot_3class_auc_fold_long.csv", index=False)
    tmin, tmax = _ylim_step(three["auc"].to_numpy(dtype=float), step=0.1)
    draw_threeclass_auc_boxplot_pdf(
        three,
        gse_ids=gse_ids,
        method_colors=METHOD_COLORS_3CLASS_AUC,
        out_path=out_dir / "threeclass_auc_boxplot.pdf",
        network_label=NETWORK,
        ymin=tmin,
        ymax=tmax,
        y_step=0.1,
    )
    print("[plot] threeclass_auc_boxplot.pdf")

    # 5) delta bars
    _draw_delta_bars(
        means,
        gse_ids,
        out_dir / "per_gse_delta_trip_minus_deep.pdf",
        f"{NETWORK}: TripGRN - DeepFGRN (cohort means)",
    )
    print("[plot] per_gse_delta_trip_minus_deep.pdf")
    print(f"[plot] done -> {out_dir}")


if __name__ == "__main__":
    main()
