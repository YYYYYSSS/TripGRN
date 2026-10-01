#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GSE189357 transfer: macro ROC, macro PR, and per-class ROC PDFs."""

from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from grn.evaluation.logit_calibration import apply_temperature_and_bias
from grn.reproducibility import set_seed
from grn.scripts.evaluate_step1_on_external_expression import _collect_logits_numpy
from grn.scripts.regenerate_unseen_run_plots_from_checkpoint import _build_pipeline
from grn.scripts.train_unseen_samples_no_cv import _predict_arrays

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/old/benchmark_luad_gse253718_step3_cross_dataset_gse189357"
FIG = EXP / "figures"

# Model column order: NoReg, Repression, Activation
CLASS_FULL = ("No regulation", "Activation", "Repression")
CLASS_MODEL_IDX = (0, 2, 1)
CLASS_COLORS = {
    "No regulation": "#4E79A7",
    "Activation": "#59A14F",
    "Repression": "#E15759",
}
FPR_GRID = np.linspace(0.0, 1.0, 201)
RECALL_GRID = np.linspace(0.0, 1.0, 201)


def _load_test_probs() -> Tuple[np.ndarray, np.ndarray]:
    cfg = json.loads((EXP / "run_config.json").read_text(encoding="utf-8"))
    calib = json.loads((EXP / "posthoc_calibration.json").read_text(encoding="utf-8"))
    ckpt = EXP / "best_model_fold0.pth"
    if not ckpt.is_file():
        ckpt = EXP / "fusion_finetuned_best.pth"
    set_seed(int(cfg.get("seed", 42)))
    model, _, _val_loader, test_loader, device, use_amp_bf16 = _build_pipeline(
        Namespace(**cfg),
        norm_fit_all_columns=True,
        column_splits=None,
    )
    import torch

    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    model.eval()
    logits, y = _collect_logits_numpy(model, test_loader, device, use_amp_bf16)
    T = float(calib["temperature"])
    bias = np.asarray(calib["bias"], dtype=np.float32)
    prob = apply_temperature_and_bias(logits, T, bias)
    return np.asarray(y, dtype=np.int64), np.asarray(prob, dtype=np.float64)


def _per_class_roc(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, Dict[str, np.ndarray | float]]:
    out: Dict[str, Dict[str, np.ndarray | float]] = {}
    for name, idx in zip(CLASS_FULL, CLASS_MODEL_IDX):
        y_bin = (y_true == idx).astype(int)
        score = y_prob[:, idx]
        fpr, tpr, _ = roc_curve(y_bin, score)
        try:
            auc_v = float(roc_auc_score(y_bin, score))
        except ValueError:
            auc_v = float("nan")
        tpr_i = np.interp(FPR_GRID, fpr, tpr)
        tpr_i[0] = 0.0
        out[name] = {"fpr": FPR_GRID, "tpr": tpr_i, "fpr_raw": fpr, "tpr_raw": tpr, "auc": auc_v}
    return out


def _per_class_pr(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, Dict[str, np.ndarray | float]]:
    out: Dict[str, Dict[str, np.ndarray | float]] = {}
    for name, idx in zip(CLASS_FULL, CLASS_MODEL_IDX):
        y_bin = (y_true == idx).astype(int)
        score = y_prob[:, idx]
        precision, recall, _ = precision_recall_curve(y_bin, score)
        try:
            ap = float(average_precision_score(y_bin, score))
        except ValueError:
            ap = float("nan")
        # interpolate precision vs increasing recall for macro average
        order = np.argsort(recall)
        r_s = recall[order]
        p_s = precision[order]
        # precision_recall_curve is decreasing recall; unique for interp
        r_u, idx_u = np.unique(r_s, return_index=True)
        p_u = p_s[idx_u]
        p_i = np.interp(RECALL_GRID, r_u, p_u)
        out[name] = {
            "recall": recall,
            "precision": precision,
            "recall_grid": RECALL_GRID,
            "precision_i": p_i,
            "auprc": ap,
        }
    return out


def _style_axes(ax) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.28, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_macro_roc_pdf(per: Dict, out: Path) -> None:
    tprs = [np.asarray(per[c]["tpr"], dtype=float) for c in CLASS_FULL]
    aucs = [float(per[c]["auc"]) for c in CLASS_FULL]
    mean_tpr = np.mean(np.vstack(tprs), axis=0)
    macro_auc = float(np.mean(aucs))

    plt.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.0, label="Chance")
    ax.plot(
        FPR_GRID,
        mean_tpr,
        color="#4C78A8",
        linewidth=2.2,
        label=f"Macro-average ROC (AUROC={macro_auc:.3f})",
    )
    _style_axes(ax)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("GSE253718 -> GSE189357: macro-AUROC ROC curve")
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}  macro_auc={macro_auc:.6f}")


def draw_macro_pr_pdf(per: Dict, out: Path) -> None:
    precs = [np.asarray(per[c]["precision_i"], dtype=float) for c in CLASS_FULL]
    aps = [float(per[c]["auprc"]) for c in CLASS_FULL]
    mean_p = np.mean(np.vstack(precs), axis=0)
    macro_ap = float(np.mean(aps))

    plt.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot(
        RECALL_GRID,
        mean_p,
        color="#E15759",
        linewidth=2.2,
        label=f"Macro-average PR (AUPRC={macro_ap:.3f})",
    )
    _style_axes(ax)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("GSE253718 -> GSE189357: macro-AUPRC PR curve")
    ax.legend(loc="lower left", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}  macro_auprc={macro_ap:.6f}")


def draw_perclass_roc_pdf(per: Dict, out: Path) -> None:
    plt.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.0, label="Chance")
    for name in CLASS_FULL:
        d = per[name]
        ax.plot(
            np.asarray(d["fpr_raw"], dtype=float),
            np.asarray(d["tpr_raw"], dtype=float),
            color=CLASS_COLORS[name],
            linewidth=2.0,
            label=f"{name} (AUROC={float(d['auc']):.3f})",
        )
    _style_axes(ax)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("GSE253718 -> GSE189357: class-specific ROC (one-vs-rest)")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def draw_other_metrics_radar_pdf(m: dict, out: Path) -> None:
    """Radar: start at prior F1 angle, then clockwise Macro-AUROC ... F1(macro)."""
    import math

    from reportlab.lib.colors import Color, HexColor, black, white
    from reportlab.pdfgen import canvas

    macro_auprc = float(sum(m["auprc"]) / len(m["auprc"]))
    # Clockwise order starting at the previous F1 axis position.
    specs = [
        ("Macro-AUROC", float(m["macro_auc"])),
        ("Macro-AUPRC", macro_auprc),
        ("MCC", float(m["mcc"])),
        ("Accuracy", float(m["accuracy"])),
        ("Precision (macro)", float(m["precision_macro"])),
        ("Recall (macro)", float(m["recall_macro"])),
        ("F1 (macro)", float(m["f1_macro"])),
    ]
    labels = [s[0] for s in specs]
    values = [s[1] for s in specs]
    n = len(labels)

    page_w, page_h = 600.0, 600.0
    c = canvas.Canvas(str(out), pagesize=(page_w, page_h))
    c.setFillColor(white)
    c.rect(0, 0, page_w, page_h, fill=1, stroke=0)

    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(page_w / 2, page_h - 34, "Cross-dataset transfer: metrics radar")
    c.setFont("Helvetica", 9)
    c.setFillColor(HexColor("#555555"))
    c.drawCentredString(
        page_w / 2,
        page_h - 50,
        "GSE253718 -> GSE189357 (calibrated test); axes = metrics; scale 0-1",
    )

    cx, cy = page_w / 2, page_h / 2 - 10
    radius = min(page_w, page_h) * 0.28
    ymin, ymax = 0.0, 1.0
    # Previous layout: i=0 at top, counterclockwise; F1 was index 4.
    # Keep that angle as the first axis, then advance clockwise.
    start_angle = 2 * math.pi * 4 / n - math.pi / 2
    angles = [start_angle - 2 * math.pi * i / n for i in range(n)]

    def polar(i: int, v: float) -> tuple[float, float]:
        t = max(0.0, min(1.0, (v - ymin) / (ymax - ymin)))
        a = angles[i]
        return cx + radius * t * math.cos(a), cy + radius * t * math.sin(a)

    grid = HexColor("#DDDDDD")
    muted = HexColor("#555555")
    c.setFont("Helvetica", 8)
    for ring in (0.25, 0.5, 0.75, 1.0):
        pts = [polar(i, ring) for i in range(n)]
        p = c.beginPath()
        p.moveTo(pts[0][0], pts[0][1])
        for x, y in pts[1:]:
            p.lineTo(x, y)
        p.close()
        c.setStrokeColor(grid)
        c.setLineWidth(0.8)
        c.drawPath(p, fill=0, stroke=1)
        c.setFillColor(muted)
        tx, ty = pts[0]
        c.drawRightString(tx - 4, ty - 3, f"{ring:.2f}")

    for i, lab in enumerate(labels):
        x, y = polar(i, ymax)
        c.setStrokeColor(HexColor("#CCCCCC"))
        c.setLineWidth(0.8)
        c.line(cx, cy, x, y)
        lx = cx + (radius + 44) * math.cos(angles[i])
        ly = cy + (radius + 44) * math.sin(angles[i])
        c.setFillColor(black)
        c.setFont("Helvetica", 8)
        if "(" in lab:
            a, b = lab.split(" (", 1)
            c.drawCentredString(lx, ly + 4, a)
            c.drawCentredString(lx, ly - 8, "(" + b)
        else:
            c.drawCentredString(lx, ly - 3, lab)

    pts = [polar(i, v) for i, v in enumerate(values)]
    p = c.beginPath()
    p.moveTo(pts[0][0], pts[0][1])
    for x, y in pts[1:]:
        p.lineTo(x, y)
    p.close()
    fill = HexColor("#4C78A8")
    soft = Color(76 / 255.0, 120 / 255.0, 168 / 255.0, alpha=0.28)
    c.setFillColor(soft)
    c.setStrokeColor(fill)
    c.setLineWidth(2.0)
    c.drawPath(p, fill=1, stroke=1)

    for pt, v in zip(pts, values):
        c.setFillColor(fill)
        c.circle(pt[0], pt[1], 3.2, fill=1, stroke=0)
        c.setFillColor(black)
        c.setFont("Helvetica", 7.5)
        c.drawCentredString(pt[0], pt[1] + 6, f"{v:.3f}")

    c.setFillColor(fill)
    c.circle(page_w / 2 - 70, 28, 5, fill=1, stroke=0)
    c.setFillColor(black)
    c.setFont("Helvetica", 9)
    c.drawString(page_w / 2 - 60, 25, "TripGRN (finetune + calib.)")
    c.showPage()
    c.save()
    print(f"[fig] {out}")
    for lab, v in specs:
        print(f"  {lab}: {v:.6f}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--radar-only", action="store_true", help="Only redraw metrics radar from saved JSON")
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    if args.radar_only:
        m = json.loads((EXP / "unseen_samples_test_metrics.json").read_text(encoding="utf-8"))
        draw_other_metrics_radar_pdf(m, FIG / "fig_gse189357_other_metrics_radar.pdf")
        return

    print("[load] predicting calibrated test probs ...", flush=True)
    y_true, y_prob = _load_test_probs()
    print(f"[load] n_test={len(y_true)}  class_counts={np.bincount(y_true, minlength=3)}", flush=True)

    per_roc = _per_class_roc(y_true, y_prob)
    per_pr = _per_class_pr(y_true, y_prob)

    for stale in (
        "fig_gse189357_macro_auroc_auprc.pdf",
        "fig_gse189357_perclass_auroc.pdf",
    ):
        p = FIG / stale
        if p.is_file():
            p.unlink()
            print(f"[rm] {p.name}")

    draw_macro_roc_pdf(per_roc, FIG / "fig_gse189357_macro_roc.pdf")
    draw_macro_pr_pdf(per_pr, FIG / "fig_gse189357_macro_pr.pdf")
    draw_perclass_roc_pdf(per_roc, FIG / "fig_gse189357_perclass_roc.pdf")
    m = json.loads((EXP / "unseen_samples_test_metrics.json").read_text(encoding="utf-8"))
    draw_other_metrics_radar_pdf(m, FIG / "fig_gse189357_other_metrics_radar.pdf")


if __name__ == "__main__":
    main()
