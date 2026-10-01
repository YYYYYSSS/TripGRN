#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redraw step01 radars: 5-method binary + TripGRN/DeepFGRN 3-class PDFs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from grn.data.pancancer_paths import PANCANCER_GSE_IDS, grn_root_from_here
from grn.scripts.plot_step01_pancancer_method_comparison import (
    CLASS_LABELS,
    FGRN_CM_IDX,
    TRIP_CM_IDX,
    _parse_cm,
)
from grn.scripts.plot_step01_trrust_metrics import (
    METHOD_COLORS_BINARY,
    METHODS_BINARY,
    METHODS_3CLASS,
    METHOD_COLORS_3CLASS_AUC,
    _gse_method_means,
    draw_metric_radar_pdf,
)

# Keep GSE145825 out of radar axes (matches previously updated boxplots/radars).
EXCLUDE_GSE = {"GSE145825"}

# Binary 5-method radars (overwrite existing radar_*.pdf names where applicable).
BINARY_RADAR_SPECS: List[Tuple[str, str, str]] = [
    # metric_col, metric_label, out_stem
    ("accuracy", "Accuracy (binary)", "radar_accuracy"),
    ("precision", "Precision (binary)", "radar_precision"),
    ("recall", "Recall (binary)", "radar_recall"),
    ("f1", "F1 (binary)", "radar_f1"),
    ("mcc", "MCC (binary)", "radar_mcc"),
    ("auprc", "AUPRC (binary)", "radar_auprc"),
    ("auc", "AUC (binary)", "radar_auc"),
]

# 3-class TripGRN / DeepFGRN radars.
THREECLASS_RADAR_SPECS: List[Tuple[str, str, str]] = [
    ("accuracy", "Accuracy (3-class)", "radar_3class_accuracy"),
    ("precision_macro", "Precision macro (3-class)", "radar_3class_precision"),
    ("recall_macro", "Recall macro (3-class)", "radar_3class_recall"),
    ("f1_macro", "F1 macro (3-class)", "radar_3class_f1"),
    ("mcc", "MCC (3-class)", "radar_3class_mcc"),
    ("auprc_macro", "AUPRC macro (3-class)", "radar_3class_auprc"),
    ("macro_auc", "Macro-AUC (3-class)", "radar_3class_macro_auc"),
    ("auc_ovo", "AUC-OvO (3-class)", "radar_3class_auc_ovo"),
]

# Per-class 3-class radars: one PDF per (class, metric); axes = GSE datasets.
CLASS_PREFIX = {
    "No regulation": "noreg",
    "Activation": "activation",
    "Repression": "repression",
}
CLASS_SHORT = {
    "No regulation": "NoReg",
    "Activation": "Act",
    "Repression": "Rep",
}
PERCLASS_METRIC_SPECS: List[Tuple[str, str]] = [
    ("auprc", "AUPRC"),
    ("mcc", "MCC"),
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
]
THREE_RAW_FILES = {
    "TripGRN": "tripgrn_10cv_fold_metrics.csv",
    "DeepFGRN": "deepfgrn_10cv_fold_metrics.csv",
}


def _network_label(experiment_name: str) -> str:
    name = experiment_name.lower()
    if "dorothea" in name:
        return "DoRothEA A-C"
    if "trrust" in name:
        return "TRRUST TF-target"
    return "KEGG PPrel"


def _safe_f1(precision: float, recall: float) -> float:
    if not (np.isfinite(precision) and np.isfinite(recall)):
        return float("nan")
    denom = precision + recall
    if denom <= 0:
        return 0.0
    return float(2.0 * precision * recall / denom)


def _cm_to_y(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt: List[int] = []
    yp: List[int] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            n = int(cm[i, j])
            yt.extend([i] * n)
            yp.extend([j] * n)
    return np.asarray(yt), np.asarray(yp)


def _ovr_mcc_from_cm(cm: np.ndarray, class_idx: int) -> float:
    yt, yp = _cm_to_y(cm)
    yt_b = (yt == class_idx).astype(int)
    yp_b = (yp == class_idx).astype(int)
    if len(np.unique(yt_b)) < 2 or len(np.unique(yp_b)) < 2:
        return float("nan")
    return float(matthews_corrcoef(yt_b, yp_b))


def _enrich_perclass_f1_mcc(three_df: pd.DataFrame, summary: Path) -> pd.DataFrame:
    """Add per-class F1 (from P/R) and OvR MCC (from confusion matrices)."""
    out = three_df.copy()
    for cls in CLASS_LABELS:
        pfx = CLASS_PREFIX[cls]
        prec = pd.to_numeric(out[f"{pfx}_precision"], errors="coerce")
        rec = pd.to_numeric(out[f"{pfx}_recall"], errors="coerce")
        out[f"{pfx}_f1"] = [_safe_f1(float(p), float(r)) for p, r in zip(prec, rec)]
        out[f"{pfx}_mcc"] = np.nan

    mcc_rows: List[dict] = []
    for method, fname in THREE_RAW_FILES.items():
        path = summary / fname
        if not path.is_file():
            continue
        raw = pd.read_csv(path)
        idx_map = TRIP_CM_IDX if method == "TripGRN" else FGRN_CM_IDX
        for _, r in raw.iterrows():
            cm = _parse_cm(r["confusion_matrix"])
            row = {
                "gse_id": str(r["gse_id"]),
                "method": method,
                "fold": int(r["fold"]),
            }
            for cls in CLASS_LABELS:
                row[f"{CLASS_PREFIX[cls]}_mcc"] = _ovr_mcc_from_cm(cm, idx_map[cls])
            mcc_rows.append(row)
    if not mcc_rows:
        return out

    mcc_df = pd.DataFrame(mcc_rows)
    key = ["gse_id", "method", "fold"]
    out = out.drop(columns=[f"{CLASS_PREFIX[c]}_mcc" for c in CLASS_LABELS], errors="ignore")
    out = out.merge(mcc_df, on=key, how="left")
    return out


def _perclass_radar_specs() -> List[Tuple[str, str, str]]:
    """(column, label, out_stem) for each class x metric; axes remain GSE."""
    specs: List[Tuple[str, str, str]] = []
    for cls in CLASS_LABELS:
        pfx = CLASS_PREFIX[cls]
        short = CLASS_SHORT[cls]
        for metric_key, metric_label in PERCLASS_METRIC_SPECS:
            col = f"{pfx}_{metric_key}"
            label = f"{metric_label} ({short}, 3-class)"
            stem = f"radar_3class_{pfx}_{metric_key}"
            specs.append((col, label, stem))
    return specs


def _plot_perclass_metric_radars(
    three_df: pd.DataFrame,
    *,
    summary: Path,
    methods: Sequence[str],
    method_colors: Dict[str, str],
    gse_ids: Sequence[str],
    out_dir: Path,
    network_label: str,
) -> None:
    enriched = _enrich_perclass_f1_mcc(three_df, summary)
    _plot_radar_set(
        enriched,
        specs=_perclass_radar_specs(),
        methods=methods,
        method_colors=method_colors,
        gse_ids=gse_ids,
        out_dir=out_dir,
        network_label=network_label,
        combined_csv_name="radar_3class_perclass_all_metrics_gse_means.csv",
    )


def _plot_radar_set(
    fold_df: pd.DataFrame,
    *,
    specs: Sequence[Tuple[str, str, str]],
    methods: Sequence[str],
    method_colors: Dict[str, str],
    gse_ids: Sequence[str],
    out_dir: Path,
    network_label: str,
    combined_csv_name: str,
) -> None:
    mean_rows: List[dict] = []
    for metric_key, metric_label, stem in specs:
        if metric_key not in fold_df.columns:
            print(f"[skip] missing column {metric_key}")
            continue
        mean_df = _gse_method_means(fold_df, metric_key, methods, gse_ids)
        mean_df.to_csv(out_dir / f"{stem}_gse_means.csv", index=False, encoding="utf-8-sig")
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
        out_pdf = out_dir / f"{stem}.pdf"
        draw_metric_radar_pdf(
            mean_df,
            metric_key=metric_key,
            metric_label=metric_label,
            methods=methods,
            gse_ids=gse_ids,
            method_colors=method_colors,
            out_path=out_pdf,
            network_label=network_label,
            ymin=0.0,
            ymax=1.0,
        )
        print(f"[radar] {out_pdf.name}")

    if mean_rows:
        pd.DataFrame(mean_rows).to_csv(out_dir / combined_csv_name, index=False, encoding="utf-8-sig")


def export_one(exp: str, root: Path, *, perclass_only: bool = False) -> None:
    summary = root / "experiments" / exp / "summary"
    out_dir = root / "experiments" / exp / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    network_label = _network_label(exp)

    three_path = summary / "threeclass_trip_deep_fold_metrics.csv"
    if not three_path.is_file():
        raise FileNotFoundError(
            f"Missing {three_path}. "
            "Run: python -m grn.scripts.export_binary_and_threeclass_summary_tables"
        )
    three_df = pd.read_csv(three_path)
    three_df = three_df[~three_df["gse_id"].isin(EXCLUDE_GSE)].copy()

    gse_ids = [g for g in PANCANCER_GSE_IDS if g not in EXCLUDE_GSE and g in set(three_df["gse_id"])]
    if not gse_ids:
        gse_ids = sorted(three_df["gse_id"].unique().tolist())

    print(f"==== {exp} | network={network_label} | n_gse={len(gse_ids)} ====")

    three_colors = {
        "TripGRN": METHOD_COLORS_BINARY["TripGRN"],
        "DeepFGRN": METHOD_COLORS_BINARY["DeepFGRN"],
    }
    three_colors.update({k: METHOD_COLORS_3CLASS_AUC[k] for k in METHODS_3CLASS if k in METHOD_COLORS_3CLASS_AUC})

    if not perclass_only:
        bin_path = summary / "binary_five_method_fold_metrics.csv"
        if not bin_path.is_file():
            alt = summary / "binary_five_method_fold_metrics_strict.csv"
            if alt.is_file():
                bin_path = alt
            else:
                raise FileNotFoundError(f"Missing binary fold metrics under {summary}")
        bin_df = pd.read_csv(bin_path)
        bin_df = bin_df[~bin_df["gse_id"].isin(EXCLUDE_GSE)].copy()

        _plot_radar_set(
            bin_df,
            specs=BINARY_RADAR_SPECS,
            methods=METHODS_BINARY,
            method_colors=METHOD_COLORS_BINARY,
            gse_ids=gse_ids,
            out_dir=out_dir,
            network_label=network_label,
            combined_csv_name="radar_binary_all_metrics_gse_means.csv",
        )
        auc_pdf = out_dir / "radar_auc.pdf"
        alias = out_dir / "radar_auc_ovo.pdf"
        if auc_pdf.is_file():
            alias.write_bytes(auc_pdf.read_bytes())
            means = out_dir / "radar_auc_gse_means.csv"
            if means.is_file():
                (out_dir / "radar_auc_ovo_gse_means.csv").write_bytes(means.read_bytes())
            print(f"[radar] alias {alias.name} <- {auc_pdf.name}")

        _plot_radar_set(
            three_df,
            specs=THREECLASS_RADAR_SPECS,
            methods=METHODS_3CLASS,
            method_colors=three_colors,
            gse_ids=gse_ids,
            out_dir=out_dir,
            network_label=network_label,
            combined_csv_name="radar_3class_all_metrics_gse_means.csv",
        )

    _plot_perclass_metric_radars(
        three_df,
        summary=summary,
        methods=METHODS_3CLASS,
        method_colors=three_colors,
        gse_ids=gse_ids,
        out_dir=out_dir,
        network_label=network_label,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot binary + 3-class step01 radar PDFs")
    ap.add_argument("--experiment-name", action="append", default=None)
    ap.add_argument(
        "--perclass-only",
        action="store_true",
        help="Only draw Trip/Deep per-class (NoReg/Act/Rep) metric radars",
    )
    args = ap.parse_args()
    root = grn_root_from_here()
    exps = args.experiment_name or [
        "step01_pancancer_epithelial_10cv",
        "step01_pancancer_epithelial_10cv_dorothea_abc",
    ]
    for exp in exps:
        export_one(exp, root, perclass_only=args.perclass_only)


if __name__ == "__main__":
    main()
