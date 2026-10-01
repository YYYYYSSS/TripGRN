#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate TCGA bulk 10-fold CV metrics into summary CSV/JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from grn.data.bulk_paths import TCGA_CANCER_IDS, bulk_fold_dir, bulk_layout

METRIC_KEYS = [
    "accuracy",
    "precision",
    "precision_macro",
    "recall",
    "recall_macro",
    "f1",
    "f1_macro",
    "mcc",
    "macro_auc",
    "auc_ovo",
    "macro_auprc",
]


def _load_fold_metrics(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "unseen_samples_test_metrics.json"
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def _macro_auprc(m: Dict[str, Any]) -> float:
    """Macro AUPRC = mean of per-class one-vs-rest AUPRC."""
    vals = m.get("auprc")
    if not isinstance(vals, list) or not vals:
        return float("nan")
    arr = np.asarray(vals, dtype=float)
    return float(np.nanmean(arr))


def _flatten_row(cancer: str, fold_idx: int, m: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"cancer": cancer, "method": "TripGRN", "fold": fold_idx}
    for k in METRIC_KEYS:
        if k == "macro_auprc":
            row[k] = _macro_auprc(m)
        else:
            row[k] = m.get(k, np.nan)
    for i, name in enumerate(["无调控", "激活", "抑制"]):
        row[f"auc_ovr_{name}"] = (m.get("auc_ovr") or [np.nan] * 3)[i] if m.get("auc_ovr") else np.nan
        row[f"auprc_{name}"] = (m.get("auprc") or [np.nan] * 3)[i] if m.get("auprc") else np.nan
        row[f"acc_class_{name}"] = m.get(f"{name}_accuracy", np.nan)
    row["confusion_matrix"] = json.dumps(m.get("confusion_matrix"))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize TCGA bulk 10-fold CV")
    ap.add_argument("--cancer", default="ALL")
    args = ap.parse_args()

    layout = bulk_layout()
    grn_root = layout["grn_root"]
    summary_dir = layout["summary_dir"]
    summary_dir.mkdir(parents=True, exist_ok=True)

    cancers = list(TCGA_CANCER_IDS) if str(args.cancer).upper() == "ALL" else [str(args.cancer).strip()]
    fold_rows: List[Dict[str, Any]] = []
    cancer_summaries: List[Dict[str, Any]] = []

    for cancer in cancers:
        cancer_fold_metrics: List[Dict[str, Any]] = []
        for fold_idx in range(1, 11):
            run_dir = bulk_fold_dir(cancer, fold_idx, grn_root)
            if not (run_dir / "unseen_samples_test_metrics.json").is_file():
                print(f"[bulk-summarize] missing {cancer} fold {fold_idx:02d}")
                continue
            m = _load_fold_metrics(run_dir)
            row = _flatten_row(cancer, fold_idx, m)
            fold_rows.append(row)
            cancer_fold_metrics.append(m)

        if not cancer_fold_metrics:
            continue
        agg: Dict[str, Any] = {"cancer": cancer, "method": "TripGRN", "n_folds": len(cancer_fold_metrics)}
        for k in METRIC_KEYS:
            if k == "macro_auprc":
                vals = [_macro_auprc(x) for x in cancer_fold_metrics]
            else:
                vals = [float(x.get(k, np.nan)) for x in cancer_fold_metrics]
            agg[f"{k}_mean"] = float(np.nanmean(vals))
            agg[f"{k}_std"] = float(np.nanstd(vals))
        cancer_summaries.append(agg)

    if not fold_rows:
        raise SystemExit("No fold metrics found; run run_bulk_tcga_10cv first.")

    fold_csv = summary_dir / "bulk_10cv_fold_metrics.csv"
    cancer_csv = summary_dir / "bulk_10cv_cancer_mean_std.csv"
    pd.DataFrame(fold_rows).to_csv(fold_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(cancer_summaries).to_csv(cancer_csv, index=False, encoding="utf-8-sig")

    manifest = {
        "method": "TripGRN",
        "protocol": "10-fold CV, shuffled 5:2:3 unseen samples",
        "n_fold_rows": len(fold_rows),
        "n_cancer_summaries": len(cancer_summaries),
        "fold_metrics_csv": str(fold_csv.resolve()),
        "cancer_mean_std_csv": str(cancer_csv.resolve()),
    }
    (summary_dir / "bulk_10cv_summary_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[bulk-summarize] fold rows -> {fold_csv}")
    print(f"[bulk-summarize] cancer mean±sd -> {cancer_csv}")


if __name__ == "__main__":
    main()
