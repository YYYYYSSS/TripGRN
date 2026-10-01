#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate TripGRN 10-fold CV metrics into summary CSV/JSON (no plots)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from grn.data.pancancer_paths import PANCANCER_GSE_IDS, scRNA_layout, tripgrn_fold_dir

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
]


def _load_fold_metrics(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "unseen_samples_test_metrics.json"
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def _flatten_row(gse: str, fold_idx: int, m: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"gse_id": gse, "method": "TripGRN", "fold": fold_idx}
    for k in METRIC_KEYS:
        row[k] = m.get(k, np.nan)
    for i, name in enumerate(["无调控", "激活", "抑制"]):
        row[f"auc_ovr_{name}"] = (m.get("auc_ovr") or [np.nan] * 3)[i] if m.get("auc_ovr") else np.nan
        row[f"auprc_{name}"] = (m.get("auprc") or [np.nan] * 3)[i] if m.get("auprc") else np.nan
        row[f"specificity_{name}"] = (m.get("specificity_ovr") or [np.nan] * 3)[i] if m.get("specificity_ovr") else np.nan
        row[f"acc_class_{name}"] = m.get(f"{name}_accuracy", np.nan)
    row["confusion_matrix"] = json.dumps(m.get("confusion_matrix"))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize TripGRN pancancer 10-fold CV")
    ap.add_argument("--gse", default="ALL")
    args = ap.parse_args()

    grn_root = Path(__file__).resolve().parents[2]
    layout = scRNA_layout(grn_root)
    summary_dir = layout["experiments_summary"]
    summary_dir.mkdir(parents=True, exist_ok=True)

    gses = list(PANCANCER_GSE_IDS) if str(args.gse).upper() == "ALL" else [str(args.gse).strip()]
    fold_rows: List[Dict[str, Any]] = []
    gse_summaries: List[Dict[str, Any]] = []

    for gse in gses:
        gse_fold_metrics: List[Dict[str, Any]] = []
        for fold_idx in range(1, 11):
            run_dir = tripgrn_fold_dir(gse, fold_idx, grn_root)
            if not (run_dir / "unseen_samples_test_metrics.json").is_file():
                print(f"[summarize] missing {gse} fold {fold_idx:02d}")
                continue
            m = _load_fold_metrics(run_dir)
            row = _flatten_row(gse, fold_idx, m)
            fold_rows.append(row)
            gse_fold_metrics.append(m)

        if not gse_fold_metrics:
            continue
        agg: Dict[str, Any] = {"gse_id": gse, "method": "TripGRN", "n_folds": len(gse_fold_metrics)}
        for k in METRIC_KEYS:
            vals = [float(x.get(k, np.nan)) for x in gse_fold_metrics]
            agg[f"{k}_mean"] = float(np.nanmean(vals))
            agg[f"{k}_std"] = float(np.nanstd(vals))
        gse_summaries.append(agg)

    fold_csv = summary_dir / "tripgrn_10cv_fold_metrics.csv"
    gse_csv = summary_dir / "tripgrn_10cv_gse_mean_std.csv"
    pd.DataFrame(fold_rows).to_csv(fold_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(gse_summaries).to_csv(gse_csv, index=False, encoding="utf-8-sig")

    manifest = {
        "method": "TripGRN",
        "n_fold_rows": len(fold_rows),
        "n_gse_summaries": len(gse_summaries),
        "fold_metrics_csv": str(fold_csv.resolve()),
        "gse_mean_std_csv": str(gse_csv.resolve()),
    }
    (summary_dir / "tripgrn_10cv_summary_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[summarize] fold rows -> {fold_csv}")
    print(f"[summarize] gse mean±sd -> {gse_csv}")


if __name__ == "__main__":
    main()
