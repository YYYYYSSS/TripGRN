#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate DeepFGRN 10-fold CV metrics into summary CSV/JSON (no plots)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from grn.data.pancancer_paths import PANCANCER_GSE_IDS, deepfgrn_fold_dir, scRNA_layout

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
    row: Dict[str, Any] = {"gse_id": gse, "method": "DeepFGRN", "fold": fold_idx}
    for k in METRIC_KEYS:
        row[k] = m.get(k, np.nan)
    for i, name in enumerate(["无调控", "激活", "抑制"]):
        row[f"auc_ovr_{name}"] = (m.get("auc_ovr") or [np.nan] * 3)[i] if m.get("auc_ovr") else np.nan
        row[f"auprc_{name}"] = (m.get("auprc") or [np.nan] * 3)[i] if m.get("auprc") else np.nan
        row[f"acc_class_{name}"] = m.get(f"{name}_accuracy", np.nan)
    row["confusion_matrix"] = json.dumps(m.get("confusion_matrix"))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize DeepFGRN pancancer 10-fold CV")
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
            run_dir = deepfgrn_fold_dir(gse, fold_idx, grn_root)
            if not (run_dir / "unseen_samples_test_metrics.json").is_file():
                print(f"[summarize-deepfgrn] missing {gse} fold {fold_idx:02d}")
                continue
            m = _load_fold_metrics(run_dir)
            row = _flatten_row(gse, fold_idx, m)
            fold_rows.append(row)
            gse_fold_metrics.append(m)

        if not gse_fold_metrics:
            continue
        agg: Dict[str, Any] = {"gse_id": gse, "method": "DeepFGRN", "n_folds": len(gse_fold_metrics)}
        for k in METRIC_KEYS:
            vals = [float(x.get(k, np.nan)) for x in gse_fold_metrics if k in x]
            agg[f"{k}_mean"] = float(np.nanmean(vals)) if vals else np.nan
            agg[f"{k}_std"] = float(np.nanstd(vals)) if vals else np.nan
        gse_summaries.append(agg)

    if fold_rows:
        df_fold = pd.DataFrame(fold_rows)
        df_fold.to_csv(summary_dir / "deepfgrn_10cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    if gse_summaries:
        df_gse = pd.DataFrame(gse_summaries)
        df_gse.to_csv(summary_dir / "deepfgrn_10cv_gse_mean_std.csv", index=False, encoding="utf-8-sig")

    print(f"[summarize-deepfgrn] wrote {len(fold_rows)} fold rows, {len(gse_summaries)} GSE summaries -> {summary_dir}")


if __name__ == "__main__":
    main()
