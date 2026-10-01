#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate scMGATGRN 10-fold CV binary metrics into summary CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from grn.data.pancancer_paths import PANCANCER_GSE_IDS, scRNA_layout, scmgatgrn_fold_dir

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "mcc", "macro_auc", "auc_ovo", "auprc"]


def _load_fold_metrics(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "unseen_samples_test_metrics.json"
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize scMGATGRN pancancer 10-fold CV")
    ap.add_argument("--gse", default="ALL")
    args = ap.parse_args()

    grn_root = Path(__file__).resolve().parents[2]
    summary_dir = scRNA_layout(grn_root)["experiments_summary"]
    summary_dir.mkdir(parents=True, exist_ok=True)

    gses = list(PANCANCER_GSE_IDS) if str(args.gse).upper() == "ALL" else [str(args.gse).strip()]
    fold_rows: List[Dict[str, Any]] = []
    gse_summaries: List[Dict[str, Any]] = []

    for gse in gses:
        gse_metrics: List[Dict[str, Any]] = []
        for fold_idx in range(1, 11):
            run_dir = scmgatgrn_fold_dir(gse, fold_idx, grn_root)
            if not (run_dir / "unseen_samples_test_metrics.json").is_file():
                print(f"[summarize-scmgatgrn] missing {gse} fold {fold_idx:02d}")
                continue
            m = _load_fold_metrics(run_dir)
            row = {"gse_id": gse, "method": "scMGATGRN", "task": "binary", "fold": fold_idx}
            for k in METRIC_KEYS:
                row[k] = m.get(k, np.nan)
            row["confusion_matrix"] = json.dumps(m.get("confusion_matrix"))
            fold_rows.append(row)
            gse_metrics.append(m)
        if not gse_metrics:
            continue
        agg: Dict[str, Any] = {"gse_id": gse, "method": "scMGATGRN", "task": "binary", "n_folds": len(gse_metrics)}
        for k in METRIC_KEYS:
            vals = [float(x.get(k, np.nan)) for x in gse_metrics if k in x]
            agg[f"{k}_mean"] = float(np.nanmean(vals)) if vals else np.nan
            agg[f"{k}_std"] = float(np.nanstd(vals)) if vals else np.nan
        gse_summaries.append(agg)

    if fold_rows:
        pd.DataFrame(fold_rows).to_csv(summary_dir / "scmgatgrn_10cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    if gse_summaries:
        pd.DataFrame(gse_summaries).to_csv(summary_dir / "scmgatgrn_10cv_gse_mean_std.csv", index=False, encoding="utf-8-sig")
    print(f"[summarize-scmgatgrn] wrote {len(fold_rows)} fold rows -> {summary_dir}")


if __name__ == "__main__":
    main()
