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


def _runtime_wall(run_dir: Path) -> float:
    p = run_dir / "runtime.json"
    if not p.is_file():
        return float("nan")
    try:
        return float(json.loads(p.read_text(encoding="utf-8")).get("wall_time_sec", float("nan")))
    except Exception:
        return float("nan")


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
            row["n_genes_used"] = m.get("n_genes_used", np.nan)
            row["gene_policy"] = m.get("gene_policy", "")
            row["n_cells_train"] = m.get("n_cells_train", np.nan)
            row["wall_time_sec"] = _runtime_wall(run_dir)
            fold_rows.append(row)
            gse_metrics.append(m)
        if not gse_metrics:
            continue
        agg: Dict[str, Any] = {"gse_id": gse, "method": "scMGATGRN", "task": "binary", "n_folds": len(gse_metrics)}
        for k in METRIC_KEYS:
            vals = [float(x.get(k, np.nan)) for x in gse_metrics if k in x]
            agg[f"{k}_mean"] = float(np.nanmean(vals)) if vals else np.nan
            agg[f"{k}_std"] = float(np.nanstd(vals)) if vals else np.nan
        wall_vals = [float(r["wall_time_sec"]) for r in fold_rows if r["gse_id"] == gse]
        agg["wall_time_sec_mean"] = float(np.nanmean(wall_vals)) if wall_vals else float("nan")
        agg["wall_time_sec_sum"] = float(np.nansum(wall_vals)) if wall_vals else float("nan")
        gse_summaries.append(agg)

    if fold_rows:
        pd.DataFrame(fold_rows).to_csv(summary_dir / "scmgatgrn_10cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    if gse_summaries:
        pd.DataFrame(gse_summaries).to_csv(summary_dir / "scmgatgrn_10cv_gse_mean_std.csv", index=False, encoding="utf-8-sig")
    print(f"[summarize-scmgatgrn] wrote {len(fold_rows)} fold rows -> {summary_dir}")


if __name__ == "__main__":
    main()
