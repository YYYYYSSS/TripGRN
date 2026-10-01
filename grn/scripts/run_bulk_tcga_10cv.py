#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrate TCGA bulk cross-cancer 10-fold CV (12 cancers x 10 folds).

Uses shuffled 5:2:3 unseen-sample protocol (same as step01 scRNA 10CV).
Trainer: train_unseen_samples_no_cv (full modalities: expr + CDS + TSS + PPI).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from grn.data.bulk_paths import (
    DEFAULT_BULK_EXPERIMENT,
    TCGA_CANCER_IDS,
    bulk_exp_csv,
    bulk_fold_dir,
    bulk_layout,
    cv_splits_path,
)
from grn.data.pancancer_cv import DEFAULT_N_FOLDS, get_fold_split, load_cv_splits


def _resolve_folds(fold_arg: str, n_folds: int) -> List[int]:
    fa = str(fold_arg).strip().upper()
    if fa == "ALL":
        return list(range(1, n_folds + 1))
    return [int(fa)]


def _resolve_cancers(cancer_arg: str) -> List[str]:
    if str(cancer_arg).upper() == "ALL":
        return list(TCGA_CANCER_IDS)
    c = str(cancer_arg).strip()
    if not c.upper().startswith("TCGA-"):
        c = f"TCGA-{c}"
    return [c]


def _fold_done(cancer_id: str, fold_idx: int, grn_root: Path) -> bool:
    return (bulk_fold_dir(cancer_id, fold_idx, grn_root) / "unseen_samples_test_metrics.json").is_file()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run TCGA bulk 10-fold CV")
    ap.add_argument("--cancer", default="ALL")
    ap.add_argument("--fold", default="ALL", help="1..10 or ALL")
    ap.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--experiment-name", default=DEFAULT_BULK_EXPERIMENT)
    args = ap.parse_args()

    layout = bulk_layout()
    grn_root = layout["grn_root"]
    out_base = layout["experiment_base"]

    for key in ("grn_csv", "cds_gene", "tss_gene", "ppi_csv"):
        if not layout[key].is_file():
            raise FileNotFoundError(f"Missing {key}: {layout[key]}")

    cancers = _resolve_cancers(args.cancer)
    folds = _resolve_folds(args.fold, int(args.n_folds))
    n_run = n_skip = n_miss = 0

    for cancer in cancers:
        exp_csv = bulk_exp_csv(cancer, grn_root)
        splits_path = cv_splits_path(cancer, grn_root)
        if not exp_csv.is_file():
            print(f"[bulk-10cv] skip {cancer} — missing CSV: {exp_csv}")
            n_miss += 1
            continue
        if not splits_path.is_file():
            print(f"[bulk-10cv] skip {cancer} — missing CV splits: {splits_path}")
            n_miss += 1
            continue

        for fold_idx in folds:
            if _fold_done(cancer, fold_idx, grn_root) and not args.force:
                print(f"[bulk-10cv] skip {cancer} fold {fold_idx:02d}")
                n_skip += 1
                continue

            fold = get_fold_split(load_cv_splits(splits_path), fold_idx)
            n_train = len(fold["train_cols"])
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "grn.scripts.train_unseen_samples_no_cv",
                "--exp-csv",
                str(exp_csv),
                "--grn-csv",
                str(layout["grn_csv"]),
                "--cds-gene-feature-csv",
                str(layout["cds_gene"]),
                "--tss-gene-feature-csv",
                str(layout["tss_gene"]),
                "--ppi-csv",
                str(layout["ppi_csv"]),
                "--output-base-dir",
                str(grn_root / "experiments" / args.experiment_name),
                "--experiment-name",
                cancer,
                "--no-timestamp",
                "--method-name",
                "TripGRN",
                "--cv-splits-json",
                str(splits_path),
                "--fold-idx",
                str(fold_idx),
                "--skip-plots",
                "--seed",
                str(args.seed),
                "--batch-size",
                str(args.batch_size),
                "--epochs",
                str(args.epochs),
                "--lr",
                str(args.lr),
                "--patience",
                str(args.patience),
                "--neg-sample-ratio",
                "1.0",
            ]
            print(f"[bulk-10cv] run {cancer} fold {fold_idx:02d} (train_cols={n_train})")
            subprocess.run(cmd, check=True, cwd=str(grn_root))
            n_run += 1

    print(f"[bulk-10cv] finished: ran={n_run}, skipped={n_skip}, missing={n_miss}")


if __name__ == "__main__":
    main()
