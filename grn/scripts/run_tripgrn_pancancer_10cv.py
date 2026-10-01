#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrate TripGRN 10-fold CV on epithelial pancancer datasets.

Usage:
  python -m grn.scripts.run_tripgrn_pancancer_10cv --gse ALL --fold ALL
  python -m grn.scripts.run_tripgrn_pancancer_10cv --gse GSE103322 --fold 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np

from grn.data.pancancer_cv import DEFAULT_N_FOLDS, get_fold_split, load_cv_splits
from grn.data.pancancer_paths import (
    PANCANCER_GSE_IDS,
    cv_splits_path,
    gse_npy_triplet,
    scRNA_layout,
    step01_grn_csv,
    tripgrn_fold_dir,
)
from grn.training.pair_dataloader import suggest_batch_size


def _resolve_folds(fold_arg: str, n_folds: int) -> List[int]:
    fa = str(fold_arg).strip().upper()
    if fa == "ALL":
        return list(range(1, n_folds + 1))
    return [int(fa)]


def _resolve_gses(gse_arg: str) -> List[str]:
    if str(gse_arg).upper() == "ALL":
        return list(PANCANCER_GSE_IDS)
    return [str(gse_arg).strip()]


def _fold_done(run_dir: Path) -> bool:
    return (run_dir / "unseen_samples_test_metrics.json").is_file()


def _train_cols_for_fold(splits_path: Path, fold_idx: int) -> int:
    payload = load_cv_splits(splits_path)
    fold = get_fold_split(payload, fold_idx)
    return int(len(fold["train_cols"]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run TripGRN pancancer epithelial 10-fold CV")
    ap.add_argument("--gse", default="ALL")
    ap.add_argument("--fold", default="ALL", help="Fold index 1..10 or ALL")
    ap.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--force", action="store_true", help="Retrain even if metrics exist")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=0, help="0 = auto from train column count")
    ap.add_argument("--batch-size-cap", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument(
        "--expression-gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (slower, saves VRAM). Off by default for speed.",
    )
    ap.add_argument("--compile-model", action="store_true", help="torch.compile (PyTorch 2+)")
    args = ap.parse_args()

    grn_root = Path(__file__).resolve().parents[2]
    layout = scRNA_layout(grn_root)
    grn_csv = step01_grn_csv(grn_root)
    cds_gene = grn_root / "data" / "CDS" / "features" / "gene_shared" / "cds_gene_feature_k4.csv"
    ppi_csv = grn_root / "data" / "network" / "ppi" / "ppi_string_v12_human_score900.csv"
    out_base = layout["experiments_tripgrn"]
    print(f"[tripgrn-10cv] grn_csv={grn_csv}")
    print(f"[tripgrn-10cv] experiments={out_base.parent}")

    for p, label in ((grn_csv, "GRN CSV"), (cds_gene, "CDS"), (ppi_csv, "PPI")):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {label}: {p}")

    gses = _resolve_gses(args.gse)
    folds = _resolve_folds(args.fold, int(args.n_folds))
    n_run = 0
    n_skip = 0

    for gse in gses:
        v_path, g_path, _ = gse_npy_triplet("epithelial", gse, grn_root)
        splits_path = cv_splits_path(gse, grn_root)
        if not v_path.is_file():
            raise FileNotFoundError(f"Missing epithelial npy for {gse}: {v_path}")
        if not splits_path.is_file():
            raise FileNotFoundError(f"Missing CV splits for {gse}: {splits_path}")

        for fold_idx in folds:
            run_dir = tripgrn_fold_dir(gse, fold_idx, grn_root)
            if _fold_done(run_dir) and not args.force:
                print(f"[tripgrn-10cv] skip {gse} fold {fold_idx:02d}")
                n_skip += 1
                continue

            n_train_cols = _train_cols_for_fold(splits_path, fold_idx)
            batch_size = int(args.batch_size)
            if batch_size <= 0:
                batch_size = suggest_batch_size(n_train_cols, cap=int(args.batch_size_cap))

            exp_name = gse
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "grn.scripts.train_unseen_samples_no_cv",
                "--exp-npy-values",
                str(v_path),
                "--exp-npy-genes",
                str(g_path),
                "--grn-csv",
                str(grn_csv),
                "--cds-gene-feature-csv",
                str(cds_gene),
                "--ppi-csv",
                str(ppi_csv),
                "--output-base-dir",
                str(out_base),
                "--experiment-name",
                exp_name,
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
                str(batch_size),
                "--epochs",
                str(args.epochs),
                "--lr",
                str(args.lr),
                "--patience",
                str(args.patience),
                "--neg-sample-ratio",
                "1.0",
                "--amp-bf16",
            ]
            if args.expression_gradient_checkpointing:
                cmd.append("--expression-gradient-checkpointing")
            if args.compile_model:
                cmd.append("--compile-model")

            print(
                f"[tripgrn-10cv] run {gse} fold {fold_idx:02d} "
                f"(train_cols={n_train_cols}, batch_size={batch_size})"
            )
            proc = subprocess.run(cmd, check=False, cwd=str(grn_root))
            # Windows/CUDA teardown can return a fatal status even after metrics were written.
            if (run_dir / "unseen_samples_test_metrics.json").is_file():
                if proc.returncode != 0:
                    print(
                        f"[tripgrn-10cv] warn: fold {fold_idx:02d} exit={proc.returncode} "
                        f"but metrics exist; continuing"
                    )
                n_run += 1
            else:
                raise subprocess.CalledProcessError(proc.returncode, cmd)

    print(f"[tripgrn-10cv] finished: ran={n_run}, skipped={n_skip}")


if __name__ == "__main__":
    main()
