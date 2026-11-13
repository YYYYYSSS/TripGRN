#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestrate scMGATGRN 10-fold CV on epithelial pancancer (binary, TripGRN protocol)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from grn.data.pancancer_cv import DEFAULT_N_FOLDS
from grn.data.pancancer_paths import (
    PANCANCER_GSE_IDS,
    cv_splits_path,
    gse_npy_triplet,
    scmgatgrn_fold_dir,
)
from grn.data.pancancer_cv import get_fold_split, load_cv_splits
from grn.training.pair_dataloader import suggest_batch_size
from grn.utils.runtime import now_mono, stamp_orchestrator_runtime


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
    ap = argparse.ArgumentParser(description="Run scMGATGRN pancancer epithelial 10-fold CV")
    ap.add_argument("--gse", default="ALL")
    ap.add_argument("--fold", default="ALL")
    ap.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--batch-size-cap", type=int, default=256)
    ap.add_argument("--max-genes", type=int, default=0, help="Gene cap (<=0: all genes; refactor default)")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()

    grn_root = Path(__file__).resolve().parents[2]
    gses = _resolve_gses(args.gse)
    folds = _resolve_folds(args.fold, int(args.n_folds))
    n_run = n_skip = 0

    for gse in gses:
        v_path, _, _ = gse_npy_triplet("epithelial", gse, grn_root)
        splits_path = cv_splits_path(gse, grn_root)
        if not v_path.is_file():
            raise FileNotFoundError(f"Missing epithelial npy for {gse}: {v_path}")
        if not splits_path.is_file():
            raise FileNotFoundError(f"Missing CV splits for {gse}: {splits_path}")

        for fold_idx in folds:
            run_dir = scmgatgrn_fold_dir(gse, fold_idx, grn_root)
            if _fold_done(run_dir) and not args.force:
                print(f"[scmgatgrn-10cv] skip {gse} fold {fold_idx:02d}")
                n_skip += 1
                continue

            n_train = _train_cols_for_fold(splits_path, fold_idx)
            batch_size = int(args.batch_size)
            if batch_size <= 0:
                batch_size = suggest_batch_size(n_train, cap=int(args.batch_size_cap))

            cmd = [
                sys.executable,
                "-u",
                "-m",
                "grn.scripts.run_scmgatgrn_pancancer_10cv_fold",
                "--gse-id",
                gse,
                "--fold-idx",
                str(fold_idx),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(batch_size),
                "--max-genes",
                str(args.max_genes),
                "--device",
                str(args.device),
            ]
            print(
                f"[scmgatgrn-10cv] run {gse} fold {fold_idx:02d} "
                f"(train_cols={n_train}, batch={batch_size}, max_genes={args.max_genes})"
            )
            t0 = now_mono()
            proc = subprocess.run(cmd, check=False, cwd=str(grn_root))
            stamp_orchestrator_runtime(
                run_dir,
                float(now_mono() - t0),
                method="scMGATGRN",
                gse=gse,
                fold_idx=fold_idx,
                extra={
                    "batch_size": int(batch_size),
                    "max_genes": int(args.max_genes),
                    "cell_policy": "full_cv_splits_pad_to_max",
                },
            )
            if _fold_done(run_dir):
                if proc.returncode != 0:
                    print(
                        f"[scmgatgrn-10cv] warn: {gse} fold {fold_idx:02d} exit={proc.returncode} "
                        f"but metrics exist; continuing"
                    )
                n_run += 1
            else:
                raise subprocess.CalledProcessError(proc.returncode, cmd)

    print(f"[scmgatgrn-10cv] finished: ran={n_run}, skipped={n_skip}")


if __name__ == "__main__":
    main()
