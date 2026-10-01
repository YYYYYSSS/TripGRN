#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrate DeepFGRN 10-fold CV on epithelial pancancer datasets.

Usage:
  python -m grn.scripts.run_deepfgrn_pancancer_10cv --gse ALL --fold ALL
  python -m grn.scripts.run_deepfgrn_pancancer_10cv --gse GSE103322 --fold 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from grn.data.pancancer_cv import DEFAULT_N_FOLDS, get_fold_split, load_cv_splits
from grn.data.pancancer_paths import (
    PANCANCER_GSE_IDS,
    cv_splits_path,
    deepfgrn_fold_dir,
    gse_npy_triplet,
    scRNA_layout,
    step01_grn_csv,
)
from grn.training.pair_dataloader import suggest_keras_batch_size


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


def _train_samples_for_fold(splits_path: Path, fold_idx: int) -> int:
    payload = load_cv_splits(splits_path)
    fold = get_fold_split(payload, fold_idx)
    return int(len(fold["train_cols"]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run DeepFGRN pancancer epithelial 10-fold CV")
    ap.add_argument("--gse", default="ALL")
    ap.add_argument("--fold", default="ALL", help="Fold index 1..10 or ALL")
    ap.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--force", action="store_true", help="Retrain even if metrics exist")
    ap.add_argument("--keras-batch-size", type=int, default=0, help="0 = auto from train sample count")
    ap.add_argument("--keras-batch-size-cap", type=int, default=256)
    args = ap.parse_args()

    grn_root = Path(__file__).resolve().parents[2]
    layout = scRNA_layout(grn_root)
    kegg_csv = step01_grn_csv(grn_root)
    if not kegg_csv.is_file():
        raise FileNotFoundError(f"Missing GRN CSV: {kegg_csv}")
    print(f"[deepfgrn-10cv] grn_csv={kegg_csv}")
    print(f"[deepfgrn-10cv] experiments={layout['experiments_deepfgrn'].parent}")

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
            run_dir = deepfgrn_fold_dir(gse, fold_idx, grn_root)
            if _fold_done(run_dir) and not args.force:
                print(f"[deepfgrn-10cv] skip {gse} fold {fold_idx:02d}")
                n_skip += 1
                continue

            n_train = _train_samples_for_fold(splits_path, fold_idx)
            keras_bs = int(args.keras_batch_size)
            if keras_bs <= 0:
                keras_bs = suggest_keras_batch_size(n_train, dim_exp=0, cap=int(args.keras_batch_size_cap))

            env = os.environ.copy()
            env["DEEPFGRN_BATCH_SIZE"] = str(keras_bs)

            cmd = [
                sys.executable,
                "-u",
                "-m",
                "grn.scripts.run_deepfgrn_pancancer_cell_unseen",
                "--gse-id",
                gse,
                "--exp-npy-values",
                str(v_path),
                "--exp-npy-genes",
                str(g_path),
                "--kegg-csv",
                str(kegg_csv),
                "--cv-splits-json",
                str(splits_path),
                "--fold-idx",
                str(fold_idx),
                "--output-fold-dir",
                str(run_dir),
                "--keras-batch-size",
                str(keras_bs),
                "--skip-summary-plots",
            ]

            print(
                f"[deepfgrn-10cv] run {gse} fold {fold_idx:02d} "
                f"(train_cols={n_train}, keras_batch={keras_bs})"
            )
            proc = subprocess.run(cmd, check=False, cwd=str(grn_root), env=env)
            # Windows/CUDA teardown can return a fatal status even after training artifacts exist.
            if _fold_done(run_dir):
                if proc.returncode != 0:
                    print(
                        f"[deepfgrn-10cv] warn: {gse} fold {fold_idx:02d} exit={proc.returncode} "
                        f"but metrics exist; continuing"
                    )
                n_run += 1
                continue

            # Recover: train often finishes (best_model + CM) then CUDA dies before metrics JSON.
            ckpt = run_dir / "best_model.keras"
            if ckpt.is_file():
                recover_cmd = list(cmd) + ["--cm-only"]
                print(
                    f"[deepfgrn-10cv] recover {gse} fold {fold_idx:02d} via --cm-only "
                    f"(exit={proc.returncode}, ckpt exists)"
                )
                proc2 = subprocess.run(recover_cmd, check=False, cwd=str(grn_root), env=env)
                if _fold_done(run_dir):
                    if proc2.returncode != 0:
                        print(
                            f"[deepfgrn-10cv] warn: recover fold {fold_idx:02d} exit={proc2.returncode} "
                            f"but metrics exist; continuing"
                        )
                    n_run += 1
                    continue

            raise subprocess.CalledProcessError(proc.returncode, cmd)

    print(f"[deepfgrn-10cv] finished: ran={n_run}, skipped={n_skip}")


if __name__ == "__main__":
    main()
