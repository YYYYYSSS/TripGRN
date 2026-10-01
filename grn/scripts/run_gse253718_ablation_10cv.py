#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrate GSE253718 modality ablation 10-fold CV (7 stages x 10 folds).

Uses the same shuffled 5:2:3 unseen-cell protocol as step01 pancancer benchmarks.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from grn.data.gse253718_paths import (
    ABLATION_STAGES,
    DEFAULT_ABLATION_EXPERIMENT,
    ablation_stage_dir,
    gse253718_layout,
)
from grn.data.pancancer_cv import DEFAULT_N_FOLDS, get_fold_split, load_cv_splits


def _resolve_folds(fold_arg: str, n_folds: int) -> List[int]:
    fa = str(fold_arg).strip().upper()
    if fa == "ALL":
        return list(range(1, n_folds + 1))
    return [int(fa)]


def _resolve_stages(stage_arg: str) -> List[tuple]:
    sa = str(stage_arg).strip().upper()
    if sa == "ALL":
        return list(ABLATION_STAGES)
    out = [s for s in ABLATION_STAGES if s[0] == sa or s[1] == sa]
    if not out:
        ids = ", ".join(x[0] for x in ABLATION_STAGES)
        raise SystemExit(f"Unknown stage {stage_arg!r}; choose from: {ids}")
    return out


def _fold_done(stage_id: str, fold_idx: int, grn_root: Path) -> bool:
    return (ablation_stage_dir(stage_id, fold_idx, grn_root) / "unseen_samples_test_metrics.json").is_file()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run GSE253718 modality ablation 10-fold CV")
    ap.add_argument("--stage", default="ALL", help="Stage dir id (e.g. 04_expr_cds_ppi) or ALL")
    ap.add_argument("--fold", default="ALL", help="Fold 1..10 or ALL")
    ap.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--experiment-name", default=DEFAULT_ABLATION_EXPERIMENT)
    args = ap.parse_args()

    layout = gse253718_layout()
    grn_root = layout["grn_root"]
    splits_path = layout["cv_splits"]
    if not splits_path.is_file():
        raise FileNotFoundError(f"Missing CV splits: {splits_path}; run generate_gse253718_cv_splits first.")
    for key in ("exp_values", "exp_genes", "grn_csv", "cds_gene", "ppi_csv"):
        if not layout[key].is_file():
            raise FileNotFoundError(f"Missing {key}: {layout[key]} (epithelial subset required)")

    stages = _resolve_stages(args.stage)
    folds = _resolve_folds(args.fold, int(args.n_folds))
    n_run = n_skip = 0

    for stage_id, _label, extra_flags in stages:
        for fold_idx in folds:
            if _fold_done(stage_id, fold_idx, grn_root) and not args.force:
                print(f"[ablation-10cv] skip {stage_id} fold {fold_idx:02d}")
                n_skip += 1
                continue

            fold = get_fold_split(load_cv_splits(splits_path), fold_idx)
            n_train = len(fold["train_cols"])
            exp_name = f"{args.experiment_name}/{stage_id}"
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "grn.scripts.train_unseen_samples_no_cv",
                "--exp-npy-values",
                str(layout["exp_values"]),
                "--exp-npy-genes",
                str(layout["exp_genes"]),
                "--grn-csv",
                str(layout["grn_csv"]),
                "--cds-gene-feature-csv",
                str(layout["cds_gene"]),
                "--ppi-csv",
                str(layout["ppi_csv"]),
                "--output-base-dir",
                str(grn_root / "experiments"),
                "--experiment-name",
                exp_name,
                "--no-timestamp",
                "--cv-splits-json",
                str(splits_path),
                "--fold-idx",
                str(fold_idx),
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
                "--expression-gradient-checkpointing",
                "--amp-bf16",
                *extra_flags,
            ]
            print(f"[ablation-10cv] run {stage_id} fold {fold_idx:02d} (train_cols={n_train})")
            subprocess.run(cmd, check=True, cwd=str(grn_root))
            n_run += 1

    print(f"[ablation-10cv] finished: ran={n_run}, skipped={n_skip}")


if __name__ == "__main__":
    main()
