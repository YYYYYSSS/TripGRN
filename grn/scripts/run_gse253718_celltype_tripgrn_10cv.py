#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestrate TripGRN 10-fold CV per GSE253718 SingleR cell type."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np

from grn.data.gse253718_celltype_paths import (
    DEFAULT_CELLTYPE_EXPERIMENT,
    celltype_genes_npy,
    celltype_values_npy,
    cv_splits_path,
    gse253718_celltype_layout,
    resolve_type_ids,
    tripgrn_10cv_fold_dir,
)
from grn.data.pancancer_cv import DEFAULT_N_FOLDS, get_fold_split, load_cv_splits
from grn.evaluation.confusion_matrix_io import export_confusion_matrix_artifacts
from grn.training.pair_dataloader import suggest_batch_size


def _resolve_folds(fold_arg: str, n_folds: int) -> List[int]:
    if str(fold_arg).upper() == "ALL":
        return list(range(1, n_folds + 1))
    return [int(fold_arg)]


def _export_val_cm(run_dir: Path, *, method: str, fold_idx: int, experiment_name: str) -> None:
    p = run_dir / "unseen_samples_val_metrics.json"
    if not p.is_file():
        return
    mj = json.loads(p.read_text(encoding="utf-8"))
    cm = mj.get("confusion_matrix")
    if cm is None:
        return
    try:
        export_confusion_matrix_artifacts(
            run_dir,
            np.asarray(cm, dtype=np.int64),
            stem="confusion_matrix_val",
            title=f"{method} val confusion matrix",
            class_names=["No regulation", "Activation", "Repression"],
            extra={"method": method, "fold_idx": fold_idx, "experiment_name": experiment_name, "split": "val"},
        )
    except Exception as ex:
        print(f"[gse253718-celltype-10cv] warning: val CM export failed: {ex}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run GSE253718 cell-type TripGRN 10-fold CV")
    ap.add_argument("--type", default="ALL")
    ap.add_argument("--fold", default="ALL")
    ap.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--batch-size-cap", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-expression-cols", type=int, default=10000)
    ap.add_argument("--expression-col-reduce", choices=("subsample", "avg_pool"), default="subsample")
    ap.add_argument("--amp-bf16", action="store_true", default=True)
    ap.add_argument("--no-amp-bf16", action="store_true")
    ap.add_argument("--experiment-name", default=DEFAULT_CELLTYPE_EXPERIMENT)
    args = ap.parse_args()
    use_amp = bool(args.amp_bf16) and not bool(args.no_amp_bf16)

    layout = gse253718_celltype_layout()
    grn_root = layout["grn_root"]
    types = resolve_type_ids(args.type)
    folds = _resolve_folds(args.fold, int(args.n_folds))
    n_run = n_skip = 0

    for tid, label in types:
        splits_path = cv_splits_path(tid, fixed=False, grn_root=grn_root)
        if not splits_path.is_file():
            raise FileNotFoundError(f"Missing CV splits: {splits_path}")
        for key in ("grn_csv", "cds_gene", "ppi_csv"):
            if not layout[key].is_file():
                raise FileNotFoundError(layout[key])
        v_path = celltype_values_npy(tid, log_normalized=True, grn_root=grn_root)
        g_path = celltype_genes_npy(tid, grn_root=grn_root)
        if not v_path.is_file() or not g_path.is_file():
            raise FileNotFoundError(f"Missing npy for {tid}; run materialize + log_normalize")

        for fold_idx in folds:
            run_dir = tripgrn_10cv_fold_dir(tid, fold_idx, grn_root)
            if (run_dir / "unseen_samples_test_metrics.json").is_file() and not args.force:
                print(f"[gse253718-celltype-10cv] skip {tid} fold {fold_idx:02d}")
                n_skip += 1
                continue

            fold = get_fold_split(load_cv_splits(splits_path), fold_idx)
            n_train = len(fold["train_cols"])
            n_train_eff = min(n_train, int(args.max_expression_cols)) if int(args.max_expression_cols) > 0 else n_train
            bs = int(args.batch_size) if int(args.batch_size) > 0 else suggest_batch_size(n_train_eff, cap=int(args.batch_size_cap))
            exp_name = f"{args.experiment_name}/tripgrn_10cv/{tid}"

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
                str(bs),
                "--epochs",
                str(args.epochs),
                "--lr",
                str(args.lr),
                "--patience",
                str(args.patience),
                "--neg-sample-ratio",
                "1.0",
                "--skip-plots",
                "--max-expression-cols",
                str(int(args.max_expression_cols)),
                "--expression-col-reduce",
                str(args.expression_col_reduce),
            ]
            if use_amp:
                cmd.append("--amp-bf16")
            print(
                f"[gse253718-celltype-10cv] {tid} ({label}) fold {fold_idx:02d} "
                f"batch={bs} train_cols={n_train} expr_T_cap={args.max_expression_cols}"
            )
            proc = subprocess.run(cmd, check=False)
            metrics_ok = (run_dir / "unseen_samples_test_metrics.json").is_file()
            if metrics_ok:
                if proc.returncode != 0:
                    print(
                        f"[gse253718-celltype-10cv] warn: {tid} fold {fold_idx:02d} "
                        f"exit={proc.returncode} but metrics exist; continuing"
                    )
                    # Skip CM export after abnormal teardown (can hard-crash the parent).
                else:
                    _export_val_cm(run_dir, method="TripGRN", fold_idx=fold_idx, experiment_name=exp_name)
                n_run += 1
            else:
                raise SystemExit(
                    f"[gse253718-celltype-10cv] failed {tid} fold {fold_idx:02d} "
                    f"exit={proc.returncode} and no metrics"
                )

    print(f"[gse253718-celltype-10cv] done: ran={n_run}, skipped={n_skip}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
