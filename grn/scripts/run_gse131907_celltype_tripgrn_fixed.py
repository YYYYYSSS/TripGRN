#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retrain TripGRN on fixed seed42 split per GSE131907 step06 cell type (for GRN inference)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

from grn.data.gse131907_celltype_paths import (
    DEFAULT_STEP06_EXPERIMENT,
    celltype_genes_npy,
    celltype_values_npy,
    cv_splits_path,
    gse131907_celltype_layout,
    resolve_type_ids,
    tripgrn_fixed_run_dir,
)
from grn.evaluation.confusion_matrix_io import export_confusion_matrix_artifacts
import numpy as np


def _resolve_types(type_arg: str) -> List[tuple]:
    return resolve_type_ids(type_arg)

def _export_val_cm(run_dir: Path, *, method: str, experiment_name: str) -> None:
    p = run_dir / "unseen_samples_val_metrics.json"
    if not p.is_file():
        return
    mj = json.loads(p.read_text(encoding="utf-8"))
    cm = mj.get("confusion_matrix")
    if cm is None:
        return
    export_confusion_matrix_artifacts(
        run_dir,
        np.asarray(cm, dtype=np.int64),
        stem="confusion_matrix_val",
        title=f"{method} val confusion matrix",
        class_names=["No regulation", "Activation", "Repression"],
        extra={"method": method, "experiment_name": experiment_name, "split": "val"},
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="GSE131907 step06 TripGRN fixed split training")
    ap.add_argument("--type", default="ALL")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=0, help="0=auto from capped train cols")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-expression-cols", type=int, default=10000)
    ap.add_argument("--expression-col-reduce", choices=("subsample", "avg_pool"), default="subsample")
    ap.add_argument("--amp-bf16", action="store_true", default=True)
    ap.add_argument("--no-amp-bf16", action="store_true")
    ap.add_argument("--experiment-name", default=DEFAULT_STEP06_EXPERIMENT)
    args = ap.parse_args()
    use_amp = bool(args.amp_bf16) and not bool(args.no_amp_bf16)

    layout = gse131907_celltype_layout()
    grn_root = layout["grn_root"]

    for tid, label in _resolve_types(args.type):
        run_dir = tripgrn_fixed_run_dir(tid, grn_root)
        if (run_dir / "unseen_samples_test_metrics.json").is_file() and not args.force:
            print(f"[celltype-fixed] skip {tid}")
            continue

        splits_path = cv_splits_path(tid, fixed=True, grn_root=grn_root)
        if not splits_path.is_file():
            raise FileNotFoundError(splits_path)
        v_path = celltype_values_npy(tid, log_normalized=True, grn_root=grn_root)
        g_path = celltype_genes_npy(tid, grn_root=grn_root)
        exp_name = f"{args.experiment_name}/tripgrn_fixed_seed42/{tid}"

        from grn.data.pancancer_cv import get_fold_split, load_cv_splits
        from grn.training.pair_dataloader import suggest_batch_size

        fold = get_fold_split(load_cv_splits(splits_path), 1)
        n_train = len(fold["train_cols"])
        n_train_eff = min(n_train, int(args.max_expression_cols)) if int(args.max_expression_cols) > 0 else n_train
        bs = int(args.batch_size) if int(args.batch_size) > 0 else suggest_batch_size(n_train_eff)

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
            "1",
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
            f"[celltype-fixed] {tid} ({label}) batch={bs} train_cols={n_train} "
            f"expr_T_cap={args.max_expression_cols} amp_bf16={use_amp}"
        )
        subprocess.run(cmd, check=True)
        _export_val_cm(run_dir, method="TripGRN", experiment_name=exp_name)


if __name__ == "__main__":
    main()
