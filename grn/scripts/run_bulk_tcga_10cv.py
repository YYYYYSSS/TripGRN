#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrate TCGA bulk cross-cancer 10-fold CV (12 cancers x 10 folds).

Fig.3 default = original / pre-VAE protocol:
  - unseen-sample 5:2:3 CV splits
  - legacy random pair split (--no-gene-disjoint)
  - Conv1d expression encoder (not VAE)
  - CDS + TSS + raw STRING PPI
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Run TCGA bulk 10-fold CV (Fig.3 original protocol)")
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
    ap.add_argument(
        "--expression-encoder",
        choices=("conv", "vae"),
        default="conv",
        help="Fig.3 default=conv (original). Use vae only for rewrite pilots.",
    )
    ap.add_argument(
        "--gene-disjoint",
        action="store_true",
        help="Enable gene-disjoint (NOT default for Fig.3 original).",
    )
    ap.add_argument(
        "--ppi-nokegg",
        action="store_true",
        help="Use STRING\\KEGG PPI (default Fig.3 uses raw STRING).",
    )
    ap.add_argument(
        "--no-tss",
        action="store_true",
        help="Skip TSS features (original protocol includes TSS).",
    )
    args = ap.parse_args()

    layout = bulk_layout()
    grn_root = layout["grn_root"]
    out_base = layout["fig03"] / "experiments" / args.experiment_name
    ppi_csv = layout["ppi_csv"] if args.ppi_nokegg else layout["ppi_csv_raw"]

    for key, path in (("grn_csv", layout["grn_csv"]), ("cds_gene", layout["cds_gene"]), ("ppi_csv", ppi_csv)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {key}: {path}")

    use_tss = not bool(args.no_tss)
    if use_tss and not layout["tss_gene"].is_file():
        raise FileNotFoundError(
            f"Missing tss_gene (required for original Fig.3): {layout['tss_gene']}\n"
            "Link legacy tss_gene_feature_k4.csv into shared/data/features/cds/ or fig03/.../cds/"
        )

    cancers = _resolve_cancers(args.cancer)
    folds = _resolve_folds(args.fold, int(args.n_folds))
    n_run = n_skip = n_miss = 0

    print(f"[bulk-10cv] experiment_base={out_base}")
    print(f"[bulk-10cv] expression_encoder={args.expression_encoder}")
    print(f"[bulk-10cv] gene_disjoint={bool(args.gene_disjoint)} tss={use_tss}")
    print(f"[bulk-10cv] ppi={'nokegg' if args.ppi_nokegg else 'raw'} -> {ppi_csv}")

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
            if args.experiment_name == DEFAULT_BULK_EXPERIMENT:
                done_dir = bulk_fold_dir(cancer, fold_idx, grn_root)
            else:
                done_dir = out_base / cancer / f"fold_{int(fold_idx):02d}"
            if (done_dir / "unseen_samples_test_metrics.json").is_file() and not args.force:
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
                "--ppi-csv",
                str(ppi_csv),
                "--output-base-dir",
                str(out_base),
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
                "--expression-encoder",
                str(args.expression_encoder),
            ]
            if use_tss:
                cmd.extend(["--tss-gene-feature-csv", str(layout["tss_gene"])])
            if args.gene_disjoint:
                cmd.append("--gene-disjoint")
            else:
                cmd.append("--no-gene-disjoint")

            print(
                f"[bulk-10cv] run {cancer} fold {fold_idx:02d} "
                f"(train_cols={n_train} encoder={args.expression_encoder} "
                f"gene_disjoint={bool(args.gene_disjoint)})"
            )
            subprocess.run(cmd, check=True, cwd=str(grn_root))
            n_run += 1

    print(f"[bulk-10cv] finished: ran={n_run}, skipped={n_skip}, missing={n_miss}")


if __name__ == "__main__":
    main()
