#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSE131907 brain / non-brain domain-accuracy 10-fold CV (fusion-head finetune only).

Purpose: show stable per-domain accuracy under repeated unseen-cell splits.
Uses the same 1:2:7 train:val:test ratio as step05 application (not full GRN case study).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np

from grn.data.pancancer_cv import DEFAULT_N_FOLDS, generate_cv_splits, write_cv_splits


DOMAINS = {
    "mbrain": {
        "label": "mBrain",
        "values": "brain_values_log_normalized.npy",
    },
    "nonbrain": {
        "label": "non-mBrain",
        "values": "nonbrain_values_log_normalized.npy",
    },
}


def _resolve_folds(fold_arg: str, n_folds: int) -> List[int]:
    fa = str(fold_arg).strip().upper()
    if fa == "ALL":
        return list(range(1, n_folds + 1))
    return [int(fa)]


def _resolve_domains(domain_arg: str) -> List[str]:
    da = str(domain_arg).strip().lower()
    if da == "all":
        return list(DOMAINS.keys())
    if da not in DOMAINS:
        raise SystemExit(f"Unknown domain {domain_arg}. Use mbrain|nonbrain|ALL")
    return [da]


def _fold_done(run_dir: Path) -> bool:
    return (run_dir / "unseen_samples_test_metrics.json").is_file()


def main() -> None:
    ap = argparse.ArgumentParser(description="GSE131907 brain/nonbrain domain-acc 10CV")
    ap.add_argument("--domain", default="ALL", help="mbrain | nonbrain | ALL")
    ap.add_argument("--fold", default="ALL")
    ap.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--sample-split", default="1,2,7", help="train:val:test weights (match step05 app)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    grn_root = Path(__file__).resolve().parents[2]
    step1 = grn_root / "experiments" / "benchmark_luad_gse253718_step1"
    pre = grn_root / "data" / "exp" / "GSE131907" / "precomputed_brain_nonbrain_npy"
    genes = pre / "genes.npy"
    out_root = grn_root / "experiments" / "gse131907_brain_nonbrain_domain_acc_10cv_seed42"

    if not (step1 / "best_model_fold0.pth").is_file():
        raise FileNotFoundError(f"Missing Step1 checkpoint: {step1 / 'best_model_fold0.pth'}")
    if not genes.is_file():
        raise FileNotFoundError(genes)

    ratios = tuple(int(x.strip()) for x in str(args.sample_split).split(","))
    if len(ratios) != 3:
        raise SystemExit("--sample-split must be W1,W2,W3")

    domains = _resolve_domains(args.domain)
    folds = _resolve_folds(args.fold, int(args.n_folds))
    n_run = n_skip = 0

    for dom in domains:
        values = pre / DOMAINS[dom]["values"]
        if not values.is_file():
            raise FileNotFoundError(values)
        # n_cells from memmap shape
        arr = np.load(values, mmap_mode="r")
        n_cells = int(arr.shape[1])
        del arr

        splits_path = out_root / dom / "cv_splits_10fold.json"
        if not splits_path.is_file() or args.force:
            payload = generate_cv_splits(
                n_cells,
                n_folds=int(args.n_folds),
                sample_split=ratios,  # type: ignore[arg-type]
                cv_base_seed=int(args.seed),
                pair_seed=int(args.seed),
                gse_id=f"GSE131907_{dom}",
            )
            write_cv_splits(payload, splits_path)
            print(f"[domain-acc-10cv] wrote splits {splits_path} n_cells={n_cells}", flush=True)
        else:
            print(f"[domain-acc-10cv] reuse splits {splits_path}", flush=True)

        for fold_idx in folds:
            run_dir = out_root / dom / f"fold_{int(fold_idx):02d}"
            if _fold_done(run_dir) and not args.force:
                print(f"[domain-acc-10cv] skip {dom} fold {fold_idx:02d}")
                n_skip += 1
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "grn.scripts.finetune_step1_fusion_head_external",
                "--step1-run-dir",
                str(step1),
                "--external-exp-npy-values",
                str(values),
                "--external-exp-npy-genes",
                str(genes),
                "--output-dir",
                str(run_dir),
                "--expression-norm-fit-all-columns",
                "--epochs",
                str(int(args.epochs)),
                "--lr",
                str(float(args.lr)),
                "--patience",
                str(int(args.patience)),
                "--early-stop-metric",
                "macro_f1",
                "--sample-split",
                str(args.sample_split),
                "--cv-splits-json",
                str(splits_path),
                "--fold-idx",
                str(int(fold_idx)),
                "--eval-posthoc-calibrate",
            ]
            print(f"[domain-acc-10cv] run {dom} fold {fold_idx:02d}", flush=True)
            subprocess.run(cmd, check=True, cwd=str(grn_root))
            n_run += 1

    print(f"[domain-acc-10cv] finished: ran={n_run}, skipped={n_skip}, out={out_root}")


if __name__ == "__main__":
    main()
