#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate 10-fold CV sample-column splits for TCGA bulk FPKM matrices."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from grn.data.bulk_paths import TCGA_CANCER_IDS, bulk_exp_csv, cv_splits_path
from grn.data.pancancer_cv import DEFAULT_N_FOLDS, generate_cv_splits, write_cv_splits


def _count_samples_from_csv(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    if len(header) < 2:
        raise ValueError(f"CSV has no sample columns: {csv_path}")
    return len(header) - 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate TCGA bulk 10-fold CV splits")
    ap.add_argument("--cancer", default="ALL", help="TCGA-LUAD or ALL")
    ap.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    grn_root = Path(__file__).resolve().parents[2]
    cancers = list(TCGA_CANCER_IDS) if str(args.cancer).upper() == "ALL" else [str(args.cancer).strip()]
    n_ok = n_skip = n_miss = 0

    for cancer in cancers:
        out_path = cv_splits_path(cancer, grn_root)
        if out_path.is_file() and not args.force:
            print(f"[bulk-cv] skip existing {cancer}")
            n_skip += 1
            continue
        exp_csv = bulk_exp_csv(cancer, grn_root)
        if not exp_csv.is_file():
            print(f"[bulk-cv] missing CSV for {cancer}: {exp_csv}")
            n_miss += 1
            continue
        n_samples = _count_samples_from_csv(exp_csv)
        payload = generate_cv_splits(n_samples, n_folds=int(args.n_folds), gse_id=cancer)
        write_cv_splits(payload, out_path)
        print(f"[bulk-cv] {cancer}: n_samples={n_samples} -> {out_path}")
        n_ok += 1

    print(f"[bulk-cv] done: wrote={n_ok}, skipped={n_skip}, missing_csv={n_miss}")


if __name__ == "__main__":
    main()
