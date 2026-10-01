#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1: generate shared 10-fold CV cell-column splits per epithelial GSE."""

from __future__ import annotations

import argparse
from pathlib import Path

from grn.data.pancancer_cv import DEFAULT_N_FOLDS, generate_cv_splits, write_cv_splits
from grn.data.pancancer_paths import PANCANCER_GSE_IDS, cv_splits_path, gse_npy_triplet
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate 10-fold CV splits for epithelial pancancer matrices")
    ap.add_argument("--gse", default="ALL")
    ap.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    grn_root = Path(__file__).resolve().parents[2]
    gses = list(PANCANCER_GSE_IDS) if str(args.gse).upper() == "ALL" else [str(args.gse).strip()]

    for gse in gses:
        out_path = cv_splits_path(gse, grn_root)
        if out_path.is_file() and not args.force:
            print(f"[cv-splits] skip {gse}")
            continue
        v_path, _, _ = gse_npy_triplet("epithelial", gse, grn_root)
        if not v_path.is_file():
            raise FileNotFoundError(f"Missing epithelial matrix {v_path}; run SingleR step first.")
        n_cells = int(np.load(v_path, mmap_mode="r").shape[1])
        payload = generate_cv_splits(n_cells, n_folds=int(args.n_folds), gse_id=gse)
        write_cv_splits(payload, out_path)
        print(f"[cv-splits] {gse}: n_cells={n_cells} -> {out_path}")

    print(f"[cv-splits] done: {len(gses)} datasets")


if __name__ == "__main__":
    main()
