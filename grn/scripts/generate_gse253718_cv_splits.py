#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate 10-fold CV cell-column splits for GSE253718 ablation (shuffled 5:2:3)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from grn.data.gse253718_paths import gse253718_layout
from grn.data.pancancer_cv import DEFAULT_N_FOLDS, generate_cv_splits, write_cv_splits


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate GSE253718 10-fold CV splits")
    ap.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    layout = gse253718_layout()
    v_path = layout["exp_values"]
    out_path = layout["cv_splits"]
    if not v_path.is_file():
        raise FileNotFoundError(f"Missing epithelial expression npy: {v_path}; run preprocess_gse253718_epithelial_singler first.")
    if out_path.is_file() and not args.force:
        print(f"[gse253718-cv] skip existing {out_path}")
        return

    n_cells = int(np.load(v_path, mmap_mode="r").shape[1])
    payload = generate_cv_splits(n_cells, n_folds=int(args.n_folds), gse_id="GSE253718_epithelial")
    write_cv_splits(payload, out_path)
    print(f"[gse253718-cv] n_cells={n_cells} n_folds={args.n_folds} -> {out_path}")


if __name__ == "__main__":
    main()
