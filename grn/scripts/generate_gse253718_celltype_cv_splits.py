#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate 10-fold CV + fixed seed42 splits for GSE253718 cell types."""

from __future__ import annotations

import argparse

import numpy as np

from grn.data.gse253718_celltype_paths import (
    celltype_values_npy,
    cv_splits_path,
    gse253718_celltype_layout,
    resolve_type_ids,
)
from grn.data.pancancer_cv import DEFAULT_N_FOLDS, generate_cv_splits, write_cv_splits


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate GSE253718 cell-type CV splits")
    ap.add_argument("--type", default="ALL")
    ap.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    layout = gse253718_celltype_layout()
    layout["cv_splits_dir"].mkdir(parents=True, exist_ok=True)

    for tid, label in resolve_type_ids(args.type):
        v_path = celltype_values_npy(tid, log_normalized=True, grn_root=layout["grn_root"])
        if not v_path.is_file():
            raise FileNotFoundError(f"Missing {v_path}; run materialize + log_normalize first.")

        out_10 = cv_splits_path(tid, fixed=False, grn_root=layout["grn_root"])
        out_fix = cv_splits_path(tid, fixed=True, grn_root=layout["grn_root"])
        n_cells = int(np.load(v_path, mmap_mode="r").shape[1])

        if not out_10.is_file() or args.force:
            payload = generate_cv_splits(n_cells, n_folds=int(args.n_folds), gse_id=f"GSE253718_{tid}")
            write_cv_splits(payload, out_10)
            print(f"[gse253718-cv] {tid} ({label}): n={n_cells} -> {out_10}")
        else:
            print(f"[gse253718-cv] skip existing {out_10}")

        if not out_fix.is_file() or args.force:
            fixed = generate_cv_splits(n_cells, n_folds=1, cv_base_seed=42, gse_id=f"GSE253718_{tid}_fixed")
            write_cv_splits(fixed, out_fix)
            print(f"[gse253718-cv] {tid} fixed split -> {out_fix}")
        else:
            print(f"[gse253718-cv] skip existing {out_fix}")


if __name__ == "__main__":
    main()
