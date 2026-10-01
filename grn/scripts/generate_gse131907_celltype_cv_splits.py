#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate 10-fold CV + fixed seed42 split JSON for each GSE131907 step06 cell type."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from grn.data.gse131907_celltype_paths import (
    celltype_values_npy,
    cv_splits_path,
    gse131907_celltype_layout,
    resolve_type_ids,
)
from grn.data.pancancer_cv import DEFAULT_N_FOLDS, generate_cv_splits, write_cv_splits


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate GSE131907 step06 CV splits")
    ap.add_argument("--type", default="ALL", help="Cell type id / label / ALL")
    ap.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    layout = gse131907_celltype_layout()
    layout["cv_splits_dir"].mkdir(parents=True, exist_ok=True)

    types = resolve_type_ids(args.type)
    for tid, label in types:
        v_path = celltype_values_npy(tid, log_normalized=True, grn_root=layout["grn_root"])
        if not v_path.is_file():
            raise FileNotFoundError(f"Missing {v_path}; run materialize + log_normalize first.")

        out_10 = cv_splits_path(tid, fixed=False, grn_root=layout["grn_root"])
        out_fix = cv_splits_path(tid, fixed=True, grn_root=layout["grn_root"])

        n_cells = int(np.load(v_path, mmap_mode="r").shape[1])
        if not out_10.is_file() or args.force:
            payload = generate_cv_splits(n_cells, n_folds=int(args.n_folds), gse_id=f"GSE131907_{tid}")
            write_cv_splits(payload, out_10)
            print(f"[celltype-cv] {tid} ({label}): n={n_cells} -> {out_10}")
        else:
            print(f"[celltype-cv] skip existing {out_10}")

        if not out_fix.is_file() or args.force:
            fixed = generate_cv_splits(n_cells, n_folds=1, cv_base_seed=42, gse_id=f"GSE131907_{tid}_fixed")
            write_cv_splits(fixed, out_fix)
            print(f"[celltype-cv] {tid} fixed split -> {out_fix}")
        else:
            print(f"[celltype-cv] skip existing {out_fix}")


if __name__ == "__main__":
    main()
