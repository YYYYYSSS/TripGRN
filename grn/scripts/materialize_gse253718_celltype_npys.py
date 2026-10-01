#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Materialize GSE253718 per-SingleR-type count npy from full expression matrix.

Uses GSE253718_exp_values.npy + singler_labels.csv (pruned major types).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from grn.data.gse253718_celltype_paths import (
    CELL_TYPES,
    MAJOR_SINGLER_TYPES,
    SINGLER_TO_TYPE_ID,
    TYPE_ID_TO_SINGLER,
    gse253718_celltype_layout,
)


def _load_barcodes(csv_path: Path) -> List[str]:
    head = pd.read_csv(csv_path, nrows=0, low_memory=False)
    cols = [str(c) for c in head.columns[1:]]
    return cols


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize GSE253718 SingleR cell-type count npy")
    ap.add_argument("--min-cells", type=int, default=450, help="Drop types below this count")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    layout = gse253718_celltype_layout()
    values_path = layout["master_values_npy"]
    genes_path = layout["master_genes_npy"]
    labels_path = layout["singler_labels_csv"]
    master_csv = layout["master_exp_csv"]
    out_root = layout["celltype_data_dir"]

    for p in (values_path, genes_path, labels_path, master_csv):
        if not p.is_file():
            raise FileNotFoundError(p)

    manifest_path = out_root / "manifest.json"
    if manifest_path.is_file() and not args.force:
        ok = True
        for tid, _ in CELL_TYPES:
            if not (out_root / tid / "values.npy").is_file():
                ok = False
                break
        if ok:
            print(f"[gse253718-materialize] skip existing under {out_root}")
            return

    barcodes = _load_barcodes(master_csv)
    name_to_i = {c: i for i, c in enumerate(barcodes)}
    mm = np.load(str(values_path), mmap_mode="r")
    genes = np.load(str(genes_path), allow_pickle=True)
    if int(mm.shape[1]) != len(barcodes):
        raise ValueError(f"values cols {mm.shape[1]} != barcodes {len(barcodes)}")
    if int(mm.shape[0]) != len(genes):
        raise ValueError(f"values rows {mm.shape[0]} != genes {len(genes)}")

    labs = pd.read_csv(labels_path)
    if "cell_id" not in labs.columns or "singler_pruned" not in labs.columns:
        raise ValueError("singler_labels.csv needs cell_id and singler_pruned")

    # Keep first label per cell_id
    labs = labs.drop_duplicates(subset=["cell_id"], keep="first")
    label_map = dict(zip(labs["cell_id"].astype(str), labs["singler_pruned"].astype(str)))

    groups: Dict[str, List[str]] = {tid: [] for tid, _ in CELL_TYPES}
    unmatched = 0
    for bc in barcodes:
        lab = label_map.get(bc)
        if lab is None:
            unmatched += 1
            continue
        tid = SINGLER_TO_TYPE_ID.get(lab)
        if tid is None:
            continue
        groups[tid].append(bc)

    kept: List[tuple] = []
    for tid, display in CELL_TYPES:
        n = len(groups[tid])
        print(f"[gse253718-materialize] {tid}: n={n} ({TYPE_ID_TO_SINGLER[tid]} / {display})")
        if n < int(args.min_cells):
            print(f"[gse253718-materialize] drop {tid}: n<{args.min_cells}")
            continue
        kept.append((tid, display))

    if not kept:
        raise SystemExit("No cell types passed --min-cells")

    out_root.mkdir(parents=True, exist_ok=True)
    # Persist cells.npy for full matrix (useful downstream)
    cells_out = layout["grn_root"] / "data" / "exp" / "GSE253718_exp_cells.npy"
    if not cells_out.is_file():
        np.save(cells_out, np.array(barcodes, dtype=object))
        print(f"[gse253718-materialize] wrote {cells_out}")

    manifest_types: Dict[str, Dict[str, object]] = {}
    for tid, display in kept:
        names = groups[tid]
        idx = np.array([name_to_i[c] for c in names], dtype=np.int64)
        td = out_root / tid
        td.mkdir(parents=True, exist_ok=True)
        # Column-slice into new memmap (genes x cells_type)
        out_p = td / "values.npy"
        if out_p.is_file():
            out_p.unlink()
        mm_w = np.lib.format.open_memmap(
            str(out_p), mode="w+", dtype=np.float32, shape=(mm.shape[0], len(idx))
        )
        # Write in column chunks to limit RAM
        chunk = 256
        for j0 in range(0, len(idx), chunk):
            j1 = min(len(idx), j0 + chunk)
            mm_w[:, j0:j1] = np.asarray(mm[:, idx[j0:j1]], dtype=np.float32)
        mm_w.flush()
        del mm_w
        np.save(td / "genes.npy", genes)
        np.save(td / "column_names.npy", np.array(names, dtype=object))
        manifest_types[tid] = {
            "singler_pruned": TYPE_ID_TO_SINGLER[tid],
            "display_label": display,
            "n_cells": len(names),
            "values_npy": f"{tid}/values.npy",
            "genes_npy": f"{tid}/genes.npy",
            "column_names_npy": f"{tid}/column_names.npy",
        }
        print(f"[gse253718-materialize] OK {tid} -> {out_p}")

    # Update CELL_TYPES usage note: only kept types are in manifest; training uses manifest keys
    manifest = {
        "version": 1,
        "grouping": "singler_pruned_major",
        "major_types_requested": list(MAJOR_SINGLER_TYPES),
        "min_cells": int(args.min_cells),
        "master_values_npy": str(values_path),
        "singler_labels_csv": str(labels_path),
        "n_genes": int(mm.shape[0]),
        "n_cells_master": int(mm.shape[1]),
        "unmatched_barcodes": int(unmatched),
        "cell_types": manifest_types,
        "kept_type_ids": [t[0] for t in kept],
    }
    # Write active type list for resolve helpers
    active_path = out_root / "active_cell_types.json"
    active_path.write_text(
        json.dumps([{"type_id": t[0], "label": t[1], "singler": TYPE_ID_TO_SINGLER[t[0]]} for t in kept], indent=2),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gse253718-materialize] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
