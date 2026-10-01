#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Materialize GSE131907 step06 cell-type expression npy (counts) from master wide CSV.

One group per annotation Cell_type (no merged super-types).
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from grn.data.gse131907_celltype_paths import (
    ANNOTATION_CELL_TYPES,
    CELL_TYPES,
    LABEL_TO_TYPE_ID,
    gse131907_celltype_layout,
)
from grn.data.preprocessing import normalize_gene_symbol_series


def _load_cell_groups(ann_gz: Path) -> Dict[str, Set[str]]:
    wanted = set(ANNOTATION_CELL_TYPES)
    groups: Dict[str, Set[str]] = {tid: set() for tid, _ in CELL_TYPES}
    with gzip.open(ann_gz, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        i_idx = header.index("Index")
        i_ct = header.index("Cell_type")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_idx, i_ct):
                continue
            ct = parts[i_ct]
            if ct not in wanted:
                continue
            tid = LABEL_TO_TYPE_ID[ct]
            groups[tid].add(parts[i_idx])
    return groups


def _unique_gene_row_count(csv_path: Path) -> Tuple[int, int]:
    seen: Set[str] = set()
    n_rows = 0
    for chunk in pd.read_csv(csv_path, chunksize=4096, low_memory=False):
        genes = normalize_gene_symbol_series(chunk.iloc[:, 0]).astype(str)
        for g in genes:
            if g in seen:
                continue
            seen.add(g)
            n_rows += 1
    head = pd.read_csv(csv_path, nrows=0, low_memory=False)
    n_cols = len(head.columns) - 1
    return n_rows, n_cols


def _outputs_complete(manifest_path: Path, out_root: Path) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        n_genes = int(man["n_genes"])
        expected = {tid for tid, _ in CELL_TYPES}
        got = set(man.get("cell_types", {}).keys())
        if got != expected:
            return False
        for tid, info in man["cell_types"].items():
            vp = out_root / str(info["values_npy"])
            gp = out_root / str(info["genes_npy"])
            cp = out_root / str(info["column_names_npy"])
            if not vp.is_file() or not gp.is_file() or not cp.is_file():
                return False
            mm = np.load(vp, mmap_mode="r")
            if int(mm.shape[0]) != n_genes:
                return False
    except (KeyError, TypeError, ValueError, OSError):
        return False
    return True


def _remove_if_exists(path: Path) -> None:
    if path.is_file():
        path.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize GSE131907 step06 per-Cell_type count npy")
    ap.add_argument("--gene-chunk-rows", type=int, default=512)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    layout = gse131907_celltype_layout()
    master = layout["master_exp_csv"]
    ann = layout["cell_annotation_gz"]
    out_root = layout["celltype_data_dir"]
    if not master.is_file():
        raise FileNotFoundError(master)
    if not ann.is_file():
        raise FileNotFoundError(ann)

    manifest_path = out_root / "manifest.json"
    if _outputs_complete(manifest_path, out_root) and not args.force:
        print(f"[celltype-materialize] skip existing complete outputs under {out_root}")
        return
    if manifest_path.is_file() and not args.force:
        print(
            "[celltype-materialize] manifest exists but outputs incomplete; "
            "re-materializing (use --force to overwrite)"
        )

    groups = _load_cell_groups(ann)
    for tid, label in CELL_TYPES:
        print(f"[celltype-materialize] {tid}: n_cells={len(groups[tid])} ({label})")

    head = pd.read_csv(master, nrows=0, low_memory=False)
    gene_col = str(head.columns[0])
    data_cols: List[str] = [str(c) for c in head.columns[1:]]
    name_to_i = {c: i for i, c in enumerate(data_cols)}

    type_col_indices: Dict[str, np.ndarray] = {}
    type_col_names: Dict[str, List[str]] = {}
    all_group_cells: Set[str] = set()
    for tid, _ in CELL_TYPES:
        all_group_cells |= groups[tid]
    unmatched = sum(1 for c in data_cols if c not in all_group_cells)
    for tid, _ in CELL_TYPES:
        names = [c for c in data_cols if c in groups[tid]]
        if not names:
            raise ValueError(f"No matrix columns for {tid}")
        type_col_names[tid] = names
        type_col_indices[tid] = np.array([name_to_i[c] for c in names], dtype=np.int64)

    n_genes, n_cells = _unique_gene_row_count(master)
    if n_cells != len(data_cols):
        raise ValueError(f"Header cells={len(data_cols)} vs counted={n_cells}")

    out_root.mkdir(parents=True, exist_ok=True)
    mmaps: Dict[str, np.memmap] = {}
    for tid, _ in CELL_TYPES:
        td = out_root / tid
        td.mkdir(parents=True, exist_ok=True)
        n_c = len(type_col_names[tid])
        p = td / "values.npy"
        _remove_if_exists(p)
        mmaps[tid] = np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=(n_genes, n_c))

    seen: Set[str] = set()
    genes_out: List[str] = []
    gi = 0
    chunk_i = 0
    for chunk in pd.read_csv(master, chunksize=int(args.gene_chunk_rows), low_memory=False):
        chunk_i += 1
        chunk[gene_col] = normalize_gene_symbol_series(chunk[gene_col])
        vals = chunk.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)
        glist = chunk[gene_col].astype(str).tolist()
        for j in range(len(chunk)):
            g = str(glist[j]).strip()
            if g in seen:
                continue
            seen.add(g)
            genes_out.append(g)
            for tid, _ in CELL_TYPES:
                mmaps[tid][gi, :] = vals[j, type_col_indices[tid]]
            gi += 1
        if chunk_i == 1 or chunk_i % 10 == 0:
            print(f"[celltype-materialize] chunks={chunk_i}, genes={gi}/{n_genes}", flush=True)

    if gi != n_genes:
        raise ValueError(f"gene count mismatch: expected {n_genes}, wrote {gi}")

    manifest_types: Dict[str, Dict[str, object]] = {}
    for tid, label in CELL_TYPES:
        mmaps[tid].flush()
        del mmaps[tid]
        td = out_root / tid
        np.save(td / "genes.npy", np.array(genes_out, dtype=object))
        np.save(td / "column_names.npy", np.array(type_col_names[tid], dtype=object))
        manifest_types[tid] = {
            "annotation_cell_type": label,
            "n_cells": len(type_col_names[tid]),
            "values_npy": f"{tid}/values.npy",
            "genes_npy": f"{tid}/genes.npy",
            "column_names_npy": f"{tid}/column_names.npy",
        }
        print(f"[celltype-materialize] OK {tid} -> {td / 'values.npy'}")

    manifest = {
        "version": 2,
        "grouping": "annotation_Cell_type",
        "master_exp_csv": str(master),
        "cell_annotation_gz": str(ann),
        "gene_column": gene_col,
        "n_genes": int(n_genes),
        "n_cells_master": int(n_cells),
        "gene_chunk_rows": int(args.gene_chunk_rows),
        "cell_types": manifest_types,
        "unmatched_columns_not_in_any_group": int(unmatched),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[celltype-materialize] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
