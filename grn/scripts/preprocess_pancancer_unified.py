#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 0a: consolidate nine pancancer GSE matrices into data/scRNA/pancancer/full/{GSE}/.

Sources (first hit wins):
  1. data/scRNA/pancancer/full/{GSE}/  (already unified)
  2. data/exp/pancancer_scdata_step5/npy/{GSE}_*.npy
  3. data/exp/pancancer/{GSE}.csv  -> convert via wide_expression_csv_to_npy (temp)

Outputs per GSE:
  values.npy, genes.npy, cells.npy, source_manifest.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from grn.data.pancancer_paths import (
    PANCANCER_GSE_IDS,
    gse_full_dir,
    gse_npy_triplet,
    scRNA_layout,
    write_manifest,
)
from grn.data.preprocessing import normalize_gene_symbol_series


def _read_csv_header_cells(csv_path: Path) -> List[str]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        header = f.readline().strip().split(",")
    if len(header) < 2:
        raise ValueError(f"CSV header too short: {csv_path}")
    return [h.strip().strip('"').strip("'") for h in header[1:]]


def _default_cells(n: int) -> List[str]:
    return [f"cell_{i:06d}" for i in range(n)]


def _copy_or_link_npy(src_v: Path, src_g: Path, dst_v: Path, dst_g: Path) -> None:
    dst_v.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_v, dst_v)
    shutil.copy2(src_g, dst_g)


def _resolve_legacy_npy(gse_id: str, layout) -> Optional[Tuple[Path, Path]]:
    legacy = layout["legacy_npy"]
    v = legacy / f"{gse_id}_values.npy"
    g = legacy / f"{gse_id}_genes.npy"
    if v.is_file() and g.is_file():
        return v, g
    return None


def _resolve_legacy_csv(gse_id: str, layout) -> Optional[Path]:
    p = layout["legacy_csv_dir"] / f"{gse_id}.csv"
    return p if p.is_file() else None


def consolidate_one_gse(
    gse_id: str,
    *,
    grn_root: Path,
    filter_zero_frac: float,
    force: bool,
) -> Path:
    out_v, out_g, out_c = gse_npy_triplet("full", gse_id, grn_root)
    out_dir = out_v.parent
    manifest_path = out_dir / "source_manifest.json"

    if out_v.is_file() and out_g.is_file() and out_c.is_file() and not force:
        print(f"[unified-full] skip {gse_id} — already exists")
        return out_dir

    layout = scRNA_layout(grn_root)
    cells: List[str] = []
    source_note = ""

    legacy_npy = _resolve_legacy_npy(gse_id, layout)
    if legacy_npy is not None:
        src_v, src_g = legacy_npy
        _copy_or_link_npy(src_v, src_g, out_v, out_g)
        n_cells = int(np.load(out_v, mmap_mode="r").shape[1])
        filt = layout["legacy_npy"] / f"{gse_id}_filter_manifest.json"
        csv_from_filt = None
        if filt.is_file():
            meta = json.loads(filt.read_text(encoding="utf-8"))
            csv_from_filt = Path(meta.get("csv", ""))
        if csv_from_filt and csv_from_filt.is_file():
            try:
                cells = _read_csv_header_cells(csv_from_filt)
            except Exception:
                cells = _default_cells(n_cells)
        else:
            cells = _default_cells(n_cells)
        source_note = f"legacy_npy:{src_v}"
    else:
        csv_path = _resolve_legacy_csv(gse_id, layout)
        if csv_path is None:
            raise FileNotFoundError(
                f"No source for {gse_id}: expected {layout['legacy_npy']}/{gse_id}_values.npy "
                f"or {layout['legacy_csv_dir']}/{gse_id}.csv"
            )
        cells = _read_csv_header_cells(csv_path)
        tmp_prefix = out_dir / "_tmp_npy"
        cmd = [
            sys.executable,
            "-m",
            "grn.scripts.wide_expression_csv_to_npy",
            "--csv",
            str(csv_path),
            "--out-prefix",
            str(tmp_prefix),
            "--filter-cells-max-zero-frac",
            str(filter_zero_frac),
        ]
        print("[unified-full]", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(grn_root))
        _copy_or_link_npy(
            Path(f"{tmp_prefix}_values.npy"),
            Path(f"{tmp_prefix}_genes.npy"),
            out_v,
            out_g,
        )
        for p in tmp_prefix.parent.glob("_tmp_npy*"):
            try:
                p.unlink()
            except OSError:
                pass
        source_note = f"legacy_csv:{csv_path}"

    if len(cells) != int(np.load(out_v, mmap_mode="r").shape[1]):
        n_cells = int(np.load(out_v, mmap_mode="r").shape[1])
        print(
            f"[unified-full] warning: {gse_id} cell name count {len(cells)} != n_cells {n_cells}; "
            "using synthetic names"
        )
        cells = _default_cells(n_cells)

    np.save(out_c, np.asarray(cells, dtype=object))
    g_arr = np.load(out_g, allow_pickle=True)
    genes_norm = normalize_gene_symbol_series(pd.Series([str(x) for x in g_arr.tolist()])).tolist()
    np.save(out_g, np.asarray(genes_norm, dtype=object))

    manifest = {
        "gse_id": gse_id,
        "source": source_note,
        "n_genes": int(np.load(out_v, mmap_mode="r").shape[0]),
        "n_cells": int(np.load(out_v, mmap_mode="r").shape[1]),
        "values_npy": str(out_v.resolve()),
        "genes_npy": str(out_g.resolve()),
        "cells_npy": str(out_c.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[unified-full] {gse_id} -> {out_dir} ({manifest['n_genes']} genes, {manifest['n_cells']} cells)")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Consolidate pancancer full matrices to data/scRNA/pancancer/full/")
    ap.add_argument("--gse", default="ALL", help="GSE id or ALL")
    ap.add_argument("--filter-cells-max-zero-frac", type=float, default=0.9)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    grn_root = Path(__file__).resolve().parents[2]
    gses = list(PANCANCER_GSE_IDS) if str(args.gse).upper() == "ALL" else [str(args.gse).strip()]
    done = []
    for gse in gses:
        consolidate_one_gse(
            gse,
            grn_root=grn_root,
            filter_zero_frac=float(args.filter_cells_max_zero_frac),
            force=bool(args.force),
        )
        done.append(gse)

    write_manifest(
        {
            "gse_ids": done,
            "tier": "full",
            "n_datasets": len(done),
        },
        grn_root,
    )
    print(f"[unified-full] done: {len(done)} datasets")


if __name__ == "__main__":
    main()
