#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 0b: SingleR auto-annotation -> epithelial cell subset.

Input:  data/scRNA/pancancer/full/{GSE}/
Output: data/scRNA/pancancer/epithelial/{GSE}/
        values.npy, genes.npy, cells.npy, epithelial_indices.npy, singler/, report.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import io as spio
from scipy import sparse as sp

from grn.data.pancancer_paths import (
    PANCANCER_GSE_IDS,
    gse_epithelial_dir,
    gse_full_dir,
    gse_npy_triplet,
)


def _resolve_rscript(cli: str | None) -> Path:
    if cli:
        p = Path(cli)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Rscript not found: {cli}")
    env = os.environ.get("RSCRIPT", "").strip()
    if env and Path(env).is_file():
        return Path(env)
    for cand in (
        r"C:\Program Files\R\R-4.5.2\bin\x64\Rscript.exe",
        r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe",
    ):
        if Path(cand).is_file():
            return Path(cand)
    return Path("Rscript")


def _to_logcounts_matrix(values: np.ndarray) -> np.ndarray:
    """genes x cells -> logcounts (genes x cells) for SingleR."""
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"values must be 2D, got {x.shape}")
    if float(np.nanmax(x)) > 50.0 and float(np.nanmin(x)) >= 0.0:
        import anndata as ad

        adata = ad.AnnData(X=x.T.copy())
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        log_mat = adata.X
        if hasattr(log_mat, "toarray"):
            log_mat = log_mat.toarray()
        return np.asarray(log_mat.T, dtype=np.float32)
    return x.astype(np.float32, copy=False)


def _export_mtx_for_singler(
    values: np.ndarray,
    genes: np.ndarray,
    cells: np.ndarray,
    out_dir: Path,
) -> Path:
    """Write matrix.mtx (genes x cells) + genes.tsv + barcodes.tsv for R Matrix::readMM."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_mat = _to_logcounts_matrix(values)
    gene_names = [str(g) for g in genes.tolist()]
    cell_names = [str(c) for c in cells.tolist()]
    spio.mmwrite(out_dir / "matrix.mtx", sp.csr_matrix(log_mat))
    pd.DataFrame({"gene": gene_names}).to_csv(out_dir / "genes.tsv", sep="\t", header=False, index=False)
    pd.DataFrame({"barcode": cell_names}).to_csv(out_dir / "barcodes.tsv", sep="\t", header=False, index=False)
    meta = {
        "format": "matrix_market",
        "orientation": "genes_x_cells",
        "n_genes": len(gene_names),
        "n_cells": len(cell_names),
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def run_singler_one(
    gse_id: str,
    *,
    grn_root: Path,
    rscript: Path,
    force: bool,
    min_epithelial_cells: int,
) -> Path:
    full_v, full_g, full_c = gse_npy_triplet("full", gse_id, grn_root)
    out_dir = gse_epithelial_dir(gse_id, grn_root)
    out_v, out_g, out_c = gse_npy_triplet("epithelial", gse_id, grn_root)
    report_path = out_dir / "report.json"
    epi_idx_path = out_dir / "epithelial_indices.npy"
    singler_dir = out_dir / "singler"

    if out_v.is_file() and report_path.is_file() and not force:
        print(f"[singler-epi] skip {gse_id} — epithelial subset exists")
        return out_dir

    if not full_v.is_file():
        raise FileNotFoundError(f"Missing full matrix: {full_v}. Run preprocess_pancancer_unified first.")

    values = np.load(full_v, mmap_mode="r")
    genes = np.load(full_g, allow_pickle=True)
    cells = np.load(full_c, allow_pickle=True)
    n_genes, n_cells = values.shape

    singler_input = out_dir / "for_singler"
    _export_mtx_for_singler(values, genes, cells, singler_input)

    r_script = grn_root / "grn" / "r" / "pancancer_singler_epithelial.R"
    singler_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(rscript), str(r_script), str(singler_input), str(singler_dir)]
    print("[singler-epi]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(grn_root))

    idx_file = singler_dir / "epithelial_indices_0based.txt"
    if not idx_file.is_file():
        raise FileNotFoundError(f"SingleR did not write {idx_file}")
    epi_idx = np.array([int(x) for x in idx_file.read_text(encoding="utf-8").splitlines() if x.strip()], dtype=np.int64)
    if epi_idx.size < min_epithelial_cells:
        raise RuntimeError(
            f"{gse_id}: epithelial cells {epi_idx.size} < min {min_epithelial_cells}. "
            "Check SingleR labels in singler/singler_labels.csv"
        )

    sub_values = np.asarray(values[:, epi_idx], dtype=np.float32)
    sub_cells = cells[epi_idx]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_v, sub_values)
    np.save(out_g, genes)
    np.save(out_c, sub_cells)
    np.save(epi_idx_path, epi_idx)

    singler_report = {}
    sr = singler_dir / "singler_report.json"
    if sr.is_file():
        singler_report = json.loads(sr.read_text(encoding="utf-8"))

    report = {
        "gse_id": gse_id,
        "method": "SingleR",
        "reference": "HumanPrimaryCellAtlasData",
        "n_cells_full": int(n_cells),
        "n_cells_epithelial": int(epi_idx.size),
        "n_genes": int(n_genes),
        "epithelial_fraction": float(epi_idx.size / max(n_cells, 1)),
        "singler_report": singler_report,
        "paths": {
            "values_npy": str(out_v.resolve()),
            "genes_npy": str(out_g.resolve()),
            "cells_npy": str(out_c.resolve()),
            "epithelial_indices_npy": str(epi_idx_path.resolve()),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[singler-epi] {gse_id}: {epi_idx.size}/{n_cells} epithelial cells -> {out_dir}"
    )
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="SingleR epithelial subset for pancancer GSE")
    ap.add_argument("--gse", default="ALL")
    ap.add_argument("--rscript", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-epithelial-cells", type=int, default=50)
    args = ap.parse_args()

    grn_root = Path(__file__).resolve().parents[2]
    rscript = _resolve_rscript(args.rscript)
    gses: List[str] = list(PANCANCER_GSE_IDS) if str(args.gse).upper() == "ALL" else [str(args.gse).strip()]

    for gse in gses:
        run_singler_one(
            gse,
            grn_root=grn_root,
            rscript=rscript,
            force=bool(args.force),
            min_epithelial_cells=int(args.min_epithelial_cells),
        )
    print(f"[singler-epi] done: {len(gses)} datasets")


if __name__ == "__main__":
    main()
