#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SingleR epithelial subset for GSE253718 ablation (step02).

Input:  data/exp/GSE253718_exp_counts_{values,genes}.npy + CSV header for cell barcodes
Output: data/exp/GSE253718_epithelial_counts_{values,genes,cells}.npy
        data/exp/GSE253718_epithelial/singler/, report.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from grn.data.gse253718_paths import gse253718_layout
from grn.scripts.preprocess_pancancer_epithelial_singler import (
    _export_mtx_for_singler,
    _resolve_rscript,
)


def _read_cell_names_from_csv(csv_path: Path) -> np.ndarray:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    if len(header) < 2:
        raise ValueError(f"CSV header has no cell columns: {csv_path}")
    return np.asarray([str(c).strip() for c in header[1:]], dtype=object)


def run_gse253718_epithelial(
    *,
    grn_root: Path,
    rscript: Path,
    force: bool,
    min_epithelial_cells: int,
) -> Path:
    layout = gse253718_layout(grn_root)
    full_v = layout["exp_values_full"]
    full_g = layout["exp_genes_full"]
    exp_csv = layout["exp_csv"]
    out_dir = layout["epithelial_dir"]
    out_v = layout["exp_values"]
    out_g = layout["exp_genes"]
    out_c = layout["exp_cells"]
    report_path = out_dir / "report.json"
    epi_idx_path = out_dir / "epithelial_indices.npy"
    singler_dir = out_dir / "singler"

    if out_v.is_file() and report_path.is_file() and not force:
        print(f"[gse253718-epi] skip — epithelial subset exists: {out_v}")
        return out_dir

    if not full_v.is_file():
        raise FileNotFoundError(f"Missing full counts npy: {full_v}")
    if not full_g.is_file():
        raise FileNotFoundError(f"Missing genes npy: {full_g}")
    if not exp_csv.is_file():
        raise FileNotFoundError(f"Missing expression CSV (for cell barcodes): {exp_csv}")

    values = np.load(full_v, mmap_mode="r")
    genes = np.load(full_g, allow_pickle=True)
    cells = _read_cell_names_from_csv(exp_csv)
    n_genes, n_cells = values.shape
    if cells.size != n_cells:
        raise ValueError(
            f"Cell count mismatch: npy columns={n_cells}, CSV cells={cells.size}. "
            "Rebuild GSE253718_exp_counts npy from the same CSV."
        )

    singler_input = out_dir / "for_singler"
    _export_mtx_for_singler(values, genes, cells, singler_input)

    import subprocess

    r_script = grn_root / "grn" / "r" / "pancancer_singler_epithelial.R"
    singler_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(rscript), str(r_script), str(singler_input), str(singler_dir)]
    print("[gse253718-epi]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(grn_root))

    idx_file = singler_dir / "epithelial_indices_0based.txt"
    if not idx_file.is_file():
        raise FileNotFoundError(f"SingleR did not write {idx_file}")
    epi_idx = np.array(
        [int(x) for x in idx_file.read_text(encoding="utf-8").splitlines() if x.strip()],
        dtype=np.int64,
    )
    if epi_idx.size < min_epithelial_cells:
        raise RuntimeError(
            f"GSE253718 epithelial cells {epi_idx.size} < min {min_epithelial_cells}. "
            f"Check {singler_dir / 'singler_labels.csv'}"
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
        "gse_id": "GSE253718",
        "method": "SingleR",
        "reference": "HumanPrimaryCellAtlasData",
        "n_cells_full": int(n_cells),
        "n_cells_epithelial": int(epi_idx.size),
        "n_genes": int(n_genes),
        "epithelial_fraction": float(epi_idx.size / max(n_cells, 1)),
        "singler_report": singler_report,
        "paths": {
            "full_values_npy": str(full_v.resolve()),
            "values_npy": str(out_v.resolve()),
            "genes_npy": str(out_g.resolve()),
            "cells_npy": str(out_c.resolve()),
            "epithelial_indices_npy": str(epi_idx_path.resolve()),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gse253718-epi] {epi_idx.size}/{n_cells} epithelial cells -> {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="SingleR epithelial subset for GSE253718 ablation")
    ap.add_argument("--rscript", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-epithelial-cells", type=int, default=50)
    args = ap.parse_args()

    grn_root = Path(__file__).resolve().parents[2]
    run_gse253718_epithelial(
        grn_root=grn_root,
        rscript=_resolve_rscript(args.rscript),
        force=bool(args.force),
        min_epithelial_cells=int(args.min_epithelial_cells),
    )


if __name__ == "__main__":
    main()
