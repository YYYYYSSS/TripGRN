# -*- coding: utf-8 -*-
"""Run SingleR on GSE123902 merged counts; keep labels for all cells."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from grn.data.gse123902_paths import gse123902_layout
from grn.scripts.preprocess_pancancer_epithelial_singler import (
    _export_mtx_for_singler,
    _resolve_rscript,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rscript", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    layout = gse123902_layout(root)
    values_path = layout["counts_values"]
    genes_path = layout["counts_genes"]
    cells_path = layout["counts_cells"]
    meta_path = layout["cell_meta_csv"]
    out_dir = layout["singler_dir"]
    report_path = out_dir / "singler_pipeline_report.json"
    labels_path = out_dir / "singler_labels.csv"

    if labels_path.is_file() and report_path.is_file() and not args.force:
        print(f"[singler] skip exists {labels_path}", flush=True)
        return

    for p in (values_path, genes_path, cells_path, meta_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}; run prepare_gse123902_from_raw_tar first")

    values = np.load(values_path, mmap_mode="r")
    genes = np.load(genes_path, allow_pickle=True)
    cells = np.load(cells_path, allow_pickle=True)
    meta = pd.read_csv(meta_path)
    if meta.shape[0] != cells.size:
        raise ValueError("cell_metadata rows != cells.npy length")

    for_singler = out_dir / "for_singler"
    print(f"[singler] export mtx genes={genes.size} cells={cells.size}", flush=True)
    _export_mtx_for_singler(np.asarray(values), genes, cells, for_singler)

    rscript = _resolve_rscript(args.rscript)
    r_script = root / "grn" / "r" / "pancancer_singler_epithelial.R"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(rscript), str(r_script), str(for_singler), str(out_dir)]
    print("[singler]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(root))

    labels = pd.read_csv(labels_path)
    # merge site annotations
    merged = labels.merge(meta, left_on="cell_id", right_on="cell_id", how="left")
    merged_path = out_dir / "singler_labels_with_site.csv"
    merged.to_csv(merged_path, index=False)

    epi = merged[merged["is_epithelial"].astype(str).str.lower().isin(["true", "1"])].copy()
    by_site = epi.groupby("site").size().to_dict() if "site" in epi.columns else {}
    report = {
        "n_cells": int(cells.size),
        "n_epithelial": int(epi.shape[0]),
        "epithelial_by_site": {str(k): int(v) for k, v in by_site.items()},
        "label_table": merged["singler_pruned"].value_counts().head(30).to_dict(),
        "paths": {
            "labels": str(labels_path),
            "labels_with_site": str(merged_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print("[singler] done", flush=True)


if __name__ == "__main__":
    main()
