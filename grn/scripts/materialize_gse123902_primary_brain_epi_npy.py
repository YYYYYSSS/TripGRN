# -*- coding: utf-8 -*-
"""Materialize primary/brain epithelial log-normalized npy for TripGRN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grn.data.gse123902_paths import SITE_BRAIN, SITE_PRIMARY, gse123902_layout


def _log_normalize(counts: np.ndarray, *, target_sum: float = 1e4) -> np.ndarray:
    x = np.asarray(counts, dtype=np.float64)
    col_sum = x.sum(axis=0)
    scale = np.zeros_like(col_sum)
    pos = col_sum > 0
    scale[pos] = float(target_sum) / col_sum[pos]
    return np.log1p(x * scale).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-cells-per-site", type=int, default=50)
    ap.add_argument("--filter-cells-max-zero-frac", type=float, default=0.9,
                    help="Drop cells with zero expression in >= this fraction of genes")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    layout = gse123902_layout(root)
    out = layout["npy_dir"]
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"
    if man_path.is_file() and not args.force:
        print(f"[mat] skip exists {man_path}", flush=True)
        return

    values = np.load(layout["counts_values"], mmap_mode="r")
    genes = np.load(layout["counts_genes"], allow_pickle=True)
    cells = np.load(layout["counts_cells"], allow_pickle=True)
    lab = layout["singler_dir"] / "singler_labels_with_site.csv"
    if not lab.is_file():
        raise FileNotFoundError(f"Missing {lab}; run preprocess_gse123902_singler first")
    df = pd.read_csv(lab)
    df["is_epithelial"] = df["is_epithelial"].astype(str).str.lower().isin(["true", "1"])

    cell_to_i = {str(c): i for i, c in enumerate(cells.tolist())}
    payload = {"n_genes": int(genes.size), "sites": {}}

    np.save(out / "genes.npy", genes)

    for site, key in ((SITE_PRIMARY, "primary"), (SITE_BRAIN, "brain")):
        sub = df[(df["site"] == site) & (df["is_epithelial"])].copy()
        ids = [str(x) for x in sub["cell_id"].tolist() if str(x) in cell_to_i]
        if len(ids) < int(args.min_cells_per_site):
            raise RuntimeError(f"{site} epithelial cells {len(ids)} < min {args.min_cells_per_site}")
        idx = np.asarray([cell_to_i[c] for c in ids], dtype=np.int64)
        mat = np.asarray(values[:, idx], dtype=np.float32)

        # cell filter: zero-frac across genes
        zero_frac = (mat <= 0).mean(axis=0)
        keep = zero_frac < float(args.filter_cells_max_zero_frac)
        mat = mat[:, keep]
        ids_k = [ids[j] for j, k in enumerate(keep.tolist()) if k]
        print(f"[mat] {site}: epi {len(ids)} -> after zero-frac {mat.shape[1]}", flush=True)

        counts_path = out / f"{key}_values_counts.npy"
        log_path = out / f"{key}_values_log_normalized.npy"
        cols_path = out / f"{key}_column_names.npy"
        np.save(counts_path, mat)
        np.save(log_path, _log_normalize(mat))
        np.save(cols_path, np.asarray(ids_k, dtype=object))
        # also write cell meta subset
        sub_meta = sub[sub["cell_id"].astype(str).isin(ids_k)]
        sub_meta.to_csv(out / f"{key}_cell_metadata.csv", index=False)
        payload["sites"][key] = {
            "n_cells": int(mat.shape[1]),
            "counts": str(counts_path),
            "log_normalized": str(log_path),
            "columns": str(cols_path),
        }

    man_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print("[mat] done", flush=True)


if __name__ == "__main__":
    main()
