"""
Export 2D UMAP coordinates for GSE253718 expression cells (train/val/test column split 5:2:3).

Matches column ordering and split3() in train_unseen_samples_no_cv.py (seed only affects pair split, not columns).

Writes:
  <out-dir>/gse253718_cells_umap_coords.csv  (columns: cell_idx, split, umap1, umap2)

Usage:
  python -m grn.scripts.export_gse253718_ablation_umap_coords
  python -m grn.scripts.export_gse253718_ablation_umap_coords \\
    --exp-values data/exp/GSE253718_exp_counts_values.npy \\
    --out-dir experiments/gse253718_ablation_step1_seed42
  python -m grn.scripts.export_gse253718_ablation_umap_coords --max-cells 0   # all cells (default)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import numpy as np


def split3(n: int, w1: int, w2: int, w3: int) -> Tuple[int, int, int]:
    w = w1 + w2 + w3
    a = n * w1 // w
    b = n * w2 // w
    c = n - a - b
    return a, b, c


def _cell_splits(n_cells: int, sample_split: str) -> np.ndarray:
    w1, w2, w3 = [int(x.strip()) for x in sample_split.split(",")]
    c_tr, c_va, c_te = split3(n_cells, w1, w2, w3)
    labels = np.empty(n_cells, dtype=object)
    labels[:c_tr] = "train"
    labels[c_tr : c_tr + c_va] = "val"
    labels[c_tr + c_va :] = "test"
    return labels


def _subsample_idx(labels: np.ndarray, max_cells: int, seed: int) -> np.ndarray:
    n = labels.shape[0]
    if max_cells <= 0 or n <= max_cells:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    quota = max_cells
    uniq, counts = np.unique(labels, return_counts=True)
    for u, cnt in zip(uniq, counts):
        idx_u = np.flatnonzero(labels == u)
        take = max(1, int(round(quota * cnt / n)))
        take = min(take, idx_u.size)
        parts.append(rng.choice(idx_u, size=take, replace=False))
    idx = np.concatenate(parts)
    if idx.size > max_cells:
        idx = rng.choice(idx, size=max_cells, replace=False)
    return np.sort(idx)


def _hvg_gene_indices(
    mat: np.ndarray,
    cell_idx: np.ndarray,
    n_hvg: int,
    *,
    cell_chunk: int = 256,
    log1p_counts: bool = True,
) -> np.ndarray:
    """Variance over selected cells without loading genes × all_cells at once."""
    n_genes = mat.shape[0]
    n = int(cell_idx.size)
    if n < 2:
        raise ValueError("Need at least 2 cells for HVG selection")
    sums = np.zeros(n_genes, dtype=np.float64)
    sums_sq = np.zeros(n_genes, dtype=np.float64)
    for start in range(0, n, cell_chunk):
        cols = cell_idx[start : start + cell_chunk]
        block = np.asarray(mat[:, cols], dtype=np.float32)
        if log1p_counts and np.nanmax(block) > 50.0:
            np.log1p(block, out=block)
        sums += block.sum(axis=1, dtype=np.float64)
        sums_sq += np.square(block, dtype=np.float64).sum(axis=1, dtype=np.float64)
    mean = sums / n
    var = np.maximum(sums_sq / n - mean * mean, 0.0)
    top_k = min(n_hvg, n_genes)
    return np.argpartition(var, -top_k)[-top_k:]


def _load_cells_hvg(
    mat: np.ndarray,
    cell_idx: np.ndarray,
    hvg: np.ndarray,
    *,
    cell_chunk: int = 256,
    log1p_counts: bool = True,
) -> np.ndarray:
    """Return (n_cells, n_hvg) float32."""
    n = int(cell_idx.size)
    out = np.empty((n, hvg.size), dtype=np.float32)
    for start in range(0, n, cell_chunk):
        end = min(start + cell_chunk, n)
        cols = cell_idx[start:end]
        block = np.asarray(mat[np.ix_(hvg, cols)], dtype=np.float32).T
        if log1p_counts and np.nanmax(block) > 50.0:
            np.log1p(block, out=block)
        out[start:end, :] = block
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="UMAP coords for GSE253718 cells (column split coloring).")
    p.add_argument(
        "--exp-values",
        type=Path,
        default=Path("data/exp/GSE253718_exp_counts_values.npy"),
        help="genes × cells float32 matrix",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/gse253718_ablation_step1_seed42"),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <out-dir>/cells_umap_coords.csv)",
    )
    p.add_argument("--sample-split", default="5,2,3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-cells",
        type=int,
        default=0,
        help="0 = all cells; >0 = stratified subsample cap (for quick tests only)",
    )
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--n-pca", type=int, default=30)
    p.add_argument("--cell-chunk", type=int, default=256, help="Cells per block when reading memmap")
    args = p.parse_args()

    vals_path = args.exp_values.resolve()
    if not vals_path.is_file():
        raise SystemExit(f"Missing expression npy: {vals_path}")

    mat = np.load(vals_path, mmap_mode="r")
    if mat.ndim != 2:
        raise SystemExit(f"Expected 2D matrix, got shape {mat.shape}")
    n_genes, n_cells = mat.shape
    print(f"[umap-export] matrix {n_genes} genes × {n_cells} cells (memmap)")

    splits = _cell_splits(n_cells, args.sample_split)
    cell_idx = _subsample_idx(splits, args.max_cells, args.seed)
    if args.max_cells <= 0:
        print(f"[umap-export] using all {cell_idx.size} cells")
    else:
        print(f"[umap-export] subsample {cell_idx.size} cells (max_cells={args.max_cells})")

    log1p = True
    print(f"[umap-export] selecting top {args.n_hvg} HVG (chunked)...")
    hvg = _hvg_gene_indices(mat, cell_idx, args.n_hvg, cell_chunk=args.cell_chunk, log1p_counts=log1p)
    print(f"[umap-export] loading {cell_idx.size} × {hvg.size} matrix for PCA/UMAP...")
    X = _load_cells_hvg(mat, cell_idx, hvg, cell_chunk=args.cell_chunk, log1p_counts=log1p)

    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise SystemExit("Need scikit-learn: pip install scikit-learn") from e

    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    n_comp = min(args.n_pca, X.shape[0] - 1, X.shape[1])
    print(f"[umap-export] PCA n_components={n_comp} ...")
    emb = PCA(n_components=n_comp, random_state=args.seed).fit_transform(X)

    try:
        import umap
    except ImportError as e:
        raise SystemExit("Need umap-learn: pip install umap-learn") from e

    print(f"[umap-export] UMAP fit_transform on {emb.shape[0]} cells ...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.25,
        metric="euclidean",
        random_state=args.seed,
    )
    xy = reducer.fit_transform(emb)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = (args.out_csv.resolve() if args.out_csv else out_dir / "cells_umap_coords.csv")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_idx", "split", "umap1", "umap2"])
        for i, ci in enumerate(cell_idx):
            w.writerow([int(ci), splits[ci], float(xy[i, 0]), float(xy[i, 1])])

    print(f"[umap-export] wrote {out_csv} ({cell_idx.size} rows)")


if __name__ == "__main__":
    main()
