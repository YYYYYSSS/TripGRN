"""
Export combined UMAP for GSE131907 mBrain + non-mBrain expression (log-normalized npy).

Writes:
  <out-dir>/gse131907_brain_nonbrain_umap_coords.csv
  columns: cell_idx, domain, umap1, umap2

Usage:
  python -m grn.scripts.export_gse131907_brain_nonbrain_umap
  python -m grn.scripts.export_gse131907_brain_nonbrain_umap --max-cells 0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from grn.scripts.export_gse253718_ablation_umap_coords import (
    _hvg_gene_indices,
    _load_cells_hvg,
    _subsample_idx,
)


def main() -> None:
    p = argparse.ArgumentParser(description="UMAP for GSE131907 mBrain + non-mBrain combined.")
    p.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/exp/GSE131907/precomputed_brain_nonbrain_npy"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/gse131907_application_step4_case_nonmbrain_vs_mbrain"),
    )
    p.add_argument("--max-cells", type=int, default=0, help="0 = all cells (can be slow)")
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--n-pca", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cell-chunk", type=int, default=256)
    args = p.parse_args()

    split_dir = args.split_dir.resolve()
    brain_path = split_dir / "brain_values_log_normalized.npy"
    nb_path = split_dir / "nonbrain_values_log_normalized.npy"
    if not brain_path.is_file() or not nb_path.is_file():
        raise SystemExit(f"Missing npy under {split_dir}")

    mat_b = np.load(brain_path, mmap_mode="r")
    mat_nb = np.load(nb_path, mmap_mode="r")
    if mat_b.shape[0] != mat_nb.shape[0]:
        raise SystemExit(f"Gene count mismatch: brain {mat_b.shape[0]} vs nonbrain {mat_nb.shape[0]}")

    n_b, n_nb = mat_b.shape[1], mat_nb.shape[1]
    domains = np.array(["mBrain"] * n_b + ["non-mBrain"] * n_nb, dtype=object)
    print(f"[umap-131907] brain {n_b} + non-mBrain {n_nb} = {len(domains)} cells")

    idx = _subsample_idx(domains, args.max_cells, args.seed)
    if args.max_cells <= 0:
        print(f"[umap-131907] using all {idx.size} cells")
    else:
        print(f"[umap-131907] subsample {idx.size} cells")

    # Map global index -> (matrix, local_col)
    def _col_slice(gi: int):
        if gi < n_b:
            return mat_b, gi, "mBrain"
        return mat_nb, gi - n_b, "non-mBrain"

    # HVG on pooled subsample (chunked per matrix)
    n_genes = mat_b.shape[0]
    sums = np.zeros(n_genes, dtype=np.float64)
    sums_sq = np.zeros(n_genes, dtype=np.float64)
    n_used = 0
    chunk = max(32, int(args.cell_chunk))
    for start in range(0, idx.size, chunk):
        gidx = idx[start : start + chunk]
        for gi in gidx:
            mat, lc, _ = _col_slice(int(gi))
            col = np.asarray(mat[:, lc], dtype=np.float32)
            if np.nanmax(col) > 50.0:
                np.log1p(col, out=col)
            sums += col
            sums_sq += np.square(col, dtype=np.float64)
            n_used += 1
    mean = sums / max(n_used, 1)
    var = np.maximum(sums_sq / max(n_used, 1) - mean * mean, 0.0)
    top_k = min(args.n_hvg, n_genes)
    hvg = np.argpartition(var, -top_k)[-top_k:]

    X = np.empty((idx.size, hvg.size), dtype=np.float32)
    for start in range(0, idx.size, chunk):
        gidx = idx[start : start + chunk]
        block = np.empty((len(gidx), hvg.size), dtype=np.float32)
        for j, gi in enumerate(gidx):
            mat, lc, _ = _col_slice(int(gi))
            row = np.asarray(mat[hvg, lc], dtype=np.float32)
            if np.nanmax(row) > 50.0:
                np.log1p(row, out=row)
            block[j, :] = row
        X[start : start + len(gidx), :] = block

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(X)
    n_comp = min(args.n_pca, X.shape[0] - 1, X.shape[1])
    emb = PCA(n_components=n_comp, random_state=args.seed).fit_transform(X)

    import umap

    xy = umap.UMAP(n_neighbors=15, min_dist=0.25, random_state=args.seed).fit_transform(emb)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "gse131907_brain_nonbrain_umap_coords.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_idx", "domain", "umap1", "umap2"])
        for i, gi in enumerate(idx):
            _, _, dom = _col_slice(int(gi))
            w.writerow([int(gi), dom, float(xy[i, 0]), float(xy[i, 1])])
    print(f"[umap-131907] wrote {out_csv} ({idx.size} rows)")


if __name__ == "__main__":
    main()
