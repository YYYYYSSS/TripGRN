from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
from scipy.sparse import issparse

from grn.data.preprocessing import normalize_gene_symbol_series


def _sanitize_header(s: str) -> str:
    return str(s).strip().replace("\r", "_").replace("\n", "_").replace(",", "_")


def _resolve_matrix(adata: ad.AnnData, layer: Optional[str]):
    if layer:
        if layer not in adata.layers:
            raise ValueError(f"layer not found in h5ad: {layer}")
        return adata.layers[layer]
    return adata.X


def export_h5ad_to_genes_by_cells_csv(
    h5ad_path: Path,
    output_csv: Path,
    layer: Optional[str] = None,
    chunk_genes: int = 500,
) -> None:
    adata = ad.read_h5ad(str(h5ad_path))
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    X = _resolve_matrix(adata, layer=layer)
    if issparse(X):
        X = X.tocsr()
    else:
        X = np.asarray(X)

    genes = normalize_gene_symbol_series(adata.var_names.to_series()).tolist()
    cells = [_sanitize_header(x) for x in adata.obs_names.tolist()]
    if len(set(cells)) != len(cells):
        raise ValueError("cell headers are not unique after sanitize")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    chunk_genes = max(1, int(chunk_genes))
    n_cells, n_genes = int(adata.n_obs), int(adata.n_vars)
    float_fmt = "%.8g"

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gene"] + cells)
        for g0 in range(0, n_genes, chunk_genes):
            g1 = min(n_genes, g0 + chunk_genes)
            if issparse(X):
                block = X[:, g0:g1].toarray()
            else:
                block = np.asarray(X[:, g0:g1])
            for j, gi in enumerate(range(g0, g1)):
                row = block[:, j]
                w.writerow([genes[gi]] + [float_fmt % float(v) for v in row])

    print(
        f"[h5ad-to-csv] done: {output_csv} | cells={n_cells}, genes={n_genes}, "
        f"layer={'X' if layer is None else layer}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert .h5ad to genes-by-cells expression CSV.")
    ap.add_argument("--h5ad", required=True, help="Input h5ad path.")
    ap.add_argument("--output-csv", required=True, help="Output CSV path.")
    ap.add_argument("--layer", default=None, help="Optional layer key; default uses .X.")
    ap.add_argument("--chunk-genes", type=int, default=500, help="Gene chunk size for CSV export.")
    args = ap.parse_args()

    export_h5ad_to_genes_by_cells_csv(
        h5ad_path=Path(args.h5ad),
        output_csv=Path(args.output_csv),
        layer=args.layer,
        chunk_genes=args.chunk_genes,
    )


if __name__ == "__main__":
    main()
