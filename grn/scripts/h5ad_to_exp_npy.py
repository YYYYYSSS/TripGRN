from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Tuple

import anndata as ad
import numpy as np
from numpy.lib.format import open_memmap
from scipy.sparse import issparse

from grn.data.preprocessing import normalize_gene_symbol_series


def _pick_expression_matrix(adata: ad.AnnData, layer: Optional[str]) -> Tuple[Any, str]:
    """
    返回 (matrix cells×genes, 描述字符串)。
    - None / x：用 adata.X
    - auto：优先 layers['data']（与仓库 preprocess 导出的 log_normalized 一致），否则退回 .X 并告警
    - 其它：必须是 adata.layers 里存在的键
    """
    avail = sorted(adata.layers.keys())
    if layer is None:
        return adata.X, "X"
    key = str(layer).strip()
    low = key.lower()
    if low in ("", "x", "none"):
        return adata.X, "X"
    if low == "auto":
        if "data" in adata.layers:
            return adata.layers["data"], "layers['data']"
        print(
            "[h5ad-to-npy] WARNING: 未找到 layers['data']，改用 adata.X。\n"
            f"            当前 layers 键: {avail}\n"
            "            若 .X 仍是 counts，请先 normalize_total+log1p 并写入 layers['data']，"
            "或与 Step1 的 GSE253718_exp.csv（log1p 后）一致后再导出。"
        )
        return adata.X, "X (fallback, no layers['data'])"
    if key not in adata.layers:
        raise ValueError(
            f"layer {key!r} 不存在。当前 layers 键: {avail}。"
            "若已与 Step1 对齐的矩阵就在 .X，可改用 --layer auto 或省略 --layer。"
        )
    return adata.layers[key], f"layers[{key!r}]"


def convert_h5ad_to_npy(h5ad: Path, out_prefix: Path, layer: Optional[str], chunk_genes: int) -> None:
    adata = ad.read_h5ad(str(h5ad))
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    X, src = _pick_expression_matrix(adata, layer=layer)  # cells x genes
    genes = normalize_gene_symbol_series(adata.var_names.to_series()).tolist()
    n_cells, n_genes = int(adata.n_obs), int(adata.n_vars)

    out_values = Path(str(out_prefix) + "_values.npy")
    out_genes = Path(str(out_prefix) + "_genes.npy")
    out_values.parent.mkdir(parents=True, exist_ok=True)

    # output for training/eval pipeline expects genes x cells float32
    mm = open_memmap(out_values, mode="w+", dtype=np.float32, shape=(n_genes, n_cells))
    chunk_genes = max(1, int(chunk_genes))

    if issparse(X):
        X = X.tocsr()
        for g0 in range(0, n_genes, chunk_genes):
            g1 = min(n_genes, g0 + chunk_genes)
            block = X[:, g0:g1].toarray().astype(np.float32, copy=False)  # cells x chunk
            mm[g0:g1, :] = block.T
    else:
        X = np.asarray(X, dtype=np.float32)
        for g0 in range(0, n_genes, chunk_genes):
            g1 = min(n_genes, g0 + chunk_genes)
            mm[g0:g1, :] = X[:, g0:g1].T

    mm.flush()
    np.save(out_genes, np.asarray(genes, dtype=object))
    print(
        f"[h5ad-to-npy] done values={out_values} genes={out_genes} "
        f"shape=({n_genes}, {n_cells}) source={src}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert h5ad to (genes x cells) npy + genes npy.")
    ap.add_argument("--h5ad", required=True, help="Input h5ad path")
    ap.add_argument("--out-prefix", required=True, help="Output prefix for *_values.npy and *_genes.npy")
    ap.add_argument(
        "--layer",
        default=None,
        help="表达矩阵来源：省略或 x=adata.X；data=layers['data']；auto=优先 data 否则 X（缺 data 时会打印警告）",
    )
    ap.add_argument("--chunk-genes", type=int, default=500, help="Gene chunk size")
    args = ap.parse_args()

    convert_h5ad_to_npy(
        h5ad=Path(args.h5ad),
        out_prefix=Path(args.out_prefix),
        layer=args.layer,
        chunk_genes=args.chunk_genes,
    )


if __name__ == "__main__":
    main()
