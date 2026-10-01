"""
将两个「基因×细胞」表达 npy（与 wide_expression_csv_to_npy 一致）按基因交集水平拼接列。

左矩阵列 + 右矩阵列 -> 输出 out_prefix_values.npy / out_prefix_genes.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from grn.data.preprocessing import normalize_gene_symbol_series


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--left-values", required=True)
    ap.add_argument("--left-genes", required=True)
    ap.add_argument("--right-values", required=True)
    ap.add_argument("--right-genes", required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    gL = np.load(args.left_genes, allow_pickle=True)
    gR = np.load(args.right_genes, allow_pickle=True)
    genes_l = normalize_gene_symbol_series(pd.Series([str(x) for x in gL.tolist()])).tolist()
    genes_r = normalize_gene_symbol_series(pd.Series([str(x) for x in gR.tolist()])).tolist()
    map_l = {g: i for i, g in enumerate(genes_l)}
    map_r = {g: i for i, g in enumerate(genes_r)}
    common = [g for g in genes_l if g in map_r]
    if not common:
        raise SystemExit("左右基因无交集")
    idx_l = [map_l[g] for g in common]
    idx_r = [map_r[g] for g in common]

    vl = np.load(args.left_values, mmap_mode="r")
    vr = np.load(args.right_values, mmap_mode="r")
    if vl.shape[0] != len(genes_l) or vr.shape[0] != len(genes_r):
        raise SystemExit("基因向量长度与 values 行数不一致")

    sub_l = np.asarray(vl[idx_l, :], dtype=np.float32)
    sub_r = np.asarray(vr[idx_r, :], dtype=np.float32)
    out = np.concatenate([sub_l, sub_r], axis=1)

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(prefix) + "_values.npy", out)
    np.save(str(prefix) + "_genes.npy", np.asarray(common, dtype=object))
    print(
        f"[concat-expr-npy] genes={len(common)} cols={out.shape[1]} "
        f"(left={sub_l.shape[1]} + right={sub_r.shape[1]}) -> {prefix}_*.npy"
    )


if __name__ == "__main__":
    main()
