"""
从 GSE131907 原始 UMI 矩阵 + 细胞注释中拆分两条表达谱（基因 × 细胞），供 step1 case study 使用。

划分规则（与 GEO 注释列一致）：
  - 恶性上皮：Cell_type == \"Epithelial cells\" 且 Cell_subtype == \"Malignant cells\"
  - 非上皮：Cell_type != \"Epithelial cells\"（其余上皮亚型不包含在任一侧）

矩阵格式：首行 ``Index\\t<cell1>\\t<cell2>...``，数据行 ``<gene>\\t<UMI>...``（与官方 txt.gz 一致）。

非上皮细胞数量极大时，默认对非上皮侧做随机子采样以控制下游内存；可用参数关闭或调整。

用法：
  python -m grn.scripts.gse131907_split_malignant_epi_nonepi_matrix ^
    --umi-matrix-gz data/exp/GSE131907/GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz ^
    --cell-annotation-gz data/exp/GSE131907/GSE131907_Lung_Cancer_cell_annotation.txt.gz ^
    --out-malignant-epi-csv-gz experiments/gse131907_case_inputs/expression_malignant_epithelium.csv.gz ^
    --out-non-epi-csv-gz experiments/gse131907_case_inputs/expression_non_epithelial.csv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import pandas as pd


def _load_cell_groups(
    ann_gz: Path,
    epithelial_cell_type: str,
    malignant_subtype: str,
) -> Tuple[Set[str], Set[str]]:
    malignant_epi: Set[str] = set()
    non_epi: Set[str] = set()
    with gzip.open(ann_gz, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            i_idx = header.index("Index")
            i_ct = header.index("Cell_type")
            i_sub = header.index("Cell_subtype")
        except ValueError as e:
            raise SystemExit(f"注释表缺少必需列 Index/Cell_type/Cell_subtype: {header}") from e
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_idx, i_ct, i_sub):
                continue
            idx = parts[i_idx]
            ct = parts[i_ct]
            sub = parts[i_sub]
            if ct == epithelial_cell_type and sub == malignant_subtype:
                malignant_epi.add(idx)
            if ct != epithelial_cell_type:
                non_epi.add(idx)
    return malignant_epi, non_epi


def _ordered_column_indices(cell_cols: List[str], wanted: Set[str]) -> List[int]:
    return [i for i, c in enumerate(cell_cols) if c in wanted]


def _maybe_subsample(indices: List[int], max_cells: int, seed: int) -> List[int]:
    if max_cells <= 0 or len(indices) <= max_cells:
        return indices
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(indices), size=max_cells, replace=False)
    pick.sort()
    return [indices[int(j)] for j in pick]


def _write_subset_matrix_pandas(
    umi_gz: Path,
    col_indices: List[int],
    cell_ids: List[str],
    out_gz: Path,
    chunksize: int = 500,
) -> int:
    """仅读取所需细胞列，分块写出 CSV.gz（避免整行 split 全矩阵）。"""
    out_gz = Path(out_gz)
    out_gz.parent.mkdir(parents=True, exist_ok=True)
    usecols = [0] + [1 + i for i in col_indices]
    n_genes = 0
    with gzip.open(out_gz, "wt", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["gene"] + cell_ids)
        for chunk in pd.read_csv(
            umi_gz,
            sep="\t",
            compression="gzip",
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        ):
            genes = chunk.iloc[:, 0].astype(str).to_numpy()
            mat = chunk.iloc[:, 1:].to_numpy()
            for gi in range(len(genes)):
                w.writerow([genes[gi]] + [str(x) for x in mat[gi, :]])
            n_genes += len(genes)
    return n_genes


def main() -> None:
    p = argparse.ArgumentParser(description="GSE131907: 恶性上皮 vs 非上皮 → 两个 gene×cell CSV.gz")
    p.add_argument("--umi-matrix-gz", type=Path, required=True)
    p.add_argument("--cell-annotation-gz", type=Path, required=True)
    p.add_argument("--out-malignant-epi-csv-gz", type=Path, required=True)
    p.add_argument("--out-non-epi-csv-gz", type=Path, required=True)
    p.add_argument("--epithelial-cell-type", type=str, default="Epithelial cells")
    p.add_argument("--malignant-subtype", type=str, default="Malignant cells")
    p.add_argument(
        "--max-non-epithelial-cells",
        type=int,
        default=8000,
        help="非上皮侧最多保留细胞数；0 表示不截断（内存占用极高）",
    )
    p.add_argument(
        "--max-malignant-epithelial-cells",
        type=int,
        default=0,
        help="恶性上皮侧最多保留细胞数；0 表示不截断",
    )
    p.add_argument("--subsample-seed", type=int, default=42)
    p.add_argument("--meta-json", type=Path, default=None, help="写出拆分参数与细胞数统计")
    args = p.parse_args()

    mal_set, non_set = _load_cell_groups(
        args.cell_annotation_gz,
        args.epithelial_cell_type,
        args.malignant_subtype,
    )
    if not mal_set:
        raise SystemExit("恶性上皮细胞集合为空，请检查注释列名与取值")
    if not non_set:
        raise SystemExit("非上皮细胞集合为空")

    with gzip.open(args.umi_matrix_gz, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
    if not header or header[0] != "Index":
        raise SystemExit("矩阵首行第一列须为 Index")
    cell_cols = header[1:]
    if len(cell_cols) != len(set(cell_cols)):
        raise SystemExit("矩阵表头存在重复细胞 ID")

    mal_idx = _ordered_column_indices(cell_cols, mal_set)
    non_idx = _ordered_column_indices(cell_cols, non_set)
    if len(mal_idx) != len(mal_set):
        raise SystemExit("恶性上皮细胞在矩阵表头中缺失，集合与矩阵不一致")
    if len(non_idx) != len(non_set):
        raise SystemExit("非上皮细胞在矩阵表头中缺失")

    mal_idx = _maybe_subsample(mal_idx, args.max_malignant_epithelial_cells, args.subsample_seed)
    non_idx = _maybe_subsample(non_idx, args.max_non_epithelial_cells, args.subsample_seed + 1)
    mal_ids = [cell_cols[i] for i in mal_idx]
    non_ids = [cell_cols[i] for i in non_idx]

    print(
        f"[gse131907-split] malignant_epithelial_cells={len(mal_idx)} "
        f"(raw_in_annotation={len(mal_set)}), non_epithelial_cells={len(non_idx)} "
        f"(raw_in_annotation={len(non_set)})"
    )

    ng_m = _write_subset_matrix_pandas(args.umi_matrix_gz, mal_idx, mal_ids, args.out_malignant_epi_csv_gz)
    ng_n = _write_subset_matrix_pandas(args.umi_matrix_gz, non_idx, non_ids, args.out_non_epi_csv_gz)
    print(f"[gse131907-split] wrote genes={ng_m} -> {args.out_malignant_epi_csv_gz}")
    print(f"[gse131907-split] wrote genes={ng_n} -> {args.out_non_epi_csv_gz}")

    meta = {
        "umi_matrix_gz": str(args.umi_matrix_gz.resolve()),
        "cell_annotation_gz": str(args.cell_annotation_gz.resolve()),
        "epithelial_cell_type": args.epithelial_cell_type,
        "malignant_subtype": args.malignant_subtype,
        "n_malignant_epithelial_in_annotation": len(mal_set),
        "n_non_epithelial_in_annotation": len(non_set),
        "n_malignant_epithelial_exported": len(mal_idx),
        "n_non_epithelial_exported": len(non_idx),
        "max_malignant_epithelial_cells": args.max_malignant_epithelial_cells,
        "max_non_epithelial_cells": args.max_non_epithelial_cells,
        "subsample_seed": args.subsample_seed,
        "out_malignant_epi_csv_gz": str(Path(args.out_malignant_epi_csv_gz).resolve()),
        "out_non_epi_csv_gz": str(Path(args.out_non_epi_csv_gz).resolve()),
        "case_study_mapping": {
            "nonbrain_exp_csv_semantics": "non_epithelial_microenvironment",
            "brain_exp_csv_semantics": "malignant_epithelium",
        },
    }
    meta_path = (
        Path(args.meta_json)
        if args.meta_json is not None
        else Path(args.out_malignant_epi_csv_gz).parent / "gse131907_split_meta.json"
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gse131907-split] meta -> {meta_path}")


if __name__ == "__main__":
    main()
