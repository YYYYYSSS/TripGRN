"""
GSE131907：先剔除「非恶性上皮」（上皮且 Cell_subtype 非 Malignant cells），
再在「恶性上皮 + 非上皮」细胞池中，按 Feature Summary 的 Tissue origins 分为
  - non-mBrain（非脑转移样本来源细胞）
  - mBrain（脑转移样本来源细胞）
写出两个 gene×cell CSV.gz，供 step1 case study 两侧表达对比。

矩阵列名须与注释 Index 一致，且列名以 _<Samples> 结尾（与 train_metastasis 一致）。
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import List, Set, Tuple

from grn.data.gse131907_feature_summary import mbrain_sample_ids_from_feature_summary


def _load_pooled_cell_barcode_sets(
    ann_gz: Path,
    feature_xlsx: Path,
    epithelial_cell_type: str,
    malignant_subtype: str,
) -> Tuple[Set[str], Set[str]]:
    mbrain_samples = mbrain_sample_ids_from_feature_summary(str(feature_xlsx))
    if not mbrain_samples:
        raise SystemExit("Feature Summary 中未找到 Tissue origins == mBrain 的样本")

    non_mbrain_cells: Set[str] = set()
    mbrain_cells: Set[str] = set()
    with gzip.open(ann_gz, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            i_idx = header.index("Index")
            i_sample = header.index("Sample")
            i_ct = header.index("Cell_type")
            i_sub = header.index("Cell_subtype")
        except ValueError as e:
            raise SystemExit(f"注释表缺少 Index/Sample/Cell_type/Cell_subtype: {header}") from e
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_idx, i_sample, i_ct, i_sub):
                continue
            idx = parts[i_idx]
            sample = parts[i_sample].strip()
            ct = parts[i_ct]
            sub = parts[i_sub]
            if ct == epithelial_cell_type and sub != malignant_subtype:
                continue
            if sample in mbrain_samples:
                mbrain_cells.add(idx)
            else:
                non_mbrain_cells.add(idx)
    return non_mbrain_cells, mbrain_cells


def main() -> None:
    from grn.scripts.gse131907_split_malignant_epi_nonepi_matrix import (
        _maybe_subsample,
        _ordered_column_indices,
        _write_subset_matrix_pandas,
    )

    p = argparse.ArgumentParser(
        description="GSE131907: pooled malignant_epi+non-epi -> non-mBrain vs mBrain CSV.gz"
    )
    p.add_argument("--umi-matrix-gz", type=Path, required=True)
    p.add_argument("--cell-annotation-gz", type=Path, required=True)
    p.add_argument("--feature-summary-xlsx", type=Path, required=True)
    p.add_argument("--out-non-mbrain-csv-gz", type=Path, required=True)
    p.add_argument("--out-mbrain-csv-gz", type=Path, required=True)
    p.add_argument("--epithelial-cell-type", type=str, default="Epithelial cells")
    p.add_argument("--malignant-subtype", type=str, default="Malignant cells")
    p.add_argument(
        "--max-non-mbrain-cells",
        type=int,
        default=12000,
        help="non-mBrain 侧最多细胞数；0 不截断（内存大）",
    )
    p.add_argument(
        "--max-mbrain-cells",
        type=int,
        default=0,
        help="mBrain 侧最多细胞数；0 不截断",
    )
    p.add_argument("--subsample-seed", type=int, default=42)
    p.add_argument("--meta-json", type=Path, default=None)
    args = p.parse_args()

    nb_set, mb_set = _load_pooled_cell_barcode_sets(
        args.cell_annotation_gz,
        args.feature_summary_xlsx,
        args.epithelial_cell_type,
        args.malignant_subtype,
    )
    if not nb_set:
        raise SystemExit("non-mBrain 细胞集合为空")
    if not mb_set:
        raise SystemExit("mBrain 细胞集合为空")

    with gzip.open(args.umi_matrix_gz, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
    if not header or header[0] != "Index":
        raise SystemExit("矩阵首行第一列须为 Index")
    cell_cols = header[1:]
    if len(cell_cols) != len(set(cell_cols)):
        raise SystemExit("矩阵表头存在重复细胞 ID")

    nb_idx = _ordered_column_indices(cell_cols, nb_set)
    mb_idx = _ordered_column_indices(cell_cols, mb_set)
    if len(nb_idx) != len(nb_set):
        raise SystemExit("non-mBrain 细胞在矩阵表头中缺失")
    if len(mb_idx) != len(mb_set):
        raise SystemExit("mBrain 细胞在矩阵表头中缺失")

    nb_idx = _maybe_subsample(nb_idx, args.max_non_mbrain_cells, args.subsample_seed)
    mb_idx = _maybe_subsample(mb_idx, args.max_mbrain_cells, args.subsample_seed + 1)
    nb_ids = [cell_cols[i] for i in nb_idx]
    mb_ids = [cell_cols[i] for i in mb_idx]

    print(
        f"[gse131907-mbrain-split] non_mBrain_cells={len(nb_idx)} "
        f"(raw={len(nb_set)}), mBrain_cells={len(mb_idx)} (raw={len(mb_set)})"
    )

    ng_nb = _write_subset_matrix_pandas(args.umi_matrix_gz, nb_idx, nb_ids, args.out_non_mbrain_csv_gz)
    ng_mb = _write_subset_matrix_pandas(args.umi_matrix_gz, mb_idx, mb_ids, args.out_mbrain_csv_gz)
    print(f"[gse131907-mbrain-split] wrote genes={ng_nb} -> {args.out_non_mbrain_csv_gz}")
    print(f"[gse131907-mbrain-split] wrote genes={ng_mb} -> {args.out_mbrain_csv_gz}")

    meta = {
        "umi_matrix_gz": str(args.umi_matrix_gz.resolve()),
        "cell_annotation_gz": str(args.cell_annotation_gz.resolve()),
        "feature_summary_xlsx": str(args.feature_summary_xlsx.resolve()),
        "epithelial_cell_type": args.epithelial_cell_type,
        "malignant_subtype": args.malignant_subtype,
        "cell_filter_zh": "剔除上皮且非 Malignant cells；保留恶性上皮 + 全部非上皮",
        "n_non_mbrain_in_annotation": len(nb_set),
        "n_mbrain_in_annotation": len(mb_set),
        "n_non_mbrain_exported": len(nb_idx),
        "n_mbrain_exported": len(mb_idx),
        "max_non_mbrain_cells": args.max_non_mbrain_cells,
        "max_mbrain_cells": args.max_mbrain_cells,
        "subsample_seed": args.subsample_seed,
        "out_non_mbrain_csv_gz": str(Path(args.out_non_mbrain_csv_gz).resolve()),
        "out_mbrain_csv_gz": str(Path(args.out_mbrain_csv_gz).resolve()),
        "case_study_mapping": {
            "nonbrain_exp_csv_semantics": "non_mBrain_pooled_cells",
            "brain_exp_csv_semantics": "mBrain_pooled_cells",
        },
    }
    meta_path = (
        Path(args.meta_json)
        if args.meta_json is not None
        else Path(args.out_non_mbrain_csv_gz).parent / "gse131907_mbrain_split_meta.json"
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gse131907-mbrain-split] meta -> {meta_path}")


if __name__ == "__main__":
    main()
