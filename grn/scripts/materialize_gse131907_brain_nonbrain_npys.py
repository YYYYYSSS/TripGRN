"""
一次性将 GSE131907「基因×细胞」超宽 CSV 转为 brain / non-brain 两份 float32 .npy（按行块读入，单遍扫描主文件）。

与 train_metastasis_brain_train_nonbrain_eval 中 CSV 切分语义一致：
  - mBrain / non-mBrain 列由 Feature Summary xlsx 决定；
  - 基因去重规则与 wide_expression_csv_to_npy / 训练脚本一致：同名基因仅保留首次出现行。

输出目录（默认可由 --out-dir 指定）：
  brain_values.npy           shape (n_genes, n_brain_cols), float32
  nonbrain_values.npy        shape (n_genes, n_nonbrain_cols), float32
  genes.npy                  shape (n_genes,), object
  brain_column_names.npy     object
  nonbrain_column_names.npy  object
  manifest.json

后续与 GSE253718 Step1 对齐的表达尺度：对 counts npy 做一次 log 归一化（不经 CSV），例如

  python -u -m grn.scripts.gse131907_split_counts_npys_to_log_normalized --split-dir <out-dir>

用法：
  python -u -m grn.scripts.materialize_gse131907_brain_nonbrain_npys ^
    --master-exp-csv data/exp/GSE131907/GSE131907_exp_counts.csv ^
    --feature-summary-xlsx data/other/GSE131907/GSE131907_Lung_Cancer_Feature_Summary.xlsx ^
    --out-dir data/exp/GSE131907/precomputed_brain_nonbrain_npy ^
    --gene-chunk-rows 512
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import pandas as pd

from grn.data.preprocessing import normalize_gene_symbol_series
from grn.scripts.train_metastasis_nonbrain_brain_eval import (
    _load_sample_tissue_from_feature_summary_xlsx,
    _split_expression_columns_by_brain,
)
from grn.scripts.wide_expression_csv_to_npy import _unique_gene_row_count


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize GSE131907 brain/non-brain expression npys (single CSV pass).")
    ap.add_argument("--master-exp-csv", type=Path, required=True)
    ap.add_argument("--feature-summary-xlsx", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--gene-chunk-rows",
        type=int,
        default=512,
        help="每次从主 CSV 读入的基因行数（越大越快、峰值内存越高；224GB 机器可试 1024～2048）。",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="若输出目录已有 manifest.json，默认跳过；加此项则覆盖重写 npy/manifest。",
    )
    args = ap.parse_args()

    master = Path(args.master_exp_csv).resolve()
    xlsx = Path(args.feature_summary_xlsx).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_man = out_dir / "manifest.json"
    if existing_man.is_file() and not args.force:
        print(f"[materialize] skip: {existing_man} exists (pass --force to overwrite)", flush=True)
        return

    if not master.is_file():
        raise FileNotFoundError(str(master))
    if not xlsx.is_file():
        raise FileNotFoundError(str(xlsx))

    head = pd.read_csv(master, nrows=0, low_memory=False)
    gene_col = str(head.columns[0])
    data_cols: List[str] = [str(c) for c in head.columns[1:]]

    sample_tissue, brain_ids = _load_sample_tissue_from_feature_summary_xlsx(str(xlsx))
    all_sample_ids: Set[str] = {s for s, _ in sample_tissue}
    nb_names, br_names, unmatched = _split_expression_columns_by_brain(data_cols, brain_ids, all_sample_ids)
    if unmatched:
        print(f"[materialize] warning: {len(unmatched)} matrix columns did not match any Sample id suffix")
    if not nb_names:
        raise ValueError("No non-brain columns after split.")
    if not br_names:
        raise ValueError("No brain columns after split.")

    name_to_i = {c: i for i, c in enumerate(data_cols)}
    brain_idx = np.array([name_to_i[c] for c in br_names], dtype=np.int64)
    nb_idx = np.array([name_to_i[c] for c in nb_names], dtype=np.int64)

    n_genes, n_cells = _unique_gene_row_count(master)
    if n_cells != len(data_cols):
        raise ValueError(f"Header cells={len(data_cols)} vs counted={n_cells}")

    print(
        f"[materialize] master={master}\n"
        f"  unique_genes={n_genes}, all_cell_cols={n_cells}\n"
        f"  nonbrain_cols={len(nb_names)}, mbrain_cols={len(br_names)}\n"
        f"  gene_chunk_rows={args.gene_chunk_rows}"
    )

    brain_path = out_dir / "brain_values.npy"
    nb_path = out_dir / "nonbrain_values.npy"
    genes_path = out_dir / "genes.npy"
    br_cols_path = out_dir / "brain_column_names.npy"
    nb_cols_path = out_dir / "nonbrain_column_names.npy"

    mm_b = np.lib.format.open_memmap(brain_path, mode="w+", dtype=np.float32, shape=(n_genes, len(br_names)))
    mm_nb = np.lib.format.open_memmap(nb_path, mode="w+", dtype=np.float32, shape=(n_genes, len(nb_names)))

    seen: Set[str] = set()
    genes_out: List[str] = []
    gi = 0
    chunk_i = 0

    for chunk in pd.read_csv(master, chunksize=int(args.gene_chunk_rows), low_memory=False):
        chunk_i += 1
        chunk[gene_col] = normalize_gene_symbol_series(chunk[gene_col])
        vals = chunk.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)
        if vals.shape[1] != n_cells:
            raise ValueError(f"chunk cols {vals.shape[1]} != expected {n_cells}")
        glist = chunk[gene_col].astype(str).tolist()
        for j in range(len(chunk)):
            g = str(glist[j]).strip()
            if g in seen:
                continue
            seen.add(g)
            genes_out.append(g)
            mm_b[gi, :] = vals[j, brain_idx]
            mm_nb[gi, :] = vals[j, nb_idx]
            gi += 1
        if chunk_i == 1 or chunk_i % 10 == 0:
            print(f"[materialize] chunks_read={chunk_i}, genes_written={gi}/{n_genes}", flush=True)

    if gi != n_genes:
        raise ValueError(f"gene row count mismatch: expected unique {n_genes}, wrote {gi}")
    if len(genes_out) != n_genes:
        raise ValueError("internal error: genes_out length")

    mm_b.flush()
    mm_nb.flush()
    del mm_b
    del mm_nb

    np.save(genes_path, np.array(genes_out, dtype=object))
    np.save(br_cols_path, np.array(br_names, dtype=object))
    np.save(nb_cols_path, np.array(nb_names, dtype=object))

    manifest = {
        "version": 1,
        "master_exp_csv": str(master),
        "feature_summary_xlsx": str(xlsx),
        "gene_column": gene_col,
        "n_genes": int(n_genes),
        "n_cells_master": int(n_cells),
        "n_brain_cols": len(br_names),
        "n_nonbrain_cols": len(nb_names),
        "gene_chunk_rows": int(args.gene_chunk_rows),
        "brain_values_npy": brain_path.name,
        "nonbrain_values_npy": nb_path.name,
        "genes_npy": genes_path.name,
        "brain_column_names_npy": br_cols_path.name,
        "nonbrain_column_names_npy": nb_cols_path.name,
        "unmatched_columns_count": len(unmatched),
    }
    man_path = out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[materialize] OK genes.npy -> {genes_path}")
    print(f"[materialize] OK brain_values.npy -> {brain_path}")
    print(f"[materialize] OK nonbrain_values.npy -> {nb_path}")
    print(f"[materialize] manifest -> {man_path}")


if __name__ == "__main__":
    main()
