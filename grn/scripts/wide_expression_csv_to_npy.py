"""
将「基因 × 细胞」宽大 CSV（首列表头为 gene 或常见别名/空，其余列为细胞）流式转为 float32 .npy + 基因名 .npy，
便于训练时用 numpy memmap 映射，避免 pandas 整表读入内存。

与 train_unseen_samples_no_cv 中 ``drop_duplicates(subset=gene, keep='first')`` 一致：同名基因仅保留首次出现行。

可选 ``--filter-cells-max-zero-frac``：去掉「零表达基因占比 >= 该阈值」的细胞列（与 GSE253718 / metastasis 管线 QC 语义一致），
对流式两遍扫描 CSV（不累乘整矩阵到内存），直接写出筛选后的窄矩阵 npy，避免先写巨大筛选后 CSV。

用法：
  python -m grn.scripts.wide_expression_csv_to_npy \\
    --csv data/exp/GSE253718_exp.csv \\
    --out-prefix data/exp/GSE253718_exp
写出：
  {out-prefix}_values.npy  # shape (n_genes, n_cells), float32
  {out-prefix}_genes.npy   # shape (n_genes,), object 字符串
  {out-prefix}_filter_manifest.json  # 仅在使用细胞筛选时写出
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np


def _norm_gene_symbol(s: str) -> str:
    x = str(s).strip().replace('"', "").replace("'", "")
    return x


def _gene_column_header_ok(cell: str) -> bool:
    """首列表头：gene / 空（矩阵导出常见左上角留空）/ 常见别名。"""
    x = str(cell).strip().replace("\ufeff", "").lower()
    if x == "":
        return True
    return x in (
        "gene",
        "genes",
        "gene_symbol",
        "gene symbol",
        "feature",
        "features",
        "symbol",
    )


def _unique_gene_row_count(csv_path: Path) -> tuple[int, int]:
    """返回 (去重后基因数, 细胞列数)。"""
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header or not _gene_column_header_ok(header[0]):
            raise ValueError(
                f"首列须为 gene（或空表头），当前表头: {header[:3]}..."
            )
        n_cells = len(header) - 1
        if n_cells < 1:
            raise ValueError("除 gene 外至少需要一列细胞")
        seen: set[str] = set()
        n_uniq = 0
        for row in reader:
            if len(row) != n_cells + 1:
                raise ValueError(f"列数异常: 期望 {n_cells + 1}, 得到 {len(row)}")
            g = _norm_gene_symbol(row[0])
            if g in seen:
                continue
            seen.add(g)
            n_uniq += 1
    return n_uniq, n_cells


def convert_csv_to_npy(
    csv_path: Path,
    out_prefix: Path,
    *,
    filter_cells_max_zero_frac: Optional[float] = None,
) -> None:
    csv_path = Path(csv_path).resolve()
    out_prefix = Path(out_prefix).resolve()
    values_path = Path(str(out_prefix) + "_values.npy")
    genes_path = Path(str(out_prefix) + "_genes.npy")
    manifest_path = Path(str(out_prefix) + "_filter_manifest.json")

    n_genes, n_cells = _unique_gene_row_count(csv_path)
    print(f"[wide_csv_to_npy] unique_genes={n_genes}, cells={n_cells}")

    fz: Optional[float] = None
    if filter_cells_max_zero_frac is not None:
        fz = float(filter_cells_max_zero_frac)
        if not (fz > 0.0 and fz <= 1.0):
            raise ValueError("filter_cells_max_zero_frac must be in (0, 1]")

    if fz is None:
        mm = np.lib.format.open_memmap(values_path, mode="w+", dtype=np.float32, shape=(n_genes, n_cells))
        gene_names: list[str] = []

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            seen: set[str] = set()
            gi = 0
            for row in reader:
                if len(row) != n_cells + 1:
                    raise ValueError(f"列数异常: 期望 {n_cells + 1}, 得到 {len(row)}")
                g = _norm_gene_symbol(row[0])
                if g in seen:
                    continue
                seen.add(g)
                gene_names.append(g)
                for j, v in enumerate(row[1:]):
                    mm[gi, j] = float(v)
                gi += 1
                if gi % 2000 == 0:
                    print(f"  ... {gi}/{n_genes} genes")
            if gi != n_genes:
                raise ValueError(f"去重基因数不一致: 预期 {n_genes}, 写入 {gi}")

        mm.flush()
        del mm
        np.save(genes_path, np.array(gene_names, dtype=object))
        print(f"[OK] {values_path}")
        print(f"[OK] {genes_path}")
        return

    # --- 细胞 QC：两遍流式扫描，直接写出筛选后的矩阵（不写中间 CSV）---
    zero_counts = np.zeros(n_cells, dtype=np.int64)
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        seen: set[str] = set()
        gi = 0
        for row in reader:
            if len(row) != n_cells + 1:
                raise ValueError(f"列数异常: 期望 {n_cells + 1}, 得到 {len(row)}")
            g = _norm_gene_symbol(row[0])
            if g in seen:
                continue
            seen.add(g)
            row_f = np.asarray([float(x) for x in row[1:]], dtype=np.float32)
            if row_f.shape[0] != n_cells:
                raise ValueError(f"细胞列数不一致: 期望 {n_cells}, 得到 {row_f.shape[0]}")
            zero_counts += (row_f == 0.0).astype(np.int64)
            gi += 1
            if gi % 2000 == 0:
                print(f"  [QC pass1] {gi}/{n_genes} genes (accumulate zeros)")
        if gi != n_genes:
            raise ValueError(f"去重基因数不一致: 预期 {n_genes}, 扫描 {gi}")

    zero_frac = zero_counts.astype(np.float64) / float(n_genes)
    keep_mask = zero_frac < fz
    n_keep = int(np.sum(keep_mask))
    n_drop = n_cells - n_keep
    print(
        f"[wide_csv_to_npy] filter_cells_max_zero_frac={fz}: "
        f"keep {n_keep}/{n_cells} cells (removed {n_drop} with zero_frac>={fz})"
    )
    if n_keep < 1:
        raise ValueError("细胞筛选后无剩余列，请检查阈值或 CSV")

    keep_idx = np.flatnonzero(keep_mask).astype(np.int64)

    mm = np.lib.format.open_memmap(values_path, mode="w+", dtype=np.float32, shape=(n_genes, n_keep))
    gene_names = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        seen = set()
        gi = 0
        for row in reader:
            if len(row) != n_cells + 1:
                raise ValueError(f"列数异常: 期望 {n_cells + 1}, 得到 {len(row)}")
            g = _norm_gene_symbol(row[0])
            if g in seen:
                continue
            seen.add(g)
            gene_names.append(g)
            vals = row[1:]
            row_f = np.asarray([float(x) for x in vals], dtype=np.float32)
            if row_f.shape[0] != n_cells:
                raise ValueError(f"细胞列数不一致: 期望 {n_cells}, 得到 {row_f.shape[0]}")
            mm[gi, :] = row_f[keep_idx]
            gi += 1
            if gi % 2000 == 0:
                print(f"  [QC pass2] {gi}/{n_genes} genes (write filtered)")
        if gi != n_genes:
            raise ValueError(f"去重基因数不一致: 预期 {n_genes}, 写入 {gi}")

    mm.flush()
    del mm
    np.save(genes_path, np.array(gene_names, dtype=object))

    manifest = {
        "csv": str(csv_path),
        "filter_cells_max_zero_frac": fz,
        "zero_definition": "value == 0.0 after float() per gene row (unique genes only)",
        "n_genes_unique": int(n_genes),
        "n_cells_before": int(n_cells),
        "n_cells_after": int(n_keep),
        "n_cells_removed": int(n_drop),
        "values_npy": str(values_path),
        "genes_npy": str(genes_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {values_path}")
    print(f"[OK] {genes_path}")
    print(f"[OK] {manifest_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out-prefix", type=Path, required=True)
    p.add_argument(
        "--filter-cells-max-zero-frac",
        type=float,
        default=None,
        metavar="THR",
        help="若指定（如 0.9）：剔除「零表达基因数/基因总数 >= THR」的细胞列；两遍流式读 CSV，直接写筛选后 npy。",
    )
    args = p.parse_args()
    convert_csv_to_npy(args.csv, args.out_prefix, filter_cells_max_zero_frac=args.filter_cells_max_zero_frac)


if __name__ == "__main__":
    main()
