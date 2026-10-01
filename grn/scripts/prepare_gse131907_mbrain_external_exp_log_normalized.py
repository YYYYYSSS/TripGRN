from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from contextlib import contextmanager
from typing import Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import openpyxl

from grn.data.preprocessing import normalize_gene_symbol_series


@contextmanager
def _openpyxl_workbook_skip_bad_custom_props(path: str) -> Iterator:
    """
    部分 Excel 文件的 docProps/custom.xml 会触发 openpyxl 解析错误；
    复制为临时 xlsx 并跳过该文件后再打开。
    """
    fd, tmp_path = tempfile.mkstemp(suffix="_metastasis_xlsx.xlsx")
    os.close(fd)
    try:
        with zipfile.ZipFile(path, "r") as zin:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                for info in zin.infolist():
                    if info.filename == "docProps/custom.xml":
                        continue
                    zout.writestr(info, zin.read(info.filename))
        wb = openpyxl.load_workbook(tmp_path, read_only=True, data_only=True)
        try:
            yield wb
        finally:
            wb.close()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _load_brain_sample_ids_from_feature_summary_xlsx(path: str) -> Set[str]:
    """
    从 `feature_summary_xlsx` 取：
      - Tissue origins == "mBrain" 对应的 Samples
    """
    with _openpyxl_workbook_skip_bad_custom_props(path) as wb:
        ws = wb.active
        # 训练脚本约定：第 3 行是表头
        rows_iter = ws.iter_rows(min_row=3, max_row=3, values_only=True)
        header_row = next(rows_iter)
        if not header_row:
            raise ValueError("empty header row (row 3)")
        headers = [str(h).strip() if h is not None else "" for h in header_row]
        try:
            i_samples = headers.index("Samples")
            i_tissue = headers.index("Tissue origins")
        except ValueError as e:
            raise ValueError(f"need columns 'Samples' and 'Tissue origins', got {headers}") from e

        brain_ids: Set[str] = set()
        for row in ws.iter_rows(min_row=4, values_only=True):
            if not row or row[0] is None:
                break
            first = str(row[0]).strip()
            if first.startswith("*"):
                break
            sample = row[i_samples]
            tissue = row[i_tissue]
            if sample is None or tissue is None:
                continue
            sample_s = str(sample).strip()
            tissue_s = str(tissue).strip()
            if tissue_s == "mBrain" and sample_s:
                brain_ids.add(sample_s)
        return brain_ids


def _match_column_to_sample_id(col_name: str, sample_ids: Sequence[str]) -> Optional[str]:
    """
    与 train_metastasis_brain_train_nonbrain_eval.py 一致：
    列名需要能匹配到 `_<Samples>` 后缀。
    """
    s = str(col_name).strip()
    for sid in sample_ids:
        suffix = "_" + sid
        if s.endswith(suffix):
            return sid
    return None


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Prepare GSE131907 mBrain external expression for step1 evaluation: "
        "extract mBrain columns, filter cells with >=90% zero genes, "
        "then convert counts -> normalize_total(target_sum=1e4) + log1p (log_normalized)."
    )
    ap.add_argument("--master-exp-csv", required=True, help="GSE131907_exp_counts.csv (gene x cells, first col gene)")
    ap.add_argument("--feature-summary-xlsx", required=True, help="GSE131907_Lung_Cancer_Feature_Summary.xlsx")
    ap.add_argument("--output-csv", required=True, help="Output external expression CSV (log_normalized, gene x filtered cells)")
    ap.add_argument("--filter-cells-max-zero-frac", type=float, default=0.9, help="drop cell if zero_frac >= this threshold")
    ap.add_argument("--normalize-target-sum", type=float, default=1e4, help="normalize_total target_sum (Seurat/Scanpy style)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    master_path = os.path.abspath(args.master_exp_csv)
    feature_xlsx = os.path.abspath(args.feature_summary_xlsx)
    out_path = os.path.abspath(args.output_csv)
    out_path_dir = os.path.dirname(out_path)
    if out_path_dir:
        os.makedirs(out_path_dir, exist_ok=True)

    brain_ids = _load_brain_sample_ids_from_feature_summary_xlsx(feature_xlsx)
    if not brain_ids:
        raise SystemExit("No mBrain samples found in feature summary xlsx.")

    # 只读表头，避免一次性读入全矩阵
    head = pd.read_csv(master_path, nrows=0)
    gene_col = head.columns[0]
    data_cols = [c for c in head.columns[1:]]

    # 与训练脚本一致：优先匹配更长的 Samples id
    ordered_brain_ids = sorted(brain_ids, key=len, reverse=True)
    brain_cols: List[str] = []
    unmatched: List[str] = []
    for c in data_cols:
        sid = _match_column_to_sample_id(str(c), ordered_brain_ids)
        if sid is None:
            unmatched.append(str(c))
        else:
            brain_cols.append(c)

    if not brain_cols:
        raise SystemExit("No expression columns matched mBrain sample ids (suffix _<Samples>).")

    # 仅加载 mBrain 列 + gene 列
    df = pd.read_csv(master_path, usecols=[gene_col] + brain_cols)
    df[gene_col] = normalize_gene_symbol_series(df[gene_col])
    df = df.drop_duplicates(subset=[gene_col], keep="first").set_index(gene_col)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    genes = df.index.tolist()
    mat = df.values.astype(np.float32)  # [G, C]

    if args.filter_cells_max_zero_frac is not None:
        thr = float(args.filter_cells_max_zero_frac)
        if thr > 0.0 and thr <= 1.0:
            zero_frac = (mat == 0).sum(axis=0) / float(mat.shape[0])
            keep = zero_frac < thr
            n_drop = int((~keep).sum())
            n_keep = int(keep.sum())
            if n_drop > 0:
                print(
                    f"[prep-gse131907-mbrain-step1] filter (>= {thr:.2f} zero-gene frac): "
                    f"removed {n_drop} / {mat.shape[1]} cells, kept {n_keep}"
                )
            mat = mat[:, keep]
            filtered_cols = [brain_cols[i] for i in range(len(brain_cols)) if keep[i]]
        else:
            filtered_cols = list(brain_cols)
    else:
        filtered_cols = list(brain_cols)

    # normalize_total + log1p（log_normalized）
    col_sum = mat.sum(axis=0)  # [C]
    scale = np.zeros_like(col_sum, dtype=np.float32)
    pos = col_sum > 0
    scale[pos] = float(args.normalize_target_sum) / col_sum[pos]
    mat_scaled = mat * scale[None, :]
    mat_log = np.log1p(mat_scaled).astype(np.float32)

    out_df = pd.DataFrame(mat_log, index=genes, columns=filtered_cols)
    out_df = out_df.reset_index()  # gene column first
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[prep-gse131907-mbrain-step1] wrote: {out_path} (genes={len(genes)}, cells={len(filtered_cols)})")

    meta = {
        "master_exp_csv": master_path,
        "feature_summary_xlsx": feature_xlsx,
        "mBrain_sample_ids": sorted(brain_ids),
        "n_brain_cols_matched": len(brain_cols),
        "n_unmatched_expression_columns": len(unmatched),
        "filter_cells_max_zero_frac": args.filter_cells_max_zero_frac,
        "normalize_target_sum": args.normalize_target_sum,
        "n_output_genes": len(genes),
        "n_output_cells": len(filtered_cols),
    }
    meta_path = out_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[prep-gse131907-mbrain-step1] meta -> {meta_path}")


if __name__ == "__main__":
    main()

