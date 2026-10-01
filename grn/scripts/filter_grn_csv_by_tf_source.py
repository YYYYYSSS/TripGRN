"""
从 KEGG 风格 GRN CSV 生成「仅保留 source 为转录因子」的边表。

用法示例（在项目根目录）::

    python -m grn.scripts.filter_grn_csv_by_tf_source ^
      --input data/network/grn/KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv ^
      --tf-symbols-file data/network/grn/human_tf_gene_symbols.txt ^
      --output data/network/grn/KEGG_hsa_PPrel_TF_source_only_with_symbols.csv

可选：额外剔除一批 source（如自备激酶列表）::

    python -m grn.scripts.filter_grn_csv_by_tf_source ^
      --input ... --tf-symbols-file ... ^
      --exclude-source-symbols-file data/network/grn/extra_exclude_regulator_sources.txt ^
      --output ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from grn.data.preprocessing import normalize_gene_symbol_series, resolve_grn_table_columns
from grn.data.tf_source_filter import filter_grn_by_source_symbols, load_regulator_symbol_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter GRN CSV rows to TF-only sources (or exclude lists).")
    parser.add_argument("--input", required=True, help="Input GRN CSV path.")
    parser.add_argument("--output", required=True, help="Output filtered CSV path.")
    parser.add_argument(
        "--tf-symbols-file",
        default=None,
        help="仅保留 source 在此列表中的行（每行一个 HGNC symbol）。与 --exclude-source-symbols-file 可同用。",
    )
    parser.add_argument(
        "--exclude-source-symbols-file",
        default=None,
        help="剔除 source 在此列表中的行（在 TF 白名单筛选之前执行）。",
    )
    parser.add_argument("--grn-tf-col", default="TF")
    parser.add_argument("--grn-target-col", default="target")
    parser.add_argument("--grn-type-col", default="Re_type")
    args = parser.parse_args()

    grn_df = pd.read_csv(args.input)
    tf_col, tg_col, tp_col = resolve_grn_table_columns(
        grn_df, args.grn_tf_col, args.grn_target_col, args.grn_type_col
    )
    if tf_col is None or tg_col is None or tp_col is None:
        raise ValueError(f"Could not resolve GRN columns. Columns: {list(grn_df.columns)}")

    grn_df[tf_col] = normalize_gene_symbol_series(grn_df[tf_col])
    grn_df[tg_col] = normalize_gene_symbol_series(grn_df[tg_col])

    include_tf = load_regulator_symbol_set(args.tf_symbols_file) if args.tf_symbols_file else None
    exclude = (
        load_regulator_symbol_set(args.exclude_source_symbols_file)
        if args.exclude_source_symbols_file
        else None
    )

    if include_tf is None and exclude is None:
        raise ValueError("Provide at least one of --tf-symbols-file or --exclude-source-symbols-file.")

    filtered, stats = filter_grn_by_source_symbols(
        grn_df, tf_col, include_tf=include_tf, exclude_sources=exclude
    )
    filtered.to_csv(args.output, index=False)
    outp = Path(args.output)
    meta_path = str(outp.with_name(outp.stem + "_filter_stats.json"))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[filter-grn] rows {stats['rows_input']} -> {stats['rows_output']} (written {args.output})")
    print(f"[filter-grn] stats json -> {meta_path}")


if __name__ == "__main__":
    main()
