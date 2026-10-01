#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one DeepTGI step01 fold (binary, unseen cells, KEGG-priority gene subset)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _resolve_deeptgi_root(cli: Path | None) -> Path:
    if cli is not None:
        return cli.resolve()
    env = os.environ.get("DEEPTGI_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    grn_root = Path(__file__).resolve().parents[2]
    return (grn_root.parent / "project_DeepTGI").resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one DeepTGI step01 fold")
    ap.add_argument("--gse-id", required=True)
    ap.add_argument("--fold-idx", type=int, required=True)
    ap.add_argument("--deeptgi-root", type=Path, default=None)
    ap.add_argument("--run-seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--cuda-device", type=str, default="0")
    ap.add_argument(
        "--max-genes",
        type=int,
        default=1500,
        help="KEGG-priority gene cap (PCC features are 2×N genes; full genome OOM on 16GB).",
    )
    args = ap.parse_args()

    grn_root = Path(__file__).resolve().parents[2]
    from grn.data.pancancer_paths import deeptgi_fold_dir, step01_grn_csv
    from grn.scripts.pancancer_step01_binary import (
        build_grn_matrix,
        ensure_deeptgi_cells_x_genes_csv,
        load_cv_column_splits,
        load_epithelial_expression,
        load_kegg_grn_dataframe,
        pairs_to_gene_label_df,
        select_kegg_priority_gene_subset,
        split_binary_pairs,
        subset_expression,
    )

    gse = str(args.gse_id).strip()
    fold_idx = int(args.fold_idx)
    max_genes = int(args.max_genes)
    kegg = step01_grn_csv(grn_root)
    print(f"[deeptgi-fold] grn_csv={kegg}", flush=True)
    fold_dir = deeptgi_fold_dir(gse, fold_idx, grn_root)
    fold_dir.mkdir(parents=True, exist_ok=True)

    gene_list, raw_expr = load_epithelial_expression(gse, grn_root)
    n_genes_full = len(gene_list)
    if max_genes > 0 and max_genes < n_genes_full:
        grn_df, tf_col, tg_col = load_kegg_grn_dataframe(gene_list, kegg)
        sub_genes = select_kegg_priority_gene_subset(
            gene_list, raw_expr, grn_df, tf_col, tg_col, max_genes, int(args.run_seed)
        )
        gene_list, raw_expr = subset_expression(gene_list, raw_expr, sub_genes)
        print(
            f"[deeptgi-fold] {gse} fold {fold_idx:02d}: genes {n_genes_full} -> {len(gene_list)} "
            f"(max_genes={max_genes})",
            flush=True,
        )
    else:
        print(f"[deeptgi-fold] {gse} fold {fold_idx:02d}: using all {n_genes_full} genes", flush=True)

    grn_matrix = build_grn_matrix(gene_list, kegg)
    pairs_train, pairs_val, pairs_test = split_binary_pairs(grn_matrix, seed=42, neg_ratio=1.0)
    train_cols, val_cols, test_cols = load_cv_column_splits(gse, fold_idx, grn_root)

    csv_tag = f"m{max_genes}" if max_genes > 0 else "full"
    expr_csv = ensure_deeptgi_cells_x_genes_csv(gse, gene_list, raw_expr, grn_root, tag=csv_tag)
    pairs_to_gene_label_df(pairs_train, gene_list).to_csv(fold_dir / "label_train.csv", index=False)
    pairs_to_gene_label_df(pairs_val, gene_list).to_csv(fold_dir / "label_val.csv", index=False)
    pairs_to_gene_label_df(pairs_test, gene_list).to_csv(fold_dir / "label_test.csv", index=False)
    (fold_dir / "train_cells.json").write_text(json.dumps(train_cols.tolist()), encoding="utf-8")
    (fold_dir / "val_cells.json").write_text(json.dumps(val_cols.tolist()), encoding="utf-8")
    (fold_dir / "test_cells.json").write_text(json.dumps(test_cols.tolist()), encoding="utf-8")

    deeptgi_root = _resolve_deeptgi_root(args.deeptgi_root)
    cli = deeptgi_root / "program" / "run_unseen_cells_cli.py"
    if not cli.is_file():
        raise SystemExit(f"missing DeepTGI CLI: {cli}")

    metrics = fold_dir / "unseen_samples_test_metrics.json"
    cmd = [
        sys.executable,
        "-u",
        str(cli),
        "--expr-csv",
        str(expr_csv),
        "--label-train-csv",
        str(fold_dir / "label_train.csv"),
        "--label-val-csv",
        str(fold_dir / "label_val.csv"),
        "--label-test-csv",
        str(fold_dir / "label_test.csv"),
        "--train-cells-json",
        str(fold_dir / "train_cells.json"),
        "--val-cells-json",
        str(fold_dir / "val_cells.json"),
        "--test-cells-json",
        str(fold_dir / "test_cells.json"),
        "--output-fold-dir",
        str(fold_dir),
        "--output-metrics-json",
        str(metrics),
        "--run-seed",
        str(args.run_seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--cuda-visible-devices",
        str(args.cuda_device),
    ]
    proc = subprocess.run(cmd, check=False, cwd=str(cli.parent))
    if not metrics.is_file():
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    if proc.returncode != 0:
        print(
            f"[deeptgi-fold] warn: exit={proc.returncode} but metrics exist; continuing",
            flush=True,
        )

    payload = json.loads(metrics.read_text(encoding="utf-8"))
    payload.update(
        {
            "gse_id": gse,
            "fold_idx": fold_idx,
            "n_genes_full": n_genes_full,
            "n_genes_used": len(gene_list),
            "max_genes": max_genes,
        }
    )
    metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[deeptgi-fold] done -> {fold_dir}")


if __name__ == "__main__":
    main()
