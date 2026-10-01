#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 GSE131907 step06 CV 划分（列索引）映射为细胞 barcode，导出 R 易读 CSV。

输出目录默认:
  data/exp/GSE131907/celltype_step06_bytype/cv_splits_barcodes/

每个细胞类型:
  {type_id}_10fold_long.csv   — 长表: barcode, fold, split, col_idx, ...
  {type_id}_10fold_wide.csv   — 宽表: barcode + fold_01..fold_10
  {type_id}_fixed_seed42_long.csv / _wide.csv

另写:
  all_types_10fold_long.csv
  all_types_fixed_seed42_long.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from grn.data.gse131907_celltype_paths import (
    CELL_TYPES,
    celltype_columns_npy,
    cv_splits_path,
    gse131907_celltype_layout,
    resolve_type_ids,
)
from grn.data.pancancer_cv import get_fold_split, load_cv_splits


def _load_barcodes(type_id: str, grn_root: Path) -> np.ndarray:
    path = celltype_columns_npy(type_id, grn_root=grn_root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing column_names.npy: {path}")
    cols = np.load(path, allow_pickle=True)
    return np.asarray([str(x) for x in cols], dtype=object)


def _fold_to_long(
    barcodes: np.ndarray,
    fold: Dict[str, Any],
    *,
    type_id: str,
    type_label: str,
    split_kind: str,
) -> pd.DataFrame:
    fold_idx = int(fold["fold"])
    parts: List[pd.DataFrame] = []
    for split_name, key in (("train", "train_cols"), ("val", "val_cols"), ("test", "test_cols")):
        idxs = np.asarray(fold[key], dtype=np.int64)
        if idxs.size and (idxs.min() < 0 or idxs.max() >= len(barcodes)):
            raise IndexError(
                f"{type_id} fold {fold_idx} {split_name}: index out of range "
                f"(n_barcodes={len(barcodes)}, max_idx={int(idxs.max())})"
            )
        parts.append(
            pd.DataFrame(
                {
                    "type_id": type_id,
                    "type_label": type_label,
                    "split_kind": split_kind,
                    "fold": fold_idx,
                    "split": split_name,
                    "col_idx": idxs,
                    "barcode": barcodes[idxs],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """One row per barcode; columns fold_01..fold_K with train/val/test."""
    if df_long.empty:
        return pd.DataFrame(columns=["type_id", "type_label", "split_kind", "barcode", "col_idx"])
    meta = (
        df_long[["type_id", "type_label", "split_kind", "barcode", "col_idx"]]
        .drop_duplicates(subset=["barcode"])
        .sort_values("col_idx")
        .reset_index(drop=True)
    )
    piv = (
        df_long.assign(fold_col=df_long["fold"].map(lambda k: f"fold_{int(k):02d}"))
        .pivot_table(index="barcode", columns="fold_col", values="split", aggfunc="first")
        .reset_index()
    )
    fold_cols = sorted([c for c in piv.columns if c.startswith("fold_")])
    out = meta.merge(piv, on="barcode", how="left")
    return out[["type_id", "type_label", "split_kind", "barcode", "col_idx", *fold_cols]]


def export_one_type(
    type_id: str,
    type_label: str,
    *,
    grn_root: Path,
    out_dir: Path,
    which: str,
    write_json: bool = False,
) -> Dict[str, Path]:
    barcodes = _load_barcodes(type_id, grn_root)
    written: Dict[str, Path] = {}

    jobs: List[tuple[str, Path, bool]] = []
    if which in ("10fold", "both"):
        jobs.append(("10fold", cv_splits_path(type_id, fixed=False, grn_root=grn_root), False))
    if which in ("fixed", "both"):
        jobs.append(("fixed_seed42", cv_splits_path(type_id, fixed=True, grn_root=grn_root), True))

    for split_kind, split_path, _fixed in jobs:
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing CV split JSON: {split_path}")
        payload = load_cv_splits(split_path)
        if int(payload["n_cells"]) != len(barcodes):
            raise ValueError(
                f"{type_id}: n_cells mismatch split={payload['n_cells']} vs barcodes={len(barcodes)}"
            )

        long_parts = [
            _fold_to_long(
                barcodes,
                get_fold_split(payload, int(f["fold"])),
                type_id=type_id,
                type_label=type_label,
                split_kind=split_kind,
            )
            for f in payload["folds"]
        ]
        df_long = pd.concat(long_parts, ignore_index=True)
        df_wide = _long_to_wide(df_long)

        p_long = out_dir / f"{type_id}_{split_kind}_long.csv"
        p_wide = out_dir / f"{type_id}_{split_kind}_wide.csv"
        df_long.to_csv(p_long, index=False, encoding="utf-8")
        df_wide.to_csv(p_wide, index=False, encoding="utf-8")
        written[f"{split_kind}_long"] = p_long
        written[f"{split_kind}_wide"] = p_wide

        if write_json:
            barcode_payload = {
                "type_id": type_id,
                "type_label": type_label,
                "split_kind": split_kind,
                "n_cells": int(len(barcodes)),
                "n_folds": int(payload["n_folds"]),
                "sample_split": payload.get("sample_split"),
                "source_cv_splits_json": str(split_path),
                "folds": [],
            }
            for f in payload["folds"]:
                fold = get_fold_split(payload, int(f["fold"]))
                barcode_payload["folds"].append(
                    {
                        "fold": int(fold["fold"]),
                        "n_train": int(fold["n_train"]),
                        "n_val": int(fold["n_val"]),
                        "n_test": int(fold["n_test"]),
                        "train_barcodes": barcodes[
                            np.asarray(fold["train_cols"], dtype=np.int64)
                        ].tolist(),
                        "val_barcodes": barcodes[
                            np.asarray(fold["val_cols"], dtype=np.int64)
                        ].tolist(),
                        "test_barcodes": barcodes[
                            np.asarray(fold["test_cols"], dtype=np.int64)
                        ].tolist(),
                    }
                )
            p_json = out_dir / f"{type_id}_{split_kind}_barcodes.json"
            p_json.write_text(
                json.dumps(barcode_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            written[f"{split_kind}_json"] = p_json

    return written


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export GSE131907 celltype CV splits as barcodes (R-friendly CSV)."
    )
    ap.add_argument("--type", default="ALL", help="Cell type id / label / ALL")
    ap.add_argument(
        "--which",
        choices=["10fold", "fixed", "both"],
        default="both",
        help="Export 10-fold, fixed seed42, or both",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: .../celltype_step06_bytype/cv_splits_barcodes)",
    )
    ap.add_argument(
        "--write-json",
        action="store_true",
        help="Also write per-type barcode JSON (large; CSV is enough for R)",
    )
    args = ap.parse_args()

    layout = gse131907_celltype_layout()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        layout["celltype_data_dir"] / "cv_splits_barcodes"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    types = resolve_type_ids(args.type)
    all_long_10: List[pd.DataFrame] = []
    all_long_fixed: List[pd.DataFrame] = []

    for tid, label in types:
        written = export_one_type(
            tid,
            label,
            grn_root=layout["grn_root"],
            out_dir=out_dir,
            which=args.which,
            write_json=bool(args.write_json),
        )
        print(f"[barcode-splits] {tid}: " + ", ".join(f"{k}={v.name}" for k, v in written.items()))
        if "10fold_long" in written:
            all_long_10.append(pd.read_csv(written["10fold_long"]))
        if "fixed_seed42_long" in written:
            all_long_fixed.append(pd.read_csv(written["fixed_seed42_long"]))

    if all_long_10:
        p = out_dir / "all_types_10fold_long.csv"
        pd.concat(all_long_10, ignore_index=True).to_csv(p, index=False, encoding="utf-8")
        print(f"[barcode-splits] combined -> {p}")
    if all_long_fixed:
        p = out_dir / "all_types_fixed_seed42_long.csv"
        pd.concat(all_long_fixed, ignore_index=True).to_csv(p, index=False, encoding="utf-8")
        print(f"[barcode-splits] combined -> {p}")

    readme = {
        "description": "GSE131907 step06 train/val/test splits mapped to cell barcodes",
        "out_dir": str(out_dir),
        "column_names_source": "data/exp/GSE131907/celltype_step06_bytype/<type_id>/column_names.npy",
        "cv_splits_source": "data/exp/GSE131907/celltype_step06_bytype/cv_splits/",
        "r_usage": {
            "long": "read.csv('.../epithelial_cells_10fold_long.csv'); subset by fold & split",
            "wide": "read.csv('.../epithelial_cells_10fold_wide.csv'); columns fold_01..fold_10",
            "example": (
                "sp <- read.csv('epithelial_cells_10fold_long.csv'); "
                "test_bc <- sp$barcode[sp$fold == 1 & sp$split == 'test']"
            ),
        },
        "types": [{"type_id": tid, "type_label": lab} for tid, lab in CELL_TYPES],
    }
    (out_dir / "README.json").write_text(json.dumps(readme, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[barcode-splits] done -> {out_dir}")


if __name__ == "__main__":
    main()
