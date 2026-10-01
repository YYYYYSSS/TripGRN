#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Identify type-unique GRN edges from saved per-type test probabilities.

Default rule for type A unique:
  - regulated in A: max(P_Repress, P_Activate) >= min_reg_prob
  - low regulation in every other type: max(P_Repress, P_Activate) < max_other_reg_prob
    OR P_NoReg >= min_other_noreg_prob

All thresholds are CLI-configurable for downstream re-analysis without re-inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from grn.data.gse131907_celltype_paths import CELL_TYPES, gse131907_celltype_layout, grn_prob_csv_path


PROB_COLS = ("P_NoReg", "P_Repress", "P_Activate")


def _load_probs(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    need = {"TF", "Target", *PROB_COLS}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df.copy()
    df["TF"] = df["TF"].astype(str)
    df["Target"] = df["Target"].astype(str)
    for c in PROB_COLS:
        df[c] = df[c].astype(float)
    df["max_reg"] = df[["P_Repress", "P_Activate"]].max(axis=1)
    return df


def _merge_type_probs(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for tid, df in tables.items():
        sub = df[["TF", "Target", *PROB_COLS, "max_reg"]].rename(
            columns={c: f"{tid}__{c}" for c in (*PROB_COLS, "max_reg")}
        )
        merged = sub if merged is None else merged.merge(sub, on=["TF", "Target"], how="inner")
    if merged is None or merged.empty:
        raise ValueError("No overlapping TF-Target pairs across types")
    return merged


def _is_regulated(row: pd.Series, prefix: str, *, min_reg_prob: float) -> bool:
    return float(row[f"{prefix}__max_reg"]) >= float(min_reg_prob)


def _is_low_regulation(
    row: pd.Series,
    prefix: str,
    *,
    max_reg_prob: float,
    min_noreg_prob: float,
) -> bool:
    max_reg = float(row[f"{prefix}__max_reg"])
    noreg = float(row[f"{prefix}__P_NoReg"])
    return max_reg < float(max_reg_prob) or noreg >= float(min_noreg_prob)


def _unique_for_type(
    merged: pd.DataFrame,
    focus: str,
    others: List[str],
    *,
    min_reg_prob: float,
    max_other_reg_prob: float,
    min_other_noreg_prob: float,
) -> pd.DataFrame:
    mask = merged.apply(_is_regulated, axis=1, prefix=focus, min_reg_prob=min_reg_prob)
    for o in others:
        mask &= merged.apply(
            _is_low_regulation,
            axis=1,
            prefix=o,
            max_reg_prob=max_other_reg_prob,
            min_noreg_prob=min_other_noreg_prob,
        )
    out = merged.loc[mask].copy()
    out["unique_type"] = focus
    out["rule_min_reg_prob"] = float(min_reg_prob)
    out["rule_max_other_reg_prob"] = float(max_other_reg_prob)
    out["rule_min_other_noreg_prob"] = float(min_other_noreg_prob)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="GSE131907 step06 type-unique edges from saved probs")
    ap.add_argument("--min-reg-prob", type=float, default=0.45, help="Focus type: min max(Repress, Activate)")
    ap.add_argument("--max-other-reg-prob", type=float, default=0.35, help="Other types: max reg prob below this")
    ap.add_argument("--min-other-noreg-prob", type=float, default=0.45, help="Other types: NoReg at least this")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    layout = gse131907_celltype_layout()
    out_dir = layout["unique_edges_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_path = out_dir / "merged_all_type_probs.csv"
    if merged_path.is_file() and not args.force:
        merged = pd.read_csv(merged_path)
    else:
        tables = {tid: _load_probs(grn_prob_csv_path(tid, layout["grn_root"])) for tid, _ in CELL_TYPES}
        merged = _merge_type_probs(tables)
        merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
        print(f"[celltype-diff] merged {len(merged)} pairs -> {merged_path}")

    type_ids = [t[0] for t in CELL_TYPES]
    summary: Dict[str, int] = {}
    for focus in type_ids:
        others = [t for t in type_ids if t != focus]
        unique = _unique_for_type(
            merged,
            focus,
            others,
            min_reg_prob=float(args.min_reg_prob),
            max_other_reg_prob=float(args.max_other_reg_prob),
            min_other_noreg_prob=float(args.min_other_noreg_prob),
        )
        out_csv = out_dir / f"unique_edges_{focus}.csv"
        unique.to_csv(out_csv, index=False, encoding="utf-8-sig")
        summary[focus] = int(len(unique))
        print(f"[celltype-diff] {focus}: {len(unique)} unique edges -> {out_csv}")

    meta = {
        "min_reg_prob": float(args.min_reg_prob),
        "max_other_reg_prob": float(args.max_other_reg_prob),
        "min_other_noreg_prob": float(args.min_other_noreg_prob),
        "n_merged_pairs": int(len(merged)),
        "n_unique": summary,
        "merged_csv": merged_path.name,
    }
    (out_dir / "unique_edges_summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
