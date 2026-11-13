#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove STRING PPI edges that overlap gold-standard GRN pairs (undirected).

Writes a decontaminated PPI CSV so topology cannot short-cut KEGG/DoRothEA labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from grn.data.pancancer_paths import ppi_string_csv, step01_grn_csv
from grn.data.preprocessing import normalize_gene_symbol_series, resolve_grn_table_columns


def _grn_undirected_pairs(grn_csv: Path) -> set[tuple[str, str]]:
    grn = pd.read_csv(grn_csv)
    tf_col, tg_col, tp_col = resolve_grn_table_columns(grn, "TF", "target", "Re_type")
    if tf_col is None or tg_col is None:
        raise ValueError(f"Cannot resolve TF/target columns in {grn_csv}")
    grn = grn.copy()
    grn[tf_col] = normalize_gene_symbol_series(grn[tf_col])
    grn[tg_col] = normalize_gene_symbol_series(grn[tg_col])
    if tp_col is not None:
        rel = grn[tp_col].astype(str).str.lower().str.strip().replace(
            {
                "activation": "activator",
                "activate": "activator",
                "positive": "activator",
                "upregulation": "activator",
                "inhibition": "repressor",
                "inhibit": "repressor",
                "repression": "repressor",
                "negative": "repressor",
                "downregulation": "repressor",
            }
        )
        grn = grn[rel.isin(["activator", "repressor"])]
    pairs: set[tuple[str, str]] = set()
    for a, b in zip(grn[tf_col].tolist(), grn[tg_col].tolist()):
        if not a or not b or a == b:
            continue
        pairs.add((a, b) if a <= b else (b, a))
    return pairs


def decontaminate(
    ppi_csv: Path,
    grn_csv: Path,
    output_csv: Path,
) -> dict:
    ppi = pd.read_csv(ppi_csv)
    if not {"from", "to"}.issubset(ppi.columns):
        raise ValueError(f"PPI CSV needs from,to columns; got {list(ppi.columns)}")
    ppi = ppi.copy()
    ppi["from"] = normalize_gene_symbol_series(ppi["from"])
    ppi["to"] = normalize_gene_symbol_series(ppi["to"])
    n_in = int(len(ppi))

    banned = _grn_undirected_pairs(grn_csv)
    keys = [
        (a, b) if a <= b else (b, a)
        for a, b in zip(ppi["from"].tolist(), ppi["to"].tolist())
    ]
    keep = [k not in banned for k in keys]
    out = ppi.loc[keep].reset_index(drop=True)
    n_out = int(len(out))
    n_rm = n_in - n_out

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    stats = {
        "ppi_input": str(ppi_csv.resolve()),
        "grn_csv": str(grn_csv.resolve()),
        "ppi_output": str(output_csv.resolve()),
        "n_ppi_edges_in": n_in,
        "n_ppi_edges_out": n_out,
        "n_ppi_edges_removed": n_rm,
        "frac_removed": float(n_rm / n_in) if n_in else 0.0,
        "n_grn_undirected_pairs": int(len(banned)),
        "rule": "drop PPI undirected edge if same undirected pair appears in GRN activator/repressor",
    }
    stats_path = output_csv.with_suffix(output_csv.suffix + ".decontam_stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Decontaminate STRING PPI vs gold GRN")
    ap.add_argument("--ppi-csv", default=None, help="Input STRING PPI CSV (default: shared path)")
    ap.add_argument("--grn-csv", default=None, help="Gold GRN CSV (default: step01 KEGG)")
    ap.add_argument(
        "--output-csv",
        default=None,
        help="Output PPI CSV (default: *_nokegg.csv next to input)",
    )
    args = ap.parse_args()

    ppi_in = Path(args.ppi_csv) if args.ppi_csv else ppi_string_csv()
    grn = Path(args.grn_csv) if args.grn_csv else step01_grn_csv()
    if args.output_csv:
        out = Path(args.output_csv)
    else:
        out = ppi_in.with_name(ppi_in.stem + "_nokegg.csv")

    stats = decontaminate(ppi_in, grn, out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[decontam-ppi] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
