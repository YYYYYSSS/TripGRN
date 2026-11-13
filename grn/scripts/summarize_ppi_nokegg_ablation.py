#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare gene-disjoint ablations: contaminated STRING vs STRING\\KEGG PPI."""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path
from typing import Any, Dict, List, Optional

from grn.data.pancancer_paths import scRNA_layout

# (dirname, label, ppi_kind)
SETTINGS = [
    ("TripGRN_vae_gene_disjoint", "full", "contaminated"),
    ("TripGRN_vae_gd_ppi_only", "PPI only", "contaminated"),
    ("TripGRN_vae_gd_no_expr", "w/o expr", "contaminated"),
    ("TripGRN_vae_gd_no_cds", "w/o CDS", "contaminated"),
    ("TripGRN_vae_gd_no_ppi", "w/o PPI", "contaminated"),
    ("TripGRN_vae_gd_nokegg_full", "full", "nokegg"),
    ("TripGRN_vae_gd_nokegg_ppi_only", "PPI only", "nokegg"),
    ("TripGRN_vae_gd_nokegg_no_expr", "w/o expr", "nokegg"),
    ("TripGRN_vae_gd_nokegg_no_cds", "w/o CDS", "nokegg"),
    ("TripGRN_vae_gd_nokegg_no_ppi", "w/o PPI", "nokegg"),
]


def _ms(vals: List[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(st.mean(vals)), "std": float(st.pstdev(vals)), "n": len(vals)}


def _load(exp_root: Path, dirname: str, gse: str) -> List[Dict[str, Any]]:
    rows = []
    for fold in range(1, 11):
        d = exp_root / dirname / gse / f"fold_{fold:02d}"
        tm = d / "unseen_samples_test_metrics.json"
        if not tm.is_file():
            continue
        t = json.loads(tm.read_text(encoding="utf-8"))
        rpath = d / "runtime.json"
        r = json.loads(rpath.read_text(encoding="utf-8")) if rpath.is_file() else {}
        auprc = t.get("auprc_macro")
        if auprc is None and isinstance(t.get("auprc"), list):
            auprc = float(st.mean(t["auprc"]))
        rows.append(
            {
                "fold": fold,
                "accuracy": float(t["accuracy"]),
                "f1": float(t.get("f1", float("nan"))),
                "mcc": float(t.get("mcc", float("nan"))),
                "macro_auc": float(t.get("macro_auc", float("nan"))),
                "auprc_macro": float(auprc) if auprc is not None else float("nan"),
                "wall_time_sec": float(r.get("wall_time_sec", float("nan"))),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gse", default="GSE156632")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    exp_root = scRNA_layout(root)["experiments_root"]
    out_dir = exp_root / "summary" / "ppi_nokegg_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_rows: List[Dict[str, Any]] = []
    print(f"[nokegg-summary] gse={args.gse}")
    for dirname, label, kind in SETTINGS:
        folds = _load(exp_root, dirname, args.gse)
        print(f"  [{kind:13s}] {label:10s} {dirname}: {len(folds)}/10")
        if not folds:
            continue
        row: Dict[str, Any] = {
            "ppi_kind": kind,
            "setting": label,
            "dirname": dirname,
            "gse": args.gse,
            "n_folds": len(folds),
        }
        for key in ("accuracy", "f1", "mcc", "macro_auc", "auprc_macro", "wall_time_sec"):
            ms = _ms([float(x[key]) for x in folds if x[key] == x[key]])
            row[f"{key}_mean"] = ms["mean"]
            row[f"{key}_std"] = ms["std"]
        mean_rows.append(row)
        print(
            f"    acc={row['accuracy_mean']:.4f} mcc={row['mcc_mean']:.4f} "
            f"macro_auc={row['macro_auc_mean']:.4f}"
        )

    # Side-by-side delta nokegg - contaminated for matched settings
    print("[nokegg-summary] delta (nokegg - contaminated):")
    by = {(r["setting"], r["ppi_kind"]): r for r in mean_rows}
    for setting in ("full", "PPI only", "w/o expr", "w/o CDS", "w/o PPI"):
        a = by.get((setting, "contaminated"))
        b = by.get((setting, "nokegg"))
        if not a or not b:
            continue
        print(
            f"  {setting:10s}  d_acc={b['accuracy_mean']-a['accuracy_mean']:+.4f}  "
            f"d_mcc={b['mcc_mean']-a['mcc_mean']:+.4f}  "
            f"d_auc={b['macro_auc_mean']-a['macro_auc_mean']:+.4f}"
        )

    out = out_dir / f"{args.gse}_mean_std.csv"
    if mean_rows:
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(mean_rows[0].keys()))
            w.writeheader()
            w.writerows(mean_rows)
    print(f"[nokegg-summary] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
