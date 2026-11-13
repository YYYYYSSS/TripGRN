#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize gene-disjoint + VAE modality ablations for one GSE (10CV)."""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path
from typing import Any, Dict, List, Optional

from grn.data.pancancer_paths import scRNA_layout

# dirname under kegg_epithelial_10cv/ → short label
ABLATION_DIRS = [
    ("TripGRN_vae_gene_disjoint", "full"),
    ("TripGRN_vae_gd_no_expr", "w/o expr"),
    ("TripGRN_vae_gd_no_cds", "w/o CDS"),
    ("TripGRN_vae_gd_no_ppi", "w/o PPI"),
    ("TripGRN_vae_gd_expr_only", "expr only"),
    ("TripGRN_vae_gd_cds_only", "CDS only"),
    ("TripGRN_vae_gd_ppi_only", "PPI only"),
]


def _mean_std(vals: List[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(st.mean(vals)), "std": float(st.pstdev(vals)), "n": len(vals)}


def _load_setting(exp_root: Path, dirname: str, gse: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fold in range(1, 11):
        d = exp_root / dirname / gse / f"fold_{fold:02d}"
        tm = d / "unseen_samples_test_metrics.json"
        rt = d / "runtime.json"
        if not tm.is_file():
            continue
        t = json.loads(tm.read_text(encoding="utf-8"))
        r = json.loads(rt.read_text(encoding="utf-8")) if rt.is_file() else {}
        auprc = t.get("auprc_macro")
        if auprc is None and isinstance(t.get("auprc"), list):
            auprc = float(st.mean(t["auprc"]))
        rows.append(
            {
                "fold": fold,
                "accuracy": float(t["accuracy"]),
                "f1": float(t.get("f1", t.get("f1_macro", float("nan")))),
                "mcc": float(t.get("mcc", float("nan"))),
                "macro_auc": float(t.get("macro_auc", float("nan"))),
                "auprc_macro": float(auprc) if auprc is not None else float("nan"),
                "wall_time_sec": float(r.get("wall_time_sec", float("nan"))),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize modality ablation (gene-disjoint+VAE)")
    ap.add_argument("--gse", default="GSE156632")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    exp_root = scRNA_layout(root)["experiments_root"]
    out_dir = exp_root / "summary" / "modality_ablation_gene_disjoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: List[Dict[str, Any]] = []
    mean_rows: List[Dict[str, Any]] = []

    print(f"[ablation-summary] exp_root={exp_root}")
    print(f"[ablation-summary] gse={args.gse}")

    for dirname, label in ABLATION_DIRS:
        folds = _load_setting(exp_root, dirname, args.gse)
        print(f"  {label:12s}  {dirname}: {len(folds)}/10 folds")
        for fr in folds:
            fold_rows.append({"setting": label, "dirname": dirname, "gse": args.gse, **fr})
        if not folds:
            continue
        mrow: Dict[str, Any] = {
            "setting": label,
            "dirname": dirname,
            "gse": args.gse,
            "n_folds": len(folds),
        }
        for key in ("accuracy", "f1", "mcc", "macro_auc", "auprc_macro", "wall_time_sec"):
            ms = _mean_std([float(x[key]) for x in folds if x[key] == x[key]])
            mrow[f"{key}_mean"] = ms["mean"]
            mrow[f"{key}_std"] = ms["std"]
        mean_rows.append(mrow)
        if mrow.get("accuracy_mean") is not None:
            print(
                f"    acc={mrow['accuracy_mean']:.4f}  mcc={mrow['mcc_mean']:.4f}  "
                f"macro_auc={mrow['macro_auc_mean']:.4f}  wall={mrow['wall_time_sec_mean']:.1f}s"
            )

    # Drop vs full (contribution = full - w/o modality)
    full = next((r for r in mean_rows if r["setting"] == "full"), None)
    if full and full.get("accuracy_mean") is not None:
        print("[ablation-summary] contribution approx = full - w/o modality (acc/mcc/macro_auc):")
        for r in mean_rows:
            if not str(r["setting"]).startswith("w/o "):
                continue
            print(
                f"  {r['setting']:12s}  Δacc={full['accuracy_mean']-r['accuracy_mean']:+.4f}  "
                f"Δmcc={full['mcc_mean']-r['mcc_mean']:+.4f}  "
                f"Δauc={full['macro_auc_mean']-r['macro_auc_mean']:+.4f}"
            )

    fold_csv = out_dir / f"{args.gse}_fold_metrics.csv"
    mean_csv = out_dir / f"{args.gse}_mean_std.csv"
    if fold_rows:
        with fold_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(fold_rows[0].keys()))
            w.writeheader()
            w.writerows(fold_rows)
    if mean_rows:
        with mean_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(mean_rows[0].keys()))
            w.writeheader()
            w.writerows(mean_rows)
    print(f"[ablation-summary] wrote {fold_csv}")
    print(f"[ablation-summary] wrote {mean_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
