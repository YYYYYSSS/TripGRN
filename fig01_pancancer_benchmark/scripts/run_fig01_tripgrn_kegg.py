#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig.1 launchers: TripGRN KEGG epithelial 10CV (+ path smoke check)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cmd_smoke() -> int:
    """Validate Fig.1 paths / GSE list / CV splits without training."""
    root = _repo_root()
    sys.path.insert(0, str(root))
    from grn.data.pancancer_paths import (
        EXCLUDED_GSE_IDS,
        PANCANCER_GSE_IDS,
        cds_gene_feature_csv,
        cv_splits_path,
        gse_npy_triplet,
        ppi_string_csv,
        scRNA_layout,
        step01_grn_csv,
    )

    layout = scRNA_layout(root)
    checks = {
        "grn_csv": step01_grn_csv(root),
        "cds": cds_gene_feature_csv(root),
        "ppi": ppi_string_csv(root),
        "epithelial": layout["epithelial"],
        "experiments": layout["experiments_root"],
    }
    print(json.dumps({k: str(v) for k, v in checks.items()}, ensure_ascii=False, indent=2))
    print(f"PANCANCER_GSE_IDS ({len(PANCANCER_GSE_IDS)}): {PANCANCER_GSE_IDS}")
    print(f"EXCLUDED_GSE_IDS: {EXCLUDED_GSE_IDS}")
    assert "GSE145825" not in PANCANCER_GSE_IDS
    for p, label in (
        (checks["grn_csv"], "GRN"),
        (checks["cds"], "CDS"),
        (checks["ppi"], "PPI"),
    ):
        if not Path(p).is_file():
            raise FileNotFoundError(f"Missing {label}: {p}")
    missing = []
    for gse in PANCANCER_GSE_IDS:
        v, g, c = gse_npy_triplet("epithelial", gse, root)
        cv = cv_splits_path(gse, root)
        for p in (v, g, c, cv):
            if not p.is_file():
                missing.append(str(p))
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))
    print("[fig01-smoke] OK: paths + 9 GSE npy/cv present")
    return 0


def cmd_run(gse: str, fold: str, force: bool, epochs: int) -> int:
    root = _repo_root()
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "grn.scripts.run_tripgrn_pancancer_10cv",
        "--gse",
        gse,
        "--fold",
        fold,
        "--epochs",
        str(epochs),
    ]
    if force:
        cmd.append("--force")
    print("[fig01] RUN", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(root))


def cmd_summarize(gse: str) -> int:
    root = _repo_root()
    cmd = [sys.executable, "-u", "-m", "grn.scripts.summarize_tripgrn_pancancer_10cv", "--gse", gse]
    print("[fig01] RUN", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(root))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig.1 TripGRN KEGG helpers")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("smoke", help="Validate paths / GSE list (no training)")

    p_run = sub.add_parser("run", help="Run TripGRN 10CV (default: one GSE one fold smoke)")
    p_run.add_argument("--gse", default="GSE156632", help="GSE id or ALL")
    p_run.add_argument("--fold", default="1", help="fold 1..10 or ALL")
    p_run.add_argument("--epochs", type=int, default=150)
    p_run.add_argument("--force", action="store_true")

    p_sum = sub.add_parser("summarize", help="Aggregate TripGRN fold metrics")
    p_sum.add_argument("--gse", default="ALL")

    args = ap.parse_args()
    if args.cmd == "smoke":
        raise SystemExit(cmd_smoke())
    if args.cmd == "run":
        raise SystemExit(cmd_run(args.gse, args.fold, args.force, args.epochs))
    if args.cmd == "summarize":
        raise SystemExit(cmd_summarize(args.gse))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
