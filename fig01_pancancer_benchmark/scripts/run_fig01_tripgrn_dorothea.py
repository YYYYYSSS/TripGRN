#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig.1 launchers: TripGRN DoRothEA-ABC epithelial 10CV.

Uses the same shared ``cv_splits_10fold.json`` as KEGG TripGRN.
Outputs under ``fig01_.../experiments/dorothea_abc_epithelial_10cv/``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dorothea_csv(root: Path) -> Path:
    candidates = [
        root / "shared" / "data" / "network" / "grn" / "DoRothEA_human_ABC_activation_inhibition_with_symbols.csv",
        root / "shared" / "data" / "network" / "grn" / "dorothea_abc_activation_inhibition.csv",
        root
        / "fig01_pancancer_benchmark"
        / "data"
        / "network"
        / "grn"
        / "DoRothEA_human_ABC_activation_inhibition_with_symbols.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return candidates[0].resolve()


def _apply_dorothea_env(root: Path) -> Path:
    """Set env so pancancer_paths points at DoRothEA experiments + CSV."""
    grn_csv = _dorothea_csv(root)
    os.environ["STEP01_EXPERIMENTS_NAME"] = "dorothea_abc_epithelial_10cv"
    os.environ["STEP01_GRN_CSV"] = str(grn_csv)
    return grn_csv


def cmd_smoke() -> int:
    root = _repo_root()
    sys.path.insert(0, str(root))
    grn_csv = _apply_dorothea_env(root)
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
        "grn_csv": str(step01_grn_csv(root)),
        "expected_dorothea": str(grn_csv),
        "cds": str(cds_gene_feature_csv(root)),
        "ppi": str(ppi_string_csv(root)),
        "epithelial": str(layout["epithelial"]),
        "experiments": str(layout["experiments_root"]),
    }
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(f"PANCANCER_GSE_IDS ({len(PANCANCER_GSE_IDS)}): {PANCANCER_GSE_IDS}")
    print(f"EXCLUDED_GSE_IDS: {EXCLUDED_GSE_IDS}")
    assert "GSE145825" not in PANCANCER_GSE_IDS
    assert "dorothea" in Path(checks["grn_csv"]).name.lower() or "DoRothEA" in Path(checks["grn_csv"]).name
    assert layout["experiments_root"].name == "dorothea_abc_epithelial_10cv"
    for p, label in (
        (checks["grn_csv"], "DoRothEA GRN"),
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
        payload = json.loads(cv.read_text(encoding="utf-8"))
        if list(payload.get("sample_split", [])) != [5, 2, 3]:
            raise ValueError(f"{gse} sample_split != [5,2,3]")
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))
    print("[fig01-dorothea-smoke] OK: DoRothEA CSV + shared 9 GSE cv_splits + exp dir name")
    return 0


def cmd_run(gse: str, fold: str, force: bool, epochs: int) -> int:
    root = _repo_root()
    grn_csv = _apply_dorothea_env(root)
    env = os.environ.copy()
    env["STEP01_EXPERIMENTS_NAME"] = "dorothea_abc_epithelial_10cv"
    env["STEP01_GRN_CSV"] = str(grn_csv)
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
    print("[fig01-dorothea] GRN", grn_csv, flush=True)
    print("[fig01-dorothea] RUN", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(root), env=env)


def cmd_summarize(gse: str) -> int:
    root = _repo_root()
    grn_csv = _apply_dorothea_env(root)
    env = os.environ.copy()
    env["STEP01_EXPERIMENTS_NAME"] = "dorothea_abc_epithelial_10cv"
    env["STEP01_GRN_CSV"] = str(grn_csv)
    cmd = [sys.executable, "-u", "-m", "grn.scripts.summarize_tripgrn_pancancer_10cv", "--gse", gse]
    print("[fig01-dorothea] RUN", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(root), env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig.1 TripGRN DoRothEA-ABC helpers")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("smoke", help="Validate DoRothEA paths / shared CV")

    p_run = sub.add_parser("run", help="Run TripGRN DoRothEA 10CV")
    p_run.add_argument("--gse", default="GSE156632", help="GSE id or ALL")
    p_run.add_argument("--fold", default="1", help="fold 1..10 or ALL")
    p_run.add_argument("--epochs", type=int, default=150)
    p_run.add_argument("--force", action="store_true")

    p_sum = sub.add_parser("summarize", help="Aggregate TripGRN DoRothEA fold metrics")
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
