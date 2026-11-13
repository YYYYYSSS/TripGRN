#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig.3 launcher: TCGA bulk cross-cancer 10CV (original / pre-VAE protocol)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_path(root: Path) -> None:
    s = str(root.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def cmd_smoke() -> int:
    root = _repo_root()
    _ensure_path(root)
    from grn.data.bulk_paths import (
        TCGA_CANCER_IDS,
        bulk_exp_csv,
        bulk_layout,
        cv_splits_path,
        discover_all_bulk_exp_csvs,
    )

    layout = bulk_layout(root)
    checks = {
        "experiment_base": str(layout["experiment_base"]),
        "grn_csv": str(layout["grn_csv"]),
        "cds_gene": str(layout["cds_gene"]),
        "tss_gene": str(layout["tss_gene"]),
        "ppi_csv_raw": str(layout["ppi_csv_raw"]),
        "bulk_exp_dir": str(layout["bulk_exp_dir"]),
        "cv_splits_dir": str(layout["cv_splits_dir"]),
        "protocol": "original: conv + no-gene-disjoint + CDS/TSS + raw STRING",
    }
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    for key in ("grn_csv", "cds_gene", "tss_gene", "ppi_csv_raw"):
        if not Path(checks[key]).is_file():
            raise FileNotFoundError(f"Missing {key}: {checks[key]}")

    found = discover_all_bulk_exp_csvs(root)
    missing = []
    for ca in TCGA_CANCER_IDS:
        exp = found.get(ca)
        sp = cv_splits_path(ca, root)
        if exp is None or not Path(exp).is_file():
            missing.append(f"exp {ca}: {bulk_exp_csv(ca, root)}")
        if not sp.is_file():
            missing.append(f"splits {ca}: {sp}")
    if missing:
        raise FileNotFoundError("Missing bulk assets:\n" + "\n".join(missing))
    print(f"[fig03-smoke] cancers={len(TCGA_CANCER_IDS)} all CSV+splits+TSS OK")
    print("[fig03-smoke] OK")
    return 0


def cmd_status() -> int:
    root = _repo_root()
    _ensure_path(root)
    from grn.data.bulk_paths import TCGA_CANCER_IDS, bulk_fold_dir

    print(f"{'cancer':<12} {'done':>6}  next")
    total = 0
    for ca in TCGA_CANCER_IDS:
        n = 0
        nxt = None
        for fold in range(1, 11):
            ok = (bulk_fold_dir(ca, fold, root) / "unseen_samples_test_metrics.json").is_file()
            if ok:
                n += 1
            elif nxt is None:
                nxt = fold
        total += n
        nxt_s = "COMPLETE" if nxt is None else f"fold {nxt:02d}"
        print(f"{ca:<12} {n:3d}/10  {nxt_s}")
    print(f"[fig03-status] {total}/120 folds done")
    return 0


def cmd_run(
    cancer: str,
    fold: str,
    *,
    force: bool,
    epochs: int,
    expression_encoder: str,
    gene_disjoint: bool,
    ppi_nokegg: bool,
    no_tss: bool,
) -> int:
    root = _repo_root()
    _ensure_path(root)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "grn.scripts.run_bulk_tcga_10cv",
        "--cancer",
        cancer,
        "--fold",
        fold,
        "--epochs",
        str(epochs),
        "--expression-encoder",
        expression_encoder,
    ]
    if force:
        cmd.append("--force")
    if gene_disjoint:
        cmd.append("--gene-disjoint")
    if ppi_nokegg:
        cmd.append("--ppi-nokegg")
    if no_tss:
        cmd.append("--no-tss")
    print("[fig03] RUN", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(root))


def cmd_summarize(cancer: str) -> int:
    root = _repo_root()
    _ensure_path(root)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "grn.scripts.summarize_bulk_tcga_10cv",
        "--cancer",
        cancer,
    ]
    print("[fig03] RUN", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(root))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig.3 TCGA bulk cross-cancer (original protocol)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    sub.add_parser("status")

    p_run = sub.add_parser("run")
    p_run.add_argument("--cancer", default="ALL", help="TCGA-XXX or ALL")
    p_run.add_argument("--fold", default="ALL", help="1..10 or ALL")
    p_run.add_argument("--epochs", type=int, default=150)
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument(
        "--expression-encoder",
        choices=("conv", "vae"),
        default="conv",
        help="Default conv = original Fig.3 (no VAE).",
    )
    p_run.add_argument(
        "--gene-disjoint",
        action="store_true",
        help="Optional rewrite setting; OFF by default for original protocol.",
    )
    p_run.add_argument(
        "--ppi-nokegg",
        action="store_true",
        help="Optional; default uses raw STRING.",
    )
    p_run.add_argument(
        "--no-tss",
        action="store_true",
        help="Skip TSS (original protocol includes TSS).",
    )

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--cancer", default="ALL")

    args = ap.parse_args()
    if args.cmd == "smoke":
        raise SystemExit(cmd_smoke())
    if args.cmd == "status":
        raise SystemExit(cmd_status())
    if args.cmd == "run":
        raise SystemExit(
            cmd_run(
                args.cancer,
                args.fold,
                force=args.force,
                epochs=args.epochs,
                expression_encoder=args.expression_encoder,
                gene_disjoint=args.gene_disjoint,
                ppi_nokegg=args.ppi_nokegg,
                no_tss=args.no_tss,
            )
        )
    if args.cmd == "summarize":
        raise SystemExit(cmd_summarize(args.cancer))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
