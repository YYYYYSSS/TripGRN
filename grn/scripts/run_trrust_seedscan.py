#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run TRRUST TripGRN + DeepFGRN 10CV for one or more seeds (full 10 GSE).

Skip-if-done; writes under experiments/step01_pancancer_epithelial_10cv_trrust_seed{S}/.
CV splits come from the seed-scan hub (STEP01_CV_SPLITS_ROOT), never overwriting epithelial seed42.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from grn.data.pancancer_paths import grn_root_from_here

HUB_REL = Path("experiments") / "step01_trrust_trip_vs_deep_seedscan"
TRRUST_CSV_REL = Path("data") / "network" / "grn" / "TRRUST_human_activation_inhibition_with_symbols.csv"


def _load_manifest(hub: Path) -> dict:
    return json.loads((hub / "SEED_MANIFEST.json").read_text(encoding="utf-8"))


def _ensure_cv(grn_root: Path, seed: int) -> Path:
    cv_root = grn_root / HUB_REL / "cv_splits" / f"seed_{seed}"
    marker = cv_root / "GSE103322" / "cv_splits_10fold.json"
    if not marker.is_file():
        cmd = [sys.executable, "-u", "-m", "grn.scripts.prepare_trrust_seedscan_cv", "--seed", str(seed)]
        subprocess.run(cmd, check=True, cwd=str(grn_root))
    return cv_root


def _run_seed(grn_root: Path, seed_entry: dict, force: bool) -> None:
    seed = int(seed_entry["seed"])
    if seed_entry.get("role") == "baseline_already_run":
        print(f"[seedscan] seed={seed} is baseline; skip train (score only)")
        return

    exp_name = seed_entry["experiments_name"]
    cv_root = _ensure_cv(grn_root, seed)
    trrust = grn_root / TRRUST_CSV_REL
    if not trrust.is_file():
        raise FileNotFoundError(trrust)

    env = os.environ.copy()
    env["STEP01_GRN_CSV"] = str(trrust.resolve())
    env["STEP01_EXPERIMENTS_NAME"] = exp_name
    env["STEP01_CV_SPLITS_ROOT"] = str(cv_root.resolve())
    env["MPLBACKEND"] = "Agg"
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[seedscan] ===== seed={seed} exp={exp_name} =====")
    print(f"[seedscan] CV_SPLITS_ROOT={env['STEP01_CV_SPLITS_ROOT']}")

    trip_cmd = [
        sys.executable,
        "-u",
        "-m",
        "grn.scripts.run_tripgrn_pancancer_10cv",
        "--gse",
        "ALL",
        "--fold",
        "ALL",
        "--seed",
        str(seed),
    ]
    if force:
        trip_cmd.append("--force")
    print("[seedscan] TripGRN ...")
    subprocess.run(trip_cmd, check=True, cwd=str(grn_root), env=env)

    deep_cmd = [
        sys.executable,
        "-u",
        "-m",
        "grn.scripts.run_deepfgrn_pancancer_10cv",
        "--gse",
        "ALL",
        "--fold",
        "ALL",
    ]
    if force:
        deep_cmd.append("--force")
    print("[seedscan] DeepFGRN ...")
    subprocess.run(deep_cmd, check=True, cwd=str(grn_root), env=env)

    for mod in (
        "grn.scripts.summarize_tripgrn_pancancer_10cv",
        "grn.scripts.summarize_deepfgrn_pancancer_10cv",
    ):
        subprocess.run([sys.executable, "-u", "-m", mod, "--gse", "ALL"], check=True, cwd=str(grn_root), env=env)

    # refresh ranking
    subprocess.run([sys.executable, "-u", "-m", "grn.scripts.summarize_trrust_seedscan"], check=True, cwd=str(grn_root))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run TRRUST Trip vs Deep seed-scan")
    ap.add_argument("--seed", type=int, default=0, help="One seed to run")
    ap.add_argument("--all-pending", action="store_true", help="Run all non-baseline pending seeds")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Only print plan")
    args = ap.parse_args()

    grn_root = grn_root_from_here()
    hub = grn_root / HUB_REL
    manifest = _load_manifest(hub)

    selected: List[dict] = []
    if int(args.seed) > 0:
        hit = [s for s in manifest["seeds"] if int(s["seed"]) == int(args.seed)]
        if not hit:
            raise SystemExit(f"seed {args.seed} not in manifest")
        selected = hit
    elif args.all_pending:
        selected = [s for s in manifest["seeds"] if s.get("role") != "baseline_already_run" and s.get("status") != "done"]
    else:
        raise SystemExit("Specify --seed N or --all-pending")

    print("[seedscan] plan:", [int(s["seed"]) for s in selected])
    if args.dry_run:
        return

    # ensure all CV for planned seeds exist
    subprocess.run([sys.executable, "-u", "-m", "grn.scripts.prepare_trrust_seedscan_cv"], check=True, cwd=str(grn_root))

    for s in selected:
        _run_seed(grn_root, s, force=bool(args.force))

    subprocess.run([sys.executable, "-u", "-m", "grn.scripts.summarize_trrust_seedscan"], check=True, cwd=str(grn_root))
    print("[seedscan] finished planned seeds")


if __name__ == "__main__":
    main()
