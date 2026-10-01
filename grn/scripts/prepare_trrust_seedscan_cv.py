#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialize per-seed CV splits for TRRUST Trip vs Deep seed-scan (no training)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from grn.data.pancancer_cv import generate_cv_splits, write_cv_splits
from grn.data.pancancer_paths import PANCANCER_GSE_IDS, gse_npy_triplet, grn_root_from_here

HUB_REL = Path("experiments") / "step01_trrust_trip_vs_deep_seedscan"


def _load_manifest(hub: Path) -> dict:
    return json.loads((hub / "SEED_MANIFEST.json").read_text(encoding="utf-8"))


def _n_cells(gse: str, grn_root: Path) -> int:
    v_path, _, c_path = gse_npy_triplet("epithelial", gse, grn_root)
    if c_path.is_file():
        return int(np.load(c_path, allow_pickle=True).shape[0])
    # fallback: values columns
    arr = np.load(v_path, mmap_mode="r")
    return int(arr.shape[1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare per-seed CV splits for TRRUST seed-scan")
    ap.add_argument("--seed", type=int, default=0, help="Single seed; 0 = all candidate seeds in manifest")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    grn_root = grn_root_from_here()
    hub = grn_root / HUB_REL
    manifest = _load_manifest(hub)
    seeds = [int(s["seed"]) for s in manifest["seeds"]]
    if int(args.seed) > 0:
        seeds = [int(args.seed)]

    written = 0
    skipped = 0
    for seed in seeds:
        # seed 42 already lives under epithelial/; still materialize a copy under hub for审计
        out_root = hub / "cv_splits" / f"seed_{seed}"
        for gse in PANCANCER_GSE_IDS:
            out_path = out_root / gse / "cv_splits_10fold.json"
            if out_path.is_file() and not args.force:
                skipped += 1
                continue
            n_cells = _n_cells(gse, grn_root)
            payload = generate_cv_splits(
                n_cells,
                cv_base_seed=int(seed),
                pair_seed=int(seed),
                gse_id=gse,
            )
            write_cv_splits(payload, out_path)
            written += 1
            print(f"[seedscan-cv] wrote seed={seed} {gse} n_cells={n_cells} -> {out_path}")

    status = {
        "prepare_cv": {"written": written, "skipped": skipped, "seeds": seeds},
    }
    status_path = hub / "status.json"
    prev = {}
    if status_path.is_file():
        prev = json.loads(status_path.read_text(encoding="utf-8"))
    prev.update(status)
    status_path.write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[seedscan-cv] done written={written} skipped={skipped}")


if __name__ == "__main__":
    main()
