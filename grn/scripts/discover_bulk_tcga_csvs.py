#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scan TCGA bulk FPKM CSV locations (matches historical Step03 run_config paths)."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from grn.data.bulk_paths import (
    OLD_BULK_RUN_ROOT,
    TCGA_CANCER_IDS,
    _candidate_exp_csvs,
    _normalize_cancer_id,
    _rewrite_legacy_path,
    bulk_layout,
    discover_all_bulk_exp_csvs,
    grn_root_from_here,
)


def _count_samples(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    return max(0, len(header) - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Discover TCGA bulk FPKM CSV files on disk")
    ap.add_argument("--json-out", type=Path, default=None, help="Optional manifest JSON path")
    args = ap.parse_args()

    grn_root = grn_root_from_here()
    layout = bulk_layout(grn_root)
    found = discover_all_bulk_exp_csvs(grn_root)

    print(f"[discover-bulk] grn_root={grn_root}")
    print(f"[discover-bulk] bulk bundle dir: {layout['bulk_exp_dir']}")
    print(f"[discover-bulk] env override: TCGA_BULK_DATA_ROOT={os.environ.get('TCGA_BULK_DATA_ROOT', '') or '(unset)'}")
    print()

    rows = []
    n_ok = 0
    for cancer in TCGA_CANCER_IDS:
        path = found[cancer]
        if path is not None and path.is_file():
            n_samples = _count_samples(path)
            print(f"[OK]   {cancer}: {path}  (n_samples={n_samples})")
            rows.append({"cancer": cancer, "status": "found", "path": str(path), "n_samples": n_samples})
            n_ok += 1
            continue

        print(f"[MISS] {cancer}: expected under data/external_validation/{cancer}_FPKM_Symbol.csv")
        old_cfg = grn_root / OLD_BULK_RUN_ROOT / f"repro_extval_{cancer}_FPKM_Symbol" / "run_config.json"
        legacy_hint = ""
        if old_cfg.is_file():
            payload = json.loads(old_cfg.read_text(encoding="utf-8"))
            raw = payload.get("exp_csv", "")
            rewritten = _rewrite_legacy_path(Path(str(raw)), grn_root) if raw else None
            legacy_hint = f"old run_config -> {raw}"
            if rewritten and rewritten.is_file():
                print(f"       (legacy path exists: {rewritten})")
            else:
                print(f"       ({legacy_hint}; not found after remap to current project root)")
        tried = [str(p) for p in _candidate_exp_csvs(cancer, grn_root)]
        rows.append(
            {
                "cancer": cancer,
                "status": "missing",
                "path": "",
                "n_samples": 0,
                "legacy_exp_csv": legacy_hint,
                "candidates_tried": tried,
            }
        )

    print()
    print(f"[discover-bulk] found {n_ok}/{len(TCGA_CANCER_IDS)} cancers")
    if n_ok < len(TCGA_CANCER_IDS):
        print(
            "[discover-bulk] tip: copy missing files to "
            f"{layout['external_validation_dir']}\\TCGA-XXX_FPKM_Symbol.csv "
            "(same layout as old D:\\project_GRN\\data\\external_validation\\)"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[discover-bulk] wrote {args.json_out}")


if __name__ == "__main__":
    main()
