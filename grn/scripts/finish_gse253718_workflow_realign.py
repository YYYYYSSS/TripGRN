#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
After GSE253718 pan-cancer TripGRN folds finish:
  1) summarize step01 TripGRN (GSE253718 + refresh ALL CSV)
  2) run GSE253718 cell-type 10CV (skip finished)
  3) summarize + heatmap figures for cell types

Safe to re-run; skips completed folds.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from grn.data.gse253718_celltype_paths import resolve_type_ids, tripgrn_10cv_fold_dir
from grn.data.pancancer_paths import tripgrn_fold_dir


def _pancancer_done(grn_root: Path, n_folds: int = 10) -> bool:
    for f in range(1, n_folds + 1):
        d = tripgrn_fold_dir("GSE253718", f, grn_root)
        if not (d / "unseen_samples_test_metrics.json").is_file():
            return False
    return True


def _run(mod: str, *extra: str, allow_nonzero_if: str | None = None) -> None:
    cmd = [sys.executable, "-u", "-m", mod, *extra]
    print(f"[finish-realign] RUN {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode == 0:
        return
    if allow_nonzero_if and Path(allow_nonzero_if).is_file():
        print(
            f"[finish-realign] warn: {mod} exit={proc.returncode} "
            f"but marker exists ({allow_nonzero_if}); continuing",
            flush=True,
        )
        return
    raise subprocess.CalledProcessError(proc.returncode, cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wait", action="store_true", help="Poll until pan-cancer GSE253718 folds exist")
    ap.add_argument("--poll-sec", type=int, default=120)
    ap.add_argument("--skip-celltype-train", action="store_true")
    ap.add_argument("--max-expression-cols", type=int, default=10000)
    args = ap.parse_args()

    grn_root = Path(__file__).resolve().parents[2]

    if args.wait:
        while not _pancancer_done(grn_root):
            print("[finish-realign] waiting for GSE253718 pan-cancer folds...", flush=True)
            time.sleep(int(args.poll_sec))
        print("[finish-realign] pan-cancer folds ready", flush=True)
    elif not _pancancer_done(grn_root):
        raise SystemExit(
            "GSE253718 pan-cancer TripGRN folds incomplete. "
            "Re-run with --wait or finish run_tripgrn_pancancer_10cv first."
        )

    _run("grn.scripts.summarize_tripgrn_pancancer_10cv", "--gse", "GSE253718")
    _run("grn.scripts.summarize_tripgrn_pancancer_10cv", "--gse", "ALL")

    if not args.skip_celltype_train:
        # Child may return Windows fatal codes after successful fold writes; runner itself
        # continues and exits 0 when metrics exist. Still tolerate nonzero here.
        _run(
            "grn.scripts.run_gse253718_celltype_tripgrn_10cv",
            "--type",
            "ALL",
            "--fold",
            "ALL",
            "--max-expression-cols",
            str(int(args.max_expression_cols)),
            allow_nonzero_if=str(
                grn_root
                / "experiments"
                / "gse253718_celltype_bytype_seed42"
                / "summary"
                / "tripgrn_10cv_type_summary.csv"
            ),
        )
        # If summary not yet written, force continue to summarize from whatever folds exist.
        # Re-invoke train is idempotent (skips finished folds) only when runner survived;
        # if runner hard-crashed mid-way, call it again until complete or max retries.
        for attempt in range(1, 6):
            n_done = sum(
                1
                for tid, _ in resolve_type_ids("ALL")
                for f in range(1, 11)
                if (tripgrn_10cv_fold_dir(tid, f, grn_root) / "unseen_samples_test_metrics.json").is_file()
            )
            n_expected = len(resolve_type_ids("ALL")) * 10
            print(f"[finish-realign] celltype folds {n_done}/{n_expected} (attempt {attempt})", flush=True)
            if n_done >= n_expected:
                break
            _run(
                "grn.scripts.run_gse253718_celltype_tripgrn_10cv",
                "--type",
                "ALL",
                "--fold",
                "ALL",
                "--max-expression-cols",
                str(int(args.max_expression_cols)),
                allow_nonzero_if=str(
                    grn_root
                    / "experiments"
                    / "gse253718_celltype_bytype_seed42"
                    / "tripgrn_10cv"
                    / "macrophage"
                    / "fold_01"
                    / "unseen_samples_test_metrics.json"
                ),
            )

    # Summarize only types that have at least one fold
    types = resolve_type_ids("ALL")
    any_fold = False
    for tid, _ in types:
        for f in range(1, 11):
            if (tripgrn_10cv_fold_dir(tid, f, grn_root) / "unseen_samples_test_metrics.json").is_file():
                any_fold = True
                break
        if any_fold:
            break
    if any_fold:
        _run("grn.scripts.summarize_gse253718_celltype_10cv")
        _run("grn.scripts.plot_gse253718_celltype_heatmap_auc")
    else:
        print("[finish-realign] no cell-type folds yet; skip summarize/plot")

    print("[finish-realign] done")


if __name__ == "__main__":
    main()
