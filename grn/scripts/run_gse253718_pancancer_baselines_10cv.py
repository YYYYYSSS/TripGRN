# -*- coding: utf-8 -*-
"""Run missing step01 baseline methods on GSE253718 (10-fold), then summarize + replot.

Methods: DeepFGRN, DeepTGI, scMGATGRN, GENELink (TripGRN already complete).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


METHODS = (
    (
        "DeepFGRN",
        "grn.scripts.run_deepfgrn_pancancer_10cv",
        ["--keras-batch-size-cap", "256"],
        "grn.scripts.summarize_deepfgrn_pancancer_10cv",
    ),
    (
        "DeepTGI",
        "grn.scripts.run_deeptgi_pancancer_10cv",
        ["--batch-size-cap", "512", "--max-genes", "0"],
        "grn.scripts.summarize_deeptgi_pancancer_10cv",
    ),
    (
        "scMGATGRN",
        "grn.scripts.run_scmgatgrn_pancancer_10cv",
        ["--batch-size-cap", "256", "--max-genes", "0"],
        "grn.scripts.summarize_scmgatgrn_pancancer_10cv",
    ),
    (
        "GENELink",
        "grn.scripts.run_genelink_pancancer_10cv",
        ["--batch-size-cap", "256", "--max-genes", "0"],
        "grn.scripts.summarize_genelink_pancancer_10cv",
    ),
)


def _run(module: str, *extra: str, cwd: Path) -> None:
    cmd = [sys.executable, "-u", "-m", module, *extra]
    print("[baselines] RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd))


def main() -> None:
    ap = argparse.ArgumentParser(description="GSE253718 pan-cancer baselines 10CV")
    ap.add_argument("--gse", default="GSE253718")
    ap.add_argument("--fold", default="ALL")
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument(
        "--methods",
        default="ALL",
        help="Comma list: DeepFGRN,DeepTGI,scMGATGRN,GENELink or ALL",
    )
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-summarize", action="store_true")
    ap.add_argument("--skip-plot", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    want = {m.strip().upper() for m in str(args.methods).split(",") if m.strip()}
    if "ALL" in want:
        selected = list(METHODS)
    else:
        selected = [m for m in METHODS if m[0].upper() in want]
        if not selected:
            raise SystemExit(f"No methods matched: {args.methods}")

    for name, train_mod, train_extra, sum_mod in selected:
        if not args.skip_train:
            extras = [
                "--gse",
                args.gse,
                "--fold",
                args.fold,
                "--n-folds",
                str(args.n_folds),
                *train_extra,
            ]
            if args.force:
                extras.append("--force")
            _run(train_mod, *extras, cwd=root)

        if not args.skip_summarize:
            # Refresh ALL so ten-cohort CSVs include GSE253718 for every method
            _run(sum_mod, "--gse", "ALL", cwd=root)

    if not args.skip_plot:
        _run(
            "grn.scripts.plot_step01_trrust_metrics",
            "--experiment-name",
            "step01_pancancer_epithelial_10cv",
            "--network-label",
            "KEGG PPrel",
            "--out-dir",
            "experiments/step01_pancancer_epithelial_10cv/figures",
            cwd=root,
        )
        try:
            _run("grn.scripts.plot_step01_pancancer_method_comparison", cwd=root)
        except subprocess.CalledProcessError as exc:
            print(f"[baselines] warn: method_comparison plot failed: {exc}", flush=True)

    print("[baselines] done", flush=True)


if __name__ == "__main__":
    main()
