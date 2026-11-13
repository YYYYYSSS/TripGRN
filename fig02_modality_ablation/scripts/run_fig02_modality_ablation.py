#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig.2 launcher: GSE253718 epithelial modality ablation 10CV.

Default protocol (original / manuscript Fig.3 ablation):
  - expression encoder = conv (ResNet/CNN; not VAE)
  - legacy random pair-split (--no-gene-disjoint)
  - raw STRING PPI (not KEGG-decontaminated)
"""

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
    from grn.data.gse253718_paths import ABLATION_STAGES, gse253718_layout

    layout = gse253718_layout(root)
    checks = {k: str(layout[k]) for k in (
        "exp_values", "exp_genes", "cv_splits", "grn_csv", "cds_gene", "ppi_csv_raw", "experiment_base"
    )}
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    for key in ("exp_values", "exp_genes", "cv_splits", "grn_csv", "cds_gene", "ppi_csv_raw"):
        if not Path(checks[key]).is_file():
            raise FileNotFoundError(f"Missing {key}: {checks[key]}")
    print(f"[fig02-smoke] stages={len(ABLATION_STAGES)} experiment_base={checks['experiment_base']}")
    print("[fig02-smoke] OK")
    return 0


def cmd_status() -> int:
    root = _repo_root()
    _ensure_path(root)
    from grn.data.gse253718_paths import ABLATION_STAGES, ablation_stage_dir

    print(f"{'stage':<20} {'done':>6}  next")
    for stage_id, label, _ in ABLATION_STAGES:
        n = 0
        nxt = None
        for fold in range(1, 11):
            ok = (ablation_stage_dir(stage_id, fold, root) / "unseen_samples_test_metrics.json").is_file()
            if ok:
                n += 1
            elif nxt is None:
                nxt = fold
        nxt_s = "COMPLETE" if nxt is None else f"fold {nxt:02d}"
        print(f"{stage_id:<20} {n:3d}/10  {nxt_s}  ({label})")
    return 0


def cmd_run(
    stage: str,
    fold: str,
    *,
    force: bool,
    epochs: int,
    gene_disjoint: bool,
    ppi_nokegg: bool,
    expression_encoder: str,
) -> int:
    root = _repo_root()
    _ensure_path(root)
    from grn.data.gse253718_paths import ABLATION_STAGES, ablation_stage_dir, gse253718_layout
    from grn.training.pair_dataloader import suggest_batch_size
    from grn.data.pancancer_cv import get_fold_split, load_cv_splits

    layout = gse253718_layout(root)
    ppi_csv = layout["ppi_csv"] if ppi_nokegg else layout["ppi_csv_raw"]
    expr_enc = str(expression_encoder).strip().lower()

    stages = ABLATION_STAGES
    if str(stage).strip().upper() != "ALL":
        want = str(stage).strip()
        stages = [s for s in ABLATION_STAGES if s[0] == want or s[1] == want]
        if not stages:
            raise SystemExit(f"Unknown stage: {stage}")

    folds = list(range(1, 11)) if str(fold).strip().upper() == "ALL" else [int(fold)]
    cv_payload = load_cv_splits(layout["cv_splits"])

    for stage_id, label, extra in stages:
        for fold_idx in folds:
            run_dir = ablation_stage_dir(stage_id, fold_idx, root)
            if (run_dir / "unseen_samples_test_metrics.json").is_file() and not force:
                print(f"[fig02] skip {stage_id} fold {fold_idx:02d}", flush=True)
                continue
            n_train = len(get_fold_split(cv_payload, fold_idx)["train_cols"])
            batch_size = suggest_batch_size(n_train, cap=256)
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "grn.scripts.train_unseen_samples_no_cv",
                "--exp-npy-values",
                str(layout["exp_values"]),
                "--exp-npy-genes",
                str(layout["exp_genes"]),
                "--grn-csv",
                str(layout["grn_csv"]),
                "--output-base-dir",
                str(layout["experiment_base"]),
                "--experiment-name",
                stage_id,
                "--no-timestamp",
                "--method-name",
                f"TripGRN:{label}",
                "--cv-splits-json",
                str(layout["cv_splits"]),
                "--fold-idx",
                str(fold_idx),
                "--skip-plots",
                "--seed",
                "42",
                "--batch-size",
                str(batch_size),
                "--epochs",
                str(epochs),
                "--patience",
                "5",
                "--neg-sample-ratio",
                "1.0",
                "--amp-bf16",
                "--expression-encoder",
                expr_enc,
                *extra,
            ]
            # CDS / PPI paths only when that modality is enabled
            if "--no-cds" not in extra:
                cmd.extend(["--cds-gene-feature-csv", str(layout["cds_gene"])])
            if "--no-ppi" not in extra:
                cmd.extend(["--ppi-csv", str(ppi_csv)])
            if gene_disjoint:
                cmd.append("--gene-disjoint")
            else:
                cmd.append("--no-gene-disjoint")

            print(
                f"[fig02] run {stage_id} ({label}) fold {fold_idx:02d} "
                f"batch={batch_size} encoder={expr_enc} "
                f"gene_disjoint={gene_disjoint} ppi={'nokegg' if ppi_nokegg else 'raw'}",
                flush=True,
            )
            rc = subprocess.call(cmd, cwd=str(root))
            if rc != 0 and not (run_dir / "unseen_samples_test_metrics.json").is_file():
                return rc
            if rc != 0:
                print(f"[fig02] warn: exit={rc} but metrics exist; continuing", flush=True)
    return 0


def cmd_summarize() -> int:
    root = _repo_root()
    _ensure_path(root)
    # 10CV layout: {stage}/fold_XX/unseen_samples_test_metrics.json
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "grn.scripts.summarize_gse253718_ablation_10cv",
    ]
    print("[fig02] RUN", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(root))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig.2 GSE253718 modality ablation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    sub.add_parser("status")
    p_run = sub.add_parser("run")
    p_run.add_argument("--stage", default="ALL", help="stage id / label / ALL")
    p_run.add_argument("--fold", default="ALL", help="1..10 or ALL")
    p_run.add_argument("--epochs", type=int, default=150)
    p_run.add_argument("--force", action="store_true", help="Re-run even if metrics exist.")
    p_run.add_argument(
        "--expression-encoder",
        choices=("conv", "vae"),
        default="conv",
        help="Expression front-end (default conv = original ResNet/CNN).",
    )
    p_run.add_argument(
        "--gene-disjoint",
        action="store_true",
        help="Optional pilot: gene-disjoint pairs (default OFF = legacy pair-split).",
    )
    p_run.add_argument(
        "--ppi-nokegg",
        action="store_true",
        help="Optional pilot: STRING\\KEGG PPI (default OFF = raw STRING).",
    )
    sub.add_parser("summarize")

    args = ap.parse_args()
    if args.cmd == "smoke":
        raise SystemExit(cmd_smoke())
    if args.cmd == "status":
        raise SystemExit(cmd_status())
    if args.cmd == "run":
        raise SystemExit(
            cmd_run(
                args.stage,
                args.fold,
                force=args.force,
                epochs=args.epochs,
                gene_disjoint=bool(args.gene_disjoint),
                ppi_nokegg=bool(args.ppi_nokegg),
                expression_encoder=str(args.expression_encoder),
            )
        )
    if args.cmd == "summarize":
        raise SystemExit(cmd_summarize())
    raise SystemExit(2)


if __name__ == "__main__":
    main()
