#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrate TripGRN 10-fold CV on epithelial pancancer datasets.

Usage:
  python -m grn.scripts.run_tripgrn_pancancer_10cv --gse ALL --fold ALL
  python -m grn.scripts.run_tripgrn_pancancer_10cv --gse GSE103322 --fold 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np

from grn.data.pancancer_cv import DEFAULT_N_FOLDS, get_fold_split, load_cv_splits
from grn.data.pancancer_paths import (
    PANCANCER_GSE_IDS,
    cds_gene_feature_csv,
    cv_splits_path,
    gse_npy_triplet,
    ppi_string_csv,
    scRNA_layout,
    step01_grn_csv,
    tripgrn_fold_dir,
)
from grn.training.pair_dataloader import suggest_batch_size
from grn.utils.runtime import now_mono, write_runtime_json


def _resolve_folds(fold_arg: str, n_folds: int) -> List[int]:
    fa = str(fold_arg).strip().upper()
    if fa == "ALL":
        return list(range(1, n_folds + 1))
    return [int(fa)]


def _resolve_gses(gse_arg: str) -> List[str]:
    if str(gse_arg).upper() == "ALL":
        return list(PANCANCER_GSE_IDS)
    return [str(gse_arg).strip()]


def _fold_done(run_dir: Path) -> bool:
    return (run_dir / "unseen_samples_test_metrics.json").is_file()


def _train_cols_for_fold(splits_path: Path, fold_idx: int) -> int:
    payload = load_cv_splits(splits_path)
    fold = get_fold_split(payload, fold_idx)
    return int(len(fold["train_cols"]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run TripGRN pancancer epithelial 10-fold CV")
    ap.add_argument("--gse", default="ALL")
    ap.add_argument("--fold", default="ALL", help="Fold index 1..10 or ALL")
    ap.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--force", action="store_true", help="Retrain even if metrics exist")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=0, help="0 = auto from train column count")
    ap.add_argument("--batch-size-cap", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument(
        "--expression-gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (slower, saves VRAM). Off by default for speed.",
    )
    ap.add_argument("--compile-model", action="store_true", help="torch.compile (PyTorch 2+)")
    ap.add_argument(
        "--method-dirname",
        default="TripGRN",
        help="Subdir under kegg_epithelial_10cv/ (default TripGRN). "
        "Use e.g. TripGRN_vae_gene_disjoint for pilots without overwriting Fig.1.",
    )
    ap.add_argument(
        "--no-expression",
        action="store_true",
        help="Modality ablation: disable expression branch (FusionModelAblation).",
    )
    ap.add_argument(
        "--no-cds",
        action="store_true",
        help="Modality ablation: disable CDS/sequence branch.",
    )
    ap.add_argument(
        "--no-ppi",
        action="store_true",
        help="Modality ablation: disable PPI branch.",
    )
    ap.add_argument(
        "--ppi-csv",
        default=None,
        help="Override PPI edge CSV (default: shared STRING score900). "
        "Use decontaminated *_nokegg.csv to remove GRN-overlapping edges.",
    )
    args = ap.parse_args()

    grn_root = Path(__file__).resolve().parents[2]
    layout = scRNA_layout(grn_root)
    grn_csv = step01_grn_csv(grn_root)
    cds_gene = cds_gene_feature_csv(grn_root)
    ppi_csv = Path(args.ppi_csv).expanduser().resolve() if args.ppi_csv else ppi_string_csv(grn_root)
    method_dirname = str(args.method_dirname).strip() or "TripGRN"
    out_base = layout["experiments_root"] / method_dirname
    use_expr = not bool(args.no_expression)
    use_cds = not bool(args.no_cds)
    use_ppi = not bool(args.no_ppi)
    if not (use_expr or use_cds or use_ppi):
        raise ValueError("At least one modality must remain enabled.")
    print(f"[tripgrn-10cv] grn_csv={grn_csv}")
    print(f"[tripgrn-10cv] cds={cds_gene} (enabled={use_cds})")
    print(f"[tripgrn-10cv] ppi={ppi_csv} (enabled={use_ppi})")
    print(f"[tripgrn-10cv] expression enabled={use_expr}")
    print(f"[tripgrn-10cv] experiments={out_base}")

    required = [(grn_csv, "GRN CSV")]
    if use_cds:
        required.append((cds_gene, "CDS"))
    if use_ppi:
        required.append((ppi_csv, "PPI"))
    for p, label in required:
        if not p.is_file():
            raise FileNotFoundError(f"Missing {label}: {p}")

    gses = _resolve_gses(args.gse)
    folds = _resolve_folds(args.fold, int(args.n_folds))
    n_run = 0
    n_skip = 0

    for gse in gses:
        v_path, g_path, _ = gse_npy_triplet("epithelial", gse, grn_root)
        splits_path = cv_splits_path(gse, grn_root)
        if not v_path.is_file():
            raise FileNotFoundError(f"Missing epithelial npy for {gse}: {v_path}")
        if not splits_path.is_file():
            raise FileNotFoundError(f"Missing CV splits for {gse}: {splits_path}")

        for fold_idx in folds:
            if method_dirname == "TripGRN":
                run_dir = tripgrn_fold_dir(gse, fold_idx, grn_root)
            else:
                run_dir = out_base / gse / f"fold_{int(fold_idx):02d}"
            if _fold_done(run_dir) and not args.force:
                print(f"[tripgrn-10cv] skip {gse} fold {fold_idx:02d}")
                n_skip += 1
                continue

            n_train_cols = _train_cols_for_fold(splits_path, fold_idx)
            batch_size = int(args.batch_size)
            if batch_size <= 0:
                batch_size = suggest_batch_size(n_train_cols, cap=int(args.batch_size_cap))

            exp_name = gse
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "grn.scripts.train_unseen_samples_no_cv",
                "--exp-npy-values",
                str(v_path),
                "--exp-npy-genes",
                str(g_path),
                "--grn-csv",
                str(grn_csv),
                "--output-base-dir",
                str(out_base),
                "--experiment-name",
                exp_name,
                "--no-timestamp",
                "--method-name",
                "TripGRN",
                "--cv-splits-json",
                str(splits_path),
                "--fold-idx",
                str(fold_idx),
                "--skip-plots",
                "--seed",
                str(args.seed),
                "--batch-size",
                str(batch_size),
                "--epochs",
                str(args.epochs),
                "--lr",
                str(args.lr),
                "--patience",
                str(args.patience),
                "--neg-sample-ratio",
                "1.0",
                "--amp-bf16",
            ]
            if use_cds:
                cmd.extend(["--cds-gene-feature-csv", str(cds_gene)])
            else:
                cmd.append("--no-cds")
            if use_ppi:
                cmd.extend(["--ppi-csv", str(ppi_csv)])
            else:
                cmd.append("--no-ppi")
            if not use_expr:
                cmd.append("--no-expression")
            if args.expression_gradient_checkpointing:
                cmd.append("--expression-gradient-checkpointing")
            if args.compile_model:
                cmd.append("--compile-model")

            abl = []
            if not use_expr:
                abl.append("no_expr")
            if not use_cds:
                abl.append("no_cds")
            if not use_ppi:
                abl.append("no_ppi")
            abl_tag = "+".join(abl) if abl else "full"
            print(
                f"[tripgrn-10cv] run {gse} fold {fold_idx:02d} "
                f"(train_cols={n_train_cols}, batch_size={batch_size}, ablation={abl_tag})"
            )
            t0 = now_mono()
            proc = subprocess.run(cmd, check=False, cwd=str(grn_root))
            wall = float(now_mono() - t0)
            # Windows/CUDA teardown can return a fatal status even after metrics were written.
            if (run_dir / "unseen_samples_test_metrics.json").is_file():
                if proc.returncode != 0:
                    print(
                        f"[tripgrn-10cv] warn: fold {fold_idx:02d} exit={proc.returncode} "
                        f"but metrics exist; continuing"
                    )
                # Orchestrator wall time (covers train+eval subprocess). Fold script may
                # also write a finer-grained runtime.json; keep orchestrator stamp if absent.
                rt = run_dir / "runtime.json"
                if not rt.is_file():
                    write_runtime_json(
                        run_dir,
                        wall_time_sec=wall,
                        gene_policy="all_genes",
                        extra={
                            "gse_id": gse,
                            "fold": int(fold_idx),
                            "method": "TripGRN",
                            "batch_size": int(batch_size),
                            "n_cells_train": int(n_train_cols),
                            "source": "run_tripgrn_pancancer_10cv",
                        },
                    )
                else:
                    # Always record orchestrator wall alongside existing fields.
                    try:
                        payload = json.loads(rt.read_text(encoding="utf-8"))
                    except Exception:
                        payload = {}
                    payload["orchestrator_wall_time_sec"] = wall
                    payload.setdefault("wall_time_sec", wall)
                    payload.setdefault("gene_policy", "all_genes")
                    rt.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                n_run += 1
            else:
                raise subprocess.CalledProcessError(proc.returncode, cmd)

    print(f"[tripgrn-10cv] finished: ran={n_run}, skipped={n_skip}")


if __name__ == "__main__":
    main()
