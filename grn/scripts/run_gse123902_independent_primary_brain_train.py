# -*- coding: utf-8 -*-
"""Independent TripGRN train for GSE123902 primary vs brain epithelial (no transfer)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from grn.data.gse123902_paths import gse123902_layout
from grn.training.pair_dataloader import suggest_batch_size

DOMAINS = {
    "primary": {"label": "primary_tumour", "npy": "primary_values_log_normalized.npy"},
    "brain": {"label": "brain_met", "npy": "brain_values_log_normalized.npy"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="ALL", help="primary | brain | ALL")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-split", default="5,2,3")
    ap.add_argument("--max-expression-cols", type=int, default=10000)
    ap.add_argument("--expression-col-reduce", default="subsample", choices=("subsample", "avg_pool"))
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--batch-size-cap", type=int, default=64)
    ap.add_argument("--prefer-batch-size", type=int, default=32)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    layout = gse123902_layout(root)
    data = layout["npy_dir"]
    genes = data / "genes.npy"
    grn_csv = root / "data" / "network" / "grn" / "KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv"
    cds = root / "data" / "CDS" / "features" / "gene_shared" / "cds_gene_feature_k4.csv"
    ppi = root / "data" / "network" / "ppi" / "ppi_string_v12_human_score900.csv"
    exp_base = "gse123902_independent_primary_brain_seed42"

    if not genes.is_file():
        raise FileNotFoundError(f"Missing {genes}; run materialize first")
    for p in (grn_csv, cds, ppi):
        if not p.is_file():
            raise FileNotFoundError(p)

    doms = list(DOMAINS.keys()) if str(args.domain).upper() == "ALL" else [str(args.domain).lower()]
    for d in doms:
        if d not in DOMAINS:
            raise SystemExit(f"unknown domain {d}")
        v_path = data / DOMAINS[d]["npy"]
        if not v_path.is_file():
            raise FileNotFoundError(v_path)
        run_dir = root / "experiments" / exp_base / d
        metric = run_dir / "unseen_samples_test_metrics.json"
        if metric.is_file() and not args.force:
            print(f"[train] skip {d} (exists {metric})", flush=True)
            continue

        arr = np.load(v_path, mmap_mode="r")
        n_cols = int(arr.shape[1])
        del arr
        w = [int(x) for x in str(args.sample_split).split(",")]
        n_train = n_cols * w[0] // sum(w)
        n_train_eff = min(n_train, int(args.max_expression_cols)) if int(args.max_expression_cols) > 0 else n_train
        if int(args.batch_size) > 0:
            bs = int(args.batch_size)
        else:
            suggested = suggest_batch_size(n_train_eff, cap=int(args.batch_size_cap))
            bs = min(int(args.batch_size_cap), max(int(suggested), int(args.prefer_batch_size)))

        exp_name = f"{exp_base}/{d}"
        cmd = [
            sys.executable, "-u", "-m", "grn.scripts.train_unseen_samples_no_cv",
            "--exp-npy-values", str(v_path),
            "--exp-npy-genes", str(genes),
            "--grn-csv", str(grn_csv),
            "--cds-gene-feature-csv", str(cds),
            "--ppi-csv", str(ppi),
            "--output-base-dir", str(root / "experiments"),
            "--experiment-name", exp_name,
            "--no-timestamp",
            "--seed", str(args.seed),
            "--sample-split", str(args.sample_split),
            "--batch-size", str(bs),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--patience", str(args.patience),
            "--neg-sample-ratio", "1.0",
            "--skip-plots",
            "--max-expression-cols", str(int(args.max_expression_cols)),
            "--expression-col-reduce", str(args.expression_col_reduce),
            "--amp-bf16",
        ]
        meta = {
            "domain": d,
            "label": DOMAINS[d]["label"],
            "n_cols": n_cols,
            "batch_size": bs,
            "cmd": cmd,
            "dataset": "GSE123902",
            "transfer": False,
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "indep_train_launch_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[train] {d} n_cols={n_cols} batch={bs}", flush=True)
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise SystemExit(f"train failed domain={d} exit={proc.returncode}")
        print(f"[train] done {d} -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
