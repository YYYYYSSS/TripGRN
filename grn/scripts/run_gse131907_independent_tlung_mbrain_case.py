# -*- coding: utf-8 -*-
"""After independent tLung/mBrain training: differential GRN case study (no transfer)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob-diff-threshold", type=float, default=0.5)
    ap.add_argument("--edge-mode", default="prob_threshold")
    ap.add_argument("--inference-batch-size", type=int, default=64)
    ap.add_argument(
        "--pair-mode",
        choices=("de_cartesian", "grn_default"),
        default="de_cartesian",
        help="de_cartesian: KEGG gene-panel ordered pairs (~1e7); grn_default: training GRN edges+negatives only",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    exp = root / "experiments" / "gse131907_independent_tlung_mbrain_seed42"
    tlung = exp / "tlung"
    mbrain = exp / "mbrain"
    data = root / "data" / "exp" / "GSE131907" / "precomputed_tlung_mbrain_npy"
    pre = root / "data" / "exp" / "GSE131907" / "precomputed_brain_nonbrain_npy"

    for d in (tlung, mbrain):
        if not (d / "best_model_fold0.pth").is_file():
            raise SystemExit(f"missing checkpoint under {d}; train first")
        if not (d / "run_config.json").is_file():
            raise SystemExit(f"missing run_config under {d}")

    tlung_vals = data / "tlung_values_log_normalized.npy"
    genes = data / "genes.npy" if (data / "genes.npy").is_file() else pre / "genes.npy"
    mbrain_vals = pre / "brain_values_log_normalized.npy"
    ptr = data / "mbrain_values_log_normalized.path.txt"
    if ptr.is_file():
        mbrain_vals = Path(ptr.read_text(encoding="utf-8").strip())

    out = exp / "case_tlung_vs_mbrain"
    out.mkdir(parents=True, exist_ok=True)

    # Prefer mBrain run_config as step1-run-dir template (same architecture)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "grn.scripts.case_study_step1_model_brain_nonbrain",
        "--step1-run-dir",
        str(mbrain),
        "--nonbrain-model-path",
        str(tlung / "best_model_fold0.pth"),
        "--brain-model-path",
        str(mbrain / "best_model_fold0.pth"),
        "--nonbrain-exp-npy-values",
        str(tlung_vals),
        "--nonbrain-exp-npy-genes",
        str(genes),
        "--brain-exp-npy-values",
        str(mbrain_vals),
        "--brain-exp-npy-genes",
        str(genes),
        "--output-dir",
        str(out),
        "--expression-sample-split",
        "5,2,3",
        "--expression-column-split-part",
        "test",
        "--norm-fit-domain",
        "each",
        "--group-a-label",
        "tLung_primary",
        "--group-b-label",
        "mBrain",
        "--edge-mode",
        str(args.edge_mode),
        "--prob-diff-threshold",
        str(args.prob_diff_threshold),
        "--inference-batch-size",
        str(args.inference_batch_size),
        "--max-expression-cols",
        "10000",
        "--expression-col-seed",
        "42",
        "--network-plot-title",
        "Independent models: tLung primary vs mBrain (no transfer)",
    ]
    if args.pair_mode == "de_cartesian":
        gene_list = root / "data" / "network" / "grn" / "KEGG_hsa_PPrel_all_symbols_gene_list.csv"
        if not gene_list.is_file():
            raise FileNotFoundError(gene_list)
        cmd.extend(["--de-cartesian-csv", str(gene_list), "--de-use-all-rows"])

    meta = {"cmd": cmd, "pair_mode": args.pair_mode, "transfer": False, "init": "independent_from_scratch"}
    (out / "case_launch_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[indep-case] launching case study", flush=True)
    proc = subprocess.run(cmd)
    used = out / "tables" / "differential_edges_used.csv"
    if proc.returncode != 0 and not used.is_file():
        raise SystemExit(f"case study failed exit={proc.returncode}")
    if proc.returncode != 0 and used.is_file():
        print(
            f"[indep-case] WARN process exit={proc.returncode} but differential edges exist; treat as OK",
            flush=True,
        )
    print(f"[indep-case] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
