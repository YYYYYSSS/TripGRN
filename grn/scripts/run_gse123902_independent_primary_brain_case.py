# -*- coding: utf-8 -*-
"""Differential GRN case: GSE123902 independently trained primary vs brain."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "GSE123902 primary vs brain differential case. "
            "Prefer preset pair scopes (train protocol or KEGG k-hop neighborhoods) "
            "over the full gene Cartesian product."
        )
    )
    ap.add_argument("--prob-diff-threshold", type=float, default=0.5)
    ap.add_argument("--edge-mode", default="pred_diff")
    ap.add_argument("--inference-batch-size", type=int, default=256)
    ap.add_argument(
        "--pair-mode",
        choices=("grn_default", "kegg_pos", "kegg_khop", "de_cartesian"),
        default="kegg_khop",
        help=(
            "kegg_khop: KEGG undirected k-hop neighborhood pairs from GRN sources "
            "(requires --kegg-k-hops; recommended default); "
            "grn_default: train protocol (KEGG pos + seeded neg); "
            "kegg_pos: KEGG labeled edges only; "
            "de_cartesian: full ordered Cartesian (legacy / calibration only)."
        ),
    )
    ap.add_argument(
        "--kegg-k-hops",
        type=int,
        default=1,
        help="For --pair-mode kegg_khop: neighborhood radius (1/2/3).",
    )
    ap.add_argument(
        "--neg-sample-ratio",
        type=float,
        default=None,
        help=(
            "Override negative sampling ratio for grn/kegg_khop modes. "
            "For kegg_khop default is 0 (neighborhood pairs only)."
        ),
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Override case output directory. Defaults depend on --pair-mode.",
    )
    args = ap.parse_args()

    if args.pair_mode == "kegg_khop" and int(args.kegg_k_hops) < 1:
        raise SystemExit("--kegg-k-hops must be >= 1 for kegg_khop")

    root = Path(__file__).resolve().parents[2]
    exp = root / "experiments" / "gse123902_independent_primary_brain_seed42"
    primary = exp / "primary"
    brain = exp / "brain"
    data = root / "data" / "exp" / "GSE123902" / "precomputed_primary_brain_npy"

    for d in (primary, brain):
        if not (d / "best_model_fold0.pth").is_file():
            raise SystemExit(f"missing checkpoint under {d}; train first")
        if not (d / "run_config.json").is_file():
            raise SystemExit(f"missing run_config under {d}")

    primary_vals = data / "primary_values_log_normalized.npy"
    brain_vals = data / "brain_values_log_normalized.npy"
    genes = data / "genes.npy"
    for p in (primary_vals, brain_vals, genes):
        if not p.is_file():
            raise FileNotFoundError(p)

    if args.pair_mode == "kegg_khop":
        default_out = exp / f"case_primary_vs_brain_kegg_khop_k{int(args.kegg_k_hops)}"
    else:
        default_out = {
            "grn_default": exp / "case_primary_vs_brain_grn_pairs",
            "kegg_pos": exp / "case_primary_vs_brain_kegg_pos",
            "de_cartesian": exp / "case_primary_vs_brain",
        }[args.pair_mode]
    out = Path(args.output_dir) if args.output_dir else default_out
    out.mkdir(parents=True, exist_ok=True)

    title = f"GSE123902 independent primary vs brain ({args.pair_mode}"
    if args.pair_mode == "kegg_khop":
        title += f" k={int(args.kegg_k_hops)}"
    title += ")"

    cmd = [
        sys.executable, "-u", "-m", "grn.scripts.case_study_step1_model_brain_nonbrain",
        "--step1-run-dir", str(brain),
        "--nonbrain-model-path", str(primary / "best_model_fold0.pth"),
        "--brain-model-path", str(brain / "best_model_fold0.pth"),
        "--nonbrain-exp-npy-values", str(primary_vals),
        "--nonbrain-exp-npy-genes", str(genes),
        "--brain-exp-npy-values", str(brain_vals),
        "--brain-exp-npy-genes", str(genes),
        "--output-dir", str(out),
        "--expression-sample-split", "5,2,3",
        "--expression-column-split-part", "test",
        "--norm-fit-domain", "each",
        "--group-a-label", "primary_tumour",
        "--group-b-label", "brain_met",
        "--edge-mode", str(args.edge_mode),
        "--prob-diff-threshold", str(args.prob_diff_threshold),
        "--inference-batch-size", str(args.inference_batch_size),
        "--max-expression-cols", "10000",
        "--expression-col-seed", "42",
        "--network-plot-title", title,
        "--max-nodes-plot", "60",
    ]
    if args.pair_mode == "de_cartesian":
        gene_list = root / "data" / "network" / "grn" / "KEGG_hsa_PPrel_all_symbols_gene_list.csv"
        if not gene_list.is_file():
            raise FileNotFoundError(gene_list)
        cmd.extend(["--de-cartesian-csv", str(gene_list), "--de-use-all-rows"])
    elif args.pair_mode == "kegg_pos":
        cmd.extend(["--neg-sample-ratio", "0"])
    elif args.pair_mode == "kegg_khop":
        neg = 0.0 if args.neg_sample_ratio is None else float(args.neg_sample_ratio)
        cmd.extend(
            [
                "--kegg-k-hops",
                str(int(args.kegg_k_hops)),
                "--neg-sample-ratio",
                str(neg),
            ]
        )
    elif args.neg_sample_ratio is not None:
        cmd.extend(["--neg-sample-ratio", str(float(args.neg_sample_ratio))])
    # grn_default: omit de-cartesian -> GRN positives + run_config neg ratio

    meta = {
        "cmd": cmd,
        "pair_mode": args.pair_mode,
        "kegg_k_hops": int(args.kegg_k_hops) if args.pair_mode == "kegg_khop" else None,
        "neg_sample_ratio": (
            (0.0 if args.neg_sample_ratio is None else float(args.neg_sample_ratio))
            if args.pair_mode == "kegg_khop"
            else args.neg_sample_ratio
        ),
        "dataset": "GSE123902",
        "transfer": False,
        "rationale": (
            "Preset pair scopes: train protocol or KEGG k-hop neighborhoods; "
            "avoid full gene Cartesian product for biological differential networks."
        ),
    }
    (out / "case_launch_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[case] pair_mode={args.pair_mode} -> {out}", flush=True)
    print("[case] launching", flush=True)
    proc = subprocess.run(cmd)
    used = out / "tables" / "differential_edges_used.csv"
    if proc.returncode != 0 and not used.is_file():
        raise SystemExit(f"case study failed exit={proc.returncode}")
    if proc.returncode != 0 and used.is_file():
        print(f"[case] WARN exit={proc.returncode} but edges exist; treat as OK", flush=True)
    print(f"[case] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
