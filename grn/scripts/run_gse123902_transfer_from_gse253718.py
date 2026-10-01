# -*- coding: utf-8 -*-
"""Transfer TripGRN from GSE253718 Step1 to GSE123902 primary/brain (10/20/70 column split).

Pipeline:
  1. (optional) materialize GSE123902 epithelial npy
  2. mean-shift harmonize primary/brain expression to GSE253718 reference
  3. fusion-head finetune per compartment (train 10%%, val 20%%, test 70%%)
  4. differential GRN case on held-out 70%% columns (KEGG 1-hop pairs, pred_diff)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

DOMAINS = {
    "primary": {
        "label": "primary_tumour",
        "log_npy": "primary_values_log_normalized.npy",
        "harm_npy": "primary_values_log_mean_harmonized.npy",
    },
    "brain": {
        "label": "brain_met",
        "log_npy": "brain_values_log_normalized.npy",
        "harm_npy": "brain_values_log_mean_harmonized.npy",
    },
}


def _run(cmd: List[str], *, label: str) -> None:
    print(f"[transfer-123902] {label}", flush=True)
    print(f"[transfer-123902] cmd: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"{label} failed exit={proc.returncode}")


def _ensure_npy(data_dir: Path, genes: Path) -> None:
    missing = [data_dir / DOMAINS[d]["log_npy"] for d in DOMAINS]
    if genes.is_file() and all(p.is_file() for p in missing):
        return
    print("[transfer-123902] materialize primary/brain epithelial npy ...", flush=True)
    _run(
        [sys.executable, "-u", "-m", "grn.scripts.materialize_gse123902_primary_brain_epi_npy"],
        label="materialize_gse123902_primary_brain_epi_npy",
    )
    if not genes.is_file():
        raise FileNotFoundError(genes)
    for d in DOMAINS:
        p = data_dir / DOMAINS[d]["log_npy"]
        if not p.is_file():
            raise FileNotFoundError(p)


def _harmonize(
    *,
    ref_v: Path,
    ref_g: Path,
    target_v: Path,
    target_g: Path,
    out_npy: Path,
    label: str,
) -> None:
    if out_npy.is_file():
        print(f"[transfer-123902] reuse harmonized {out_npy}", flush=True)
        return
    _run(
        [
            sys.executable,
            "-u",
            "-m",
            "grn.scripts.harmonize_target_to_reference_npy",
            "--ref-npy-values",
            str(ref_v),
            "--ref-npy-genes",
            str(ref_g),
            "--target-npy-values",
            str(target_v),
            "--target-npy-genes",
            str(target_g),
            "--out-npy",
            str(out_npy),
            "--method",
            "mean_shift",
            "--ref-max-cols",
            "4000",
            "--col-chunk",
            "256",
        ],
        label=f"harmonize {label}",
    )


def _finetune(
    *,
    step1: Path,
    harm_v: Path,
    genes: Path,
    out_dir: Path,
    sample_split: str,
    force: bool,
    epochs: int,
    lr: float,
    patience: int,
    label: str,
) -> None:
    metric = out_dir / "unseen_samples_test_metrics.json"
    if metric.is_file() and not force:
        print(f"[transfer-123902] skip finetune {label} (exists {metric})", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-u",
            "-m",
            "grn.scripts.finetune_step1_fusion_head_external",
            "--step1-run-dir",
            str(step1),
            "--external-exp-npy-values",
            str(harm_v),
            "--external-exp-npy-genes",
            str(genes),
            "--output-dir",
            str(out_dir),
            "--expression-norm-fit-all-columns",
            "--epochs",
            str(epochs),
            "--lr",
            str(lr),
            "--patience",
            str(patience),
            "--early-stop-metric",
            "macro_f1",
            "--sample-split",
            sample_split,
            "--eval-posthoc-calibrate",
        ],
        label=f"finetune {label}",
    )


def _zeroshot(
    *,
    step1: Path,
    harm_v: Path,
    genes: Path,
    out_dir: Path,
    sample_split: str,
    force: bool,
    label: str,
) -> None:
    metric = out_dir / "unseen_samples_test_metrics.json"
    if metric.is_file() and not force:
        print(f"[transfer-123902] skip zeroshot {label}", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-u",
            "-m",
            "grn.scripts.evaluate_step1_on_external_expression",
            "--step1-run-dir",
            str(step1),
            "--external-exp-npy-values",
            str(harm_v),
            "--external-exp-npy-genes",
            str(genes),
            "--output-base-dir",
            str(out_dir.parent),
            "--experiment-name",
            out_dir.name,
            "--no-timestamp",
            "--expression-norm-fit-all-columns",
            "--sample-split",
            sample_split,
            "--posthoc-calibrate",
        ],
        label=f"zeroshot {label}",
    )


def _case(
    *,
    step1: Path,
    primary_dir: Path,
    brain_dir: Path,
    primary_h: Path,
    brain_h: Path,
    genes: Path,
    out_dir: Path,
    sample_split: str,
    kegg_k_hops: int,
    force: bool,
) -> None:
    used = out_dir / "tables" / "differential_edges_used.csv"
    if used.is_file() and not force:
        print(f"[transfer-123902] skip case (exists {used})", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    title = (
        f"GSE123902 transfer GSE253718 primary vs brain "
        f"(kegg_khop k={kegg_k_hops}, test cols {sample_split})"
    )
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "grn.scripts.case_study_step1_model_brain_nonbrain",
        "--step1-run-dir",
        str(brain_dir),
        "--nonbrain-model-path",
        str(primary_dir / "best_model_fold0.pth"),
        "--brain-model-path",
        str(brain_dir / "best_model_fold0.pth"),
        "--nonbrain-exp-npy-values",
        str(primary_h),
        "--nonbrain-exp-npy-genes",
        str(genes),
        "--brain-exp-npy-values",
        str(brain_h),
        "--brain-exp-npy-genes",
        str(genes),
        "--output-dir",
        str(out_dir),
        "--expression-sample-split",
        sample_split,
        "--expression-column-split-part",
        "test",
        "--norm-fit-domain",
        "each",
        "--group-a-label",
        "primary_tumour",
        "--group-b-label",
        "brain_met",
        "--edge-mode",
        "pred_diff",
        "--prob-diff-threshold",
        "0.5",
        "--inference-batch-size",
        "256",
        "--kegg-k-hops",
        str(int(kegg_k_hops)),
        "--neg-sample-ratio",
        "0",
        "--network-plot-title",
        title,
        "--max-nodes-plot",
        "60",
    ]
    meta = {
        "transfer": True,
        "source_step1": str(step1),
        "sample_split": sample_split,
        "discovery_column_part": "test",
        "pair_mode": "kegg_khop",
        "kegg_k_hops": int(kegg_k_hops),
        "edge_mode": "pred_diff",
        "cmd": cmd,
    }
    (out_dir / "case_launch_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _run(cmd, label="differential case kegg_khop")


def main() -> None:
    ap = argparse.ArgumentParser(description="GSE253718 -> GSE123902 transfer + differential case")
    ap.add_argument(
        "--phase",
        default="all",
        choices=("prepare", "harmonize", "finetune", "zeroshot", "case", "all"),
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sample-split", default="1,2,7", help="train:val:test = 10:20:70")
    ap.add_argument("--kegg-k-hops", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    step1 = root / "experiments" / "benchmark_luad_gse253718_step1"
    ref_v = root / "data" / "exp" / "GSE253718_exp_values.npy"
    ref_g = root / "data" / "exp" / "GSE253718_exp_genes.npy"
    data_dir = root / "data" / "exp" / "GSE123902" / "precomputed_primary_brain_npy"
    genes = data_dir / "genes.npy"
    exp_root = root / "experiments" / "gse123902_transfer_from_gse253718_seed42"

    ckpt = step1 / "best_model_fold0.pth"
    if not ckpt.is_file():
        raise SystemExit(
            f"Missing Step1 checkpoint {ckpt}. "
            "Train GSE253718 Step1 first (step01 / benchmark_luad_gse253718_step1)."
        )
    if not ref_v.is_file() or not ref_g.is_file():
        raise SystemExit(f"Missing GSE253718 reference npy: {ref_v} / {ref_g}")

    phase = str(args.phase).lower()
    phases = ["prepare", "harmonize", "finetune", "zeroshot", "case"] if phase == "all" else [phase]

    launch = {
        "dataset": "GSE123902",
        "source": "GSE253718_step1",
        "sample_split": str(args.sample_split),
        "kegg_k_hops": int(args.kegg_k_hops),
        "seed": int(args.seed),
        "exp_root": str(exp_root),
    }
    exp_root.mkdir(parents=True, exist_ok=True)

    if "prepare" in phases:
        merged = root / "data" / "exp" / "GSE123902" / "merged" / "counts_values.npy"
        if not merged.is_file():
            print("[transfer-123902] merged counts missing; run step08 prepare/singler first", flush=True)
            _run(
                [sys.executable, "-u", "-m", "grn.scripts.prepare_gse123902_from_raw_tar", "--sites", "primary,brain"],
                label="prepare_gse123902_from_raw_tar",
            )
            _run(
                [sys.executable, "-u", "-m", "grn.scripts.preprocess_gse123902_singler"],
                label="preprocess_gse123902_singler",
            )
        _ensure_npy(data_dir, genes)

    harm_paths = {}
    if "harmonize" in phases or "finetune" in phases or "zeroshot" in phases or "case" in phases:
        _ensure_npy(data_dir, genes)
        for dom, spec in DOMAINS.items():
            log_p = data_dir / spec["log_npy"]
            harm_p = data_dir / spec["harm_npy"]
            harm_paths[dom] = harm_p
            if "harmonize" in phases or "finetune" in phases or "zeroshot" in phases or "case" in phases:
                _harmonize(
                    ref_v=ref_v,
                    ref_g=ref_g,
                    target_v=log_p,
                    target_g=genes,
                    out_npy=harm_p,
                    label=dom,
                )

    finetune_dirs = {}
    if "finetune" in phases:
        for dom in DOMAINS:
            out_d = exp_root / dom
            finetune_dirs[dom] = out_d
            _finetune(
                step1=step1,
                harm_v=harm_paths[dom],
                genes=genes,
                out_dir=out_d,
                sample_split=str(args.sample_split),
                force=bool(args.force),
                epochs=int(args.epochs),
                lr=float(args.lr),
                patience=int(args.patience),
                label=dom,
            )

    if "zeroshot" in phases:
        for dom in DOMAINS:
            out_d = exp_root / f"zeroshot_{dom}"
            _zeroshot(
                step1=step1,
                harm_v=harm_paths[dom],
                genes=genes,
                out_dir=out_d,
                sample_split=str(args.sample_split),
                force=bool(args.force),
                label=dom,
            )

    if "case" in phases:
        primary_dir = exp_root / "primary"
        brain_dir = exp_root / "brain"
        for d, p in (("primary", primary_dir), ("brain", brain_dir)):
            if not (p / "best_model_fold0.pth").is_file():
                raise SystemExit(f"Missing finetuned checkpoint for {d}: {p / 'best_model_fold0.pth'}")
            if not (p / "run_config.json").is_file():
                raise SystemExit(f"Missing run_config for {d}: {p / 'run_config.json'}")
        case_out = exp_root / f"case_primary_vs_brain_kegg_khop_k{int(args.kegg_k_hops)}"
        _case(
            step1=step1,
            primary_dir=primary_dir,
            brain_dir=brain_dir,
            primary_h=harm_paths["primary"],
            brain_h=harm_paths["brain"],
            genes=genes,
            out_dir=case_out,
            sample_split=str(args.sample_split),
            kegg_k_hops=int(args.kegg_k_hops),
            force=bool(args.force),
        )
        launch["case_out"] = str(case_out)

    (exp_root / "transfer_launch_meta.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
    print(f"[transfer-123902] done phase={phase} -> {exp_root}", flush=True)


if __name__ == "__main__":
    main()
