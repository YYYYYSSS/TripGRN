# -*- coding: utf-8 -*-
"""Transfer GSE253718 Step1 -> GSE131907 epithelial tLung / mBrain (seeded 1:2:7 column CV).

Uses explicit cv-splits-json (shuffle seed) for reproducible train/val/test columns.
Pair scope: KEGG 1-hop, edge mode pred_diff, case on held-out test columns.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

DOMAINS = {
    "tlung": {
        "label": "tLung_primary",
        "log_npy": "tlung_values_log_normalized.npy",
        "harm_npy": "tlung_values_log_mean_harmonized.npy",
    },
    "mbrain": {
        "label": "mBrain",
        "log_npy": "mbrain_values_log_normalized.npy",
        "harm_npy": "mbrain_values_log_mean_harmonized.npy",
    },
}


def _run(cmd: list[str], label: str) -> None:
    print(f"[transfer-131907] {label}", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"{label} failed exit={proc.returncode}")


def _make_split_json(n: int, seed: int, path: Path, ratios=(1, 2, 7)) -> dict:
    w1, w2, w3 = ratios
    s = w1 + w2 + w3
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    n_tr = int(round(w1 / s * n))
    n_va = int(round(w2 / s * n))
    tr, va, te = idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :]
    payload = {
        "n_cells": int(n),
        "sample_split": list(ratios),
        "cv_base_seed": int(seed),
        "n_folds": 1,
        "folds": [
            {
                "fold": 1,
                "cv_seed": int(seed),
                "train_cols": tr.astype(int).tolist(),
                "val_cols": va.astype(int).tolist(),
                "test_cols": te.astype(int).tolist(),
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "n_test": int(len(te)),
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _harmonize(ref_v, ref_g, target_v, target_g, out_npy, label, force) -> None:
    if out_npy.is_file() and not force:
        print(f"[transfer-131907] reuse harmonized {label}", flush=True)
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
        f"harmonize {label}",
    )


def _finetune(step1, harm_v, genes, out_dir, split_json, force, epochs, lr, patience, label) -> None:
    done = (
        (out_dir / "best_model_fold0.pth").is_file()
        and (out_dir / "run_config.json").is_file()
        and (out_dir / "finetune_config.json").is_file()
    )
    if done and not force:
        print(f"[transfer-131907] skip finetune {label}", flush=True)
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
            "1,2,7",
            "--cv-splits-json",
            str(split_json),
            "--fold-idx",
            "1",
            "--skip-plots",
        ],
        f"finetune {label}",
    )


def _slice_test_harm(harm: Path, split_json: Path, dst: Path) -> None:
    if dst.is_file():
        return
    fold = json.loads(split_json.read_text(encoding="utf-8"))["folds"][0]
    cols = np.asarray(fold["test_cols"], dtype=np.int64)
    x = np.load(harm, mmap_mode="r")
    np.save(dst, np.asarray(x[:, cols], dtype=np.float32))


def _case(tlung_dir, mbrain_dir, t_test, b_test, genes, out_dir, force) -> None:
    used = out_dir / "tables" / "differential_edges_used.csv"
    if used.is_file() and not force:
        print(f"[transfer-131907] skip case (exists {used})", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-u",
            "-m",
            "grn.scripts.case_study_step1_model_brain_nonbrain",
            "--step1-run-dir",
            str(mbrain_dir),
            "--nonbrain-model-path",
            str(tlung_dir / "best_model_fold0.pth"),
            "--brain-model-path",
            str(mbrain_dir / "best_model_fold0.pth"),
            "--nonbrain-exp-npy-values",
            str(t_test),
            "--nonbrain-exp-npy-genes",
            str(genes),
            "--brain-exp-npy-values",
            str(b_test),
            "--brain-exp-npy-genes",
            str(genes),
            "--output-dir",
            str(out_dir),
            "--expression-column-split-part",
            "all",
            "--norm-fit-domain",
            "each",
            "--group-a-label",
            "tLung_primary",
            "--group-b-label",
            "mBrain",
            "--edge-mode",
            "pred_diff",
            "--kegg-k-hops",
            "1",
            "--neg-sample-ratio",
            "0",
            "--inference-batch-size",
            "256",
            "--max-nodes-plot",
            "60",
            "--hub-top-n",
            "15",
        ],
        "differential case",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="GSE253718 -> GSE131907 epi tLung/mBrain transfer")
    ap.add_argument(
        "--phase",
        default="all",
        choices=("prepare", "harmonize", "finetune", "case", "followup", "all"),
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--split-seed", type=int, default=2, help="shuffle seed for 1:2:7 splits")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=4)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    step1 = root / "experiments/benchmark_luad_gse253718_step1"
    ref_v = root / "data/exp/GSE253718_exp_values.npy"
    ref_g = root / "data/exp/GSE253718_exp_genes.npy"
    data_dir = root / "data/exp/GSE131907/precomputed_tlung_mbrain_epi_npy"
    genes = data_dir / "genes.npy"
    exp_root = root / "experiments/gse131907_transfer_from_gse253718_epi_seed2"
    case_out = exp_root / "case_tlung_vs_mbrain_kegg_khop_k1"

    if not (step1 / "best_model_fold0.pth").is_file():
        raise SystemExit(f"missing {step1 / 'best_model_fold0.pth'}")
    if not ref_v.is_file() or not ref_g.is_file():
        raise SystemExit(f"missing ref npy {ref_v}")

    phase = str(args.phase).lower()
    phases = ["prepare", "harmonize", "finetune", "case", "followup"] if phase == "all" else [phase]
    exp_root.mkdir(parents=True, exist_ok=True)
    seed = int(args.split_seed)
    mbrain_seed = seed + 10_000_007

    if "prepare" in phases:
        man = data_dir / "manifest.json"
        if not man.is_file():
            full = root / "data/exp/GSE131907/precomputed_tlung_mbrain_npy/manifest.json"
            if not full.is_file():
                _run(
                    [sys.executable, "-u", "-m", "grn.scripts.materialize_gse131907_tlung_mbrain_npy"],
                    "materialize full tLung",
                )
            _run(
                [sys.executable, "-u", "-m", "grn.scripts.materialize_gse131907_tlung_mbrain_epi_npy"],
                "materialize epithelial npy",
            )

    harm_paths = {}
    if any(p in phases for p in ("harmonize", "finetune", "case")):
        if not genes.is_file():
            raise SystemExit(f"missing {genes}")
        for dom, spec in DOMAINS.items():
            log_p = data_dir / spec["log_npy"]
            harm_p = data_dir / spec["harm_npy"]
            if not log_p.is_file():
                raise SystemExit(f"missing {log_p}")
            harm_paths[dom] = harm_p
            if "harmonize" in phases or ("finetune" in phases and not harm_p.is_file()):
                _harmonize(ref_v, ref_g, log_p, genes, harm_p, dom, bool(args.force))

    split_tlung = exp_root / "split_tlung.json"
    split_mbrain = exp_root / "split_mbrain.json"
    if any(p in phases for p in ("finetune", "case")):
        n_t = int(np.load(harm_paths["tlung"], mmap_mode="r").shape[1])
        n_b = int(np.load(harm_paths["mbrain"], mmap_mode="r").shape[1])
        if not split_tlung.is_file() or args.force:
            _make_split_json(n_t, seed, split_tlung)
        if not split_mbrain.is_file() or args.force:
            _make_split_json(n_b, mbrain_seed, split_mbrain)

    if "finetune" in phases:
        _finetune(
            step1,
            harm_paths["tlung"],
            genes,
            exp_root / "tlung",
            split_tlung,
            bool(args.force),
            int(args.epochs),
            float(args.lr),
            int(args.patience),
            "tlung",
        )
        _finetune(
            step1,
            harm_paths["mbrain"],
            genes,
            exp_root / "mbrain",
            split_mbrain,
            bool(args.force),
            int(args.epochs),
            float(args.lr),
            int(args.patience),
            "mbrain",
        )

    if "case" in phases:
        t_test = exp_root / "tlung_test_harm.npy"
        b_test = exp_root / "mbrain_test_harm.npy"
        _slice_test_harm(harm_paths["tlung"], split_tlung, t_test)
        _slice_test_harm(harm_paths["mbrain"], split_mbrain, b_test)
        _case(exp_root / "tlung", exp_root / "mbrain", t_test, b_test, genes, case_out, bool(args.force))

    if "followup" in phases:
        _run(
            [
                sys.executable,
                "-u",
                "-m",
                "grn.scripts.analyze_gse131907_transfer_case_followup",
                "--exp-root",
                str(exp_root),
            ],
            "followup stats/figures/cytoscape",
        )

    launch = {
        "dataset": "GSE131907",
        "subset": "epithelial tLung vs mBrain",
        "source": "GSE253718_step1",
        "split_seed_tlung": seed,
        "split_seed_mbrain": mbrain_seed,
        "sample_split": "1,2,7",
        "kegg_k_hops": 1,
        "edge_mode": "pred_diff",
        "exp_root": str(exp_root),
        "case_out": str(case_out),
        "data_dir": str(data_dir),
    }
    (exp_root / "transfer_launch_meta.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
    print(f"[transfer-131907] done phase={phase} -> {exp_root}", flush=True)


if __name__ == "__main__":
    main()
