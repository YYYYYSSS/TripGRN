#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute TripGRN/DeepFGRN strict binary (regulated vs not) AUC/AUPRC from checkpoints.

Binary positive = activation OR repression.
Score = 1 - P(NoReg)  (= P(Act)+P(Rep) under softmax).

ROC-AUC equals NoReg OvR AUC at train time; this script re-infers test probs so that
AUPRC(regulated) is exact, and also records AUC from the same scores.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from grn.data.pancancer_paths import (
    PANCANCER_GSE_IDS,
    fig01_root,
    grn_root_from_here,
    step01_grn_csv,
)

METHODS = ("TripGRN", "DeepFGRN")


def _build_pair_samples_genome_wide(grn_matrix, neg_ratio: float, seed: int):
    """Legacy TripGRN negatives: any non-edge gene pair (not positive-universe matched).

    Fig.1 Trip checkpoints were trained with this sampler; matched-universe rebuild
    yields a different held-out test set and breaks AUC==NoReg-OvR checks.
    """
    import torch

    rng = np.random.RandomState(seed)
    pos = torch.where(grn_matrix > 0)
    pos_pairs = [
        (int(i.item()), int(j.item()), int(grn_matrix[i, j].item()))
        for i, j in zip(pos[0], pos[1])
    ]
    n_pos = len(pos_pairs)
    n_neg = int(n_pos * neg_ratio)
    num_genes = int(grn_matrix.shape[0])
    existing = set((i, j) for i, j, _ in pos_pairs)
    neg_pairs = []
    while len(neg_pairs) < n_neg:
        i = int(rng.randint(0, num_genes))
        j = int(rng.randint(0, num_genes))
        if (i, j) in existing:
            continue
        if int(grn_matrix[i, j].item()) != 0:
            continue
        neg_pairs.append((i, j, 0))
        existing.add((i, j))
    return pos_pairs + neg_pairs


def _binary_ranking(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """y_prob columns: [NoReg, Rep/Act, ...] with NoReg at index 0."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2:
        return float("nan"), float("nan")
    # Prefer 1-p0 to stay robust if later columns are Act/Rep in either order.
    p_reg = 1.0 - p[:, 0]
    y_reg = (y != 0).astype(int)
    if len(np.unique(y_reg)) < 2:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(y_reg, p_reg))
    auprc = float(average_precision_score(y_reg, p_reg))
    return auc, auprc


def _trip_one(fold_dir: str) -> Dict[str, Any]:
    from grn.data.pancancer_cv import get_fold_split, load_cv_splits
    import grn.scripts.regenerate_unseen_run_plots_from_checkpoint as regen
    from grn.scripts.regenerate_unseen_run_plots_from_checkpoint import (
        _build_pipeline,
        _validate_args_like_train,
    )
    from grn.scripts.train_unseen_samples_no_cv import _predict_arrays, set_seed
    import torch

    run = Path(fold_dir)
    cfg = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    args = Namespace(**cfg)
    _validate_args_like_train(args)
    set_seed(int(args.seed))
    # Must use the same shared CV cell columns as training; default split3 is wrong for Fig.1.
    column_splits = None
    if getattr(args, "cv_splits_json", None) and int(getattr(args, "fold_idx", 0) or 0) >= 1:
        cv_payload = load_cv_splits(Path(str(args.cv_splits_json)))
        fold = get_fold_split(cv_payload, int(args.fold_idx))
        column_splits = (
            np.asarray(fold["train_cols"], dtype=np.int64),
            np.asarray(fold["val_cols"], dtype=np.int64),
            np.asarray(fold["test_cols"], dtype=np.int64),
        )
    # Trip Fig.1 folds used genome-wide negatives at train time.
    _orig_pairs = regen._build_pair_samples
    regen._build_pair_samples = _build_pair_samples_genome_wide  # type: ignore[attr-defined]
    try:
        model, _, _, test_loader, device, use_amp = _build_pipeline(
            args, column_splits=column_splits
        )
    finally:
        regen._build_pair_samples = _orig_pairs  # type: ignore[attr-defined]
    state = torch.load(str(run / "best_model_fold0.pth"), map_location=device)
    model.load_state_dict(state)
    model.eval()
    y_true, y_prob = _predict_arrays(model, test_loader, device, use_amp_bf16=use_amp)
    auc, auprc = _binary_ranking(y_true, y_prob)
    # parse gse/fold from path .../TripGRN/GSExxx/fold_YY
    fold_idx = int(run.name.split("_")[1])
    gse = run.parent.name
    return {
        "method": "TripGRN",
        "gse_id": gse,
        "fold": fold_idx,
        "auc": auc,
        "auprc": auprc,
        "status": "ok",
        "fold_dir": str(run),
        "used_cv_splits": bool(column_splits is not None),
        "negative_protocol": "genome_wide_legacy_trip",
        "n_test": int(len(y_true)),
    }


def _deep_one(payload: Dict[str, str]) -> Dict[str, Any]:
    """Run DeepFGRN cm-only for one fold and compute binary ranking from exported probs."""
    import subprocess
    import tempfile

    fold_dir = Path(payload["fold_dir"])
    grn_root = Path(payload["grn_root"])
    gse = payload["gse_id"]
    fold = int(payload["fold"])
    exp_name = payload["experiment_name"]

    from grn.data.pancancer_paths import cv_splits_path, gse_npy_triplet

    v_path, g_path, _ = gse_npy_triplet("epithelial", gse, grn_root)
    cv_path = cv_splits_path(gse, grn_root)
    metrics_path = fold_dir / "unseen_samples_test_metrics.json"
    cm_path = fold_dir / "confusion_matrix_test.json"
    metrics_bak = metrics_path.read_bytes() if metrics_path.is_file() else None
    cm_bak = cm_path.read_bytes() if cm_path.is_file() else None
    with tempfile.TemporaryDirectory(prefix="deep_bin_") as tmp:
        tmp_p = Path(tmp)
        cmd = [
            os.environ.get("PROJECT_GRN_PYTHON", "python"),
            "-u",
            "-m",
            "grn.scripts.run_deepfgrn_pancancer_cell_unseen",
            "--cm-only",
            "--skip-summary-plots",
            "--gse-id",
            gse,
            "--fold-idx",
            str(fold),
            "--cv-splits-json",
            str(cv_path),
            "--exp-npy-values",
            str(v_path),
            "--exp-npy-genes",
            str(g_path),
            "--output-fold-dir",
            str(fold_dir),
            "--run-dir",
            str(fold_dir),
            "--export-test-probs",
            str(tmp_p),
        ]
        # Prefer the same interpreter
        cmd[0] = str(Path(payload["python"]))
        env = os.environ.copy()
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        env["STEP01_EXPERIMENTS_NAME"] = str(exp_name)
        # Never rewrite fold metrics during AUPRC recompute (cm-only uses wrong GRN if mis-set).
        env["DEEPFGRN_PRESERVE_METRICS"] = "1"
        grn_csv = payload.get("grn_csv") or ""
        if not grn_csv and "dorothea" in str(exp_name).lower():
            raise RuntimeError(
                f"DoRothEA Deep AUPRC recompute requires grn_csv in payload (got experiment={exp_name})"
            )
        if grn_csv:
            env["STEP01_GRN_CSV"] = str(grn_csv)
            cmd.extend(["--kegg-csv", str(grn_csv)])
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(grn_root),
                env=env,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        finally:
            # Always restore original metrics/CM even if subprocess rewrote them.
            if metrics_bak is not None:
                metrics_path.write_bytes(metrics_bak)
            if cm_bak is not None:
                cm_path.write_bytes(cm_bak)
        y_true_p = tmp_p / "test_y_true.npy"
        y_prob_p = tmp_p / "test_y_prob.npy"
        if not y_true_p.is_file() or not y_prob_p.is_file():
            return {
                "method": "DeepFGRN",
                "gse_id": gse,
                "fold": fold,
                "auc": float("nan"),
                "auprc": float("nan"),
                "status": f"fail_exit={proc.returncode}",
                "fold_dir": str(fold_dir),
                "stderr": (proc.stderr or "")[-2000:],
                "grn_csv": str(grn_csv) if grn_csv else "",
            }
        y_true = np.load(y_true_p)
        y_prob = np.load(y_prob_p)
        # DeepFGRN class order: NoReg, Repression, Activation -> NoReg at 0
        auc, auprc = _binary_ranking(y_true, y_prob)
        return {
            "method": "DeepFGRN",
            "gse_id": gse,
            "fold": fold,
            "auc": auc,
            "auprc": auprc,
            "status": "ok",
            "fold_dir": str(fold_dir),
            "experiment_name": exp_name,
            "grn_csv": str(grn_csv) if grn_csv else "",
        }


def _discover_jobs(
    exp_root: Path, *, grn_csv: Optional[Path] = None
) -> Tuple[List[str], List[Dict[str, str]]]:
    trip_dirs: List[str] = []
    deep_jobs: List[Dict[str, str]] = []
    for method in METHODS:
        mroot = exp_root / method
        if not mroot.is_dir():
            continue
        for gse in PANCANCER_GSE_IDS:
            for fold in range(1, 11):
                fd = mroot / gse / f"fold_{fold:02d}"
                if method == "TripGRN":
                    if (fd / "best_model_fold0.pth").is_file() and (fd / "run_config.json").is_file():
                        trip_dirs.append(str(fd))
                else:
                    if (fd / "best_model.keras").is_file():
                        job = {
                            "fold_dir": str(fd),
                            "gse_id": gse,
                            "fold": str(fold),
                            "experiment_name": exp_root.name,
                        }
                        if grn_csv is not None:
                            job["grn_csv"] = str(grn_csv)
                        deep_jobs.append(job)
    return trip_dirs, deep_jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-name", action="append", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--methods", default="TripGRN,DeepFGRN")
    ap.add_argument("--max-jobs", type=int, default=0, help="Debug: limit jobs per method (0=all)")
    args = ap.parse_args()
    root = grn_root_from_here()
    py = str(Path(os.environ.get("PROJECT_GRN_PYTHON", "")).resolve()) if os.environ.get("PROJECT_GRN_PYTHON") else None
    if not py or not Path(py).is_file():
        import sys

        py = sys.executable

    exps = args.experiment_name or [
        "kegg_epithelial_10cv",
        "dorothea_abc_epithelial_10cv",
    ]
    want = {m.strip() for m in str(args.methods).split(",") if m.strip()}

    for exp in exps:
        # Fig.1 layout: fig01_pancancer_benchmark/experiments/<exp>/
        exp_root = fig01_root(root) / "experiments" / exp
        if not exp_root.is_dir():
            # Legacy flat experiments/ fallback
            alt = root / "experiments" / exp
            exp_root = alt if alt.is_dir() else exp_root
        out_cache = exp_root / "summary" / "trip_deep_strict_binary_ranking_cache.csv"
        out_cache.parent.mkdir(parents=True, exist_ok=True)
        grn_csv: Optional[Path] = None
        if "dorothea" in exp.lower():
            # Must match fig01 run_fig01_deepfgrn_dorothea._dorothea_csv order:
            # training embeddings are keyed by GRN stem (DoRothEA_human_ABC_... vs dorothea_abc_...).
            candidates = [
                root / "shared" / "data" / "network" / "grn" / "DoRothEA_human_ABC_activation_inhibition_with_symbols.csv",
                root / "shared" / "data" / "network" / "grn" / "dorothea_abc_activation_inhibition.csv",
                fig01_root(root)
                / "data"
                / "network"
                / "grn"
                / "DoRothEA_human_ABC_activation_inhibition_with_symbols.csv",
                fig01_root(root) / "data" / "network" / "grn" / "dorothea_abc_activation_inhibition.csv",
            ]
            for p in candidates:
                if p.is_file():
                    grn_csv = p
                    break
            if grn_csv is None:
                raise FileNotFoundError(
                    f"DoRothEA GRN CSV not found for {exp}; refused to fall back to KEGG"
                )
            print(f"[DeepFGRN] using GRN CSV: {grn_csv}")
        trip_dirs, deep_jobs = _discover_jobs(exp_root, grn_csv=grn_csv)
        if args.max_jobs > 0:
            trip_dirs = trip_dirs[: args.max_jobs]
            deep_jobs = deep_jobs[: args.max_jobs]
        rows: List[Dict[str, Any]] = []
        if out_cache.is_file():
            prev = pd.read_csv(out_cache)
            rows.extend(prev.to_dict(orient="records"))
            done = {(str(r.get("method")), str(r.get("gse_id")), int(r.get("fold"))) for r in rows if str(r.get("status")) == "ok"}
        else:
            done = set()

        print(f"==== {exp} cache={out_cache} done={len(done)} ====")

        if "TripGRN" in want:
            todo = [d for d in trip_dirs if ("TripGRN", Path(d).parent.name, int(Path(d).name.split("_")[1])) not in done]
            print(f"[TripGRN] todo={len(todo)}")
            if args.workers <= 1:
                for d in todo:
                    try:
                        rows.append(_trip_one(d))
                        print(f"  OK Trip {rows[-1]['gse_id']} fold{rows[-1]['fold']} auc={rows[-1]['auc']:.4f} auprc={rows[-1]['auprc']:.4f}")
                    except Exception as exc:
                        print(f"  FAIL Trip {d}: {exc}")
                        traceback.print_exc()
                        rows.append(
                            {
                                "method": "TripGRN",
                                "gse_id": Path(d).parent.name,
                                "fold": int(Path(d).name.split("_")[1]),
                                "auc": float("nan"),
                                "auprc": float("nan"),
                                "status": f"error:{type(exc).__name__}",
                                "fold_dir": d,
                            }
                        )
                    pd.DataFrame(rows).to_csv(out_cache, index=False, encoding="utf-8-sig")
            else:
                with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
                    futs = {ex.submit(_trip_one, d): d for d in todo}
                    for fut in as_completed(futs):
                        d = futs[fut]
                        try:
                            rows.append(fut.result())
                            print(f"  OK Trip {rows[-1]['gse_id']} fold{rows[-1]['fold']}")
                        except Exception as exc:
                            print(f"  FAIL Trip {d}: {exc}")
                            rows.append(
                                {
                                    "method": "TripGRN",
                                    "gse_id": Path(d).parent.name,
                                    "fold": int(Path(d).name.split("_")[1]),
                                    "auc": float("nan"),
                                    "auprc": float("nan"),
                                    "status": f"error:{type(exc).__name__}",
                                    "fold_dir": d,
                                }
                            )
                        pd.DataFrame(rows).to_csv(out_cache, index=False, encoding="utf-8-sig")

        if "DeepFGRN" in want:
            for j in deep_jobs:
                j["grn_root"] = str(root)
                j["python"] = py
            todo = [
                j
                for j in deep_jobs
                if ("DeepFGRN", j["gse_id"], int(j["fold"])) not in done
            ]
            print(f"[DeepFGRN] todo={len(todo)}")
            # DeepFGRN/TF is safer single-process
            for j in todo:
                try:
                    rows.append(_deep_one(j))
                    r = rows[-1]
                    print(f"  {r['status']} Deep {r['gse_id']} fold{r['fold']} auc={r.get('auc')} auprc={r.get('auprc')}")
                except Exception as exc:
                    print(f"  FAIL Deep {j}: {exc}")
                    traceback.print_exc()
                    rows.append(
                        {
                            "method": "DeepFGRN",
                            "gse_id": j["gse_id"],
                            "fold": int(j["fold"]),
                            "auc": float("nan"),
                            "auprc": float("nan"),
                            "status": f"error:{type(exc).__name__}",
                            "fold_dir": j["fold_dir"],
                        }
                    )
                pd.DataFrame(rows).to_csv(out_cache, index=False, encoding="utf-8-sig")

        pd.DataFrame(rows).drop_duplicates(subset=["method", "gse_id", "fold"], keep="last").to_csv(
            out_cache, index=False, encoding="utf-8-sig"
        )
        print(f"[done] {out_cache}")


if __name__ == "__main__":
    main()
