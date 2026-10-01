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

from grn.data.pancancer_paths import PANCANCER_GSE_IDS, grn_root_from_here

METHODS = ("TripGRN", "DeepFGRN")


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
    model, _, _, test_loader, device, use_amp = _build_pipeline(args)
    state = torch.load(str(run / "best_model_fold0.pth"), map_location=device)
    model.load_state_dict(state)
    model.eval()
    y_true, y_prob = _predict_arrays(model, test_loader, device, use_amp_bf16=use_amp)
    auc, auprc = _binary_ranking(y_true, y_prob)
    # parse gse/fold from path .../TripGRN/GSExxx/fold_YY
    fold = int(run.name.split("_")[1])
    gse = run.parent.name
    return {
        "method": "TripGRN",
        "gse_id": gse,
        "fold": fold,
        "auc": auc,
        "auprc": auprc,
        "status": "ok",
        "fold_dir": str(run),
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
        proc = subprocess.run(cmd, cwd=str(grn_root), env=env, check=False, capture_output=True, text=True)
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
        }


def _discover_jobs(exp_root: Path) -> Tuple[List[str], List[Dict[str, str]]]:
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
                        deep_jobs.append(
                            {
                                "fold_dir": str(fd),
                                "gse_id": gse,
                                "fold": str(fold),
                                "experiment_name": exp_root.name,
                            }
                        )
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
        "step01_pancancer_epithelial_10cv",
        "step01_pancancer_epithelial_10cv_dorothea_abc",
    ]
    want = {m.strip() for m in str(args.methods).split(",") if m.strip()}

    for exp in exps:
        exp_root = root / "experiments" / exp
        out_cache = exp_root / "summary" / "trip_deep_strict_binary_ranking_cache.csv"
        out_cache.parent.mkdir(parents=True, exist_ok=True)
        trip_dirs, deep_jobs = _discover_jobs(exp_root)
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
