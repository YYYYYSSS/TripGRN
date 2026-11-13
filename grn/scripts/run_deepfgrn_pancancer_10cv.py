#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrate DeepFGRN 10-fold CV on epithelial pancancer datasets.

Uses the same per-GSE ``cv_splits_10fold.json`` as TripGRN (sample_split 5:2:3 cells)
and held-out gene pairs (~64/16/20 stratified; default in cell_unseen runner).

Pending folds for one GSE are launched in a **single** subprocess so TF/dGGAN/
expression load once (avoids ~10–15 min idle overhead per fold restart).

Usage:
  python -m grn.scripts.run_deepfgrn_pancancer_10cv --gse ALL --fold ALL
  python -m grn.scripts.run_deepfgrn_pancancer_10cv --gse GSE103322 --fold 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from grn.data.pancancer_cv import DEFAULT_N_FOLDS, get_fold_split, load_cv_splits
from grn.data.pancancer_paths import (
    PANCANCER_GSE_IDS,
    cv_splits_path,
    deepfgrn_fold_dir,
    gse_npy_triplet,
    scRNA_layout,
    step01_grn_csv,
)
from grn.training.batch_size import suggest_keras_batch_size
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


def _train_samples_for_fold(splits_path: Path, fold_idx: int) -> int:
    payload = load_cv_splits(splits_path)
    fold = get_fold_split(payload, fold_idx)
    return int(len(fold["train_cols"]))


def _stamp_orchestrator_runtime(
    run_dir: Path, wall: float, *, gse: str, fold_idx: int, keras_bs: int, n_train: int
) -> None:
    rt = run_dir / "runtime.json"
    if not rt.is_file():
        write_runtime_json(
            run_dir,
            wall_time_sec=wall,
            gene_policy="all_genes",
            extra={
                "gse_id": gse,
                "fold": int(fold_idx),
                "method": "DeepFGRN",
                "keras_batch_size": int(keras_bs),
                "n_cells_train_full": int(n_train),
                "split_source": "shared_cv_splits_10fold",
                "source": "run_deepfgrn_pancancer_10cv",
            },
        )
        return
    try:
        payload = json.loads(rt.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    payload["orchestrator_wall_time_sec"] = wall
    payload.setdefault("wall_time_sec", wall)
    payload.setdefault("gene_policy", "all_genes")
    payload.setdefault("split_source", "shared_cv_splits_10fold")
    payload.setdefault("keras_batch_size", int(keras_bs))
    rt.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _recover_fold(
    *,
    cmd_base: List[str],
    run_dir: Path,
    gse: str,
    fold_idx: int,
    grn_root: Path,
    env: dict,
    keras_bs: int,
    n_train: int,
    wall_so_far: float,
) -> bool:
    ckpt = run_dir / "best_model.keras"
    if not ckpt.is_file():
        return False
    recover_cmd = list(cmd_base) + [
        "--fold-idx",
        str(fold_idx),
        "--output-fold-dir",
        str(run_dir),
        "--cm-only",
    ]
    print(
        f"[deepfgrn-10cv] recover {gse} fold {fold_idx:02d} via --cm-only (ckpt exists)",
        flush=True,
    )
    t1 = now_mono()
    proc2 = subprocess.run(recover_cmd, check=False, cwd=str(grn_root), env=env)
    wall2 = wall_so_far + float(now_mono() - t1)
    if _fold_done(run_dir):
        if proc2.returncode != 0:
            print(
                f"[deepfgrn-10cv] warn: recover fold {fold_idx:02d} exit={proc2.returncode} "
                f"but metrics exist; continuing",
                flush=True,
            )
        _stamp_orchestrator_runtime(
            run_dir, wall2, gse=gse, fold_idx=fold_idx, keras_bs=keras_bs, n_train=n_train
        )
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Run DeepFGRN pancancer epithelial 10-fold CV")
    ap.add_argument("--gse", default="ALL")
    ap.add_argument("--fold", default="ALL", help="Fold index 1..10 or ALL")
    ap.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--force", action="store_true", help="Retrain even if metrics exist")
    ap.add_argument("--keras-batch-size", type=int, default=0, help="0 = auto from train sample count")
    ap.add_argument("--keras-batch-size-cap", type=int, default=256)
    ap.add_argument(
        "--deepfgrn-root",
        type=Path,
        default=None,
        help="project_DeepFGRN root (or set DEEPFGRN_ROOT). Default: auto-detect.",
    )
    args = ap.parse_args()

    grn_root = Path(__file__).resolve().parents[2]
    layout = scRNA_layout(grn_root)
    kegg_csv = step01_grn_csv(grn_root)
    if not kegg_csv.is_file():
        raise FileNotFoundError(f"Missing GRN CSV: {kegg_csv}")
    print(f"[deepfgrn-10cv] grn_csv={kegg_csv}")
    print(f"[deepfgrn-10cv] experiments={layout['experiments_deepfgrn'].parent}")
    print("[deepfgrn-10cv] cells=shared cv_splits_10fold.json; pairs=held-out (~64/16/20, TripGRN-aligned)")
    print("[deepfgrn-10cv] mode=one-process-per-GSE (shared TF/dGGAN/expression across pending folds)")

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

        pending: List[int] = []
        for fold_idx in folds:
            run_dir = deepfgrn_fold_dir(gse, fold_idx, grn_root)
            if _fold_done(run_dir) and not args.force:
                print(f"[deepfgrn-10cv] skip {gse} fold {fold_idx:02d}")
                n_skip += 1
                continue
            pending.append(fold_idx)

        if not pending:
            continue

        # Batch size from first pending fold (cap is what matters for GPU fill).
        n_train = _train_samples_for_fold(splits_path, pending[0])
        keras_bs = int(args.keras_batch_size)
        if keras_bs <= 0:
            keras_bs = suggest_keras_batch_size(n_train, dim_exp=0, cap=int(args.keras_batch_size_cap))

        env = os.environ.copy()
        env["DEEPFGRN_BATCH_SIZE"] = str(keras_bs)
        if args.deepfgrn_root is not None:
            env["DEEPFGRN_ROOT"] = str(Path(args.deepfgrn_root).resolve())

        gse_dir = deepfgrn_fold_dir(gse, pending[0], grn_root).parent
        gse_dir.mkdir(parents=True, exist_ok=True)
        fold_list_str = ",".join(str(i) for i in pending)

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "grn.scripts.run_deepfgrn_pancancer_cell_unseen",
            "--gse-id",
            gse,
            "--exp-npy-values",
            str(v_path),
            "--exp-npy-genes",
            str(g_path),
            "--kegg-csv",
            str(kegg_csv),
            "--cv-splits-json",
            str(splits_path),
            "--fold-list",
            fold_list_str,
            "--output-gse-dir",
            str(gse_dir),
            "--keras-batch-size",
            str(keras_bs),
            "--skip-summary-plots",
        ]
        if args.deepfgrn_root is not None:
            cmd.extend(["--deepfgrn-root", str(Path(args.deepfgrn_root).resolve())])

        # Base for single-fold recover (no fold-list).
        cmd_recover_base = [
            sys.executable,
            "-u",
            "-m",
            "grn.scripts.run_deepfgrn_pancancer_cell_unseen",
            "--gse-id",
            gse,
            "--exp-npy-values",
            str(v_path),
            "--exp-npy-genes",
            str(g_path),
            "--kegg-csv",
            str(kegg_csv),
            "--cv-splits-json",
            str(splits_path),
            "--keras-batch-size",
            str(keras_bs),
            "--skip-summary-plots",
        ]
        if args.deepfgrn_root is not None:
            cmd_recover_base.extend(["--deepfgrn-root", str(Path(args.deepfgrn_root).resolve())])

        print(
            f"[deepfgrn-10cv] run {gse} folds=[{fold_list_str}] "
            f"(train_cols~{n_train}, keras_batch={keras_bs}, one process)",
            flush=True,
        )
        t0 = now_mono()
        proc = subprocess.run(cmd, check=False, cwd=str(grn_root), env=env)
        wall_batch = float(now_mono() - t0)
        wall_each = wall_batch / max(len(pending), 1)

        failed: List[int] = []
        for fold_idx in pending:
            run_dir = deepfgrn_fold_dir(gse, fold_idx, grn_root)
            n_tr = _train_samples_for_fold(splits_path, fold_idx)
            if _fold_done(run_dir):
                if proc.returncode != 0:
                    print(
                        f"[deepfgrn-10cv] warn: {gse} fold {fold_idx:02d} batch exit={proc.returncode} "
                        f"but metrics exist; continuing",
                        flush=True,
                    )
                _stamp_orchestrator_runtime(
                    run_dir, wall_each, gse=gse, fold_idx=fold_idx, keras_bs=keras_bs, n_train=n_tr
                )
                n_run += 1
                continue
            if _recover_fold(
                cmd_base=cmd_recover_base,
                run_dir=run_dir,
                gse=gse,
                fold_idx=fold_idx,
                grn_root=grn_root,
                env=env,
                keras_bs=keras_bs,
                n_train=n_tr,
                wall_so_far=wall_each,
            ):
                n_run += 1
                continue
            failed.append(fold_idx)

        if failed:
            raise SystemExit(
                f"[deepfgrn-10cv] {gse} incomplete folds={failed} "
                f"(batch exit={proc.returncode})"
            )

    print(f"[deepfgrn-10cv] finished: ran={n_run}, skipped={n_skip}")


if __name__ == "__main__":
    main()
