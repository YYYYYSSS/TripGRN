# -*- coding: utf-8 -*-
"""10-fold unseen-sample (cell column) splits for pancancer benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

DEFAULT_N_FOLDS = 10
DEFAULT_SAMPLE_SPLIT = (5, 2, 3)
DEFAULT_PAIR_SEED = 42
DEFAULT_CV_BASE_SEED = 42


def split_cells_unseen_523(
    n_cells: int,
    fold_seed: int,
    rng: np.random.Generator,
    *,
    ratios: Tuple[int, int, int] = DEFAULT_SAMPLE_SPLIT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle cell indices then split train/val/test by weight ratios (default 5:2:3).
    Matches project_DeepFGRN.gse253718_kegg_utils.split_cells_unseen_523 semantics.
    """
    w1, w2, w3 = ratios
    w_sum = w1 + w2 + w3
    idx = np.arange(n_cells, dtype=np.int64)
    rng.shuffle(idx)
    n_train = int(round(w1 / w_sum * n_cells))
    n_val = int(round(w2 / w_sum * n_cells))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def generate_cv_splits(
    n_cells: int,
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    sample_split: Tuple[int, int, int] = DEFAULT_SAMPLE_SPLIT,
    cv_base_seed: int = DEFAULT_CV_BASE_SEED,
    pair_seed: int = DEFAULT_PAIR_SEED,
    gse_id: str = "",
) -> Dict[str, Any]:
    folds: List[Dict[str, Any]] = []
    for k in range(n_folds):
        rng = np.random.default_rng(int(cv_base_seed + k))
        tr, va, te = split_cells_unseen_523(n_cells, k, rng, ratios=sample_split)
        folds.append(
            {
                "fold": k + 1,
                "cv_seed": int(cv_base_seed + k),
                "train_cols": tr.astype(int).tolist(),
                "val_cols": va.astype(int).tolist(),
                "test_cols": te.astype(int).tolist(),
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "n_test": int(len(te)),
            }
        )
    return {
        "gse_id": gse_id,
        "n_cells": int(n_cells),
        "n_folds": int(n_folds),
        "sample_split": list(sample_split),
        "pair_seed": int(pair_seed),
        "cv_base_seed": int(cv_base_seed),
        "folds": folds,
    }


def load_cv_splits(path: Path | str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_fold_split(payload: Dict[str, Any], fold_idx: int) -> Dict[str, Any]:
    """fold_idx: 1-based."""
    fi = int(fold_idx)
    for fold in payload["folds"]:
        if int(fold["fold"]) == fi:
            return fold
    raise KeyError(f"fold {fold_idx} not in cv splits (n_folds={payload.get('n_folds')})")


def write_cv_splits(payload: Dict[str, Any], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
