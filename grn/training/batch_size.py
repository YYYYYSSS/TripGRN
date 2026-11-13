# -*- coding: utf-8 -*-
"""Batch-size heuristics (no torch / TF dependency)."""

from __future__ import annotations


def suggest_batch_size(
    n_train_cols: int,
    *,
    cap: int = 256,
    vram_gb: int = 16,
) -> int:
    """Heuristic batch size from train column count (expression vector length T).

    Expression encoder Conv1d cost / VRAM scale ~ O(B * T); for T≫4k must use tiny B.
    Prefer capping T via --max-expression-cols for step06-scale matrices.
    """
    t = int(n_train_cols)
    if vram_gb >= 16:
        if t <= 128:
            bs = 256
        elif t <= 512:
            bs = 128
        elif t <= 2048:
            bs = 64
        elif t <= 4096:
            bs = 32
        elif t <= 8192:
            bs = 16
        elif t <= 16384:
            bs = 8
        else:
            bs = 4
    else:
        if t <= 64:
            bs = 128
        elif t <= 256:
            bs = 64
        elif t <= 1024:
            bs = 32
        elif t <= 4096:
            bs = 16
        elif t <= 8192:
            bs = 8
        else:
            bs = 4
    return min(int(cap), max(4, bs))


def suggest_keras_batch_size(n_train_samples: int, dim_exp: int, *, cap: int = 256) -> int:
    """DeepFGRN CorrResNet mini-batch cap (fit uses min(n/10, batch_size))."""
    _ = dim_exp  # reserved for future VRAM-aware tuning
    if n_train_samples >= 8000:
        bs = 256
    elif n_train_samples >= 2000:
        bs = 128
    else:
        bs = 64
    return min(int(cap), max(8, bs))
