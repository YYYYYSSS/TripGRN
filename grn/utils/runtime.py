# -*- coding: utf-8 -*-
"""Wall-clock timing helpers for Fig.1 method benchmarks."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


def now_mono() -> float:
    return time.perf_counter()


@contextmanager
def timed_section() -> Iterator[Dict[str, float]]:
    """Context manager that fills ``elapsed_sec`` on exit."""
    box: Dict[str, float] = {"elapsed_sec": 0.0}
    t0 = now_mono()
    try:
        yield box
    finally:
        box["elapsed_sec"] = float(now_mono() - t0)


def write_runtime_json(
    out_dir: Path | str,
    *,
    wall_time_sec: float,
    train_time_sec: Optional[float] = None,
    eval_time_sec: Optional[float] = None,
    n_genes_used: Optional[int] = None,
    gene_policy: str = "all_genes",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``runtime.json`` next to fold metrics (single source for timing plots)."""
    payload: Dict[str, Any] = {
        "wall_time_sec": float(wall_time_sec),
        "gene_policy": str(gene_policy),
    }
    if train_time_sec is not None:
        payload["train_time_sec"] = float(train_time_sec)
    if eval_time_sec is not None:
        payload["eval_time_sec"] = float(eval_time_sec)
    if n_genes_used is not None:
        payload["n_genes_used"] = int(n_genes_used)
    if extra:
        payload.update(extra)
    path = Path(out_dir) / "runtime.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def stamp_orchestrator_runtime(
    run_dir: Path | str,
    wall: float,
    *,
    method: str,
    gse: str,
    fold_idx: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write runtime.json if missing; otherwise add orchestrator wall-clock."""
    run_dir = Path(run_dir)
    payload_extra: Dict[str, Any] = {
        "gse_id": gse,
        "fold": int(fold_idx),
        "method": method,
        "gene_policy": "all_genes",
        "cell_policy": "full_cv_splits",
        "split_source": "shared_cv_splits_10fold",
        "source": f"run_{method.lower()}_pancancer_10cv",
    }
    if extra:
        payload_extra.update(extra)
    rt = run_dir / "runtime.json"
    if not rt.is_file():
        write_runtime_json(run_dir, wall_time_sec=wall, gene_policy="all_genes", extra=payload_extra)
        return
    try:
        payload = json.loads(rt.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    payload["orchestrator_wall_time_sec"] = float(wall)
    payload.setdefault("wall_time_sec", float(wall))
    payload.setdefault("gene_policy", "all_genes")
    payload.setdefault("cell_policy", "full_cv_splits")
    payload.setdefault("split_source", "shared_cv_splits_10fold")
    rt.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
