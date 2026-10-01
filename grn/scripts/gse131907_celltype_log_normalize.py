#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert step06 cell-type count npy -> log_normalized npy (normalize_total + log1p)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from grn.data.gse131907_celltype_paths import gse131907_celltype_layout, resolve_type_ids


def _convert_matrix(mm_in: np.ndarray, mm_out: np.ndarray, *, target_sum: float, col_chunk: int) -> None:
    n_genes, n_cols = int(mm_in.shape[0]), int(mm_in.shape[1])
    cc = max(8, int(col_chunk))
    for j0 in range(0, n_cols, cc):
        j1 = min(n_cols, j0 + cc)
        block = np.asarray(mm_in[:, j0:j1], dtype=np.float64)
        col_sum = block.sum(axis=0)
        scale = np.zeros_like(col_sum)
        pos = col_sum > 0
        scale[pos] = float(target_sum) / col_sum[pos]
        mm_out[:, j0:j1] = np.log1p(block * scale).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="GSE131907 step06 cell-type counts -> log_normalized")
    ap.add_argument("--normalize-target-sum", type=float, default=1e4)
    ap.add_argument("--col-chunk", type=int, default=256)
    ap.add_argument("--type", default="ALL", help="Cell type id / label / ALL")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    layout = gse131907_celltype_layout()
    base = layout["celltype_data_dir"]
    man_path = base / "manifest.json"
    if not man_path.is_file():
        raise FileNotFoundError(f"Run materialize first: {man_path}")

    type_ids = [t[0] for t in resolve_type_ids(args.type)]
    for tid in type_ids:
        td = base / tid
        inp = td / "values.npy"
        outp = td / "values_log_normalized.npy"
        if not inp.is_file():
            raise FileNotFoundError(inp)
        if outp.is_file() and not args.force:
            print(f"[celltype-log-norm] skip {outp}")
            continue
        mm_r = np.load(str(inp), mmap_mode="r")
        mm_w = np.lib.format.open_memmap(str(outp), mode="w+", dtype=np.float32, shape=mm_r.shape)
        print(f"[celltype-log-norm] {tid}: {inp.name} -> {outp.name} shape={mm_r.shape}", flush=True)
        _convert_matrix(mm_r, mm_w, target_sum=float(args.normalize_target_sum), col_chunk=int(args.col_chunk))
        mm_w.flush()
        del mm_w

    aux = {
        "normalize_target_sum": float(args.normalize_target_sum),
        "transform": "normalize_total_then_log1p",
        "types": type_ids,
    }
    aux_path = base / "manifest_log_normalized.json"
    aux_path.write_text(json.dumps(aux, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[celltype-log-norm] wrote {aux_path}")


if __name__ == "__main__":
    main()
