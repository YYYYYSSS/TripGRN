#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert GSE253718 cell-type count npy -> log_normalized npy."""

from __future__ import annotations

import argparse
import json

import numpy as np

from grn.data.gse253718_celltype_paths import gse253718_celltype_layout, resolve_type_ids
from grn.scripts.gse131907_celltype_log_normalize import _convert_matrix


def main() -> None:
    ap = argparse.ArgumentParser(description="GSE253718 cell-type counts -> log_normalized")
    ap.add_argument("--normalize-target-sum", type=float, default=1e4)
    ap.add_argument("--col-chunk", type=int, default=256)
    ap.add_argument("--type", default="ALL")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    layout = gse253718_celltype_layout()
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
            print(f"[gse253718-log-norm] skip {outp}")
            continue
        mm_r = np.load(str(inp), mmap_mode="r")
        mm_w = np.lib.format.open_memmap(str(outp), mode="w+", dtype=np.float32, shape=mm_r.shape)
        print(f"[gse253718-log-norm] {tid}: {inp.name} -> {outp.name} shape={mm_r.shape}", flush=True)
        _convert_matrix(mm_r, mm_w, target_sum=float(args.normalize_target_sum), col_chunk=int(args.col_chunk))
        mm_w.flush()
        del mm_w

    aux = {
        "normalize_target_sum": float(args.normalize_target_sum),
        "transform": "normalize_total_then_log1p",
        "types": type_ids,
    }
    (base / "manifest_log_normalized.json").write_text(
        json.dumps(aux, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[gse253718-log-norm] done")


if __name__ == "__main__":
    main()
