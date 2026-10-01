"""
将 materialize_gse131907_brain_nonbrain_npys 输出的 counts memmap 转为与 GSE253718 Step1 一致的
「normalize_total(target_sum) + log1p」浮点矩阵（仍按基因×细胞），写出独立 .npy 便于评估时 mmap。

单次扫描按列块读写，避免整表进内存。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _convert_matrix(mm_in: np.ndarray, mm_out: np.ndarray, *, target_sum: float, col_chunk: int) -> None:
    n_genes, n_cols = int(mm_in.shape[0]), int(mm_in.shape[1])
    if mm_out.shape != (n_genes, n_cols):
        raise ValueError(f"shape mismatch out={mm_out.shape} vs in={(n_genes, n_cols)}")
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
    ap = argparse.ArgumentParser(description="GSE131907 brain/nonbrain counts npy -> log_normalized npy (Scanpy-style).")
    ap.add_argument("--split-dir", type=Path, required=True, help="materialize 输出目录（含 manifest.json）")
    ap.add_argument("--normalize-target-sum", type=float, default=1e4)
    ap.add_argument("--col-chunk", type=int, default=256, help="每次处理的列数（越大越快、峰值内存越高）")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 log_normalized npy")
    args = ap.parse_args()

    split_dir = Path(args.split_dir).resolve()
    man_path = split_dir / "manifest.json"
    if not man_path.is_file():
        raise FileNotFoundError(str(man_path))
    man = json.loads(man_path.read_text(encoding="utf-8"))

    brain_counts = split_dir / man["brain_values_npy"]
    nb_counts = split_dir / man["nonbrain_values_npy"]
    brain_out = split_dir / "brain_values_log_normalized.npy"
    nb_out = split_dir / "nonbrain_values_log_normalized.npy"

    if not brain_counts.is_file() or not nb_counts.is_file():
        raise FileNotFoundError(f"missing counts npy under {split_dir}")

    extra = {
        "brain_values_log_normalized_npy": brain_out.name,
        "nonbrain_values_log_normalized_npy": nb_out.name,
        "normalize_target_sum": float(args.normalize_target_sum),
        "transform": "normalize_total_then_log1p",
        "source_manifest": "manifest.json",
    }

    for inp, outp in ((brain_counts, brain_out), (nb_counts, nb_out)):
        if outp.is_file() and not args.force:
            print(f"[log-norm] skip existing {outp}")
            continue
        mm_r = np.load(str(inp), mmap_mode="r")
        mm_w = np.lib.format.open_memmap(str(outp), mode="w+", dtype=np.float32, shape=mm_r.shape)
        print(f"[log-norm] {inp.name} -> {outp.name} shape={mm_r.shape}", flush=True)
        _convert_matrix(mm_r, mm_w, target_sum=float(args.normalize_target_sum), col_chunk=int(args.col_chunk))
        mm_w.flush()
        del mm_w

    aux_path = split_dir / "manifest_log_normalized.json"
    aux_path.write_text(json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[log-norm] wrote {aux_path}")


if __name__ == "__main__":
    main()
