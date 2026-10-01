"""
将「目标」表达矩阵（如 GSE131907 log_normalized npy）按参考（如 GSE253718）做批次对齐。
无需 harmonypy。

- mean_shift（默认）：按基因对齐均值 target -= (mean_target - mean_ref)，内存友好、支持宽矩阵分块写盘。
- combat：使用已有依赖 scanpy.pp.combat（与 prepare_gse131907_mbrain_combat_external_csv 一致），
  需将 ref+target 堆叠矩阵一次性载入内存；列数过大时会失败或需自行减小 --ref-max-cols / --target-max-cols。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from grn.data.preprocessing import normalize_gene_symbol_series


def _subset_cols(n_cols: int, max_cols: int, seed: int) -> np.ndarray:
    if max_cols <= 0 or n_cols <= max_cols:
        return np.arange(n_cols, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_cols, size=max_cols, replace=False).astype(np.int64))


def _gene_intersection_indices(
    ref_genes: Sequence[str],
    tgt_genes: Sequence[str],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    ref_norm = normalize_gene_symbol_series(pd.Series([str(x) for x in ref_genes])).tolist()
    tgt_norm = normalize_gene_symbol_series(pd.Series([str(x) for x in tgt_genes])).tolist()
    ref_set = set(ref_norm)
    shared = [g for g in tgt_norm if g in ref_set]
    if not shared:
        raise ValueError("参考与目标基因无交集")
    r_map = {g: i for i, g in enumerate(ref_norm)}
    t_map = {g: i for i, g in enumerate(tgt_norm)}
    idx_r = np.array([r_map[g] for g in shared], dtype=np.int64)
    idx_t = np.array([t_map[g] for g in shared], dtype=np.int64)
    return shared, idx_r, idx_t


def run_mean_shift(
    ref_mm: np.ndarray,
    tgt_mm: np.ndarray,
    idx_r: np.ndarray,
    idx_t: np.ndarray,
    *,
    ref_cols: np.ndarray,
    col_chunk: int,
    out_path: Path,
) -> None:
    ref_blk = np.asarray(ref_mm[idx_r, :][:, ref_cols], dtype=np.float64)
    mu_ref = ref_blk.mean(axis=1)
    n_tar_cols = int(tgt_mm.shape[1])
    acc = np.zeros(len(idx_t), dtype=np.float64)
    cc = max(32, int(col_chunk))
    for j0 in range(0, n_tar_cols, cc):
        j1 = min(n_tar_cols, j0 + cc)
        acc += np.asarray(tgt_mm[idx_t, j0:j1], dtype=np.float64).sum(axis=1)
    mu_tar = acc / float(n_tar_cols)
    delta = (mu_tar - mu_ref).astype(np.float32)

    delta_full = np.zeros(int(tgt_mm.shape[0]), dtype=np.float32)
    delta_full[idx_t] = delta

    G, S = int(tgt_mm.shape[0]), int(tgt_mm.shape[1])
    mm_out = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.float32, shape=(G, S))
    for j0 in range(0, S, cc):
        j1 = min(S, j0 + cc)
        blk = np.asarray(tgt_mm[:, j0:j1], dtype=np.float32)
        blk = blk - delta_full[:, None]
        mm_out[:, j0:j1] = blk
    mm_out.flush()


def run_combat(
    ref_mm: np.ndarray,
    tgt_mm: np.ndarray,
    idx_r: np.ndarray,
    idx_t: np.ndarray,
    *,
    ref_cols: np.ndarray,
    tgt_cols: np.ndarray,
    out_path: Path,
) -> None:
    import anndata as ad
    import scanpy as sc

    X_ref = np.asarray(ref_mm[idx_r, :][:, ref_cols], dtype=np.float32).T
    X_tar = np.asarray(tgt_mm[idx_t, :][:, tgt_cols], dtype=np.float32).T
    X_all = np.vstack([X_ref, X_tar]).astype(np.float32, copy=False)
    batch = np.array(["ref"] * X_ref.shape[0] + ["tgt"] * X_tar.shape[0], dtype=object)
    adata = ad.AnnData(X=X_all)
    adata.obs["batch"] = pd.Categorical(batch)
    sc.pp.combat(adata, key="batch")
    X_corr = np.asarray(adata.X[X_all.shape[0] - X_tar.shape[0] :, :], dtype=np.float32)

    G_full, S_full = int(tgt_mm.shape[0]), int(tgt_mm.shape[1])
    mm_out = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.float32, shape=(G_full, S_full))
    cc = 256
    for j0 in range(0, S_full, cc):
        j1 = min(S_full, j0 + cc)
        mm_out[:, j0:j1] = np.asarray(tgt_mm[:, j0:j1], dtype=np.float32)
    corr_T = X_corr.T
    for j_local, j_global in enumerate(tgt_cols.tolist()):
        mm_out[idx_t, int(j_global)] = corr_T[:, j_local]
    mm_out.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description="Harmonize target npy to reference npy (mean_shift or scanpy ComBat).")
    ap.add_argument("--ref-npy-values", type=Path, required=True)
    ap.add_argument("--ref-npy-genes", type=Path, required=True)
    ap.add_argument("--target-npy-values", type=Path, required=True)
    ap.add_argument("--target-npy-genes", type=Path, required=True)
    ap.add_argument("--out-npy", type=Path, required=True)
    ap.add_argument(
        "--method",
        choices=("mean_shift", "combat"),
        default="mean_shift",
        help="mean_shift: 仅 numpy；combat: scanpy（需内存容纳拼接矩阵）",
    )
    ap.add_argument("--ref-max-cols", type=int, default=4000, help="参考矩阵最多使用的列数（0=全部）")
    ap.add_argument("--target-max-cols", type=int, default=0, help="ComBat 时目标最多列数（0=全部；极大时可能 OOM）")
    ap.add_argument("--col-chunk", type=int, default=256, help="mean_shift 写盘列块大小")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ref_p = Path(args.ref_npy_values).resolve()
    tgt_p = Path(args.target_npy_values).resolve()
    out_p = Path(args.out_npy).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    g_ref = np.load(str(args.ref_npy_genes), allow_pickle=True)
    g_tgt = np.load(str(args.target_npy_genes), allow_pickle=True)
    mm_r = np.load(str(ref_p), mmap_mode="r")
    mm_t = np.load(str(tgt_p), mmap_mode="r")
    shared, idx_r, idx_t = _gene_intersection_indices(
        [str(x) for x in g_ref.tolist()],
        [str(x) for x in g_tgt.tolist()],
    )
    ref_cols = _subset_cols(mm_r.shape[1], int(args.ref_max_cols), int(args.seed))
    tgt_cols = _subset_cols(mm_t.shape[1], int(args.target_max_cols), int(args.seed) + 1)

    meta: Dict[str, object] = {
        "method": str(args.method),
        "ref_npy_values": str(ref_p),
        "target_npy_values": str(tgt_p),
        "out_npy": str(out_p),
        "n_shared_genes": int(len(shared)),
        "ref_cols_used": int(len(ref_cols)),
        "target_cols_used_for_stats": int(mm_t.shape[1]),
        "note": "harmonypy not required; combat uses scanpy",
    }

    if args.method == "mean_shift":
        print(f"[harmonize] mean_shift shared_genes={len(shared)} ref_cols={len(ref_cols)} -> {out_p}", flush=True)
        run_mean_shift(mm_r, mm_t, idx_r, idx_t, ref_cols=ref_cols, col_chunk=int(args.col_chunk), out_path=out_p)
    else:
        est_gb = (len(ref_cols) + len(tgt_cols)) * len(shared) * 4 / 1e9
        print(f"[harmonize] combat estimated dense ~{est_gb:.2f} GB for ComBat slice", flush=True)
        run_combat(mm_r, mm_t, idx_r, idx_t, ref_cols=ref_cols, tgt_cols=tgt_cols, out_path=out_p)
        meta["target_cols_used_for_combat"] = int(len(tgt_cols))

    meta_path = out_p.with_suffix(".harmonize_meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[harmonize] wrote {out_p} meta={meta_path}", flush=True)


if __name__ == "__main__":
    main()
