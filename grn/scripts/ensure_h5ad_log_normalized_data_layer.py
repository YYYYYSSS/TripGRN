"""
检查 AnnData 的 layers / .X，并在缺少 layers['data'] 时按 Scanpy 惯例写入：
  normalize_total(target_sum=1e4) -> log1p -> 存入 layers['data']

与仓库 preprocess_gse253718_sc_to_exp_csv 中「log_normalized = layers['data']」一致。
不会默认覆盖已有 layers['data']（可用 --force）。

用法：
  python -m grn.scripts.ensure_h5ad_log_normalized_data_layer --h5ad path.h5ad --inspect-only
  python -m grn.scripts.ensure_h5ad_log_normalized_data_layer --h5ad path.h5ad --in-place
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.sparse import issparse


def _sample_nonzero_values(X: Any, rng: np.random.Generator, max_n: int = 8000) -> np.ndarray:
    if issparse(X):
        X = X.tocoo()
        if X.nnz == 0:
            return np.array([], dtype=np.float64)
        n = min(max_n, X.nnz)
        idx = rng.choice(X.nnz, size=n, replace=False)
        return np.asarray(X.data[idx], dtype=np.float64)
    arr = np.asarray(X)
    nz = arr[arr != 0]
    if nz.size == 0:
        return np.array([], dtype=np.float64)
    n = min(max_n, nz.size)
    idx = rng.choice(nz.size, size=n, replace=False)
    return nz[idx].astype(np.float64)


def _looks_like_raw_counts(samp: np.ndarray) -> bool:
    """启发式：非零采样值是否更像整数计数。"""
    if samp.size < 50:
        return True
    near_int = np.mean(np.abs(samp - np.round(samp)) < 1e-5)
    med = float(np.median(samp))
    mx = float(np.max(samp))
    # log1p 后通常 max 不会特别大；原始 count 常有大整数
    if mx > 80.0 and near_int > 0.85:
        return True
    if med > 1.5 and mx > 50.0 and near_int > 0.7:
        return True
    return False


def inspect_adata(adata: ad.AnnData) -> None:
    rng = np.random.default_rng(0)
    print(f"obs={adata.n_obs:,} var={adata.n_vars:,}")
    print(f"layers: {sorted(adata.layers.keys())}")
    X = adata.X
    print(f"X: sparse={issparse(X)} dtype={getattr(X, 'dtype', None)}")
    samp = _sample_nonzero_values(X, rng)
    if samp.size:
        print(
            f"X nonzero sample: min={float(samp.min()):.4g} med={float(np.median(samp)):.4g} "
            f"max={float(samp.max()):.4g} near_int_frac={float(np.mean(np.abs(samp - np.round(samp)) < 1e-5)):.3f}"
        )
        print(f"heuristic 'raw_counts_like': {_looks_like_raw_counts(samp)}")
    else:
        print("X: no nonzeros sampled")
    if "counts" in adata.layers:
        c = adata.layers["counts"]
        print(f"layers['counts']: sparse={issparse(c)} dtype={getattr(c, 'dtype', None)}")
        sc = _sample_nonzero_values(c, rng)
        if sc.size:
            print(
                f"counts nonzero sample: min={float(sc.min()):.4g} med={float(np.median(sc)):.4g} "
                f"max={float(sc.max()):.4g} near_int_frac={float(np.mean(np.abs(sc - np.round(sc)) < 1e-5)):.3f}"
            )
            print(f"counts heuristic 'raw_counts_like': {_looks_like_raw_counts(sc)}")
    if "data" in adata.layers:
        d = adata.layers["data"]
        s2 = _sample_nonzero_values(d, rng)
        if s2.size:
            print(
                f"layers['data'] nonzero sample: min={float(s2.min()):.4g} med={float(np.median(s2)):.4g} "
                f"max={float(s2.max()):.4g}"
            )


def _matrix_for_normalize(adata: ad.AnnData) -> tuple[Any, str]:
    """优先用 layers['counts']（避免误用已 scale 的 .X）。"""
    if "counts" in adata.layers:
        return adata.layers["counts"], "layers['counts']"
    X = adata.X
    if not issparse(X):
        xflat = np.asarray(X).ravel()
        if xflat.size and float(np.min(xflat)) < -1e-6:
            raise ValueError(
                ".X 出现负值，疑似 ScaleData/z-score 矩阵，不能用来做 normalize_total+log1p。"
                "请提供 layers['counts'] 或未经 scale 的原始计数矩阵。"
            )
    samp = _sample_nonzero_values(X, np.random.default_rng(42))
    if samp.size and not _looks_like_raw_counts(samp):
        raise ValueError(
            ".X 不像原始计数；且不存在 layers['counts']。"
            "请把原始计数存入 adata.layers['counts'] 后再运行本脚本。"
        )
    return adata.X, "X"


def ensure_data_layer(
    adata: ad.AnnData,
    *,
    force: bool,
    target_sum: float,
) -> bool:
    if "data" in adata.layers and not force:
        print("[ensure-data] 已存在 layers['data']，跳过（使用 --force 可覆盖）。")
        return False

    base, src = _matrix_for_normalize(adata)

    tmp = ad.AnnData(X=base.copy() if not issparse(base) else base.copy())
    sc.pp.normalize_total(tmp, target_sum=target_sum)
    sc.pp.log1p(tmp)
    adata.layers["data"] = tmp.X.copy()
    print(f"[ensure-data] 已从 {src} 写入 layers['data']（target_sum={target_sum} + log1p）。")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5ad", required=True, type=Path)
    ap.add_argument("--inspect-only", action="store_true", help="只打印信息，不改文件")
    ap.add_argument("--in-place", action="store_true", help="写回同一 h5ad（默认需指定）")
    ap.add_argument("--out", type=Path, default=None, help="另存为新 h5ad（与 --in-place 二选一）")
    ap.add_argument("--force", action="store_true", help="覆盖已有 layers['data']")
    ap.add_argument("--target-sum", type=float, default=1e4)
    args = ap.parse_args()

    h5 = args.h5ad.expanduser().resolve()
    if not h5.is_file():
        raise SystemExit(f"文件不存在: {h5}")

    print(f"[ensure-data] 读取 {h5} …")
    adata = ad.read_h5ad(str(h5))
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    print("--- inspect ---")
    inspect_adata(adata)

    if args.inspect_only:
        return

    if not args.in_place and args.out is None:
        raise SystemExit("请指定 --in-place 或 --out path.h5ad")

    changed = ensure_data_layer(
        adata,
        force=args.force,
        target_sum=float(args.target_sum),
    )
    if not changed and not args.force:
        out_path = h5 if args.in_place else args.out
        print(f"[ensure-data] 无需写入，不保存。若需覆盖可加 --force。")
        return

    if args.in_place:
        out_path = h5
    else:
        out_path = args.out.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ensure-data] 保存 -> {out_path}")
    adata.write_h5ad(out_path)


if __name__ == "__main__":
    main()
