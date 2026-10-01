"""
GSE253718（或同结构 GEO 单细胞）原始 10x 三文件 → Scanpy 流程 → 深度学习用表达 CSV。

对齐 show.R（Seurat）主流程，但不含 JackStraw：
  - 多样本 10x：按前缀合并
  - min.cells=3 / min.features=200（Scanpy: filter_genes min_cells, filter_cells min_genes）
  - 额外 QC（默认）：去掉计数矩阵中「≥90% 基因为零表达」的细胞（`--filter-cells-max-zero-frac`）
  - 线粒体比例 QC 指标（不过滤）
  - NormalizeData 等价：normalize_total + log1p
  - FindVariableFeatures：highly_variable_genes, n_top_genes=3000, flavor=seurat
  - ScaleData → RunPCA(高变基因) → FindNeighbors → FindClusters(res 0.5 与 2) → UMAP / t-SNE

导出 CSV 形状：**行 = 基因，列 = 细胞**（首列 ``gene``，其余列为各细胞 barcode / obs 名）。
  - ``log_normalized``（默认）：Seurat ``layer="data"``，即 normalize_total + log1p 之后、scale 之前。
  - ``scaled``：Seurat ScaleData 等价（``sc.pp.scale``，clip 10）后的值，按基因 z-score 风格。
  - ``counts``：仅 QC 后的原始计数矩阵。

  大矩阵按基因分块写盘，避免一次性构造整块稠密矩阵（见 ``--chunk-genes``）。

用法示例：
  python -m grn.scripts.preprocess_gse253718_sc_to_exp_csv \\
    --raw-dir data/exp/GSE253718 \\
    --output-csv data/exp/GSE253718_exp.csv

  python -m grn.scripts.preprocess_gse253718_sc_to_exp_csv --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import io as spio
from scipy.sparse import coo_matrix, issparse


def _discover_10x_prefixes(raw_dir: Path) -> List[str]:
    """匹配 PREFIX_barcodes.tsv.gz（或 .tsv）；标准 Cell Ranger 单样本目录则返回 sample1。"""
    names = [p.name for p in raw_dir.iterdir() if p.is_file()]
    pat = re.compile(r"^(.+)_barcodes\.tsv(\.gz)?$", re.IGNORECASE)
    prefs: List[str] = []
    for n in names:
        m = pat.match(n)
        if m:
            prefs.append(m.group(1))
    prefs = sorted(set(prefs))
    if prefs:
        return prefs
    std = {"barcodes.tsv.gz", "barcodes.tsv", "matrix.mtx.gz", "matrix.mtx"}
    if std.intersection(names) and any(
        x in names for x in ("features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv")
    ):
        return ["sample1"]
    return []


def _materialize_10x_dir(raw_dir: Path, prefix: str, dest: Path) -> None:
    """将 GEO 风格文件名复制为 Cell Ranger 风格，便于 read_10x_mtx。"""
    dest.mkdir(parents=True, exist_ok=True)
    candidates = [
        (f"{prefix}_matrix.mtx.gz", "matrix.mtx.gz"),
        (f"{prefix}_matrix.mtx", "matrix.mtx"),
        (f"{prefix}_barcodes.tsv.gz", "barcodes.tsv.gz"),
        (f"{prefix}_barcodes.tsv", "barcodes.tsv"),
        (f"{prefix}_features.tsv.gz", "features.tsv.gz"),
        (f"{prefix}_features.tsv", "features.tsv"),
        (f"{prefix}_genes.tsv.gz", "genes.tsv.gz"),
        (f"{prefix}_genes.tsv", "genes.tsv"),
    ]
    if prefix == "sample1":
        candidates += [
            ("matrix.mtx.gz", "matrix.mtx.gz"),
            ("matrix.mtx", "matrix.mtx"),
            ("barcodes.tsv.gz", "barcodes.tsv.gz"),
            ("barcodes.tsv", "barcodes.tsv"),
            ("features.tsv.gz", "features.tsv.gz"),
            ("features.tsv", "features.tsv"),
            ("genes.tsv.gz", "genes.tsv.gz"),
            ("genes.tsv", "genes.tsv"),
        ]
    copied = set()
    for src_name, dst_name in candidates:
        src = raw_dir / src_name
        if src.is_file():
            shutil.copy2(src, dest / dst_name)
            copied.add(dst_name)
    need_mtx = "matrix.mtx.gz" in copied or "matrix.mtx" in copied
    need_bc = "barcodes.tsv.gz" in copied or "barcodes.tsv" in copied
    need_ft = (
        "features.tsv.gz" in copied
        or "features.tsv" in copied
        or "genes.tsv.gz" in copied
        or "genes.tsv" in copied
    )
    if not (need_mtx and need_bc and need_ft):
        raise FileNotFoundError(
            f"未在 {raw_dir} 为前缀 {prefix!r} 凑齐 10x 三件套（matrix + barcodes + features/genes）。"
            f" 已复制: {sorted(copied)}"
        )


def _read_10x_one_sample(raw_dir: Path, prefix: str) -> ad.AnnData:
    with tempfile.TemporaryDirectory(prefix="gse253718_10x_") as td:
        tdir = Path(td)
        _materialize_10x_dir(raw_dir, prefix, tdir)
        adata = sc.read_10x_mtx(tdir, var_names="gene_symbols", cache=False)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    adata.obs["sample_id"] = prefix
    return adata


def _collapse_duplicate_genes(adata: ad.AnnData, layer_key: str = "counts") -> ad.AnnData:
    """按 var 名合并重复基因（AnnData 为 cells × genes，对同名列求和）。"""
    names = np.asarray(adata.var_names.astype(str))
    if len(names) == len(np.unique(names)):
        return adata
    X = adata.layers[layer_key]
    if issparse(X):
        Xcoo = X.tocoo()
    else:
        Xcoo = coo_matrix(X)
    # COO: row = cell, col = gene index
    _, inv = np.unique(names, return_inverse=True)
    col_new = inv[Xcoo.col]
    n_cells, n_genes_u = adata.n_obs, int(inv.max()) + 1
    X_new = coo_matrix((Xcoo.data, (Xcoo.row, col_new)), shape=(n_cells, n_genes_u)).tocsr()
    unique_names = np.unique(names)
    var_new = pd.DataFrame(index=unique_names)
    out = ad.AnnData(X=X_new, obs=adata.obs.copy(), var=var_new)
    for k in adata.obs.columns:
        out.obs[k] = adata.obs[k].values
    out.layers["counts"] = X_new.copy()
    return out


def _merge_and_collapse(raw_dir: Path, prefixes: List[str]) -> ad.AnnData:
    adatas = [_read_10x_one_sample(raw_dir, p) for p in prefixes]
    adata = ad.concat(adatas, label="batch", keys=prefixes, merge="same")
    adata.obs_names_make_unique()
    adata.layers["counts"] = adata.X.copy() if not issparse(adata.X) else adata.X.copy()
    X0 = adata.layers["counts"]
    tmp = ad.AnnData(X=X0, obs=adata.obs.copy(), var=adata.var.copy())
    tmp.layers["counts"] = X0.copy() if not issparse(X0) else X0.copy()
    return _collapse_duplicate_genes(tmp, layer_key="counts")


def _resolve_expression_matrix(adata: ad.AnnData, export_values: str):
    """返回 (X_cells_x_genes, layer_label)。"""
    if export_values == "log_normalized":
        if "data" not in adata.layers:
            raise ValueError("缺少 layers['data']")
        return adata.layers["data"], "data"
    if export_values == "scaled":
        if "scaled" not in adata.layers:
            raise ValueError("缺少 layers['scaled']（scale 步骤未写入）")
        return adata.layers["scaled"], "scaled"
    if export_values == "counts":
        if "counts" not in adata.layers:
            raise ValueError("缺少 layers['counts']")
        return adata.layers["counts"], "counts"
    raise ValueError("export_values 须为 log_normalized | scaled | counts")


def _sanitize_cell_header(name: str) -> str:
    """CSV 列名：避免逗号、换行破坏格式。"""
    s = str(name).strip().replace("\r", "_").replace("\n", "_").replace(",", "_")
    return s


def write_genes_by_cells_csv(
    adata: ad.AnnData,
    out_path: Path,
    export_values: str = "log_normalized",
    chunk_genes: int = 500,
    float_format: str = "%.8g",
) -> None:
    """
    写出：首列 gene，其余列为细胞；矩阵为 X[n_cells, n_genes] 的转置。
    按基因分块从稀疏矩阵取列子集再写行，控制峰值内存。
    """
    X, _label = _resolve_expression_matrix(adata, export_values)
    n_cells, n_genes = adata.n_obs, adata.n_vars
    if issparse(X):
        X = X.tocsr()
    genes = [str(g) for g in adata.var_names.tolist()]
    cells = [_sanitize_cell_header(c) for c in adata.obs_names.tolist()]
    if len(set(cells)) != len(cells):
        raise ValueError("细胞列名在去逗号后不唯一，请检查 barcode 或改用 obs 唯一化策略。")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_genes = max(1, int(chunk_genes))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gene"] + cells)
        for g0 in range(0, n_genes, chunk_genes):
            g1 = min(n_genes, g0 + chunk_genes)
            if issparse(X):
                block = X[:, g0:g1].toarray()
            else:
                block = np.asarray(X[:, g0:g1])
            # block: n_cells x (g1-g0) -> 逐基因一行
            for j, gi in enumerate(range(g0, g1)):
                row = block[:, j]
                w.writerow([genes[gi]] + [float_format % float(v) for v in row])


def _filter_cells_by_max_zero_fraction(adata: ad.AnnData, max_zero_frac: Optional[float]) -> ad.AnnData:
    """
    在计数矩阵上：去掉「零表达基因占比 >= max_zero_frac」的细胞（稀疏矩阵用每行 nnz 推断零占比）。
    须在 normalize 之前、与现有 min_genes/min_cells 筛选衔接。
    """
    if max_zero_frac is None:
        return adata
    thr = float(max_zero_frac)
    if thr <= 0.0 or thr > 1.0:
        return adata
    n_obs, n_genes = adata.n_obs, adata.n_vars
    if n_obs == 0 or n_genes == 0:
        return adata
    X = adata.X
    if issparse(X):
        Xcsr = X.tocsr()
        nnz_per_cell = np.diff(Xcsr.indptr).astype(np.float64)
        zero_frac = 1.0 - nnz_per_cell / float(n_genes)
    else:
        Xd = np.asarray(X)
        zero_frac = (Xd == 0).mean(axis=1, dtype=np.float64)
    keep = zero_frac < thr
    n_drop = int(np.sum(~keep))
    if n_drop:
        print(
            f"[QC] filter_cells zero_frac>={thr:.4g}: removed {n_drop} / {n_obs} cells "
            f"(keep cells with <{thr:.0%} zero-expression genes)"
        )
        adata = adata[keep].copy()
    else:
        print(f"[QC] filter_cells zero_frac>={thr:.4g}: no cells removed (n={n_obs})")
    return adata


def run_scanpy_workflow(
    adata: ad.AnnData,
    n_top_genes: int = 3000,
    n_pcs: int = 20,
    random_state: int = 0,
    min_genes: int = 200,
    min_cells: int = 3,
    skip_tsne: bool = False,
    filter_cells_max_zero_frac: Optional[float] = 0.9,
) -> ad.AnnData:
    """QC → 同步 counts → 归一化 → HVG → scale → PCA → neighbors → leiden×2 → umap/tsne。"""
    adata.obs_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata = _filter_cells_by_max_zero_fraction(adata, filter_cells_max_zero_frac)

    adata.layers["counts"] = adata.X.copy() if not issparse(adata.X) else adata.X.copy()

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-") | adata.var_names.str.startswith(
        "MT."
    )
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # Seurat ``layer="data"``：log 归一化矩阵（scale 会改写 .X，故先固化到 layer）
    adata.layers["data"] = adata.X.copy() if not issparse(adata.X) else adata.X.copy()

    n_hvg = min(n_top_genes, max(2, adata.n_vars - 1))
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_hvg,
        flavor="seurat",
        subset=False,
    )

    sc.pp.scale(adata, max_value=10)
    adata.layers["scaled"] = adata.X.copy() if not issparse(adata.X) else adata.X.copy()

    n_hvg_sel = int(adata.var["highly_variable"].sum())
    # sklearn PCA：n_components 须严格小于 min(n_samples, n_features)
    pca_max = min(adata.n_obs, max(n_hvg_sel, 1))
    if pca_max <= 1:
        adata.obsm["X_pca"] = np.zeros((adata.n_obs, 1), dtype=np.float64)
    else:
        n_comp = max(1, min(50, pca_max - 1))
        sc.tl.pca(
            adata,
            svd_solver="arpack",
            n_comps=n_comp,
            mask_var="highly_variable",
            random_state=random_state,
        )

    n_neighbors = max(2, min(15, adata.n_obs - 1))
    n_pcs_use = max(1, min(n_pcs, adata.obsm["X_pca"].shape[1]))
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs_use,
        use_rep="X_pca",
        random_state=random_state,
    )
    sc.tl.leiden(adata, resolution=0.5, key_added="leiden_0.5", random_state=random_state)
    sc.tl.leiden(adata, resolution=2.0, key_added="leiden_2", random_state=random_state)
    sc.tl.umap(adata, random_state=random_state)
    if not skip_tsne:
        # 大数据集上 t-SNE 极慢；导出 CSV 不依赖此步，可用 --skip-tsne
        perplexity = min(30, max(1, (adata.n_obs - 1) // 2))
        sc.tl.tsne(
            adata,
            n_pcs=n_pcs_use,
            perplexity=perplexity,
            random_state=random_state,
            use_rep="X_pca",
        )

    return adata


def preprocess_raw_dir_to_csv(
    raw_dir: Path,
    output_csv: Path,
    h5ad_out: Optional[Path] = None,
    random_state: int = 0,
    min_genes: int = 200,
    min_cells: int = 3,
    export_values: str = "log_normalized",
    skip_tsne: bool = False,
    chunk_genes: int = 500,
    filter_cells_max_zero_frac: Optional[float] = 0.9,
) -> Tuple[Path, Optional[Path]]:
    raw_dir = raw_dir.resolve()
    output_csv = Path(output_csv).resolve()
    prefixes = _discover_10x_prefixes(raw_dir)
    if not prefixes:
        raise FileNotFoundError(
            f"在 {raw_dir} 未发现 10x 三文件（期望 *barcodes.tsv.gz + *features/genes* + *matrix.mtx.gz）。"
        )

    adata = _merge_and_collapse(raw_dir, prefixes)
    adata = run_scanpy_workflow(
        adata,
        random_state=random_state,
        min_genes=min_genes,
        min_cells=min_cells,
        skip_tsne=skip_tsne,
        filter_cells_max_zero_frac=filter_cells_max_zero_frac,
    )

    write_genes_by_cells_csv(
        adata,
        output_csv,
        export_values=export_values,
        chunk_genes=chunk_genes,
    )

    h5ad_path = None
    if h5ad_out:
        h5ad_out = Path(h5ad_out).resolve()
        h5ad_out.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(h5ad_out)
        h5ad_path = h5ad_out

    return output_csv, h5ad_path


def _write_minimal_10x(out_dir: Path, prefix: str = "GSM_SMOKE") -> None:
    """小型 10x：10 细胞 × 30 基因，配合 --smoke-test（低 QC 阈值，但足够跑 PCA/图）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n_genes, n_cells = 30, 10
    mat = rng.integers(0, 80, size=(n_genes, n_cells)).astype(np.float64)
    mat[0, :] += 5.0  # MT- mock：首基因为线粒体样式命名
    barcodes = [f"CELL{i:04d}-1" for i in range(n_cells)]
    gene_syms = [f"G{i:02d}" for i in range(n_genes)]
    gene_syms[0] = "MT-TEST1"
    features = pd.DataFrame(
        {
            "gene_id": [f"ENSG{i:05d}" for i in range(n_genes)],
            "gene_symbol": gene_syms,
            "feature_type": ["Gene Expression"] * n_genes,
        }
    )
    d = tempfile.mkdtemp()
    try:
        p = Path(d)
        spio.mmwrite(p / "matrix.mtx", coo_matrix(mat))
        features.to_csv(p / "features.tsv", sep="\t", header=False, index=False)
        with open(p / "barcodes.tsv", "w", encoding="utf-8") as f:
            f.write("\n".join(barcodes) + "\n")
        for name, dst in [
            ("matrix.mtx", f"{prefix}_matrix.mtx.gz"),
            ("features.tsv", f"{prefix}_features.tsv.gz"),
            ("barcodes.tsv", f"{prefix}_barcodes.tsv.gz"),
        ]:
            src = p / name
            with open(src, "rb") as sf:
                data = sf.read()
            with gzip.open(out_dir / dst, "wb") as zf:
                zf.write(data)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="GSE253718 10x 单细胞预处理 → 基因×细胞 表达 CSV")
    p.add_argument("--raw-dir", type=Path, default=Path("data/exp/GSE253718"), help="原始 10x 文件目录")
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/exp/GSE253718_exp.csv"),
        help="输出：首列 gene，其余列为各细胞",
    )
    p.add_argument(
        "--export-values",
        choices=("log_normalized", "scaled", "counts"),
        default="log_normalized",
        help="log_normalized=Seurat data；scaled=ScaleData 后；counts=QC 后原始计数",
    )
    p.add_argument(
        "--chunk-genes",
        type=int,
        default=500,
        help="写 CSV 时每批处理的基因数（降低峰值内存；稀疏矩阵仍按块 toarray）",
    )
    p.add_argument(
        "--skip-tsne",
        action="store_true",
        help="跳过 t-SNE（大数据上很慢；不影响导出的表达 CSV）",
    )
    p.add_argument("--h5ad", type=Path, default=None, help="可选：写出完整 AnnData（含 UMAP/leiden 等）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-genes", type=int, default=200, help="每细胞最少检测基因数（对应 Seurat min.features）")
    p.add_argument("--min-cells", type=int, default=3, help="每基因最少覆盖细胞数（对应 Seurat min.cells）")
    p.add_argument(
        "--filter-cells-max-zero-frac",
        type=float,
        default=0.9,
        help="QC：去掉计数矩阵中「零表达基因占比 >= 该阈值」的细胞；设为 0 或 >1 则关闭此步",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="在临时目录生成最小 10x 并跑通流程（不依赖真实 GSE253718）",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    sc.settings.verbosity = 2
    sc.settings.set_figure_params(dpi=80, facecolor="white")

    if args.smoke_test:
        tmp = Path(tempfile.mkdtemp(prefix="gse253718_smoke_"))
        try:
            _write_minimal_10x(tmp, prefix="GSM_SMOKE")
            out_csv = tmp / "out.csv"
            out_h5 = tmp / "out.h5ad"
            fz = args.filter_cells_max_zero_frac
            if fz <= 0.0 or fz > 1.0:
                fz = None
            preprocess_raw_dir_to_csv(
                tmp,
                out_csv,
                h5ad_out=out_h5,
                random_state=args.seed,
                min_genes=1,
                min_cells=1,
                export_values=args.export_values,
                skip_tsne=args.skip_tsne,
                chunk_genes=args.chunk_genes,
                filter_cells_max_zero_frac=fz,
            )
            df = pd.read_csv(out_csv, nrows=5)
            full = pd.read_csv(out_csv, usecols=[0])
            assert "gene" in df.columns
            assert df.shape[1] == 11  # gene + 10 cells
            assert len(full) == 30
            print("[smoke-test] OK:", out_csv)
            print(df.iloc[:, :4])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return

    fz = args.filter_cells_max_zero_frac
    if fz <= 0.0 or fz > 1.0:
        fz = None
    preprocess_raw_dir_to_csv(
        args.raw_dir,
        args.output_csv,
        h5ad_out=args.h5ad,
        random_state=args.seed,
        min_genes=args.min_genes,
        min_cells=args.min_cells,
        export_values=args.export_values,
        skip_tsne=args.skip_tsne,
        chunk_genes=args.chunk_genes,
        filter_cells_max_zero_frac=fz,
    )
    print(
        f"[OK] 已写出: {args.output_csv.resolve()}（export_values={args.export_values}, 行=基因, 列=细胞）"
    )
    if args.h5ad:
        print(f"[OK] AnnData: {args.h5ad.resolve()}")


if __name__ == "__main__":
    main()
