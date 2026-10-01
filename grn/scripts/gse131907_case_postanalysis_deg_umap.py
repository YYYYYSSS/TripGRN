"""
GSE131907 case 后分析：
1) 差异网络基因在 mBrain vs non-mBrain 两侧表达（Wilcoxon rank-sum，FDR-BH）。
2) 在差异网络基因子空间 + 随机下采样细胞上做 UMAP，按 Cell_type 与 mBrain/non-mBrain 着色。

依赖：pandas numpy scipy matplotlib scikit-learn；可选 umap-learn。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Set

import gzip
import numpy as np
import pandas as pd


def _read_expr_gz(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip")
    gcol = df.columns[0]
    from grn.data.preprocessing import normalize_gene_symbol_series

    df[gcol] = normalize_gene_symbol_series(df[gcol].astype(str))
    df = df.drop_duplicates(subset=[gcol], keep="first").set_index(gcol)
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _load_ann_celltype(ann_gz: Path) -> pd.Series:
    rows = []
    with gzip.open(ann_gz, "rt", encoding="utf-8", errors="replace") as f:
        h = f.readline().rstrip("\n").split("\t")
        i_idx = h.index("Index")
        i_ct = h.index("Cell_type")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) > max(i_idx, i_ct):
                rows.append((p[i_idx], p[i_ct]))
    df = pd.DataFrame(rows, columns=["Index", "Cell_type"])
    return df.drop_duplicates("Index").set_index("Index")["Cell_type"]


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = np.empty(m, dtype=float)
    ranked[order] = np.arange(1, m + 1)
    q = p * m / ranked
    return np.minimum.accumulate(np.sort(q)[::-1])[::-1][np.argsort(order)]


def main() -> None:
    p = argparse.ArgumentParser(description="GSE131907 case: DE on diff-net genes + UMAP")
    p.add_argument("--non-mbrain-csv-gz", type=Path, required=True)
    p.add_argument("--mbrain-csv-gz", type=Path, required=True)
    p.add_argument("--cell-annotation-gz", type=Path, required=True)
    p.add_argument("--case-tables-dir", type=Path, required=True, help="含 differential_edges_used.csv")
    p.add_argument("--out-tables-dir", type=Path, default=None)
    p.add_argument("--out-figures-dir", type=Path, default=None)
    p.add_argument("--max-umap-cells", type=int, default=25000)
    p.add_argument("--max-umap-genes", type=int, default=400, help="UMAP 用差异网络基因上限")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tab = Path(args.case_tables_dir).resolve()
    out_t = Path(args.out_tables_dir).resolve() if args.out_tables_dir else tab
    out_f = Path(args.out_figures_dir).resolve() if args.out_figures_dir else tab.parent / "figures"
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)

    edges_path = tab / "differential_edges_used.csv"
    if not edges_path.is_file():
        raise SystemExit(f"missing {edges_path}")
    edges = pd.read_csv(edges_path)
    genes_net: Set[str] = set()
    for c in ("TF", "Target"):
        if c in edges.columns:
            genes_net.update(edges[c].astype(str).unique())
    genes_net.discard("")
    if len(genes_net) < 2:
        raise SystemExit("too few genes in differential network")

    print(f"[post] loading expression …")
    nb = _read_expr_gz(Path(args.non_mbrain_csv_gz))
    mb = _read_expr_gz(Path(args.mbrain_csv_gz))
    common = sorted(genes_net & set(nb.index) & set(mb.index))
    if len(common) < 2:
        raise SystemExit("diff-net genes missing in both expression matrices")

    from scipy.stats import mannwhitneyu

    rng = np.random.default_rng(int(args.seed))
    p_list: List[float] = []
    log2fc: List[float] = []
    mean_nb: List[float] = []
    mean_mb: List[float] = []
    for g in common:
        x = nb.loc[g].values.astype(float)
        y = mb.loc[g].values.astype(float)
        _, pv = mannwhitneyu(x, y, alternative="two-sided")
        p_list.append(float(pv))
        mean_nb.append(float(np.mean(x)))
        mean_mb.append(float(np.mean(y)))
        log2fc.append(float(np.log2((np.mean(y) + 1.0) / (np.mean(x) + 1.0))))
    p_arr = np.array(p_list, dtype=float)
    q_arr = _bh_fdr(p_arr)
    de_df = pd.DataFrame(
        {
            "gene": common,
            "mean_non_mBrain": mean_nb,
            "mean_mBrain": mean_mb,
            "log2FC_mBrain_vs_non_mBrain": log2fc,
            "pvalue_wilcox": p_list,
            "padj_BH": q_arr.tolist(),
        }
    ).sort_values("padj_BH", ascending=True)
    de_path = out_t / "diffnet_genes_mBrain_vs_non_mBrain_DE.csv"
    de_df.to_csv(de_path, index=False, encoding="utf-8-sig")
    print(f"[post] DE -> {de_path} (n_genes={len(de_df)})")

    ct_series = _load_ann_celltype(Path(args.cell_annotation_gz))
    max_c = int(args.max_umap_cells)
    nb_cols = list(nb.columns)
    mb_cols = list(mb.columns)

    def _pick(cols: List[str], cap: int) -> List[str]:
        if len(cols) <= cap:
            return cols
        return list(rng.choice(cols, size=cap, replace=False))

    half = max(1, max_c // 2)
    snb = _pick(nb_cols, half)
    smb = _pick(mb_cols, max_c - len(snb))
    use_cols = [(c, "non-mBrain") for c in snb] + [(c, "mBrain") for c in smb]
    genes_umap = common[: int(args.max_umap_genes)]

    X = np.zeros((len(use_cols), len(genes_umap)), dtype=np.float64)
    cell_types: List[str] = []
    groups: List[str] = []
    for i, (col, grp) in enumerate(use_cols):
        if col in nb.columns:
            row = nb.loc[genes_umap, col].values.astype(float)
        else:
            row = mb.loc[genes_umap, col].values.astype(float)
        X[i, :] = row
        cell_types.append(str(ct_series[col]) if col in ct_series.index else "unknown")
        groups.append(grp)

    X = np.log1p(X)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(X)
    n_comp = min(30, Z.shape[0] - 1, Z.shape[1])
    pca = PCA(n_components=max(2, n_comp), random_state=int(args.seed))
    U = pca.fit_transform(Z)
    emb = U[:, :2].copy()
    try:
        import umap

        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=int(args.seed))
        emb = reducer.fit_transform(U[:, : min(30, U.shape[1])])
    except Exception:
        pass

    import matplotlib.pyplot as plt

    def _scatter_color(cat: List[str], title: str, fname: str) -> None:
        uq = sorted(set(cat))
        cmap = plt.cm.tab20(np.linspace(0, 1, 20))
        col_map = {lab: cmap[i % len(cmap)] for i, lab in enumerate(uq)}
        colors = [col_map[c] for c in cat]
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=4, alpha=0.65, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        show = uq[:20]
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col_map[l], markersize=6, label=l) for l in show
        ]
        if len(uq) > 20:
            handles.append(
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="0.5", markersize=6, label=f"+{len(uq)-20} types")
            )
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        fig.tight_layout()
        fig.savefig(out_f / fname, dpi=160, bbox_inches="tight")
        plt.close(fig)

    _scatter_color(cell_types, "UMAP (diff-net genes) colored by Cell_type", "umap_cell_type.png")
    _scatter_color(groups, "UMAP (diff-net genes) colored by non-mBrain vs mBrain", "umap_mBrain_group.png")

    meta = {
        "n_diffnet_genes_de": int(len(de_df)),
        "n_umap_cells": int(len(use_cols)),
        "n_umap_genes": int(len(genes_umap)),
        "de_csv": de_path.name,
        "figures": ["umap_cell_type.png", "umap_mBrain_group.png"],
    }
    (out_t / "postanalysis_deg_umap_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[post] figures -> {out_f}")


if __name__ == "__main__":
    main()
