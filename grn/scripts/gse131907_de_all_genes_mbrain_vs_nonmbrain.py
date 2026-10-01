"""
对 non-mBrain / mBrain 两侧表达矩阵的共有基因，逐基因 Wilcoxon（mBrain vs non-mBrain），
输出全基因 DE 表，供 case study 按 p 阈值筛基因后做有向笛卡尔积建对。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from grn.data.preprocessing import normalize_gene_symbol_series


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = np.empty(m, dtype=float)
    ranked[order] = np.arange(1, m + 1)
    q = p * m / ranked
    return np.minimum.accumulate(np.sort(q)[::-1])[::-1][np.argsort(order)]


def _load_expr_gz(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip")
    gcol = df.columns[0]
    df[gcol] = normalize_gene_symbol_series(df[gcol].astype(str))
    df = df.drop_duplicates(subset=[gcol], keep="first").set_index(gcol)
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def main() -> None:
    p = argparse.ArgumentParser(description="GSE131907: all genes DE mBrain vs non-mBrain (Wilcoxon)")
    p.add_argument("--non-mbrain-csv-gz", type=Path, required=True)
    p.add_argument("--mbrain-csv-gz", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    args = p.parse_args()

    nb = _load_expr_gz(Path(args.non_mbrain_csv_gz))
    mb = _load_expr_gz(Path(args.mbrain_csv_gz))
    genes = sorted(set(nb.index) & set(mb.index))
    if len(genes) < 2:
        raise SystemExit("no shared genes")

    rows = []
    for gi, g in enumerate(genes):
        if gi % 2000 == 0:
            print(f"[de-all] {gi}/{len(genes)} …")
        x = nb.loc[g].values.astype(float)
        y = mb.loc[g].values.astype(float)
        _, pv = mannwhitneyu(x, y, alternative="two-sided")
        mx = float(np.mean(x))
        my = float(np.mean(y))
        rows.append(
            {
                "gene": g,
                "mean_non_mBrain": mx,
                "mean_mBrain": my,
                "log2FC_mBrain_vs_non_mBrain": float(np.log2((my + 1.0) / (mx + 1.0))),
                "pvalue_wilcox": float(pv),
            }
        )

    de = pd.DataFrame(rows)
    de["padj_BH"] = _bh_fdr(de["pvalue_wilcox"].values)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    de.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[de-all] wrote n={len(de)} -> {out}")


if __name__ == "__main__":
    main()
