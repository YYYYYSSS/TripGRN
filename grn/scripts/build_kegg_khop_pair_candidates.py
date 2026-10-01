"""
仅用 KEGG 调控边（不引入 PPI）在「表达基因面板」上构无向图，
对每个在面板内且曾作为 GRN source 出现的基因 s，取图距离 ≤k 的邻居 t，
生成候选正样本对 (s, t), t≠s；再按 neg_sample_ratio 采样负样本对。

实现见 grn.data.kegg_khop_pairs；本脚本为 CLI 包装。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd

from grn.data.kegg_khop_pairs import (
    build_adj,
    collect_khop_positive_pairs,
    grn_sources_in_panel,
    kegg_edges_in_panel,
    sample_khop_negatives,
)


def _load_panel_genes(
    exp_csv: str | None,
    exp_npy_genes: str | None,
    cds_csv: str | None,
) -> List[str]:
    from grn.data.preprocessing import normalize_gene_symbol_series

    if exp_csv:
        df = pd.read_csv(exp_csv)
        gcol = df.columns[0]
        df[gcol] = normalize_gene_symbol_series(df[gcol])
        genes: Set[str] = set(df[gcol].astype(str))
    elif exp_npy_genes:
        g_arr = np.load(exp_npy_genes, allow_pickle=True)
        genes = set(normalize_gene_symbol_series(pd.Series([str(x) for x in g_arr.tolist()])).astype(str))
    else:
        raise SystemExit("须指定 --exp-csv 或 --exp-npy-genes")

    if cds_csv:
        cdf = pd.read_csv(cds_csv)
        c0 = cdf.columns[0]
        cdf[c0] = normalize_gene_symbol_series(cdf[c0].astype(str))
        genes &= set(cdf[c0].astype(str).tolist())
    return sorted(genes)


def main() -> None:
    p = argparse.ArgumentParser(description="KEGG-only k-hop pair candidates + negative sampling (no PPI).")
    p.add_argument("--grn-csv", default=None, help="KEGG 边表；可省略若提供 --step1-run-dir 且其中有 grn_csv")
    p.add_argument("--grn-tf-col", default="TF")
    p.add_argument("--grn-target-col", default="target")
    p.add_argument("--grn-type-col", default="Re_type")
    p.add_argument("--exp-csv", default=None)
    p.add_argument("--exp-npy-genes", default=None)
    p.add_argument("--cds-gene-feature-csv", default=None, help="与训练一致时可传入 CDS 特征表以缩小面板")
    p.add_argument("--step1-run-dir", default=None, help="若给定，从 run_config.json 读取 grn/cds 列名与路径（可被显式参数覆盖）")
    p.add_argument("--k-hops", type=int, default=3)
    p.add_argument("--neg-sample-ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--neg-tf-from-sources-only",
        action="store_true",
        help="负样本仅随机选 GRN source 作第一端点（更像 TF→target）；默认两端均在面板内均匀随机",
    )
    p.add_argument("--out-csv", default=None, help="写出 TF,Target,label 列（label 1=正 0=负）；大文件慎用")
    p.add_argument("--count-only", action="store_true", help="只打印统计，不写 CSV")
    args = p.parse_args()

    grn_csv = args.grn_csv
    tf_c, tg_c, tp_c = args.grn_tf_col, args.grn_target_col, args.grn_type_col
    cds_path = args.cds_gene_feature_csv

    if args.step1_run_dir:
        rc = Path(args.step1_run_dir) / "run_config.json"
        if rc.is_file():
            cfg = json.loads(rc.read_text(encoding="utf-8"))
            if grn_csv is None and cfg.get("grn_csv"):
                grn_csv = str(cfg["grn_csv"])
            if args.cds_gene_feature_csv is None and cfg.get("cds_gene_feature_csv"):
                cds_path = str(cfg["cds_gene_feature_csv"])
            tf_c = str(cfg.get("grn_tf_col", tf_c))
            tg_c = str(cfg.get("grn_target_col", tg_c))
            tp_c = str(cfg.get("grn_type_col", tp_c))

    if not grn_csv:
        raise SystemExit("请指定 --grn-csv 或提供含 grn_csv 的 --step1-run-dir")

    panel_list = _load_panel_genes(args.exp_csv, args.exp_npy_genes, cds_path)
    panel: Set[str] = set(panel_list)
    if not panel_list:
        raise SystemExit("基因面板为空")

    edges = kegg_edges_in_panel(grn_csv, tf_c, tg_c, tp_c, panel)
    sources = grn_sources_in_panel(grn_csv, tf_c, tg_c, tp_c, panel)
    adj = build_adj(edges)
    pos_pairs = collect_khop_positive_pairs(sources, panel, adj, int(args.k_hops))

    n_pos = len(pos_pairs)
    n_neg = int(n_pos * float(args.neg_sample_ratio))
    rng = np.random.RandomState(int(args.seed))
    g2i = {g: i for i, g in enumerate(panel_list)}
    sources_idx = {g2i[s] for s in sources if s in g2i}
    neg_pairs = sample_khop_negatives(
        rng,
        panel_list,
        pos_pairs,
        n_neg,
        neg_tf_from_sources_only=bool(args.neg_tf_from_sources_only),
        sources_idx=sources_idx,
    )

    out = {
        "grn_csv": grn_csv,
        "n_panel_genes": len(panel_list),
        "n_kegg_undirected_edges_in_panel": len(edges),
        "k_hops": int(args.k_hops),
        "n_grn_sources_in_panel": len(sources),
        "n_positive_pairs_khop": n_pos,
        "neg_sample_ratio": float(args.neg_sample_ratio),
        "n_negative_pairs_sampled": len(neg_pairs),
        "n_total_pairs": n_pos + len(neg_pairs),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.count_only or not args.out_csv:
        return

    rows = [{"TF": a, "Target": b, "label": 1} for a, b in sorted(pos_pairs)]
    rows.extend([{"TF": a, "Target": b, "label": 0} for a, b in neg_pairs])
    pd.DataFrame(rows).to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[ok] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
