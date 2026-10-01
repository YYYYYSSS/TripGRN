"""
从 STRING protein.links + protein.info 生成与旧 PPI.csv 同列式的边表（from,to,combined_score），
供 PPIGraphBuilder 快速 pd.read_csv 加载。推荐生成**普适人类网络**（不按某张表达矩阵筛基因）：
训练/推理时用当前任务的 gene_list 在内存中自动取子图。

示例（全库符号边；links 文件内 combined_score 为 0–1000 整数，对应 STRING 常用分档约
0.15/0.4/0.7/0.9 → 150/400/700/900；默认取最强档 ≥900）：
  python -m grn.scripts.build_ppi_csv_from_string \\
    --links data/network/ppi/9606.protein.links.v12.0.txt.gz \\
    --output-csv data/network/ppi/ppi_string_v12_human_score900.csv

可选 --reference-gene-list-csv：仅当需要**单独导出**某基因面板的小 CSV（例如发布附件）时再使用；
日常训练应指向上面的普适 CSV，由代码按表达谱基因筛选。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
from typing import Dict, Optional, Set, Tuple


def _resolve_info_path(links_path: str) -> str:
    d = os.path.dirname(os.path.abspath(links_path))
    base = os.path.basename(links_path)
    if "protein.links" not in base:
        raise ValueError(f"links 文件名需含 protein.links，便于推断 protein.info：{base!r}")
    info_name = base.replace("protein.links", "protein.info")
    candidates = [os.path.join(d, info_name)]
    if info_name.endswith(".txt.gz"):
        candidates.append(os.path.join(d, info_name[:-3]))
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "需要同目录下的 STRING protein.info（9606.ENSP… → preferred_name）。\n"
        f"尝试过: {candidates}"
    )


def _load_protein_id_to_symbol(info_path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    open_fn = gzip.open if info_path.endswith(".gz") else open
    mode = "rt" if info_path.endswith(".gz") else "r"
    with open_fn(info_path, mode, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            pid, name = parts[0].strip(), parts[1].strip()
            if not name or name == "-":
                continue
            out[pid] = name
    return out


def _read_reference_genes(path: str) -> Set[str]:
    import pandas as pd

    df = pd.read_csv(path, usecols=[0])
    g = set(df.iloc[:, 0].astype(str).str.strip())
    g.discard("")
    return g


def main() -> None:
    parser = argparse.ArgumentParser(description="STRING links → gene-symbol PPI CSV（from,to,combined_score）")
    parser.add_argument("--links", required=True, help="9606.protein.links.v12.0.txt.gz")
    parser.add_argument(
        "--info",
        default=None,
        help="9606.protein.info.v12.0.txt.gz；默认与 links 同目录且文件名替换 protein.links→protein.info",
    )
    parser.add_argument("--output-csv", required=True, help="输出 CSV 路径")
    parser.add_argument(
        "--reference-gene-list-csv",
        default=None,
        help="可选：只导出两端均在该 CSV 第一列的边（缩小单文件体积）；默认可不传，用全库普适网络 + 训练时按 gene_list 筛选",
    )
    parser.add_argument(
        "--min-combined-score",
        type=int,
        default=900,
        help="STRING combined_score 下限（0–1000，约等于官网分档×1000；默认 900 对应最强档 0.9）。",
    )
    args = parser.parse_args()

    links_path = os.path.abspath(args.links)
    info_path = os.path.abspath(args.info) if args.info else _resolve_info_path(links_path)
    out_path = os.path.abspath(args.output_csv)

    gene_filter: Optional[Set[str]] = None
    if args.reference_gene_list_csv:
        gene_filter = _read_reference_genes(args.reference_gene_list_csv)
        print(f"[build-ppi-csv] reference genes: {len(gene_filter)}")

    print(f"[build-ppi-csv] loading protein.info: {info_path}")
    pid_sym = _load_protein_id_to_symbol(info_path)
    print(f"[build-ppi-csv] id→symbol entries: {len(pid_sym)}")

    # 无向去重：键为排序后的 (a,b)，combined_score 取同对上的 max（与拓扑一致，减小文件）
    pair_best: Dict[Tuple[str, str], int] = {}
    n_in = 0
    n_mapped = 0
    n_pass_gene_filter = 0

    open_fn = gzip.open if links_path.endswith(".gz") else open
    mode = "rt" if links_path.endswith(".gz") else "r"
    print(f"[build-ppi-csv] scanning links: {links_path}")
    with open_fn(links_path, mode, encoding="utf-8", errors="replace") as f:
        header = f.readline()
        if not (header.lstrip().startswith("protein1") and "protein2" in header):
            raise ValueError(f"表头异常: {header[:120]!r}")
        for line in f:
            n_in += 1
            parts = line.split()
            if len(parts) < 3:
                continue
            p1, p2, sc_s = parts[0], parts[1], parts[2]
            g1 = pid_sym.get(p1)
            g2 = pid_sym.get(p2)
            if g1 is None or g2 is None:
                continue
            n_mapped += 1
            if g1 == g2:
                continue
            if gene_filter is not None and (g1 not in gene_filter or g2 not in gene_filter):
                continue
            try:
                sc = int(float(sc_s))
            except ValueError:
                continue
            if sc < int(args.min_combined_score):
                continue
            n_pass_gene_filter += 1
            a, b = (g1, g2) if g1 <= g2 else (g2, g1)
            key = (a, b)
            prev = pair_best.get(key)
            if prev is None or sc > prev:
                pair_best[key] = sc

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["from", "to", "combined_score"])
        for (a, b), sc in sorted(pair_best.items()):
            w.writerow([a, b, sc])

    print(
        f"[build-ppi-csv] lines_read={n_in} mapped_pairs={n_mapped} "
        f"lines_pass_gene_filter={n_pass_gene_filter} unique_undirected_rows={len(pair_best)}"
    )
    print(f"[build-ppi-csv] wrote {out_path}")


if __name__ == "__main__":
    main()
