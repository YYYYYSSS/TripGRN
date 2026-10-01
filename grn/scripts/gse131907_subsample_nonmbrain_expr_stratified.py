"""
GSE131907：对 non-mBrain 表达矩阵按 Feature Summary 中样本的「组织来源」分层，
每层随机保留约 fraction 的细胞（每层至少 1 个），列名须可匹配到 Samples（与 gse131907_feature_summary 一致）。

流式读写，避免整表进内存。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

from grn.data.gse131907_feature_summary import load_sample_tissue_pairs, match_column_to_sample_id


def _sample_indices(idxs: List[int], fraction: float, rng: random.Random) -> List[int]:
    n = len(idxs)
    if n <= 0:
        return []
    k = max(1, int(n * fraction))
    if k >= n:
        return list(idxs)
    return rng.sample(idxs, k)


def _build_tissue_groups(
    cell_cols: List[str],
    sample_to_tissue: Dict[str, str],
    sample_ids: List[str],
) -> Tuple[Dict[str, List[int]], int]:
    """列索引（0..len(cell_cols)-1）按组织来源分组；无法匹配的列归入 __UNMAPPED__。"""
    groups: DefaultDict[str, List[int]] = defaultdict(list)
    n_bad = 0
    for j, col in enumerate(cell_cols):
        sid = match_column_to_sample_id(col, sample_ids)
        if sid is None or sid not in sample_to_tissue:
            groups["__UNMAPPED__"].append(j)
            if sid is None:
                n_bad += 1
        else:
            groups[sample_to_tissue[sid]].append(j)
    return dict(groups), n_bad


def main() -> None:
    p = argparse.ArgumentParser(description="non-mBrain expr：按 Tissue origins 分层随机下采样列（流式）")
    p.add_argument("--expr-csv", type=Path, required=True, help="gene×cell CSV，首列为基因名")
    p.add_argument("--feature-xlsx", type=Path, required=True, help="GSE131907_Lung_Cancer_Feature_Summary.xlsx")
    p.add_argument("--fraction", type=float, default=0.3, help="每层保留比例（默认 0.3）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument(
        "--meta-json",
        type=Path,
        default=None,
        help="写出分层与抽样统计 JSON",
    )
    args = p.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        raise SystemExit("--fraction 须在 (0,1]")

    pairs = load_sample_tissue_pairs(str(args.feature_xlsx))
    sample_to_tissue = {s: t for s, t in pairs}
    sample_ids = sorted(sample_to_tissue.keys(), key=len, reverse=True)

    expr_path = args.expr_csv
    if not expr_path.is_file():
        raise SystemExit(f"缺少表达矩阵: {expr_path}")

    with expr_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise SystemExit("表达矩阵为空")
        if len(header) < 2:
            raise SystemExit("表头至少需要基因列 + 1 个细胞列")
        gene_col = header[0]
        cell_cols = header[1:]
        groups, n_unmapped_cols = _build_tissue_groups(cell_cols, sample_to_tissue, sample_ids)

    rng = random.Random(int(args.seed))
    picked: List[int] = []
    stats_layers: Dict[str, Dict[str, int]] = {}
    for tissue, idxs in sorted(groups.items(), key=lambda x: x[0]):
        before = len(idxs)
        take = _sample_indices(idxs, float(args.fraction), rng)
        picked.extend(take)
        stats_layers[tissue] = {"n_cells_before": before, "n_cells_kept": len(take)}
    picked.sort()

    kept_cell_names = [cell_cols[i] for i in picked]
    out_header = [gene_col] + kept_cell_names

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with expr_path.open("r", encoding="utf-8", errors="replace", newline="") as fin, args.out_csv.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout, lineterminator="\n")
        _ = next(reader, None)
        writer.writerow(out_header)
        pick_set = set(picked)
        for row in reader:
            if not row:
                continue
            g = row[0]
            vals = row[1 : len(cell_cols) + 1]
            if len(vals) < len(cell_cols):
                vals = vals + [""] * (len(cell_cols) - len(vals))
            sub = [vals[i] for i in picked]
            writer.writerow([g] + sub)

    meta = {
        "expr_csv": str(expr_path.resolve()),
        "feature_xlsx": str(args.feature_xlsx.resolve()),
        "fraction": float(args.fraction),
        "seed": int(args.seed),
        "n_cell_cols_in": len(cell_cols),
        "n_cell_cols_out": len(picked),
        "n_unmapped_cell_cols": len(groups.get("__UNMAPPED__", [])),
        "n_columns_no_sample_suffix_match": n_unmapped_cols,
        "strata": stats_layers,
    }
    if args.meta_json:
        args.meta_json.parent.mkdir(parents=True, exist_ok=True)
        args.meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[subsample] in_cells={len(cell_cols)} out_cells={len(picked)} strata={len(stats_layers)} -> {args.out_csv}"
    )


if __name__ == "__main__":
    main()
