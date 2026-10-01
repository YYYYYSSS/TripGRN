# -*- coding: utf-8 -*-
"""
在单套 pancancer 宽大表达 CSV 上跑 DeepTGI（project_DeepTGI/program/main_mHSC.py）的未见细胞流程。

- 表达：GRN 仓库 data/exp/pancancer 常见格式为「基因×细胞」；此处转置为 DeepTGI 所需「细胞×基因」。
- 标签：KEGG activation/inhibition 为正例；负例仅在「正例边涉及的基因池」内随机采样，避免把全基因组拉进 PCC 特征。
  可选 --max-genes-pcc 限制池中基因数（默认 4096），否则特征长度 2×基因数×样本数 会爆内存。
- 训练：调用 DeepTGI 的 Pearson PCC 特征 + 5:2:3 细胞划分；默认仅跑 1 次划分，`--run-seed 42` 固定细胞打乱与 torch/numpy。

依赖：conda 环境需含 torch、pandas、numpy、scikit-learn、openpyxl；与 DeepTGI main_mHSC 一致。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import pandas as pd


from grn.scripts.pancancer_step01_paths import load_run_meta, step01_layout


def _resolve_deeptgi_root(cli: Path | None) -> Path:
    if cli is not None:
        return cli.resolve()
    env = os.environ.get("DEEPTGI_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    grn_root = Path(__file__).resolve().parents[2]
    return (grn_root.parent / "project_DeepTGI").resolve()


def _gene_canonical_map(genes_raw: List[str]) -> Tuple[dict, List[str]]:
    """upper -> first-seen canonical name; preserves list order for sampling."""
    upper_to_canon: dict = {}
    order: List[str] = []
    for g in genes_raw:
        s = str(g).strip()
        if not s:
            continue
        u = s.upper()
        if u not in upper_to_canon:
            upper_to_canon[u] = s
            order.append(s)
    return upper_to_canon, order


def load_kegg_positive_pairs(kegg_path: Path, upper_to_canon: dict) -> List[Tuple[str, str]]:
    """Return list of (tf_canon, gene_canon) for activation/inhibition, deduped."""
    df = pd.read_csv(kegg_path)
    if "source_symbol" not in df.columns or "target_symbol" not in df.columns:
        raise SystemExit("KEGG CSV must contain source_symbol, target_symbol, relation_subtype")
    seen: Set[Tuple[str, str]] = set()
    out: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        rel = str(row.get("relation_subtype", "")).strip().lower()
        if rel not in ("activation", "inhibition"):
            continue
        su = str(row.get("source_symbol", "")).strip().upper()
        tu = str(row.get("target_symbol", "")).strip().upper()
        if not su or not tu or su == tu:
            continue
        if su not in upper_to_canon or tu not in upper_to_canon:
            continue
        key = (su, tu)
        if key in seen:
            continue
        seen.add(key)
        out.append((upper_to_canon[su], upper_to_canon[tu]))
    return out


def sample_negatives(
    positives_upper: Set[Tuple[str, str]],
    genes_canon: List[str],
    upper_to_canon: dict,
    n_neg: int,
    seed: int,
) -> List[Tuple[str, str]]:
    rng = np.random.default_rng(seed)
    n = len(genes_canon)
    if n < 2:
        return []
    neg: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set(positives_upper)
    max_attempts = max(50 * n_neg, 500_000)
    attempts = 0
    while len(neg) < n_neg and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        a = genes_canon[i]
        b = genes_canon[j]
        ua, ub = a.strip().upper(), b.strip().upper()
        key = (ua, ub)
        if key in seen:
            continue
        seen.add(key)
        neg.append((upper_to_canon[ua], upper_to_canon[ub]))
    return neg


def restrict_gene_pool_for_pcc(
    pos_pairs: List[Tuple[str, str]],
    max_genes: int,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    DeepTGI 特征长度 = 2 × PCC 矩阵基因个数 × 样本数；必须把参与 PCC 的基因限制在较小集合。
    在正例边涉及的基因里，若超过 max_genes，则按在正例中的度数优先保留，再过滤正例边。
    """
    if not pos_pairs:
        return [], []
    deg: Counter[str] = Counter()
    for tf, g in pos_pairs:
        deg[tf] += 1
        deg[g] += 1
    pool_all = sorted(set(tf for tf, _ in pos_pairs) | set(g for _, g in pos_pairs))
    if max_genes <= 0 or len(pool_all) <= max_genes:
        return pos_pairs, pool_all
    ranked = sorted(pool_all, key=lambda x: (-deg[x], x))
    kept = set(ranked[:max_genes])
    pos_filt = [(tf, g) for tf, g in pos_pairs if tf in kept and g in kept]
    return pos_filt, sorted(kept)


def build_label_dataframe(
    kegg_path: Path,
    upper_to_canon: dict,
    genes_ordered: List[str],
    neg_ratio: float,
    label_seed: int,
    max_genes_pcc: int,
) -> pd.DataFrame:
    expressed_upper = {str(g).strip().upper() for g in genes_ordered}
    pos_pairs = load_kegg_positive_pairs(kegg_path, upper_to_canon)
    pos_pairs = [
        (tf, g)
        for tf, g in pos_pairs
        if tf.strip().upper() in expressed_upper and g.strip().upper() in expressed_upper
    ]
    if not pos_pairs:
        raise SystemExit("No KEGG positive edges mapped to expression genes (check symbols vs Ensembl).")

    pos_pairs, gene_pool = restrict_gene_pool_for_pcc(pos_pairs, max_genes_pcc)
    if not pos_pairs:
        raise SystemExit(
            "After --max-genes-pcc filtering, no positive pairs left; increase --max-genes-pcc "
            "or check KEGG mapping."
        )

    pos_upper = {(p[0].strip().upper(), p[1].strip().upper()) for p in pos_pairs}
    n_pos = len(pos_pairs)
    n_neg_target = int(min(500_000, max(int(n_pos * neg_ratio), n_pos)))
    # 负例仅在 gene_pool 内采样，使标签中出现的基因 ⊆ gene_pool，PCC 只加载这些列
    negs = sample_negatives(pos_upper, gene_pool, upper_to_canon, n_neg_target, label_seed)

    rows = [(tf, g, 1) for tf, g in pos_pairs] + [(tf, g, 0) for tf, g in negs]
    df = pd.DataFrame(rows, columns=["tf", "gene", "label"])
    return df


def wide_genes_cells_to_cells_genes(wide_csv: Path, out_csv: Path) -> List[str]:
    """Wide genes×cells -> DeepTGI cells×genes. Returns gene names (canonical row index order)."""
    wide = pd.read_csv(wide_csv, index_col=0)
    wide.index = wide.index.astype(str).str.strip()
    genes = list(wide.index)
    expr_t = wide.astype(np.float32).T
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    expr_t.to_csv(out_csv, index=False)
    return genes


def npy_genes_cells_to_cells_genes(values_npy: Path, genes_npy: Path, out_csv: Path) -> List[str]:
    """pancancer_scdata_step5 npy (genes×cells) -> DeepTGI cells×genes CSV."""
    g_arr = np.load(genes_npy, allow_pickle=True)
    raw = np.load(values_npy, mmap_mode="r")
    gene_list = [str(x).strip() for x in g_arr.tolist()]
    if raw.shape[0] != len(gene_list):
        raise ValueError("genes.npy row count != values.npy rows")
    expr_t = pd.DataFrame(np.asarray(raw, dtype=np.float32).T, columns=gene_list)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    expr_t.to_csv(out_csv, index=False)
    return gene_list


def append_cross_cancer_row(summary_csv: Path, row: dict) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_csv.is_file()
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepTGI pancancer one GSE driver.")
    grn_root = Path(__file__).resolve().parents[2]
    layout = step01_layout(grn_root)
    ap.add_argument("--exp-wide-csv", type=Path, default=None, help="step01: data/exp/pancancer/<GSE>.csv")
    ap.add_argument("--exp-npy-values", type=Path, default=None, help="pancancer_scdata_step5/<GSE>_values.npy")
    ap.add_argument("--exp-npy-genes", type=Path, default=None, help="pancancer_scdata_step5/<GSE>_genes.npy")
    ap.add_argument(
        "--reuse-run-dir",
        type=Path,
        default=None,
        help="Reuse prepared expr/label in an existing DeepTGI run (skip CSV/npy prep).",
    )
    ap.add_argument(
        "--kegg-csv",
        type=Path,
        default=layout["kegg_csv"],
        help="step01 KEGG CSV (data/network/grn/KEGG_hsa_PPrel_...).",
    )
    ap.add_argument(
        "--output-base",
        type=Path,
        default=layout["out_tgi"],
        help="step01: experiments/deeptgi_pancancer",
    )
    ap.add_argument("--gse-id", type=str, required=True)
    ap.add_argument("--deeptgi-root", type=Path, default=None)
    ap.add_argument("--neg-ratio", type=float, default=5.0)
    ap.add_argument(
        "--max-genes-pcc",
        type=int,
        default=4096,
        help="Cap genes in PCC or label union; feature dim ≈ 2×this. Use 0 to disable cap not recommended.",
    )
    ap.add_argument("--label-seed", type=int, default=42)
    ap.add_argument("--cv-repeats", type=int, default=1, help="Cell 5:2:3 split repeats; 1 = single holdout.")
    ap.add_argument(
        "--run-seed",
        type=int,
        default=42,
        help="DeepTGI train/split RNG (numpy, torch, cell shuffle).",
    )
    ap.add_argument("--cuda-device", type=str, default="0")
    ap.add_argument("--deeptgi-epochs", type=int, default=12, help="BERT_train max epochs early stopping may finish earlier.")
    ap.add_argument("--deeptgi-batch-size", type=int, default=512, help="Increase if VRAM allows.")
    ap.add_argument(
        "--deeptgi-loader-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader workers try 2 to 8 if CPU is idle.",
    )
    ap.add_argument(
        "--cross-cancer-summary",
        type=Path,
        default=layout["tgi_summary_csv"],
        help="Append one CSV row with mean metrics after success (step01 SUMMARY).",
    )
    args = ap.parse_args()

    deeptgi_root = _resolve_deeptgi_root(args.deeptgi_root)
    cli = deeptgi_root / "program" / "run_unseen_cells_cli.py"
    if not cli.is_file():
        raise SystemExit(f"missing DeepTGI CLI: {cli}")

    gse = str(args.gse_id).strip()

    if args.reuse_run_dir is not None:
        run_dir = Path(args.reuse_run_dir).resolve()
        meta = load_run_meta(run_dir)
        expr_t_path = Path(str(meta["expr_cells_genes_csv"]))
        label_path = Path(str(meta["label_csv"]))
        metrics_xlsx = run_dir / "deeptgi_metrics.xlsx"
        if not expr_t_path.is_file() or not label_path.is_file():
            raise SystemExit(f"reuse-run-dir missing expr/label under {run_dir}")
        cv_repeats = int(meta.get("cv_repeats", args.cv_repeats))
        run_seed = int(meta.get("run_seed", args.run_seed))
        deeptgi_epochs = int(meta.get("deeptgi_epochs", args.deeptgi_epochs))
        deeptgi_batch_size = int(meta.get("deeptgi_batch_size", args.deeptgi_batch_size))
        deeptgi_loader_workers = int(meta.get("deeptgi_dataloader_workers", args.deeptgi_loader_workers))
    else:
        has_npy = bool(args.exp_npy_values) and bool(args.exp_npy_genes)
        if bool(args.exp_npy_values) ^ bool(args.exp_npy_genes):
            raise SystemExit("--exp-npy-values 与 --exp-npy-genes 须同时指定")
        if args.exp_wide_csv is None and not has_npy:
            v_def = layout["npy_dir"] / f"{gse}_values.npy"
            g_def = layout["npy_dir"] / f"{gse}_genes.npy"
            if v_def.is_file() and g_def.is_file():
                args = argparse.Namespace(**{**vars(args), "exp_npy_values": v_def, "exp_npy_genes": g_def})
                has_npy = True
            else:
                raise SystemExit("Need --exp-wide-csv, --exp-npy-values/genes, or --reuse-run-dir")

        log_time = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())
        run_dir = Path(args.output_base).resolve() / gse / ("deeptgi_%s" % log_time)
        run_dir.mkdir(parents=True, exist_ok=True)

        expr_t_path = run_dir / ("%s_cells_x_genes.csv" % gse)
        label_path = run_dir / "label.csv"
        meta_path = run_dir / "run_meta.json"
        metrics_xlsx = run_dir / "deeptgi_metrics.xlsx"

        if has_npy:
            genes = npy_genes_cells_to_cells_genes(
                Path(args.exp_npy_values).resolve(),
                Path(args.exp_npy_genes).resolve(),
                expr_t_path,
            )
            exp_src = str(Path(args.exp_npy_values).resolve())
        else:
            genes = wide_genes_cells_to_cells_genes(Path(args.exp_wide_csv).resolve(), expr_t_path)
            exp_src = str(Path(args.exp_wide_csv).resolve())

        upper_to_canon, genes_ordered = _gene_canonical_map(genes)

        label_df = build_label_dataframe(
            Path(args.kegg_csv).resolve(),
            upper_to_canon,
            genes_ordered,
            neg_ratio=float(args.neg_ratio),
            label_seed=int(args.label_seed),
            max_genes_pcc=int(args.max_genes_pcc),
        )
        label_df.to_csv(label_path, index=False)

        meta_path.write_text(
            json.dumps(
                {
                    "gse_id": gse,
                    "deeptgi_root": str(deeptgi_root),
                    "exp_wide_csv": exp_src if not has_npy else None,
                    "exp_npy_values": str(Path(args.exp_npy_values).resolve()) if has_npy else None,
                    "exp_npy_genes": str(Path(args.exp_npy_genes).resolve()) if has_npy else None,
                    "kegg_csv": str(Path(args.kegg_csv).resolve()),
                    "expr_cells_genes_csv": str(expr_t_path),
                    "label_csv": str(label_path),
                    "n_label_rows": int(len(label_df)),
                    "n_positive": int((label_df["label"] == 1).sum()),
                    "n_negative": int((label_df["label"] == 0).sum()),
                    "neg_ratio_requested": float(args.neg_ratio),
                    "max_genes_pcc": int(args.max_genes_pcc),
                    "cv_repeats": int(args.cv_repeats),
                    "run_seed": int(args.run_seed),
                    "label_seed": int(args.label_seed),
                    "deeptgi_epochs": int(args.deeptgi_epochs),
                    "deeptgi_batch_size": int(args.deeptgi_batch_size),
                    "deeptgi_dataloader_workers": int(args.deeptgi_loader_workers),
                    "protocol": "DeepTGI main_mHSC: PCC, single or repeated 5:2:3 unseen cell splits",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        cv_repeats = int(args.cv_repeats)
        run_seed = int(args.run_seed)
        deeptgi_epochs = int(args.deeptgi_epochs)
        deeptgi_batch_size = int(args.deeptgi_batch_size)
        deeptgi_loader_workers = int(args.deeptgi_loader_workers)

    cmd = [
        sys.executable,
        str(cli),
        "--expr-csv",
        str(expr_t_path),
        "--label-csv",
        str(label_path),
        "--output-xlsx",
        str(metrics_xlsx),
        "--cv-repeats",
        str(cv_repeats),
        "--run-seed",
        str(run_seed),
        "--cuda-visible-devices",
        str(args.cuda_device),
        "--epochs",
        str(deeptgi_epochs),
        "--batch-size",
        str(deeptgi_batch_size),
        "--dataloader-workers",
        str(deeptgi_loader_workers),
    ]
    subprocess.check_call(cmd, cwd=str(cli.parent))

    if args.cross_cancer_summary is not None and args.reuse_run_dir is None:
        summary_df = pd.read_excel(metrics_xlsx, sheet_name="summary")
        mean_row = summary_df.loc[summary_df["stat"] == "mean"].iloc[0]
        out_row = {
            "gse_id": gse,
            "run_dir": str(run_dir),
            "AUC_mean": float(mean_row["AUC"]),
            "AUPR_mean": float(mean_row["AUPR"]),
            "ACC_mean": float(mean_row["ACC"]),
            "Recall_mean": float(mean_row["Recall"]),
            "Precision_mean": float(mean_row["Precision"]),
            "F1_mean": float(mean_row["F1"]),
            "MCC_mean": float(mean_row["MCC"]),
        }
        append_cross_cancer_row(Path(args.cross_cancer_summary).resolve(), out_row)

    print("[deeptgi-pancancer] done ->", run_dir)


if __name__ == "__main__":
    main()
