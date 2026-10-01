"""
统计「GRN 中出现过的 source（TF）× 当前基因面板中全部基因」的对数，并对 Step1 模型做短时推理试跑以估算总耗时。

不写入实验结果目录；仅打印统计与耗时。标签在试跑中固定为 0（仅测前向）。
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import cycle
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from grn.data.preprocessing import (
    log2_minmax_per_gene_fit,
    log2_minmax_per_gene_transform,
    normalize_gene_symbol_series,
    resolve_grn_table_columns,
)
from grn.models.fusion_model import FusionModelAblation
from grn.scripts.train_unseen_samples_no_cv import (
    PairSubsetColumnDataset,
    _forward_logits,
    _load_gene_feature_csv,
    _prepare_batch,
)


def _load_expr_gene_list(
    exp_csv: str | None,
    exp_npy_values: str | None,
    exp_npy_genes: str | None,
) -> tuple[np.ndarray, List[str]]:
    if bool(exp_npy_values) ^ bool(exp_npy_genes):
        raise SystemExit("--exp-npy-values 与 --exp-npy-genes 须同时指定或同时省略")
    if exp_csv and exp_npy_values:
        raise SystemExit("请勿同时使用 --exp-csv 与 npy")
    if not exp_csv and not exp_npy_values:
        raise SystemExit("请指定 --exp-csv，或同时指定 --exp-npy-values 与 --exp-npy-genes")

    if exp_npy_values:
        g_arr = np.load(exp_npy_genes, allow_pickle=True)
        raw = np.load(exp_npy_values, mmap_mode="r")
        gene_list = normalize_gene_symbol_series(pd.Series([str(x) for x in g_arr.tolist()])).tolist()
        if len(gene_list) != raw.shape[0]:
            raise SystemExit(f"基因数与矩阵行数不一致: {len(gene_list)} vs {raw.shape[0]}")
        raw = np.asarray(raw[:, :], dtype=np.float32)
    else:
        df = pd.read_csv(exp_csv)
        gcol = df.columns[0]
        df[gcol] = normalize_gene_symbol_series(df[gcol])
        df = df.drop_duplicates(subset=[gcol], keep="first").set_index(gcol)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        gene_list = df.index.tolist()
        raw = df.values.astype(np.float32)
    return raw, gene_list


def _grn_source_genes_in_panel(
    grn_csv: str,
    tf_arg: str,
    tg_arg: str,
    tp_arg: str,
    gene_set: Set[str],
) -> Set[str]:
    grn_df = pd.read_csv(grn_csv)
    tf_col, tg_col, tp_col = resolve_grn_table_columns(grn_df, tf_arg, tg_arg, tp_arg)
    if tf_col is None or tp_col is None:
        raise SystemExit("GRN 列解析失败")
    grn_df[tf_col] = normalize_gene_symbol_series(grn_df[tf_col])
    rel = grn_df[tp_col].astype(str).str.lower().str.strip().replace(
        {
            "activation": "activator",
            "activate": "activator",
            "positive": "activator",
            "upregulation": "activator",
            "inhibition": "repressor",
            "inhibit": "repressor",
            "repression": "repressor",
            "negative": "repressor",
            "downregulation": "repressor",
        }
    )
    grn_df = grn_df[rel.isin(["activator", "repressor"])].copy()
    src = set(grn_df[tf_col].astype(str)) & gene_set
    return src


def _build_tf_x_genome_samples(
    gene_list: List[str],
    source_symbols: Set[str],
    exclude_self: bool,
    max_pairs: int | None,
) -> List[Tuple[int, int, int]]:
    g2i = {g: i for i, g in enumerate(gene_list)}
    src_idx = sorted({g2i[s] for s in source_symbols if s in g2i})
    n = len(gene_list)
    out: List[Tuple[int, int, int]] = []
    for i in src_idx:
        for j in range(n):
            if exclude_self and j == i:
                continue
            out.append((i, j, 0))
            if max_pairs is not None and len(out) >= max_pairs:
                return out
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="TF x genome pair count + Step1 forward timing")
    p.add_argument("--step1-run-dir", required=True)
    p.add_argument("--exp-csv", default=None)
    p.add_argument("--exp-npy-values", default=None)
    p.add_argument("--exp-npy-genes", default=None)
    p.add_argument(
        "--include-self-loops",
        action="store_true",
        help="保留 TF 与 target 为同一基因的对（默认排除自环）",
    )
    p.add_argument("--count-only", action="store_true", help="只打印对数，不加载模型")
    p.add_argument("--max-pairs", type=int, default=16384, help="试跑最多前 K 对（默认 16384）")
    p.add_argument("--warmup-batches", type=int, default=2)
    p.add_argument("--time-batches", type=int, default=30, help="正式计时的 batch 数")
    args = p.parse_args()

    run_dir = Path(args.step1_run_dir)
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    raw0, gl0 = _load_expr_gene_list(args.exp_csv, args.exp_npy_values, args.exp_npy_genes)
    genes_set: Set[str] = set(gl0)

    use_cds = not bool(cfg.get("no_cds", False))
    cds_feat = _load_gene_feature_csv(cfg["cds_gene_feature_csv"]) if cfg.get("cds_gene_feature_csv") else None
    tss_feat = _load_gene_feature_csv(cfg["tss_gene_feature_csv"]) if cfg.get("tss_gene_feature_csv") else None
    if use_cds:
        if cds_feat is not None:
            genes_set &= set(cds_feat.index.tolist())
        if tss_feat is not None:
            genes_set &= set(tss_feat.index.tolist())
    gene_list = [g for g in gl0 if g in genes_set]
    g_to_i0 = {g: i for i, g in enumerate(gl0)}
    idx_keep = [g_to_i0[g] for g in gene_list]
    raw = np.asarray(raw0[np.array(idx_keep, dtype=np.intp), :], dtype=np.float32)

    src_in_panel = _grn_source_genes_in_panel(
        str(cfg["grn_csv"]),
        str(cfg.get("grn_tf_col", "TF")),
        str(cfg.get("grn_target_col", "target")),
        str(cfg.get("grn_type_col", "Re_type")),
        set(gene_list),
    )
    n_genes = len(gene_list)
    n_src = len(src_in_panel)
    exclude_self = not bool(args.include_self_loops)
    if exclude_self:
        n_total_pairs = n_src * n_genes - n_src
    else:
        n_total_pairs = n_src * n_genes

    print(
        json.dumps(
            {
                "n_genes_in_panel": n_genes,
                "n_grn_sources_in_panel": n_src,
                "exclude_self_loop": exclude_self,
                "n_pairs_tf_x_genome": n_total_pairs,
                "approx_order": f"{n_total_pairs:.2e}",
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.count_only:
        return

    use_ppi = not bool(cfg.get("no_ppi", False))
    use_expr = not bool(cfg.get("no_expression", False))
    if use_ppi and not cfg.get("ppi_csv"):
        raise SystemExit("step1 config requires ppi_csv")
    if use_cds and not (cfg.get("cds_gene_feature_csv") or cfg.get("tss_gene_feature_csv")):
        raise SystemExit("step1 requires cds/tss when use_cds")

    if use_cds:
        blocks = []
        if cds_feat is not None:
            blocks.append(cds_feat.loc[gene_list].values.astype(np.float32))
        if tss_feat is not None:
            blocks.append(tss_feat.loc[gene_list].values.astype(np.float32))
        seq_features = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(gene_list), 20), dtype=np.float32)
    else:
        seq_features = np.zeros((len(gene_list), 20), dtype=np.float32)
    seq_dim = int(seq_features.shape[1])

    ppi_emb_dim = 160
    ppi_extractor = None
    if use_ppi:
        from grn.features.ppi_features import PPIFeatureExtractor

        ppi_extractor = PPIFeatureExtractor(cfg["ppi_csv"], gene_list=gene_list, emb_dim=ppi_emb_dim, device=None)

    x_min, x_max = log2_minmax_per_gene_fit(raw, offset=1.0)
    exp_t = log2_minmax_per_gene_transform(raw, x_min, x_max, offset=1.0)

    samples = _build_tf_x_genome_samples(
        gene_list,
        src_in_panel,
        exclude_self=exclude_self,
        max_pairs=int(args.max_pairs),
    )
    bs = int(cfg.get("batch_size", 32))
    ds = PairSubsetColumnDataset(
        exp_t,
        samples,
        seq_features,
        ppi_extractor,
        device=None,
        use_ppi=use_ppi,
        use_expression=use_expr,
        use_cds=use_cds,
        ppi_feat_dim=ppi_emb_dim,
    )
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModelAblation(
        seq_dim=seq_dim, num_classes=3, use_expression=use_expr, use_cds=use_cds, use_ppi=use_ppi
    ).to(device)
    model_path = run_dir / "best_model_fold0.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    def run_batches(n_batches: int) -> None:
        it = cycle(loader)
        for _ in range(n_batches):
            batch = next(it)
            x = _prepare_batch(batch, device)
            with torch.no_grad():
                _ = _forward_logits(model, x)

    run_batches(int(args.warmup_batches))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_batches(int(args.time_batches))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n_timed = min(len(samples), int(args.time_batches) * bs)
    per_pair = elapsed / max(n_timed, 1)
    est_full = per_pair * n_total_pairs

    print(
        json.dumps(
            {
                "device": str(device),
                "batch_size": bs,
                "timed_pairs": n_timed,
                "warmup_batches": int(args.warmup_batches),
                "time_batches": int(args.time_batches),
                "seconds_for_timed_window": round(elapsed, 4),
                "seconds_per_pair_est": round(per_pair, 6),
                "estimated_seconds_full_tf_x_genome": round(est_full, 1),
                "estimated_hours_full": round(est_full / 3600.0, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
