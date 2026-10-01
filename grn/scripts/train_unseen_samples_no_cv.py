import argparse
import json
import os
from contextlib import nullcontext
from datetime import datetime
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from grn.data.preprocessing import (
    get_GRN,
    log_expression_tensor,
    log2_minmax_per_gene_fit,
    log2_minmax_per_gene_transform,
    normalize_gene_symbol_series,
    resolve_grn_table_columns,
)
from grn.data.tf_source_filter import filter_grn_by_source_symbols, load_regulator_symbol_set
from grn.evaluation.metrics import calculate_metrics
from grn.evaluation.confusion_matrix_io import export_confusion_matrix_artifacts
from grn.evaluation.training_plots import (
    per_class_ovr_accuracy_precision_recall_from_cm,
    save_accuracy_bar_val_test,
    save_confusion_matrix_fig,
    save_grouped_metrics_val_test,
    save_learning_curve,
    save_ovr_roc_pr_curves,
    save_per_class_auprc_bar,
    save_per_class_metric_bars,
)
from grn.models.fusion_model import FusionModel, FusionModelAblation
from grn.reproducibility import set_seed
from grn.training.pair_dataloader import PairSubsetColumnDataset, build_pair_dataloader


def _autocast_bf16(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def split3(n: int, w1: int, w2: int, w3: int) -> Tuple[int, int, int]:
    w = w1 + w2 + w3
    a = n * w1 // w
    b = n * w2 // w
    c = n - a - b
    return a, b, c


def _build_pair_samples(grn_matrix: torch.Tensor, neg_ratio: float, seed: int):
    rng = np.random.RandomState(seed)
    pos = torch.where(grn_matrix > 0)
    pos_pairs = [(int(i.item()), int(j.item()), int(grn_matrix[i, j].item())) for i, j in zip(pos[0], pos[1])]

    n_pos = len(pos_pairs)
    n_neg = int(n_pos * neg_ratio)
    num_genes = grn_matrix.shape[0]
    existing = set((i, j) for i, j, _ in pos_pairs)
    neg_pairs = []
    while len(neg_pairs) < n_neg:
        i = int(rng.randint(0, num_genes))
        j = int(rng.randint(0, num_genes))
        if (i, j) in existing:
            continue
        if int(grn_matrix[i, j].item()) != 0:
            continue
        neg_pairs.append((i, j, 0))
        existing.add((i, j))
    return pos_pairs + neg_pairs


def _prepare_batch(batch, device, *, non_blocking: bool = False):
    out = {
        "exp_tf": batch["tf"].to(device, non_blocking=non_blocking),
        "exp_target": batch["target"].to(device, non_blocking=non_blocking),
        "seq_tf": batch["seq_tf"].to(device, non_blocking=non_blocking),
        "seq_target": batch["seq_target"].to(device, non_blocking=non_blocking),
        "ppi_tf": batch["ppi_tf"].to(device, non_blocking=non_blocking),
        "ppi_target": batch["ppi_target"].to(device, non_blocking=non_blocking),
    }
    return out


def _forward_logits(model, x: dict):
    return model(
        x["exp_tf"],
        x["exp_target"],
        x["seq_tf"],
        x["seq_target"],
        x["ppi_tf"],
        x["ppi_target"],
    )


def _evaluate(model, loader, device, use_amp_bf16: bool = False, non_blocking: bool = False):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = _prepare_batch(batch, device, non_blocking=non_blocking)
            y = batch["label"].cpu().numpy()
            with _autocast_bf16(device, use_amp_bf16):
                logits = _forward_logits(model, x)
            prob = torch.softmax(logits.float(), dim=1).cpu().numpy()
            pred = np.argmax(prob, axis=1)
            all_labels.extend(y)
            all_preds.extend(pred)
            all_probs.append(prob)
    all_probs = np.vstack(all_probs)
    return calculate_metrics(np.array(all_labels), np.array(all_preds), all_probs)


def _predict_arrays(model, loader, device, use_amp_bf16: bool = False, non_blocking: bool = False):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            x = _prepare_batch(batch, device, non_blocking=non_blocking)
            y = batch["label"].cpu().numpy()
            with _autocast_bf16(device, use_amp_bf16):
                logits = _forward_logits(model, x)
            prob = torch.softmax(logits.float(), dim=1).cpu().numpy()
            all_labels.extend(y)
            all_probs.append(prob)
    y_true = np.array(all_labels)
    y_prob = np.vstack(all_probs)
    return y_true, y_prob


def _load_gene_feature_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"feature csv needs >=2 columns (gene + features): {path}")
    gene_col = df.columns[0]
    df[gene_col] = normalize_gene_symbol_series(df[gene_col])
    df = df.drop_duplicates(subset=[gene_col], keep="first").set_index(gene_col)
    feat = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return feat


def main():
    parser = argparse.ArgumentParser(description="No-CV training with unseen sample split (column disjoint).")
    parser.add_argument(
        "--exp-csv",
        default=None,
        help="表达矩阵 CSV（基因行、细胞列）；与 --exp-npy-values / --exp-npy-genes 二选一。",
    )
    parser.add_argument(
        "--exp-npy-values",
        default=None,
        help="基因×细胞 float32 矩阵 .npy（可由 wide_expression_csv_to_npy 生成）；可与 memmap 配合。",
    )
    parser.add_argument(
        "--exp-npy-genes",
        default=None,
        help="与 _values.npy 行顺序一致的基因名 .npy（object 数组）。",
    )
    parser.add_argument("--reference-gene-list-csv", default=None)
    parser.add_argument("--grn-csv", required=True)
    parser.add_argument("--cds-file", default=None, help="CDS feature CSV (legacy mode: row-aligned by reference gene list).")
    parser.add_argument("--tss-file", default=None, help="Optional precomputed TSS feature CSV (same gene order as reference).")
    parser.add_argument(
        "--cds-gene-feature-csv",
        default=None,
        help="Optional CDS gene feature CSV (first col gene, rest features). When provided, overrides --cds-file alignment mode.",
    )
    parser.add_argument(
        "--tss-gene-feature-csv",
        default=None,
        help="Optional TSS gene feature CSV (first col gene, rest features).",
    )
    parser.add_argument(
        "--ppi-csv",
        default=None,
        help="PPI 边表 CSV（from,to）；除非 --no-ppi。可用 build_ppi_csv_from_string 从 STRING 生成。",
    )
    parser.add_argument("--grn-tf-col", default="TF", help="TF symbol column in GRN CSV.")
    parser.add_argument("--grn-target-col", default="target", help="Target symbol column in GRN CSV.")
    parser.add_argument("--grn-type-col", default="Re_type", help="Relation type column in GRN CSV.")
    parser.add_argument(
        "--grn-tf-symbols-file",
        default=None,
        help="可选：仅保留 GRN 中 source 列在此列表（每行一个基因符号）的边，用于排除非 TF 介导的 KEGG 关系（如激酶磷酸化被标成 activation）。",
    )
    parser.add_argument(
        "--grn-exclude-source-symbols-file",
        default=None,
        help="可选：剔除 source 在此列表中的边（在 TF 白名单之前执行）；可与 --grn-tf-symbols-file 联用。",
    )

    parser.add_argument("--output-base-dir", default="experiments")
    parser.add_argument("--experiment-name", default="repro_unseen_samples_no_cv")
    parser.add_argument("--no-timestamp", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--neg-sample-ratio", type=float, default=1.0)
    parser.add_argument("--sample-split", default="5,2,3", help="train,val,test column split ratio")
    parser.add_argument(
        "--cv-splits-json",
        default=None,
        help="10-fold CV manifest (from generate_pancancer_cv_splits). Requires --fold-idx.",
    )
    parser.add_argument(
        "--fold-idx",
        type=int,
        default=None,
        help="1-based fold index when using --cv-splits-json.",
    )
    parser.add_argument(
        "--method-name",
        default="TripGRN",
        help="Method label stored in run_config / confusion matrix metadata.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip matplotlib training/ROC plots (metrics JSON still written).",
    )
    parser.add_argument(
        "--no-ppi",
        action="store_true",
        help="Ablation: PPI inputs are all-zero (same FusionModel and 6×emb fusion as baseline; PPI branches still run).",
    )
    parser.add_argument(
        "--no-cds",
        action="store_true",
        help="Ablation: sequence inputs are all-zero (same FusionModel; CDS/TSS branches still run).",
    )
    parser.add_argument(
        "--no-expression",
        action="store_true",
        help="Ablation: expression inputs are all-zero (same FusionModel; expression encoders still run).",
    )
    parser.add_argument(
        "--expression-gradient-checkpointing",
        action="store_true",
        help="表达编码器梯度检查点（省显存但更慢；上皮子集/大 batch 时通常可关闭）。",
    )
    parser.add_argument(
        "--max-expression-cols",
        type=int,
        default=0,
        help="若 train/val/test 列数超过该值则压缩（0=不限制）。"
        " step06 大类如 t_lymphocytes (~8万列) 强烈建议 2048–4096，否则每 epoch 可达小时级。",
    )
    parser.add_argument(
        "--expression-col-reduce",
        choices=("subsample", "avg_pool"),
        default="subsample",
        help="超过 --max-expression-cols 时：subsample=按 seed 随机子采样细胞列；"
        "avg_pool=adaptive average pool 到固定长度（更快但混合细胞）。",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        help="DataLoader workers（Windows 大矩阵建议 0；Linux 可试 2–4）。",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="DataLoader pin_memory（CUDA 时默认开启，可用 --no-pin-memory 关闭）。",
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="关闭 pin_memory。",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="torch.backends.cudnn.benchmark=True（CUDA 默认开启）。",
    )
    parser.add_argument(
        "--no-cudnn-benchmark",
        action="store_true",
        help="关闭 cudnn benchmark。",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="torch.compile 包装模型（PyTorch 2+，首次 epoch 较慢）。",
    )
    parser.add_argument(
        "--amp-bf16",
        action="store_true",
        help="在 CUDA 上对前向使用 bfloat16 autocast（权重保持 fp32；需 Ampere 级及以上且支持 bf16）。",
    )
    args = parser.parse_args()

    use_expr = not args.no_expression
    use_cds_modal = not args.no_cds
    use_ppi = not args.no_ppi
    if not (use_expr or use_cds_modal or use_ppi):
        raise ValueError("At least one modality must be enabled (expression, CDS, or PPI).")
    if use_ppi and not args.ppi_csv:
        raise ValueError("--ppi-csv is required unless --no-ppi is set.")
    if use_cds_modal and args.cds_file is None and args.cds_gene_feature_csv is None:
        raise ValueError("Either --cds-file or --cds-gene-feature-csv must be provided when CDS is enabled.")
    if bool(args.exp_npy_values) ^ bool(args.exp_npy_genes):
        raise ValueError("--exp-npy-values 与 --exp-npy-genes 须同时指定或同时省略")
    if args.exp_csv and args.exp_npy_values:
        raise ValueError("请勿同时使用 --exp-csv 与 --exp-npy-values/--exp-npy-genes")
    if not args.exp_csv and not args.exp_npy_values:
        raise ValueError("须指定 --exp-csv，或同时指定 --exp-npy-values 与 --exp-npy-genes")

    set_seed(args.seed)
    if args.no_ppi:
        print("[unseen-samples] Ablation: zero PPI inputs (FusionModel unchanged; fusion dim=6×emb).")
    if args.no_cds:
        print("[unseen-samples] Ablation: zero sequence inputs (FusionModel unchanged).")
    if args.no_expression:
        print("[unseen-samples] Ablation: zero expression inputs (FusionModel unchanged).")
    if args.no_timestamp:
        run_dir = os.path.join(args.output_base_dir, args.experiment_name)
    else:
        run_dir = os.path.join(args.output_base_dir, args.experiment_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    if args.cv_splits_json:
        if args.fold_idx is None:
            raise ValueError("--fold-idx is required when --cv-splits-json is set")
        run_dir = os.path.join(run_dir, f"fold_{int(args.fold_idx):02d}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # expression anchor（CSV 或 npy；宽大矩阵推荐 npy + memmap）
    if args.exp_npy_values:
        g_arr = np.load(args.exp_npy_genes, allow_pickle=True)
        raw_expr = np.load(args.exp_npy_values, mmap_mode="r")
        gene_list = normalize_gene_symbol_series(
            pd.Series([str(x) for x in g_arr.tolist()])
        ).tolist()
        if len(gene_list) != raw_expr.shape[0]:
            raise ValueError(f"基因数 {len(gene_list)} 与矩阵行数 {raw_expr.shape[0]} 不一致")
        print(
            f"[unseen-samples] expression from npy: genes={len(gene_list)}, cells={raw_expr.shape[1]} "
            f"(memmap={isinstance(raw_expr, np.memmap)})"
        )
    else:
        expr_df = pd.read_csv(args.exp_csv)
        gene_col = expr_df.columns[0]
        expr_df[gene_col] = normalize_gene_symbol_series(expr_df[gene_col])
        expr_df = expr_df.drop_duplicates(subset=[gene_col], keep="first").set_index(gene_col)
        expr_df = expr_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        gene_list = expr_df.index.tolist()
        raw_expr = expr_df.values.astype(np.float32)
        del expr_df
        print(f"[unseen-samples] expression from csv: genes={len(gene_list)}, cells={raw_expr.shape[1]}")

    def _subset_gene_rows(kept_genes: List[str]) -> None:
        nonlocal gene_list, raw_expr
        g_to_i = {g: i for i, g in enumerate(gene_list)}
        idx = [g_to_i[g] for g in kept_genes]
        raw_expr = np.asarray(raw_expr[idx, :], dtype=np.float32)
        gene_list = kept_genes

    ref_indices = None
    if args.reference_gene_list_csv:
        ref = normalize_gene_symbol_series(
            pd.read_csv(args.reference_gene_list_csv, usecols=[0]).iloc[:, 0]
        ).tolist()
        ref_map = {g: i for i, g in enumerate(ref)}
        kept = [g for g in gene_list if g in ref_map]
        if not kept:
            raise ValueError("参考基因列表与表达矩阵无交集")
        _subset_gene_rows(kept)
        ref_indices = [ref_map[g] for g in kept]
        print(f"[unseen-samples] anchor genes={len(kept)} (no fill-zero)")

    # gene-level feature alignment (by gene symbol) if provided
    cds_gene_feat = None
    tss_gene_feat = None
    if args.cds_gene_feature_csv:
        cds_gene_feat = _load_gene_feature_csv(args.cds_gene_feature_csv)
    if args.tss_gene_feature_csv:
        tss_gene_feat = _load_gene_feature_csv(args.tss_gene_feature_csv)
    if use_cds_modal and (cds_gene_feat is not None or tss_gene_feat is not None):
        feat_genes = set(gene_list)
        if cds_gene_feat is not None:
            feat_genes &= set(cds_gene_feat.index.tolist())
        if tss_gene_feat is not None:
            feat_genes &= set(tss_gene_feat.index.tolist())
        kept = [g for g in gene_list if g in feat_genes]
        if not kept:
            raise ValueError("表达与 CDS/TSS 特征无基因交集")
        _subset_gene_rows(kept)
        print(f"[unseen-samples] genes after expression and feature alignment: {len(kept)}")

    num_samples = raw_expr.shape[1]
    est_gb = (len(gene_list) * num_samples * 4 * 3) / (1024**3)
    if est_gb > 6.0:
        print(
            f"[unseen-samples] warning: 估计 train/val/test 三份表达张量约 {est_gb:.1f} GiB（float32），"
            "请确保系统内存充足；宽大 CSV 请改用 npy + memmap 并开启显存优化参数。"
        )

    # GRN filter by expression genes
    grn_df = pd.read_csv(args.grn_csv)
    tf_col, tg_col, tp_col = resolve_grn_table_columns(
        grn_df, args.grn_tf_col, args.grn_target_col, args.grn_type_col
    )
    if tf_col is None or tg_col is None or tp_col is None:
        raise ValueError(
            f"GRN columns not found. got columns={list(grn_df.columns)}; "
            f"expect tf={args.grn_tf_col}, target={args.grn_target_col}, type={args.grn_type_col} "
            "or fallback source_symbol/target_symbol/relation_subtype."
        )
    grn_df[tf_col] = normalize_gene_symbol_series(grn_df[tf_col])
    grn_df[tg_col] = normalize_gene_symbol_series(grn_df[tg_col])
    if args.grn_tf_symbols_file or args.grn_exclude_source_symbols_file:
        include_tf = (
            load_regulator_symbol_set(args.grn_tf_symbols_file)
            if args.grn_tf_symbols_file
            else None
        )
        exclude_src = (
            load_regulator_symbol_set(args.grn_exclude_source_symbols_file)
            if args.grn_exclude_source_symbols_file
            else None
        )
        grn_df, _st = filter_grn_by_source_symbols(
            grn_df, tf_col, include_tf=include_tf, exclude_sources=exclude_src
        )
        print(
            f"[unseen-samples] GRN source filter: rows {_st['rows_input']} -> {_st['rows_output']} "
            f"(tf_file={args.grn_tf_symbols_file!r}, exclude_file={args.grn_exclude_source_symbols_file!r})"
        )
    gene_set = set(gene_list)
    grn_df = grn_df[grn_df[tf_col].isin(gene_set) & grn_df[tg_col].isin(gene_set)].copy()
    rel = grn_df[tp_col].astype(str).str.lower().str.strip()
    rel = rel.replace(
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
    rel = rel[grn_df.index]
    known_grn = (grn_df[tf_col].values, grn_df[tg_col].values, rel.values)
    grn_matrix = get_GRN(known_grn, gene_list)

    # pairs split (still pair split), columns split (unseen samples)
    all_pairs = _build_pair_samples(grn_matrix, args.neg_sample_ratio, args.seed)
    if len(all_pairs) == 0:
        raise ValueError(
            "GRN 在表达基因交集上未得到任何正样本边（TF–Target 对为空）。"
            "常见原因：表达矩阵基因列含引号或与 GRN 符号体系不一致；"
            "已自动 strip 引号，若仍为空请检查基因 ID（如 Ensembl vs symbol）。"
            "若为 ENSG… Ensembl ID：请运行 python -m grn.scripts.remap_wide_npy_ensembl_to_symbol "
            "将 npy 转为 symbol 后再训练。"
        )
    labels = np.array([lab for _, _, lab in all_pairs], dtype=np.int64)
    idx = np.arange(len(all_pairs))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=args.seed, stratify=labels)
    y_tr = labels[tr_idx]
    tr_idx2, va_idx = train_test_split(tr_idx, test_size=0.2, random_state=args.seed, stratify=y_tr)

    pairs_train = [all_pairs[i] for i in tr_idx2]
    pairs_val = [all_pairs[i] for i in va_idx]
    pairs_test = [all_pairs[i] for i in te_idx]

    if args.cv_splits_json:
        from grn.data.pancancer_cv import get_fold_split, load_cv_splits

        cv_payload = load_cv_splits(args.cv_splits_json)
        fold = get_fold_split(cv_payload, int(args.fold_idx))
        train_cols = np.asarray(fold["train_cols"], dtype=np.int64)
        val_cols = np.asarray(fold["val_cols"], dtype=np.int64)
        test_cols = np.asarray(fold["test_cols"], dtype=np.int64)
        print(
            f"[unseen-samples] CV fold {args.fold_idx}: columns train/val/test = "
            f"{len(train_cols)}/{len(val_cols)}/{len(test_cols)}"
        )
        exp_tr_src = raw_expr[:, train_cols]
        exp_va_src = raw_expr[:, val_cols]
        exp_te_src = raw_expr[:, test_cols]
    else:
        w1, w2, w3 = [int(x.strip()) for x in args.sample_split.split(",")]
        c_tr, c_va, c_te = split3(num_samples, w1, w2, w3)
        sl_tr = slice(0, c_tr)
        sl_va = slice(c_tr, c_tr + c_va)
        sl_te = slice(c_tr + c_va, num_samples)
        print(f"[unseen-samples] columns split train/val/test = {c_tr}/{c_va}/{c_te}")
        exp_tr_src = raw_expr[:, sl_tr]
        exp_va_src = raw_expr[:, sl_va]
        exp_te_src = raw_expr[:, sl_te]

    ppi_emb_dim = 160
    ppi_extractor = None
    if use_ppi:
        from grn.features.ppi_features import PPIFeatureExtractor

        ppi_extractor = PPIFeatureExtractor(args.ppi_csv, gene_list=gene_list, emb_dim=ppi_emb_dim, device=None)

    if not use_cds_modal:
        seq_features = np.zeros((len(gene_list), 20), dtype=np.float32)
        print(f"[unseen-samples] CDS ablation: dummy sequence features, shape={seq_features.shape}")
    elif cds_gene_feat is not None or tss_gene_feat is not None:
        feat_blocks = []
        if cds_gene_feat is not None:
            feat_blocks.append(cds_gene_feat.loc[gene_list].values.astype(np.float32))
        elif args.cds_file:
            cds_df = pd.read_csv(args.cds_file, header=None, sep=",", skiprows=1)
            cds_all = cds_df.values.astype(np.float32)
            cds_features = cds_all[ref_indices] if ref_indices is not None else cds_all
            if len(cds_features) != len(gene_list):
                raise ValueError(f"CDS and gene_list mismatch: {len(cds_features)} vs {len(gene_list)}")
            feat_blocks.append(cds_features)

        if tss_gene_feat is not None:
            feat_blocks.append(tss_gene_feat.loc[gene_list].values.astype(np.float32))
        elif args.tss_file:
            tss_df = pd.read_csv(args.tss_file, header=None, sep=",", skiprows=1)
            tss_all = tss_df.values.astype(np.float32)
            tss_features = tss_all[ref_indices] if ref_indices is not None else tss_all
            if len(tss_features) != len(gene_list):
                raise ValueError(f"TSS and gene_list mismatch: {len(tss_features)} vs {len(gene_list)}")
            feat_blocks.append(tss_features)

        seq_features = np.concatenate(feat_blocks, axis=1)
        print(f"[unseen-samples] using gene-aligned sequence features, dim={seq_features.shape[1]}")
    else:
        cds_df = pd.read_csv(args.cds_file, header=None, sep=",", skiprows=1)
        cds_all = cds_df.values.astype(np.float32)
        if ref_indices is not None:
            cds_features = cds_all[ref_indices]
        else:
            cds_features = cds_all
        if len(cds_features) != len(gene_list):
            raise ValueError(f"CDS and gene_list mismatch: {len(cds_features)} vs {len(gene_list)}")

        if args.tss_file:
            tss_df = pd.read_csv(args.tss_file, header=None, sep=",", skiprows=1)
            tss_all = tss_df.values.astype(np.float32)
            if ref_indices is not None:
                tss_features = tss_all[ref_indices]
            else:
                tss_features = tss_all
            if len(tss_features) != len(gene_list):
                raise ValueError(f"TSS and gene_list mismatch: {len(tss_features)} vs {len(gene_list)}")
            seq_features = np.concatenate([cds_features, tss_features], axis=1)
            print(f"[unseen-samples] using CDS+TSS sequence features, dim={seq_features.shape[1]}")
        else:
            seq_features = cds_features
            print(f"[unseen-samples] using CDS-only sequence features, dim={seq_features.shape[1]}")

    # 表达归一化：用 train 列拟合“按基因”的 log2(min-max) 归一化，
    # 再变换 val/test，避免列分布导致表达坍缩。
    x_min, x_max = log2_minmax_per_gene_fit(exp_tr_src, offset=1.0)
    exp_tr = log2_minmax_per_gene_transform(exp_tr_src, x_min, x_max, offset=1.0).contiguous()
    exp_va = log2_minmax_per_gene_transform(exp_va_src, x_min, x_max, offset=1.0).contiguous()
    exp_te = log2_minmax_per_gene_transform(exp_te_src, x_min, x_max, offset=1.0).contiguous()

    max_expr_cols = int(args.max_expression_cols)
    if max_expr_cols > 0:

        def _reduce_cols(mat: torch.Tensor, *, split_name: str) -> torch.Tensor:
            t = int(mat.shape[1])
            if t <= max_expr_cols:
                return mat
            mode = str(args.expression_col_reduce)
            if mode == "avg_pool":
                # [G, T] -> [1, G, T] -> pool on last dim -> [G, max]
                packed = mat.unsqueeze(0)
                pooled = torch.nn.functional.adaptive_avg_pool1d(packed, max_expr_cols).squeeze(0)
                print(
                    f"[unseen-samples] {split_name}: avg_pool expression cols {t} -> {max_expr_cols}",
                    flush=True,
                )
                return pooled.contiguous()
            rng = np.random.default_rng(int(args.seed) + (0 if split_name == "train" else (1 if split_name == "val" else 2)))
            pick = np.sort(rng.choice(t, size=max_expr_cols, replace=False))
            print(
                f"[unseen-samples] {split_name}: subsample expression cols {t} -> {max_expr_cols} "
                f"(seed={args.seed})",
                flush=True,
            )
            return mat[:, torch.as_tensor(pick, dtype=torch.long)].contiguous()

        exp_tr = _reduce_cols(exp_tr, split_name="train")
        exp_va = _reduce_cols(exp_va, split_name="val")
        exp_te = _reduce_cols(exp_te, split_name="test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = bool(device.type == "cuda" and not args.no_pin_memory)
    if args.pin_memory:
        pin_memory = True
    non_blocking = bool(pin_memory and device.type == "cuda")
    if device.type == "cuda" and not args.no_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    ppi_node_features = None
    if use_ppi and ppi_extractor is not None:
        ppi_node_features = ppi_extractor.extract_node_features().detach().cpu()
        # Persist for case-study / transfer inference (PPINet weights are not in FusionModel ckpt).
        ppi_save = {
            "gene_list": list(gene_list),
            "features": ppi_node_features.contiguous(),
            "emb_dim": int(ppi_emb_dim),
            "ppi_csv": str(args.ppi_csv),
            "seed": int(args.seed),
        }
        ppi_path = os.path.join(run_dir, "ppi_node_features.pt")
        torch.save(ppi_save, ppi_path)
        print(f"[unseen-samples] saved PPI node features -> {ppi_path} shape={tuple(ppi_node_features.shape)}")

    ds_train = PairSubsetColumnDataset(
        exp_tr,
        pairs_train,
        seq_features,
        ppi_extractor,
        ppi_node_features=ppi_node_features,
        device=None,
        use_ppi=use_ppi,
        use_expression=use_expr,
        use_cds=use_cds_modal,
        ppi_feat_dim=ppi_emb_dim,
    )
    ds_val = PairSubsetColumnDataset(
        exp_va,
        pairs_val,
        seq_features,
        ppi_extractor,
        ppi_node_features=ppi_node_features,
        device=None,
        use_ppi=use_ppi,
        use_expression=use_expr,
        use_cds=use_cds_modal,
        ppi_feat_dim=ppi_emb_dim,
    )
    ds_test = PairSubsetColumnDataset(
        exp_te,
        pairs_test,
        seq_features,
        ppi_extractor,
        ppi_node_features=ppi_node_features,
        device=None,
        use_ppi=use_ppi,
        use_expression=use_expr,
        use_cds=use_cds_modal,
        ppi_feat_dim=ppi_emb_dim,
    )

    _bf16_fn = getattr(torch.cuda, "is_bf16_supported", None)
    _bf16_ok = bool(_bf16_fn()) if callable(_bf16_fn) and device.type == "cuda" else False
    use_amp_bf16 = bool(args.amp_bf16 and _bf16_ok)
    if args.amp_bf16 and device.type == "cuda" and not _bf16_ok:
        print("[unseen-samples] warning: 当前环境不支持 bf16 autocast，忽略 --amp-bf16")
    if use_amp_bf16:
        print("[unseen-samples] AMP: bfloat16 autocast 已启用（前向）")
    if args.expression_gradient_checkpointing:
        print("[unseen-samples] 表达编码器 gradient checkpointing 已启用（较慢，省显存）")
    else:
        print("[unseen-samples] gradient checkpointing 关闭（更快）")

    seq_dim = int(seq_features.shape[1])
    model = FusionModelAblation(
        seq_dim=seq_dim,
        num_classes=3,
        use_expression=use_expr,
        use_cds=use_cds_modal,
        use_ppi=use_ppi,
        expression_gradient_checkpointing=args.expression_gradient_checkpointing,
    ).to(device)
    if args.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("[unseen-samples] torch.compile 已启用")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    y_train = np.array([lab for _, _, lab in pairs_train], dtype=np.int64)
    counts = np.bincount(y_train, minlength=3)
    w = 1.0 / (counts + 1e-6)
    sample_w = w[y_train]
    sampler = WeightedRandomSampler(torch.tensor(sample_w, dtype=torch.double), len(sample_w), replacement=True)

    n_train_pairs = len(ds_train)
    n_steps = (n_train_pairs + args.batch_size - 1) // args.batch_size
    print(
        f"[unseen-samples] train pairs={n_train_pairs}, batch_size={args.batch_size}, "
        f"steps/epoch≈{n_steps}, pin_memory={pin_memory}, workers={args.dataloader_num_workers}"
    )

    train_loader = build_pair_dataloader(
        ds_train,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=int(args.dataloader_num_workers),
        pin_memory=pin_memory,
    )
    val_loader = build_pair_dataloader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.dataloader_num_workers),
        pin_memory=pin_memory,
    )
    test_loader = build_pair_dataloader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.dataloader_num_workers),
        pin_memory=pin_memory,
    )

    best_val = -1.0
    best_path = os.path.join(run_dir, "best_model_fold0.pth")
    patience_counter = 0
    plot_dir = os.path.join(run_dir, "plots")
    epoch_hist: List[int] = []
    train_loss_hist: List[float] = []
    val_acc_hist: List[float] = []
    val_f1_hist: List[float] = []

    for ep in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {ep+1}"):
            x = _prepare_batch(batch, device, non_blocking=non_blocking)
            y = batch["label"].long().to(device, non_blocking=non_blocking)
            with _autocast_bf16(device, use_amp_bf16):
                logits = _forward_logits(model, x)
            loss = criterion(logits.float(), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_metrics_ep = _evaluate(
            model, val_loader, device, use_amp_bf16=use_amp_bf16, non_blocking=non_blocking
        )
        val_acc = float(val_metrics_ep["accuracy"])
        val_f1 = float(val_metrics_ep["f1"])
        avg_loss = train_loss / max(len(train_loader), 1)
        epoch_hist.append(ep + 1)
        train_loss_hist.append(avg_loss)
        val_acc_hist.append(val_acc)
        val_f1_hist.append(val_f1)
        print(f"Epoch {ep+1}/{args.epochs} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[Early Stop] patience={args.patience}, stop at epoch {ep+1}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    val_metrics = _evaluate(
        model, val_loader, device, use_amp_bf16=use_amp_bf16, non_blocking=non_blocking
    )
    test_metrics = _evaluate(
        model, test_loader, device, use_amp_bf16=use_amp_bf16, non_blocking=non_blocking
    )
    y_test_true, y_test_prob = _predict_arrays(
        model, test_loader, device, use_amp_bf16=use_amp_bf16, non_blocking=non_blocking
    )

    with open(os.path.join(run_dir, "unseen_samples_val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in val_metrics.items()}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "unseen_samples_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in test_metrics.items()}, f, ensure_ascii=False, indent=2)

    try:
        export_confusion_matrix_artifacts(
            run_dir,
            np.asarray(test_metrics["confusion_matrix"], dtype=np.int64),
            title=f"{args.method_name} test confusion matrix",
            class_names=["No regulation", "Activation", "Repression"],
            extra={
                "method": args.method_name,
                "fold_idx": args.fold_idx,
                "experiment_name": args.experiment_name,
                "split": "test",
            },
        )
        export_confusion_matrix_artifacts(
            run_dir,
            np.asarray(val_metrics["confusion_matrix"], dtype=np.int64),
            stem="confusion_matrix_val",
            title=f"{args.method_name} val confusion matrix",
            class_names=["No regulation", "Activation", "Repression"],
            extra={
                "method": args.method_name,
                "fold_idx": args.fold_idx,
                "experiment_name": args.experiment_name,
                "split": "val",
            },
        )
    except Exception as ex:
        print(f"[unseen-samples] warning: confusion matrix export failed: {ex}")

    if not args.skip_plots:
        try:
            save_learning_curve(plot_dir, epoch_hist, train_loss_hist, val_acc_hist, val_f1_hist)
            save_accuracy_bar_val_test(
                plot_dir, float(val_metrics["accuracy"]), float(test_metrics["accuracy"])
            )
            save_grouped_metrics_val_test(plot_dir, val_metrics, test_metrics)
            cm_v = np.asarray(val_metrics["confusion_matrix"])
            cm_t = np.asarray(test_metrics["confusion_matrix"])
            save_confusion_matrix_fig(plot_dir, cm_v, "Validation confusion matrix", "confusion_matrix_val.png")
            save_confusion_matrix_fig(plot_dir, cm_t, "Test confusion matrix", "confusion_matrix_test.png")
            disp_names = ["No regulation", "Activation", "Repression"]
            disp_idx = [0, 2, 1]
            acc_ovr, precision_pc, recall_pc = per_class_ovr_accuracy_precision_recall_from_cm(cm_t)
            save_per_class_metric_bars(
                plot_dir,
                disp_names,
                [float(acc_ovr[i]) for i in disp_idx],
                [float(precision_pc[i]) for i in disp_idx],
                [float(recall_pc[i]) for i in disp_idx],
            )
            save_per_class_auprc_bar(
                plot_dir,
                disp_names,
                [float(x) for x in test_metrics["auprc"]],
            )
            save_ovr_roc_pr_curves(
                plot_dir,
                y_test_true,
                y_test_prob,
                disp_names,
                disp_idx,
                legend_roc_ovr=[float(x) for x in test_metrics["auc_ovr"]],
                legend_auprc=[float(x) for x in test_metrics["auprc"]],
            )
            print(f"[unseen-samples] plots -> {plot_dir}")
        except Exception as ex:
            print(f"[unseen-samples] warning: plot export failed: {ex}")
    else:
        print("[unseen-samples] skip-plots: metrics JSON only")

    print(f"[unseen-samples] best_model: {best_path}")
    print(f"[unseen-samples] val_acc={val_metrics['accuracy']:.4f}, test_acc={test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()

