"""
脑转移（mBrain）列上训练（5:2:3 列划分 + 与 non-brain 脚本相同的 pair 划分），
在非脑列与可选独立表达矩阵（如 EMTAB）上做应用评估；差异边均与「全脑列」预测对比。

与 train_metastasis_nonbrain_brain_eval 方向对偶：
  - 归一化：仅在 mBrain 的 train 列上按基因拟合 log2(min-max)，再变换脑 val/test、非脑全列、外部队列。
  - 应用侧指标：nonbrain_external_* / emtab_external_*（同一 pairs_test）。
  - 预测差异 CSV：应用侧 vs mBrain 全列（参考臂一致）。

独立数据批次效应：默认不做 ComBat；若跨平台/跨批次，建议先用训练域可得的样本
拟合 harmonization，再传入 --external-exp-csv 指向校正后的矩阵（见 run bat 注释）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from grn.data.preprocessing import (
    get_GRN,
    log2_minmax_per_gene_fit,
    log2_minmax_per_gene_transform,
    normalize_gene_symbol_series,
    resolve_grn_table_columns,
)
from grn.evaluation.training_plots import (
    save_accuracy_bar_val_test,
    save_confusion_matrix_fig,
    save_grouped_metrics_val_test,
    save_learning_curve,
    save_multi_accuracy_bar,
)
from grn.models.fusion_model import FusionModelAblation
from grn.reproducibility import set_seed

from grn.scripts.train_metastasis_nonbrain_brain_eval import (
    CLASS_LABEL_NAMES,
    _build_seq_features,
    _class_label_name,
    _collect_predictions,
    _load_sample_tissue_from_feature_summary_xlsx,
    _split_expression_columns_by_brain,
    _resolve_metastasis_run_dir,
)
from grn.scripts.train_unseen_samples_no_cv import (
    PairSubsetColumnDataset,
    _build_pair_samples,
    _evaluate,
    _forward_logits,
    _load_gene_feature_csv,
    _prepare_batch,
    split3,
)


def _autocast_bf16(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _export_master_exp_csv_to_subsets_streaming(
    master_path: str,
    gene_col: str,
    non_brain_cols: List[str],
    brain_cols: List[str],
    non_brain_path: str,
    brain_path: str,
) -> None:
    """
    从「基因×细胞」大 CSV 流式切出 mBrain / non_mBrain 子表，避免 pd.read_csv 整表进内存导致 MemoryError。
    对主表做两次只读扫表（各只读所需列 + 分块行），按列数自动调小行块以控制单块内存。
    """
    for tag, out_path, cols in (
        ("non-mBrain", non_brain_path, [gene_col] + list(non_brain_cols)),
        ("mBrain", brain_path, [gene_col] + list(brain_cols)),
    ):
        n_data = max(0, len(cols) - 1)
        row_chunksize = _safe_row_chunk_for_wide_matrix(n_data) if n_data else 1024
        print(
            f"[metastasis-brain-train] streaming subset CSV: {tag} -> {out_path} "
            f"(cols={len(cols)}, row_chunksize={row_chunksize}) ...",
            flush=True,
        )
        first = True
        n_chunk = 0
        for chunk in pd.read_csv(
            master_path,
            usecols=cols,
            chunksize=row_chunksize,
            low_memory=False,
        ):
            n_chunk += 1
            if n_chunk == 1 or n_chunk % 50 == 0:
                print(
                    f"[metastasis-brain-train]   {tag}: chunk {n_chunk} (rows_in_chunk={len(chunk)})",
                    flush=True,
                )
            for c in chunk.columns[1:]:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype(np.float32)
            chunk.to_csv(out_path, mode="w" if first else "a", index=False, header=first)
            first = False
        print(f"[metastasis-brain-train]   {tag}: done, total_chunks={n_chunk}", flush=True)


def _safe_row_chunk_for_wide_matrix(n_data_cols: int) -> int:
    """极宽时减小每块行数，使 chunk 的数值区约 < 40MB 量级，避免读块时 OOM。"""
    if n_data_cols <= 0:
        return 2000
    return max(32, min(2000, max(1, 8_000_000 // (n_data_cols * 4))))


def _filter_dense_cells_by_max_zero_fraction(
    mat: np.ndarray,
    max_zero_frac: Optional[float],
    *,
    label: str,
) -> np.ndarray:
    """
    mat: [n_genes, n_cells] 的稠密计数矩阵。
    过滤掉：零表达（mat==0）在「基因维度」占比 >= max_zero_frac 的细胞列。
    """
    if max_zero_frac is None:
        return mat
    thr = float(max_zero_frac)
    if thr <= 0.0 or thr > 1.0:
        return mat
    if mat.size == 0:
        return mat

    n_genes = mat.shape[0]
    # 每列：0 的数量 / 基因数
    zero_frac = (mat == 0).sum(axis=0) / float(n_genes)
    keep = zero_frac < thr
    n_drop = int((~keep).sum())
    if n_drop > 0:
        print(
            f"[metastasis] filter_cells ({label}) zero_frac>={thr:.4g}: removed {n_drop} / {mat.shape[1]} cells "
            f"(keep <{thr:.0%} zero-expression genes)"
        )
        return mat[:, keep]
    return mat


def _submatrix_rows_from_mmap(
    mmap_arr: np.ndarray,
    genes_full: np.ndarray,
    gene_list: Sequence[str],
) -> np.ndarray:
    """按 gene_list 顺序从 memmap (n_genes, n_cells) 取行子矩阵（稠密 float32）。"""
    g_to_i = {str(genes_full[i]): i for i in range(int(len(genes_full)))}
    rows = np.array([g_to_i[g] for g in gene_list], dtype=np.int64)
    return np.asarray(mmap_arr[rows, :], dtype=np.float32)


def _load_wide_gene_x_cell_dataframe(
    path: str,
    gene_col: str,
    *,
    row_chunksize: int = 0,
) -> pd.DataFrame:
    """
    宽表（每行一基因、首列为基因，其余为细胞）分块读入，按与 pandas 一致的方式「首次出现的基因行」保留，得到 DataFrame。
    避免 pd.read_csv 单表一次读入时内存峰值过大。
    """
    head = pd.read_csv(path, nrows=0)
    data_cols = [c for c in head.columns[1:]]
    n_c = len(data_cols)
    if n_c == 0:
        raise ValueError(f"No data columns in {path}")
    if not row_chunksize or row_chunksize < 0:
        row_chunksize = _safe_row_chunk_for_wide_matrix(n_c)

    order: List[str] = []
    seen: Set[str] = set()
    for chunk in pd.read_csv(
        path,
        usecols=[gene_col],
        chunksize=row_chunksize,
        low_memory=False,
    ):
        gser = normalize_gene_symbol_series(chunk[gene_col])
        for g in gser:
            g = str(g)
            if g not in seen:
                seen.add(g)
                order.append(g)
    n_g = len(order)
    gi = {g: i for i, g in enumerate(order)}
    mat = np.zeros((n_g, n_c), dtype=np.float32)
    filled = np.zeros(n_g, dtype=bool)
    for chunk in pd.read_csv(path, chunksize=row_chunksize, low_memory=False):
        gser = normalize_gene_symbol_series(chunk[gene_col])
        val = chunk.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        for r in range(len(gser)):
            g = str(gser.iloc[r])
            j = gi.get(g)
            if j is not None and not filled[j]:
                mat[j, :] = val[r]
                filled[j] = True
    return pd.DataFrame(mat, index=order, columns=data_cols)


def _load_wide_expr_reordered_submatrix(
    path: str,
    gene_list: List[str],
    gene_col: str,
    *,
    row_chunksize: int = 0,
) -> np.ndarray:
    """
    按 gene_list 顺序从宽表取行（基因首次出现为有效行，与 drop_duplicates(keep='first') 一致），
    返回 float32 矩阵 (len(gene_list), n_cell_cols)。不构建完整 pandas 表。
    """
    head = pd.read_csv(path, nrows=0)
    data_cols = [c for c in head.columns[1:]]
    n_c = len(data_cols)
    n_r = len(gene_list)
    if n_c == 0:
        raise ValueError(f"No data columns in {path}")
    if not row_chunksize or row_chunksize < 0:
        row_chunksize = _safe_row_chunk_for_wide_matrix(n_c)
    g_idx = {str(g): i for i, g in enumerate(gene_list)}
    mat = np.zeros((n_r, n_c), dtype=np.float32)
    filled = np.zeros(n_r, dtype=bool)
    for chunk in pd.read_csv(path, chunksize=row_chunksize, low_memory=False):
        gser = normalize_gene_symbol_series(chunk[gene_col])
        val = chunk.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        for r in range(len(gser)):
            g = str(gser.iloc[r])
            j = g_idx.get(g)
            if j is not None and not filled[j]:
                mat[j, :] = val[r]
                filled[j] = True
    miss = [gene_list[i] for i in range(n_r) if not filled[i]]
    if miss:
        raise ValueError(
            f"Submatrix {path} missing {len(miss)} genes from brain list (e.g. {miss[:5]})"
        )
    return mat


def _load_emtab_like_matrix(csv_path: str, gene_list: Sequence[str]) -> Tuple[np.ndarray, int, int]:
    """
    读取独立表达矩阵（基因×样本），行名对齐到 gene_list；缺失基因填 0。
    返回 raw [n_genes, n_cols], n_matched_genes, n_cols。
    """
    df = pd.read_csv(csv_path)
    gcol = df.columns[0]
    df[gcol] = normalize_gene_symbol_series(df[gcol])
    df = df.drop_duplicates(subset=[gcol], keep="first").set_index(gcol)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    data_cols = [c for c in df.columns]
    n_cols = len(data_cols)
    if n_cols == 0:
        raise ValueError(f"No sample columns in {csv_path}")
    gset = set(df.index.astype(str))
    mat = np.zeros((len(gene_list), n_cols), dtype=np.float32)
    n_hit = 0
    for i, g in enumerate(gene_list):
        if g in gset:
            mat[i, :] = df.loc[g].values.astype(np.float32)
            n_hit += 1
    return mat, n_hit, n_cols


def _export_apply_vs_brain_pred_diff(
    run_dir: str,
    csv_basename: str,
    summary_basename: str,
    prob_csv_basename: str,
    prob_summary_basename: str,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    gene_list: List[str],
    pairs_test: List[Tuple[int, int, int]],
    exp_apply: torch.Tensor,
    exp_brain_ref: torch.Tensor,
    apply_tag: str,
    brain_tag: str,
    seq_features: np.ndarray,
    ppi_extractor,
    use_ppi: bool,
    use_expression: bool,
    use_cds_modal: bool,
    ppi_emb_dim: int,
) -> Tuple[int, int]:
    ds_a = PairSubsetColumnDataset(
        exp_apply,
        pairs_test,
        seq_features,
        ppi_extractor,
        device=None,
        use_ppi=use_ppi,
        use_expression=use_expression,
        use_cds=use_cds_modal,
        ppi_feat_dim=ppi_emb_dim,
    )
    ds_b = PairSubsetColumnDataset(
        exp_brain_ref,
        pairs_test,
        seq_features,
        ppi_extractor,
        device=None,
        use_ppi=use_ppi,
        use_expression=use_expression,
        use_cds=use_cds_modal,
        ppi_feat_dim=ppi_emb_dim,
    )
    la = DataLoader(ds_a, batch_size=batch_size, shuffle=False)
    lb = DataLoader(ds_b, batch_size=batch_size, shuffle=False)

    yt, p_a, pr_a, tfn, tgn = _collect_predictions(model, la, device, gene_list)
    yt2, p_b, pr_b, tfn2, tgn2 = _collect_predictions(model, lb, device, gene_list)
    if not (np.array_equal(yt, yt2) and tfn == tfn2 and tgn == tgn2):
        raise RuntimeError("misaligned pairs_test order between apply and brain loaders")

    diff_mask = p_a != p_b
    n_diff = int(diff_mask.sum())
    n_total = int(len(yt))

    csv_path = os.path.join(run_dir, csv_basename)
    summary_path = os.path.join(run_dir, summary_basename)

    cols = {
        "TF": [tfn[i] for i in range(n_total) if diff_mask[i]],
        "Target": [tgn[i] for i in range(n_total) if diff_mask[i]],
        "True_Label": yt[diff_mask],
        "True_Label_Name": [_class_label_name(int(yt[i])) for i in range(n_total) if diff_mask[i]],
        f"Pred_{apply_tag}": p_a[diff_mask],
        f"Pred_{apply_tag}_Name": [_class_label_name(int(p_a[i])) for i in range(n_total) if diff_mask[i]],
        f"Pred_{brain_tag}": p_b[diff_mask],
        f"Pred_{brain_tag}_Name": [_class_label_name(int(p_b[i])) for i in range(n_total) if diff_mask[i]],
    }
    for k, name in enumerate(CLASS_LABEL_NAMES):
        cols[f"P_{apply_tag}_{name}"] = pr_a[diff_mask, k]
        cols[f"P_{brain_tag}_{name}"] = pr_b[diff_mask, k]

    pd.DataFrame(cols).to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_pairs_test": n_total,
                "n_pred_diff": n_diff,
                "fraction_pred_diff": (n_diff / n_total) if n_total else 0.0,
                "apply_arm": apply_tag,
                "brain_reference_arm": brain_tag,
                "csv": os.path.basename(csv_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[metastasis-brain-train] pred diff {apply_tag} vs {brain_tag}: {n_diff}/{n_total} -> {csv_path}")

    abs_diff = np.abs(pr_a - pr_b)
    max_diff = abs_diff.max(axis=1)
    topk = min(200, n_total)
    top_idx = np.argsort(-max_diff)[:topk]

    prob_diff_csv = os.path.join(run_dir, prob_csv_basename)
    prob_diff_summary = os.path.join(run_dir, prob_summary_basename)

    prob_cols = {
        "TF": [tfn[i] for i in top_idx],
        "Target": [tgn[i] for i in top_idx],
        "True_Label": [int(yt[i]) for i in top_idx],
        "True_Label_Name": [_class_label_name(int(yt[i])) for i in top_idx],
        f"Pred_{apply_tag}": [int(p_a[i]) for i in top_idx],
        f"Pred_{brain_tag}": [int(p_b[i]) for i in top_idx],
        "max_prob_diff": [float(max_diff[i]) for i in top_idx],
    }
    for k, name in enumerate(CLASS_LABEL_NAMES):
        prob_cols[f"P_{apply_tag}_{name}"] = [float(pr_a[i, k]) for i in top_idx]
        prob_cols[f"P_{brain_tag}_{name}"] = [float(pr_b[i, k]) for i in top_idx]

    pd.DataFrame(prob_cols).to_csv(prob_diff_csv, index=False, encoding="utf-8-sig")
    with open(prob_diff_summary, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_pairs_test": n_total,
                "topk": topk,
                "csv": os.path.basename(prob_diff_csv),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[metastasis-brain-train] prob diff top -> {prob_diff_csv}")
    return n_diff, n_total


def _build_brain_train_context(
    args: argparse.Namespace,
    run_dir: str,
    *,
    write_subset_and_manifests: bool,
) -> Dict[str, Any]:
    use_expr = not args.no_expression
    use_cds_modal = not args.no_cds
    use_ppi = not args.no_ppi

    sample_tissue, brain_ids = _load_sample_tissue_from_feature_summary_xlsx(args.feature_summary_xlsx)
    all_sample_ids = {s for s, _ in sample_tissue}
    brain_sample_ids_sorted = sorted(brain_ids)
    non_brain_sample_ids = sorted(all_sample_ids - brain_ids)
    if not brain_ids:
        raise ValueError("No mBrain samples found in feature summary xlsx.")

    non_brain_path = os.path.join(run_dir, "expression_non_brain_mBrain_excluded_subset.csv")
    brain_path = os.path.join(run_dir, "expression_brain_mBrain_only_subset.csv")

    brain_mm_arr: Optional[np.ndarray] = None
    precomputed_nb_mm: Optional[np.ndarray] = None
    precomputed_genes_full: Optional[np.ndarray] = None
    br_col_names_arr: Optional[np.ndarray] = None

    if write_subset_and_manifests:
        pdir = getattr(args, "precomputed_split_dir", None)
        if pdir:
            pdir = os.path.abspath(str(pdir))
            man_path = os.path.join(pdir, "manifest.json")
            if not os.path.isfile(man_path):
                raise FileNotFoundError(f"--precomputed-split-dir missing manifest.json: {man_path}")
            with open(man_path, encoding="utf-8") as f:
                man = json.load(f)
            master_resolved = os.path.abspath(args.master_exp_csv)
            man_master = str(man.get("master_exp_csv", ""))
            if os.path.abspath(man_master) != master_resolved:
                print(
                    f"[metastasis-brain-train] warning: manifest master_exp_csv mismatch:\n"
                    f"  manifest: {man_master}\n  args:     {master_resolved}",
                    flush=True,
                )
            gene_col = man["gene_column"]
            precomputed_genes_full = np.load(os.path.join(pdir, man["genes_npy"]), allow_pickle=True)
            brain_mm_arr = np.load(os.path.join(pdir, man["brain_values_npy"]), mmap_mode="r")
            precomputed_nb_mm = np.load(os.path.join(pdir, man["nonbrain_values_npy"]), mmap_mode="r")
            br_col_names_arr = np.load(os.path.join(pdir, man["brain_column_names_npy"]), allow_pickle=True)
            manifest = {
                "brain_sample_ids": brain_sample_ids_sorted,
                "non_brain_sample_count": len(non_brain_sample_ids),
                "expression_columns_non_brain": int(man["n_nonbrain_cols"]),
                "expression_columns_brain": int(man["n_brain_cols"]),
                "unmatched_expression_columns_count": int(man.get("unmatched_columns_count", 0)),
                "pair_split_note": "Train/val/test column split on mBrain only; apply on non-brain all cols and optional external CSV.",
                "precomputed_split_dir": pdir,
                "precomputed_manifest_version": man.get("version"),
            }
            with open(os.path.join(run_dir, "metastasis_split_manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            print(
                f"[metastasis-brain-train] using precomputed npy split from {pdir} "
                f"(skip streaming CSV subset export).",
                flush=True,
            )
        else:
            head0 = pd.read_csv(args.master_exp_csv, nrows=0, low_memory=False)
            gene_col = head0.columns[0]
            data_cols = [c for c in head0.columns[1:]]
            non_brain_cols, brain_cols, unmatched = _split_expression_columns_by_brain(
                data_cols, brain_ids, all_sample_ids
            )
            if unmatched:
                print(f"[metastasis-brain-train] warning: {len(unmatched)} columns unmatched to Excel Samples")
            if not non_brain_cols:
                raise ValueError("No non-brain expression columns after filtering.")
            if not brain_cols:
                raise ValueError("No brain expression columns after filtering.")
            print(
                f"[metastasis-brain-train] splitting master matrix into subset CSVs "
                f"(non-mBrain_cols={len(non_brain_cols)}, mBrain_cols={len(brain_cols)}). "
                f"This scans the full master file twice and can take a long time on large disks.",
                flush=True,
            )
            _export_master_exp_csv_to_subsets_streaming(
                args.master_exp_csv, gene_col, non_brain_cols, brain_cols, non_brain_path, brain_path
            )
            manifest = {
                "brain_sample_ids": brain_sample_ids_sorted,
                "non_brain_sample_count": len(non_brain_sample_ids),
                "expression_columns_non_brain": len(non_brain_cols),
                "expression_columns_brain": len(brain_cols),
                "unmatched_expression_columns_count": len(unmatched),
                "pair_split_note": "Train/val/test column split on mBrain only; apply on non-brain all cols and optional external CSV.",
            }
            with open(os.path.join(run_dir, "metastasis_split_manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
    else:
        if not os.path.isfile(non_brain_path) or not os.path.isfile(brain_path):
            raise FileNotFoundError(f"subset CSV missing under {run_dir}")
        peek = pd.read_csv(brain_path, nrows=0)
        gene_col = peek.columns[0]

    # Gene list from BRAIN matrix：预计算 npy 直接构表，或 CSV 分块读入（避免大宽表 OOM）
    if brain_mm_arr is not None:
        assert precomputed_genes_full is not None and br_col_names_arr is not None
        expr_df = pd.DataFrame(
            np.asarray(brain_mm_arr, dtype=np.float32),
            index=[str(g) for g in precomputed_genes_full.tolist()],
            columns=[str(c) for c in br_col_names_arr.tolist()],
        )
        expr_df.index.name = str(gene_col)
    else:
        expr_df = _load_wide_gene_x_cell_dataframe(brain_path, str(gene_col))
    expr_df = expr_df.fillna(0.0)

    ref_indices = None
    if args.reference_gene_list_csv:
        ref = normalize_gene_symbol_series(
            pd.read_csv(args.reference_gene_list_csv, usecols=[0]).iloc[:, 0]
        ).tolist()
        ref_map = {g: i for i, g in enumerate(ref)}
        kept = [g for g in expr_df.index.tolist() if g in ref_map]
        expr_df = expr_df.loc[kept]
        ref_indices = [ref_map[g] for g in kept]
        print(f"[metastasis-brain-train] anchor genes={len(kept)} (no fill-zero)")

    cds_gene_feat = None
    tss_gene_feat = None
    if args.cds_gene_feature_csv:
        cds_gene_feat = _load_gene_feature_csv(args.cds_gene_feature_csv)
    if args.tss_gene_feature_csv:
        tss_gene_feat = _load_gene_feature_csv(args.tss_gene_feature_csv)
    if use_cds_modal and (cds_gene_feat is not None or tss_gene_feat is not None):
        feat_genes = set(expr_df.index.tolist())
        if cds_gene_feat is not None:
            feat_genes &= set(cds_gene_feat.index.tolist())
        if tss_gene_feat is not None:
            feat_genes &= set(tss_gene_feat.index.tolist())
        kept = [g for g in expr_df.index.tolist() if g in feat_genes]
        expr_df = expr_df.loc[kept]
        print(f"[metastasis-brain-train] genes after expression and feature alignment: {len(kept)}")

    gene_list = expr_df.index.tolist()
    raw_expr_brain = expr_df.values.astype(np.float32)
    raw_expr_brain = _filter_dense_cells_by_max_zero_fraction(
        raw_expr_brain,
        getattr(args, "filter_cells_max_zero_frac", None),
        label="mBrain",
    )
    num_samples = raw_expr_brain.shape[1]

    grn_df = pd.read_csv(args.grn_csv)
    tf_col, tg_col, tp_col = resolve_grn_table_columns(
        grn_df, args.grn_tf_col, args.grn_target_col, args.grn_type_col
    )
    if tf_col is None or tg_col is None or tp_col is None:
        raise ValueError(f"GRN columns not found. columns={list(grn_df.columns)}")

    grn_df[tf_col] = normalize_gene_symbol_series(grn_df[tf_col])
    grn_df[tg_col] = normalize_gene_symbol_series(grn_df[tg_col])
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

    all_pairs = _build_pair_samples(grn_matrix, args.neg_sample_ratio, args.seed)
    if len(all_pairs) == 0:
        raise ValueError(
            "GRN 在表达基因交集上未得到任何边对；请检查表达矩阵基因列是否与 GRN 符号一致（已 strip 引号）。"
        )
    labels = np.array([lab for _, _, lab in all_pairs], dtype=np.int64)
    idx = np.arange(len(all_pairs))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=args.seed, stratify=labels)
    y_tr = labels[tr_idx]
    tr_idx2, va_idx = train_test_split(tr_idx, test_size=0.2, random_state=args.seed, stratify=y_tr)
    pairs_train = [all_pairs[i] for i in tr_idx2]
    pairs_val = [all_pairs[i] for i in va_idx]
    pairs_test = [all_pairs[i] for i in te_idx]

    if write_subset_and_manifests:
        split_info = {
            "n_pairs_total": len(all_pairs),
            "n_pairs_train": len(pairs_train),
            "n_pairs_val": len(pairs_val),
            "n_pairs_test": len(pairs_test),
            "seed": args.seed,
        }
        with open(os.path.join(run_dir, "pair_split_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(split_info, f, ensure_ascii=False, indent=2)

    w1, w2, w3 = [int(x.strip()) for x in args.sample_split.split(",")]
    c_tr, c_va, c_te = split3(num_samples, w1, w2, w3)
    sl_tr = slice(0, c_tr)
    sl_va = slice(c_tr, c_tr + c_va)
    sl_te = slice(c_tr + c_va, num_samples)
    print(f"[metastasis-brain-train] mBrain columns train/val/test = {c_tr}/{c_va}/{c_te}")

    ppi_emb_dim = 160
    ppi_extractor = None
    if use_ppi:
        from grn.features.ppi_features import PPIFeatureExtractor

        ppi_extractor = PPIFeatureExtractor(args.ppi_csv, gene_list=gene_list, emb_dim=ppi_emb_dim, device=None)

    seq_features = _build_seq_features(gene_list, ref_indices, args, use_cds_modal, cds_gene_feat, tss_gene_feat)
    print(f"[metastasis-brain-train] seq_features shape={seq_features.shape}")

    x_min, x_max = log2_minmax_per_gene_fit(raw_expr_brain[:, sl_tr], offset=1.0)
    exp_tr = log2_minmax_per_gene_transform(raw_expr_brain[:, sl_tr], x_min, x_max, offset=1.0)
    exp_va = log2_minmax_per_gene_transform(raw_expr_brain[:, sl_va], x_min, x_max, offset=1.0)
    exp_te = log2_minmax_per_gene_transform(raw_expr_brain[:, sl_te], x_min, x_max, offset=1.0)
    exp_brain_full = log2_minmax_per_gene_transform(raw_expr_brain, x_min, x_max, offset=1.0)

    if precomputed_nb_mm is not None:
        assert precomputed_genes_full is not None
        raw_nb = _submatrix_rows_from_mmap(precomputed_nb_mm, precomputed_genes_full, gene_list)
    else:
        raw_nb = _load_wide_expr_reordered_submatrix(non_brain_path, gene_list, str(gene_col))
    raw_nb = _filter_dense_cells_by_max_zero_fraction(
        raw_nb,
        getattr(args, "filter_cells_max_zero_frac", None),
        label="non-brain",
    )
    exp_nonbrain_full = log2_minmax_per_gene_transform(raw_nb, x_min, x_max, offset=1.0)

    exp_emtab_full: Optional[torch.Tensor] = None
    emtab_info: Dict[str, Any] = {}
    if args.external_exp_csv:
        raw_em, n_hit, n_ecols = _load_emtab_like_matrix(args.external_exp_csv, gene_list)
        exp_emtab_full = log2_minmax_per_gene_transform(raw_em, x_min, x_max, offset=1.0)
        emtab_info = {
            "external_exp_csv": os.path.abspath(args.external_exp_csv),
            "n_genes_matched": n_hit,
            "n_genes_total_model": len(gene_list),
            "n_sample_columns": n_ecols,
        }
        print(
            f"[metastasis-brain-train] external matrix genes matched={n_hit}/{len(gene_list)}, cols={n_ecols}"
        )
        with open(os.path.join(run_dir, "external_expression_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(emtab_info, f, ensure_ascii=False, indent=2)

    return {
        "gene_col": gene_col,
        "gene_list": gene_list,
        "pairs_train": pairs_train,
        "pairs_val": pairs_val,
        "pairs_test": pairs_test,
        "exp_tr": exp_tr,
        "exp_va": exp_va,
        "exp_te": exp_te,
        "exp_brain_full": exp_brain_full,
        "exp_nonbrain_full": exp_nonbrain_full,
        "exp_emtab_full": exp_emtab_full,
        "emtab_info": emtab_info,
        "seq_features": seq_features,
        "seq_dim": int(seq_features.shape[1]),
        "ppi_extractor": ppi_extractor,
        "ppi_emb_dim": ppi_emb_dim,
        "use_ppi": use_ppi,
        "use_expr": use_expr,
        "use_cds_modal": use_cds_modal,
        "non_brain_path": non_brain_path,
        "brain_path": brain_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train on mBrain columns; evaluate on non-brain and optional external expression (vs brain ref)."
    )
    parser.add_argument("--master-exp-csv", required=True)
    parser.add_argument("--feature-summary-xlsx", required=True)
    parser.add_argument("--reference-gene-list-csv", default=None)
    parser.add_argument("--grn-csv", required=True)
    parser.add_argument("--cds-file", default=None)
    parser.add_argument("--tss-file", default=None)
    parser.add_argument("--cds-gene-feature-csv", default=None)
    parser.add_argument("--tss-gene-feature-csv", default=None)
    parser.add_argument(
        "--ppi-csv",
        default=None,
        help="PPI 边表 CSV（from,to[,combined_score]）；推荐普适 ppi_string_v12_human_score900.csv。",
    )
    parser.add_argument("--external-exp-csv", default=None, help="Optional independent matrix, e.g. EMTAB_ep_80.csv")
    parser.add_argument("--grn-tf-col", default="TF")
    parser.add_argument("--grn-target-col", default="target")
    parser.add_argument("--grn-type-col", default="Re_type")
    parser.add_argument("--output-base-dir", default="experiments")
    parser.add_argument("--experiment-name", default="repro_metastasis_brain_train_nonbrain_extval_gse131907")
    parser.add_argument("--experiment-subdir", default=None)
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--neg-sample-ratio", type=float, default=1.0)
    parser.add_argument("--sample-split", default="5,2,3")
    parser.add_argument(
        "--filter-cells-max-zero-frac",
        type=float,
        default=None,
        help="QC：去掉计数矩阵中「零表达基因占比 >= 阈值」的细胞（对 brain/non-brain 列分别过滤）。设为 0 或 >1 则关闭。",
    )
    parser.add_argument(
        "--precomputed-split-dir",
        default=None,
        help=(
            "materialize_gse131907_brain_nonbrain_npys 输出目录（含 manifest.json 与各 npy）："
            "跳过主表两遍流式 CSV 切分，brain/non-brain 从 mmap 加载。"
        ),
    )
    parser.add_argument("--no-ppi", action="store_true")
    parser.add_argument("--no-cds", action="store_true")
    parser.add_argument("--no-expression", action="store_true")
    parser.add_argument(
        "--expression-gradient-checkpointing",
        action="store_true",
        help="表达编码器 gradient checkpointing（与 train_unseen_samples_no_cv 一致，降显存）。",
    )
    parser.add_argument(
        "--amp-bf16",
        action="store_true",
        help="CUDA 上前向 bfloat16 autocast（与 train_unseen_samples_no_cv 一致）。",
    )
    args = parser.parse_args()

    print(
        f"[metastasis-brain-train] start pid={os.getpid()} experiment={args.experiment_name!r}",
        flush=True,
    )

    use_expr = not args.no_expression
    use_cds_modal = not args.no_cds
    use_ppi = not args.no_ppi
    if not (use_expr or use_cds_modal or use_ppi):
        raise ValueError("At least one modality must be enabled.")
    if use_ppi and not args.ppi_csv:
        raise ValueError("--ppi-csv is required unless --no-ppi is set.")
    if use_cds_modal and args.cds_file is None and args.cds_gene_feature_csv is None:
        raise ValueError("Either --cds-file or --cds-gene-feature-csv must be provided when CDS is enabled.")

    set_seed(args.seed)
    run_dir = _resolve_metastasis_run_dir(
        args.output_base_dir, args.experiment_name, args.experiment_subdir, args.no_timestamp
    )
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

    master_cfg = {**vars(args), "run_dir": run_dir}
    with open(os.path.join(run_dir, "run_config_master.json"), "w", encoding="utf-8") as f:
        json.dump(master_cfg, f, ensure_ascii=False, indent=2)

    print(f"[metastasis-brain-train] run_dir={run_dir} (config written), building tensors ...", flush=True)
    sys.stdout.flush()

    ctx = _build_brain_train_context(args, run_dir, write_subset_and_manifests=True)
    gene_list = ctx["gene_list"]
    pairs_train = ctx["pairs_train"]
    pairs_val = ctx["pairs_val"]
    pairs_test = ctx["pairs_test"]
    exp_tr = ctx["exp_tr"]
    exp_va = ctx["exp_va"]
    exp_te = ctx["exp_te"]
    exp_brain_full = ctx["exp_brain_full"]
    exp_nonbrain_full = ctx["exp_nonbrain_full"]
    exp_emtab_full = ctx["exp_emtab_full"]
    seq_features = ctx["seq_features"]
    ppi_extractor = ctx["ppi_extractor"]
    ppi_emb_dim = ctx["ppi_emb_dim"]

    def _ds(ex):
        return PairSubsetColumnDataset(
            ex,
            pairs_train,
            seq_features,
            ppi_extractor,
            device=None,
            use_ppi=use_ppi,
            use_expression=use_expr,
            use_cds=use_cds_modal,
            ppi_feat_dim=ppi_emb_dim,
        )

    ds_train = _ds(exp_tr)
    ds_val = PairSubsetColumnDataset(
        exp_va,
        pairs_val,
        seq_features,
        ppi_extractor,
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
        device=None,
        use_ppi=use_ppi,
        use_expression=use_expr,
        use_cds=use_cds_modal,
        ppi_feat_dim=ppi_emb_dim,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _bf16_fn = getattr(torch.cuda, "is_bf16_supported", None)
    _bf16_ok = bool(_bf16_fn()) if callable(_bf16_fn) and device.type == "cuda" else False
    use_amp_bf16 = bool(args.amp_bf16 and _bf16_ok)
    if args.amp_bf16 and device.type == "cuda" and not _bf16_ok:
        print("[metastasis-brain-train] warning: 当前环境不支持 bf16 autocast，忽略 --amp-bf16")
    if use_amp_bf16:
        print("[metastasis-brain-train] AMP: bfloat16 autocast 已启用（前向）")
    if args.expression_gradient_checkpointing and ctx["use_expr"]:
        print("[metastasis-brain-train] 表达编码器 gradient checkpointing 已启用")

    model = FusionModelAblation(
        seq_dim=ctx["seq_dim"],
        num_classes=3,
        use_expression=ctx["use_expr"],
        use_cds=ctx["use_cds_modal"],
        use_ppi=ctx["use_ppi"],
        expression_gradient_checkpointing=bool(args.expression_gradient_checkpointing),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    y_train = np.array([lab for _, _, lab in pairs_train], dtype=np.int64)
    counts = np.bincount(y_train, minlength=3)
    w = 1.0 / (counts + 1e-6)
    sample_w = w[y_train]
    sampler = WeightedRandomSampler(torch.tensor(sample_w, dtype=torch.double), len(sample_w), replacement=True)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    best_path = os.path.join(run_dir, "best_model_brain_train_nonbrain_extval_fold0.pth")
    best_val = -1.0
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
            x = _prepare_batch(batch, device)
            y = batch["label"].long().to(device)
            with _autocast_bf16(device, use_amp_bf16):
                logits = _forward_logits(model, x)
            loss = criterion(logits.float(), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_metrics_ep = _evaluate(model, val_loader, device, use_amp_bf16=use_amp_bf16)
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
    brain_val_metrics = _evaluate(model, val_loader, device, use_amp_bf16=use_amp_bf16)
    brain_test_metrics = _evaluate(model, test_loader, device, use_amp_bf16=use_amp_bf16)

    with open(os.path.join(run_dir, "brain_mBrain_unseen_samples_val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in brain_val_metrics.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(os.path.join(run_dir, "brain_mBrain_unseen_samples_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in brain_test_metrics.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[metastasis-brain-train] mBrain val_acc={brain_val_metrics['accuracy']:.4f}, "
        f"mBrain test_acc={brain_test_metrics['accuracy']:.4f}"
    )

    try:
        save_learning_curve(plot_dir, epoch_hist, train_loss_hist, val_acc_hist, val_f1_hist)
        save_accuracy_bar_val_test(
            plot_dir,
            float(brain_val_metrics["accuracy"]),
            float(brain_test_metrics["accuracy"]),
            title="mBrain 列：验证集 vs 测试集 Accuracy",
        )
        save_grouped_metrics_val_test(
            plot_dir,
            brain_val_metrics,
            brain_test_metrics,
            fname="metrics_mbrain_val_vs_test.png",
            title="mBrain 列：验证集 vs 测试集 主要指标",
        )
        save_confusion_matrix_fig(
            plot_dir,
            np.asarray(brain_val_metrics["confusion_matrix"]),
            "mBrain·验证集混淆矩阵",
            "confusion_matrix_mbrain_val.png",
        )
        save_confusion_matrix_fig(
            plot_dir,
            np.asarray(brain_test_metrics["confusion_matrix"]),
            "mBrain·测试集混淆矩阵",
            "confusion_matrix_mbrain_test.png",
        )
    except Exception as ex:
        print(f"[metastasis-brain-train] warning: mBrain plot export failed: {ex}")

    # ----- 应用：非脑全列 -----
    ds_nb_app = PairSubsetColumnDataset(
        exp_nonbrain_full,
        pairs_test,
        seq_features,
        ppi_extractor,
        device=None,
        use_ppi=use_ppi,
        use_expression=use_expr,
        use_cds=use_cds_modal,
        ppi_feat_dim=ppi_emb_dim,
    )
    nb_loader = DataLoader(ds_nb_app, batch_size=args.batch_size, shuffle=False)
    nb_metrics = _evaluate(model, nb_loader, device, use_amp_bf16=use_amp_bf16)
    with open(os.path.join(run_dir, "nonbrain_external_allcols_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in nb_metrics.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[metastasis-brain-train] non-brain apply all-cols test_acc={nb_metrics['accuracy']:.4f}")

    try:
        save_confusion_matrix_fig(
            plot_dir,
            np.asarray(nb_metrics["confusion_matrix"]),
            "非脑全列·应用测试混淆矩阵",
            "confusion_matrix_nonbrain_external_allcols.png",
        )
        save_multi_accuracy_bar(
            plot_dir,
            ("mBrain Val", "mBrain Test", "非脑全列应用"),
            (
                float(brain_val_metrics["accuracy"]),
                float(brain_test_metrics["accuracy"]),
                float(nb_metrics["accuracy"]),
            ),
            "accuracy_mbrain_vs_nonbrain_apply.png",
            "Accuracy：mBrain 划分 vs 非脑全列应用",
        )
    except Exception as ex:
        print(f"[metastasis-brain-train] warning: apply-side plot failed: {ex}")

    _export_apply_vs_brain_pred_diff(
        run_dir,
        "pairs_test_nonbrain_all_vs_mBrain_all_pred_diff.csv",
        "pairs_test_nonbrain_all_vs_mBrain_all_pred_diff_summary.json",
        "pairs_test_nonbrain_all_vs_mBrain_all_prob_diff_top200.csv",
        "pairs_test_nonbrain_all_vs_mBrain_all_prob_diff_summary.json",
        model,
        device,
        args.batch_size,
        gene_list,
        pairs_test,
        exp_nonbrain_full,
        exp_brain_full,
        "NonBrain_AllCols",
        "mBrain_AllCols",
        seq_features,
        ppi_extractor,
        use_ppi,
        use_expr,
        use_cds_modal,
        ppi_emb_dim,
    )

    if exp_emtab_full is not None:
        ds_em = PairSubsetColumnDataset(
            exp_emtab_full,
            pairs_test,
            seq_features,
            ppi_extractor,
            device=None,
            use_ppi=use_ppi,
            use_expression=use_expr,
            use_cds=use_cds_modal,
            ppi_feat_dim=ppi_emb_dim,
        )
        em_loader = DataLoader(ds_em, batch_size=args.batch_size, shuffle=False)
        em_metrics = _evaluate(model, em_loader, device, use_amp_bf16=use_amp_bf16)
        with open(os.path.join(run_dir, "emtab_external_test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in em_metrics.items()},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[metastasis-brain-train] external EMTAB apply test_acc={em_metrics['accuracy']:.4f}")

        try:
            save_confusion_matrix_fig(
                plot_dir,
                np.asarray(em_metrics["confusion_matrix"]),
                "外部矩阵 EMTAB·应用测试混淆矩阵",
                "confusion_matrix_emtab_external.png",
            )
            save_multi_accuracy_bar(
                plot_dir,
                ("mBrain Val", "mBrain Test", "非脑应用", "EMTAB 外部"),
                (
                    float(brain_val_metrics["accuracy"]),
                    float(brain_test_metrics["accuracy"]),
                    float(nb_metrics["accuracy"]),
                    float(em_metrics["accuracy"]),
                ),
                "accuracy_with_emtab_external.png",
                "Accuracy：含 EMTAB 外部应用",
            )
        except Exception as ex:
            print(f"[metastasis-brain-train] warning: EMTAB plot failed: {ex}")

        _export_apply_vs_brain_pred_diff(
            run_dir,
            "pairs_test_EMTAB_vs_mBrain_all_pred_diff.csv",
            "pairs_test_EMTAB_vs_mBrain_all_pred_diff_summary.json",
            "pairs_test_EMTAB_vs_mBrain_all_prob_diff_top200.csv",
            "pairs_test_EMTAB_vs_mBrain_all_prob_diff_summary.json",
            model,
            device,
            args.batch_size,
            gene_list,
            pairs_test,
            exp_emtab_full,
            exp_brain_full,
            "EMTAB",
            "mBrain_AllCols",
            seq_features,
            ppi_extractor,
            use_ppi,
            use_expr,
            use_cds_modal,
            ppi_emb_dim,
        )

    print(f"[metastasis-brain-train] saved run_dir={run_dir}")
    print(f"[metastasis-brain-train] model={best_path}")


if __name__ == "__main__":
    main()
