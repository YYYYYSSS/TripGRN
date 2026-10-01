# -*- coding: utf-8 -*-
"""Shared step01 binary protocol (regulated vs no-regulation) aligned with TripGRN splits."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from grn.data.pancancer_cv import get_fold_split, load_cv_splits
from grn.data.pancancer_paths import cv_splits_path, gse_npy_triplet
from grn.data.preprocessing import (
    get_GRN,
    normalize_gene_symbol_series,
    resolve_grn_table_columns,
)
from grn.scripts.train_unseen_samples_no_cv import _build_pair_samples


def load_epithelial_expression(gse_id: str, grn_root: Path) -> Tuple[List[str], np.ndarray]:
    v_path, g_path, _ = gse_npy_triplet("epithelial", gse_id, grn_root)
    if not v_path.is_file() or not g_path.is_file():
        raise FileNotFoundError(f"Missing epithelial npy for {gse_id}: {v_path}")
    g_arr = np.load(g_path, allow_pickle=True)
    raw = np.load(v_path).astype(np.float32)
    gene_list = normalize_gene_symbol_series(pd.Series([str(x) for x in g_arr.tolist()])).tolist()
    if raw.shape[0] != len(gene_list):
        raise ValueError(f"{gse_id}: genes/values row mismatch")
    return gene_list, raw


def build_grn_matrix(
    gene_list: List[str],
    kegg_path: Path,
    *,
    grn_tf_col: str = "TF",
    grn_target_col: str = "target",
    grn_type_col: str = "Re_type",
) -> torch.Tensor:
    grn_df = pd.read_csv(kegg_path)
    tf_col, tg_col, tp_col = resolve_grn_table_columns(
        grn_df, grn_tf_col, grn_target_col, grn_type_col
    )
    if tf_col is None or tg_col is None or tp_col is None:
        raise ValueError(f"GRN columns not resolved: {list(grn_df.columns)}")
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
    if grn_df.empty:
        raise ValueError("No KEGG edges mapped to expression genes")
    known_grn = (grn_df[tf_col].values, grn_df[tg_col].values, rel.values)
    return get_GRN(known_grn, gene_list)


def load_kegg_grn_dataframe(
    gene_list: List[str],
    kegg_path: Path,
    *,
    grn_tf_col: str = "TF",
    grn_target_col: str = "target",
    grn_type_col: str = "Re_type",
) -> Tuple[pd.DataFrame, str, str]:
    grn_df = pd.read_csv(kegg_path)
    tf_col, tg_col, tp_col = resolve_grn_table_columns(
        grn_df, grn_tf_col, grn_target_col, grn_type_col
    )
    if tf_col is None or tg_col is None or tp_col is None:
        raise ValueError(f"GRN columns not resolved: {list(grn_df.columns)}")
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
    if grn_df.empty:
        raise ValueError("No KEGG edges mapped to expression genes")
    return grn_df, tf_col, tg_col


def select_kegg_priority_gene_subset(
    gene_list: List[str],
    raw_expr: np.ndarray,
    grn_df: pd.DataFrame,
    tf_col: str,
    tg_col: str,
    max_genes: int,
    seed: int,
) -> List[str]:
    """Keep KEGG-edge genes first, fill to max_genes by variance. Required for scMGATGRN (O(N^2) GAT)."""
    if max_genes <= 0:
        return list(gene_list)
    rng = np.random.RandomState(seed)
    g_upper = {g.strip().upper(): g for g in gene_list}
    on_edge = set()
    for a, b in zip(grn_df[tf_col], grn_df[tg_col]):
        ga = str(a).strip().upper()
        gb = str(b).strip().upper()
        if ga in g_upper:
            on_edge.add(g_upper[ga])
        if gb in g_upper:
            on_edge.add(g_upper[gb])
    on_edge_list = sorted(on_edge)
    if len(on_edge_list) > max_genes:
        idx = [gene_list.index(g) for g in on_edge_list]
        sub = raw_expr[idx, :].astype(np.float64, copy=False)
        var = sub.var(axis=1)
        order = np.argsort(-var)[:max_genes]
        return [on_edge_list[i] for i in order]
    pool = [g for g in gene_list if g not in on_edge]
    rem = max_genes - len(on_edge_list)
    if rem <= 0:
        return on_edge_list
    idx_pool = [gene_list.index(g) for g in pool]
    subp = raw_expr[idx_pool, :].astype(np.float64, copy=False)
    varp = subp.var(axis=1)
    take = min(rem, len(pool))
    top = np.argsort(-varp)[:take]
    extra = [pool[i] for i in top]
    rng.shuffle(extra)
    return on_edge_list + extra


def subset_expression(gene_list: List[str], raw_expr: np.ndarray, keep: List[str]) -> Tuple[List[str], np.ndarray]:
    idx_map = {g: i for i, g in enumerate(gene_list)}
    idx = [idx_map[g] for g in keep]
    return list(keep), np.asarray(raw_expr[idx, :], dtype=np.float32)


def estimate_scmgat_dense_adj_gb(n_genes: int) -> float:
    """Rough dense N×N adj + attention footprint (GB, float32)."""
    n = int(n_genes)
    return (n * n * 4 * 12) / (1024**3)


def split_binary_pairs(
    grn_matrix: torch.Tensor,
    *,
    seed: int = 42,
    neg_ratio: float = 1.0,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """TripGRN pair sampling + stratified 64/16/20 split, labels collapsed to binary."""
    all_pairs_3 = _build_pair_samples(grn_matrix, neg_ratio, seed)
    if not all_pairs_3:
        raise ValueError("No TF-target pairs after GRN sampling")
    all_pairs = [(i, j, 1 if lab in (1, 2) else 0) for i, j, lab in all_pairs_3]
    labels = np.array([lab for _, _, lab in all_pairs], dtype=np.int64)
    idx = np.arange(len(all_pairs))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels)
    y_tr = labels[tr_idx]
    tr_idx2, va_idx = train_test_split(tr_idx, test_size=0.2, random_state=seed, stratify=y_tr)
    pairs_train = [all_pairs[i] for i in tr_idx2]
    pairs_val = [all_pairs[i] for i in va_idx]
    pairs_test = [all_pairs[i] for i in te_idx]
    return pairs_train, pairs_val, pairs_test


def load_cv_column_splits(gse_id: str, fold_idx: int, grn_root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = load_cv_splits(cv_splits_path(gse_id, grn_root))
    fold = get_fold_split(payload, int(fold_idx))
    train_cols = np.asarray(fold["train_cols"], dtype=np.int64)
    val_cols = np.asarray(fold["val_cols"], dtype=np.int64)
    test_cols = np.asarray(fold["test_cols"], dtype=np.int64)
    return train_cols, val_cols, test_cols


def subsample_cols_to_common_length(
    train_cols: np.ndarray,
    val_cols: np.ndarray,
    test_cols: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Align train/val/test column counts (DeepFGRN / scMGATGRN fixed input_dim)."""
    m = int(min(len(train_cols), len(val_cols), len(test_cols)))
    if m <= 0:
        raise ValueError("Empty train/val/test column split after CV")

    def _pick(arr: np.ndarray) -> np.ndarray:
        return rng.choice(arr, size=m, replace=False) if len(arr) > m else arr

    return _pick(train_cols), _pick(val_cols), _pick(test_cols), m


def pairs_to_gene_label_df(pairs: Sequence[Tuple[int, int, int]], gene_list: List[str]) -> pd.DataFrame:
    rows = [(gene_list[i], gene_list[j], int(lab)) for i, j, lab in pairs]
    return pd.DataFrame(rows, columns=["tf", "gene", "label"])


def pairs_to_index_array(pairs: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    return np.asarray([[i, j, lab] for i, j, lab in pairs], dtype=np.int64)


def kegg_tf_indices(
    gene_list: List[str],
    kegg_path: Path,
    *,
    grn_tf_col: str = "TF",
    grn_target_col: str = "target",
    grn_type_col: str = "Re_type",
) -> np.ndarray:
    grn_df = pd.read_csv(kegg_path)
    tf_col, tg_col, tp_col = resolve_grn_table_columns(
        grn_df, grn_tf_col, grn_target_col, grn_type_col
    )
    if tf_col is None:
        raise ValueError("Cannot resolve TF column in KEGG CSV")
    grn_df[tf_col] = normalize_gene_symbol_series(grn_df[tf_col])
    grn_df[tg_col] = normalize_gene_symbol_series(grn_df[tg_col])
    gene_set = set(gene_list)
    grn_df = grn_df[grn_df[tf_col].isin(gene_set) & grn_df[tg_col].isin(gene_set)].copy()
    tf_syms = set(grn_df[tf_col].astype(str))
    tf_indices = np.array([i for i, g in enumerate(gene_list) if g in tf_syms], dtype=np.int64)
    if tf_indices.size == 0:
        raise ValueError("No TF indices found in expression gene list")
    return tf_indices


def ensure_deeptgi_cells_x_genes_csv(
    gse_id: str,
    gene_list: List[str],
    raw_expr: np.ndarray,
    grn_root: Path,
    *,
    tag: str = "",
) -> Path:
    from grn.data.pancancer_paths import gse_epithelial_dir

    suffix = f"_{tag}" if tag else ""
    out = gse_epithelial_dir(gse_id, grn_root) / f"cells_x_genes_deeptgi{suffix}.csv"
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw_expr.T.astype(np.float32), columns=gene_list).to_csv(out, index=False)
    return out


def binary_metrics_to_step01_json(
    *,
    method: str,
    gse_id: str,
    fold_idx: int,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    cm: np.ndarray,
) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    yt = y_true.astype(int)
    yp = y_prob.astype(float)
    yb = y_pred.astype(int)
    auc = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else float("nan")
    aupr = float(average_precision_score(yt, yp)) if len(np.unique(yt)) > 1 else float("nan")
    return {
        "method": method,
        "task": "binary",
        "positive_definition": "activation_or_inhibition",
        "gse_id": gse_id,
        "fold_idx": int(fold_idx),
        "accuracy": float(accuracy_score(yt, yb)),
        "precision": float(precision_score(yt, yb, zero_division=0)),
        "recall": float(recall_score(yt, yb, zero_division=0)),
        "f1": float(f1_score(yt, yb, zero_division=0)),
        "mcc": float(matthews_corrcoef(yt, yb)) if len(np.unique(yt)) > 1 else float("nan"),
        "macro_auc": auc,
        "auc_ovo": auc,
        "auprc": aupr,
        "confusion_matrix": np.asarray(cm, dtype=np.int64).tolist(),
    }
