from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch


def resolve_grn_table_columns(
    grn_df: pd.DataFrame,
    tf_arg: str,
    target_arg: str,
    type_arg: str,
) -> tuple[str | None, str | None, str | None]:
    """
    为边表选择 TF / target / 关系类型列。

    KEGG 风格 CSV 常同时含 ``target``（如 hsa:5976）与 ``target_symbol``（基因符号）。
    CLI 默认 ``--grn-target-col target`` 会误选 ID 列；若存在 ``target_symbol`` 则改用符号列。
    """
    cols = set(grn_df.columns)

    if tf_arg in cols:
        tf_col = tf_arg
    elif "source_symbol" in cols:
        tf_col = "source_symbol"
    elif "TF" in cols:
        tf_col = "TF"
    else:
        tf_col = None

    if target_arg in cols:
        if target_arg == "target" and "target_symbol" in cols:
            tg_col = "target_symbol"
        else:
            tg_col = target_arg
    elif "target_symbol" in cols:
        tg_col = "target_symbol"
    elif "target" in cols:
        tg_col = "target"
    else:
        tg_col = None

    if type_arg in cols:
        tp_col = type_arg
    elif "relation_subtype" in cols:
        tp_col = "relation_subtype"
    elif "Re_type" in cols:
        tp_col = "Re_type"
    else:
        tp_col = None

    return tf_col, tg_col, tp_col


def normalize_gene_symbol_series(s: pd.Series) -> pd.Series:
    """
    统一基因符号：去首尾空白、去掉 CSV 中常见的 ASCII 引号。
    避免表达矩阵基因为 "TP53" 而 GRN 为 TP53 时无法匹配、导致无边可训。
    """
    x = s.astype(str).str.strip()
    x = x.str.replace('"', "", regex=False)
    x = x.str.replace("'", "", regex=False)
    return x


def standard(raw_data: np.ndarray) -> torch.Tensor:
    """
    标准化基因表达数据（log2 + MinMax）。
    保持与原项目实现一致：先 log2，再 MinMax，返回 float32 tensor。
    """
    new_data = np.log2(raw_data + 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(new_data)
    return torch.tensor(scaled_data, dtype=torch.float32)


def standard_with_train_fit(raw_data: np.ndarray, train_gene_indices) -> torch.Tensor:
    """
    防止数据泄露的表达标准化：仅用训练基因的统计量拟合 MinMaxScaler，
    再把该 scaler 变换到所有基因。

    raw_data shape: [num_genes, num_features]
    """
    train_gene_indices = np.asarray(list(train_gene_indices), dtype=np.int64)
    if train_gene_indices.size == 0:
        # 边界兜底：没有训练基因时退化为全量拟合（理论上不应发生）
        return standard(raw_data)

    new_data = np.log2(raw_data + 1)
    scaler = MinMaxScaler()
    scaler.fit(new_data[train_gene_indices])
    scaled_data = scaler.transform(new_data)
    return torch.tensor(scaled_data, dtype=torch.float32)


def log_expression_tensor(raw_data: np.ndarray, offset: float = 1.0) -> torch.Tensor:
    """
    仅做 log2(x + offset) 变换（不做 MinMax）。
    用于“未见样本/未见 bulk 列”的严格评估，避免 MinMax 跨列泄露。
    """
    x = np.maximum(raw_data, 0.0)
    return torch.tensor(np.log2(x + offset), dtype=torch.float32)


def log2_minmax_per_gene_fit(
    raw_data_train: np.ndarray,
    offset: float = 1.0,
    eps: float = 1e-8,
    *,
    col_chunk: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    在“按基因”维度上做 log2(x+offset) 后的 min-max 归一化拟合。

    - raw_data_train: [num_genes, num_train_samples]（只用训练列）
    - 返回 min/max：shape 都为 [num_genes, 1]

    若 ``col_chunk`` 为正整数，则按列分块扫描（数值与整表一致），便于大矩阵时打进度；
    默认 ``None`` 为单次向量化（训练脚本保持原行为）。
    """
    if col_chunk is None or col_chunk <= 0:
        x = np.maximum(raw_data_train, 0.0)
        x_log = np.log2(x + offset)  # [G, S]
        x_min = x_log.min(axis=1, keepdims=True)
        x_max = x_log.max(axis=1, keepdims=True)
        return x_min, x_max

    g, s = int(raw_data_train.shape[0]), int(raw_data_train.shape[1])
    cc = min(int(col_chunk), s) if s > 0 else 1
    acc_dt = np.result_type(raw_data_train.dtype, np.float32)
    x_min = np.full((g, 1), np.inf, dtype=acc_dt)
    x_max = np.full((g, 1), -np.inf, dtype=acc_dt)
    n_chunks = (s + cc - 1) // cc
    report_every = max(1, n_chunks // 25)
    for ci, j0 in enumerate(range(0, s, cc)):
        j1 = min(s, j0 + cc)
        blk = raw_data_train[:, j0:j1]
        xb = np.log2(np.maximum(blk, 0.0) + offset)
        x_min = np.minimum(x_min, xb.min(axis=1, keepdims=True))
        x_max = np.maximum(x_max, xb.max(axis=1, keepdims=True))
        if progress_callback is not None and (ci % report_every == 0 or j1 == s):
            progress_callback(f"log2-minmax fit: columns {j1}/{s} ({100.0 * j1 / max(s, 1):.1f}%)")
    return x_min.astype(np.float32, copy=False), x_max.astype(np.float32, copy=False)


def log2_minmax_per_gene_transform(
    raw_data: np.ndarray,
    x_min: np.ndarray,
    x_max: np.ndarray,
    offset: float = 1.0,
    eps: float = 1e-8,
    *,
    col_chunk: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> torch.Tensor:
    """
    使用 log2_minmax_per_gene_fit 得到的 x_min/x_max 对任意 raw_data 做变换。

    raw_data: [num_genes, num_samples]（可为 val/test/brain）
    x_min/x_max: [num_genes, 1]

    ``col_chunk`` / ``progress_callback`` 语义同 ``log2_minmax_per_gene_fit``。
    """
    if col_chunk is None or col_chunk <= 0:
        x = np.maximum(raw_data, 0.0)
        x_log = np.log2(x + offset)
        scaled = (x_log - x_min) / (x_max - x_min + eps)
        return torch.tensor(scaled, dtype=torch.float32)

    g, s = int(raw_data.shape[0]), int(raw_data.shape[1])
    cc = min(int(col_chunk), s) if s > 0 else 1
    scaled = np.empty((g, s), dtype=np.float32)
    n_chunks = (s + cc - 1) // cc
    report_every = max(1, n_chunks // 25)
    for ci, j0 in enumerate(range(0, s, cc)):
        j1 = min(s, j0 + cc)
        blk = raw_data[:, j0:j1]
        x_log = np.log2(np.maximum(blk, 0.0) + offset)
        scaled[:, j0:j1] = ((x_log - x_min) / (x_max - x_min + eps)).astype(np.float32, copy=False)
        if progress_callback is not None and (ci % report_every == 0 or j1 == s):
            progress_callback(f"log2-minmax transform: columns {j1}/{s} ({100.0 * j1 / max(s, 1):.1f}%)")
    return torch.from_numpy(np.ascontiguousarray(scaled))


def get_GRN(known_GRN, gene_names):
    """
    构建基因调控关系矩阵（0: 无关系, 1: 抑制, 2: 激活）。
    保持与原项目实现一致。
    """
    if isinstance(gene_names, np.ndarray):
        gene_names = gene_names.tolist()

    tf_indices = [gene_names.index(tf) for tf in known_GRN[0]]
    target_indices = [gene_names.index(target) for target in known_GRN[1]]

    num_genes = len(gene_names)
    GRN = torch.zeros((num_genes, num_genes), dtype=torch.long)

    for tf_idx, target_idx, reg_type in zip(tf_indices, target_indices, known_GRN[2]):
        if reg_type == "activator":
            GRN[tf_idx, target_idx] = 2
        elif reg_type == "repressor":
            GRN[tf_idx, target_idx] = 1
        # 其他类型暂时视为无关系

    return GRN

