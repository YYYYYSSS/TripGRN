"""由差异基因集合构造有向笛卡尔积 (A,B) 与 (B,A)，标签来自 GRN 矩阵。"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch


def build_ordered_pairs_packed(
    gene_list: List[str],
    grn_matrix: torch.Tensor,
    de_genes: List[str],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    与 build_ordered_pairs_from_de_gene_list 语义相同，但返回 shape (N,3) 的 int32 数组
    (tf_idx, target_idx, label)，避免物化数百万 Python tuple（否则构建阶段极慢且占内存）。
    """
    g2i = {g: i for i, g in enumerate(gene_list)}
    syms = sorted({str(g) for g in de_genes if str(g) in g2i})
    if len(syms) < 2:
        raise ValueError("DE 基因在表达面板中至少需 2 个")
    idx = np.array([g2i[s] for s in syms], dtype=np.int32)
    n = int(idx.shape[0])
    ix = torch.as_tensor(idx, dtype=torch.long, device=grn_matrix.device)
    sub = grn_matrix[ix][:, ix]
    sub_np = sub.detach().cpu().numpy()
    row_idx, col_idx = np.where(~np.eye(n, dtype=bool))
    ia_out = idx[row_idx]
    ib_out = idx[col_idx]
    vals = sub_np[row_idx, col_idx]
    lab = np.where(vals > 0, vals, 0).astype(np.int32)
    pairs = np.column_stack([ia_out, ib_out, lab]).astype(np.int32, copy=False)
    stats = {"n_de_genes_in_panel": n, "n_ordered_pairs": int(pairs.shape[0])}
    return pairs, stats


def build_ordered_pairs_from_de_gene_list(
    gene_list: List[str],
    grn_matrix: torch.Tensor,
    de_genes: List[str],
) -> Tuple[List[Tuple[int, int, int]], Dict[str, int]]:
    """
    对 de_genes 与 gene_list 的交集（排序后），枚举所有有序对 (i,j), i!=j。
    标签 = grn_matrix[i,j] 若为正类调控，否则 0。
    """
    g2i = {g: i for i, g in enumerate(gene_list)}
    syms = sorted({str(g) for g in de_genes if str(g) in g2i})
    if len(syms) < 2:
        raise ValueError("DE 基因在表达面板中至少需 2 个")
    pairs: List[Tuple[int, int, int]] = []
    for ga in syms:
        ia = g2i[ga]
        for gb in syms:
            if ga == gb:
                continue
            ib = g2i[gb]
            v = int(grn_matrix[ia, ib].item())
            lab = v if v > 0 else 0
            pairs.append((ia, ib, lab))
    stats = {
        "n_de_genes_in_panel": len(syms),
        "n_ordered_pairs": len(pairs),
    }
    return pairs, stats
