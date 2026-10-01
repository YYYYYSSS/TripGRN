"""KEGG-only（无 PPI）k 跳邻居候选对 + 按比例负采样。供 case study 与 CLI 共用。"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

import numpy as np
import torch

from grn.data.preprocessing import normalize_gene_symbol_series, resolve_grn_table_columns

_REL_MAP = {
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


def kegg_edges_in_panel(
    grn_csv: str,
    tf_arg: str,
    tg_arg: str,
    tp_arg: str,
    panel: Set[str],
) -> List[Tuple[str, str]]:
    import pandas as pd

    grn_df = pd.read_csv(grn_csv)
    tf_col, tg_col, tp_col = resolve_grn_table_columns(grn_df, tf_arg, tg_arg, tp_arg)
    if tf_col is None or tg_col is None or tp_col is None:
        raise ValueError("GRN column resolution failed")
    grn_df[tf_col] = normalize_gene_symbol_series(grn_df[tf_col])
    grn_df[tg_col] = normalize_gene_symbol_series(grn_df[tg_col])
    rel = grn_df[tp_col].astype(str).str.lower().str.strip().replace(_REL_MAP)
    grn_df = grn_df[rel.isin(["activator", "repressor"])].copy()
    edges: List[Tuple[str, str]] = []
    for a, b in zip(grn_df[tf_col].astype(str), grn_df[tg_col].astype(str)):
        if a in panel and b in panel:
            edges.append((a, b))
    return edges


def grn_sources_in_panel(grn_csv: str, tf_arg: str, tg_arg: str, tp_arg: str, panel: Set[str]) -> Set[str]:
    import pandas as pd

    grn_df = pd.read_csv(grn_csv)
    tf_col, _, tp_col = resolve_grn_table_columns(grn_df, tf_arg, tg_arg, tp_arg)
    if tf_col is None or tp_col is None:
        raise ValueError("GRN column resolution failed")
    grn_df[tf_col] = normalize_gene_symbol_series(grn_df[tf_col])
    rel = grn_df[tp_col].astype(str).str.lower().str.strip().replace(_REL_MAP)
    grn_df = grn_df[rel.isin(["activator", "repressor"])].copy()
    return set(grn_df[tf_col].astype(str)) & panel


def build_adj(edges: List[Tuple[str, str]]) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def nodes_within_hops(adj: Dict[str, Set[str]], start: str, max_hops: int) -> Set[str]:
    dist = {start: 0}
    dq: deque[Tuple[str, int]] = deque([(start, 0)])
    while dq:
        u, d = dq.popleft()
        if d >= max_hops:
            continue
        nd = d + 1
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = nd
                dq.append((v, nd))
    return set(dist.keys())


def collect_khop_positive_pairs(
    sources: Set[str],
    panel: Set[str],
    adj: Dict[str, Set[str]],
    k_hops: int,
) -> Set[Tuple[str, str]]:
    pos_pairs: Set[Tuple[str, str]] = set()
    for s in sorted(sources):
        ball = nodes_within_hops(adj, s, int(k_hops))
        for t in ball:
            if t == s or t not in panel:
                continue
            pos_pairs.add((s, t))
    return pos_pairs


def sample_khop_negatives(
    rng: np.random.RandomState,
    gene_list: List[str],
    pos: Set[Tuple[str, str]],
    n_neg: int,
    neg_tf_from_sources_only: bool,
    sources_idx: Set[int],
) -> List[Tuple[str, str]]:
    n = len(gene_list)
    neg_set: Set[Tuple[str, str]] = set()
    attempts = 0
    max_attempts = max(n_neg * 80, 10_000_000)
    src_list = list(sources_idx) if sources_idx else []
    while len(neg_set) < n_neg and attempts < max_attempts:
        attempts += 1
        if neg_tf_from_sources_only and src_list:
            i = int(rng.choice(src_list))
        else:
            i = int(rng.randint(0, n))
        j = int(rng.randint(0, n))
        if i == j:
            continue
        a, b = gene_list[i], gene_list[j]
        if (a, b) in pos:
            continue
        neg_set.add((a, b))
    if len(neg_set) < n_neg:
        raise RuntimeError(
            f"Negative sampling incomplete: need {n_neg}, got {len(neg_set)}. "
            "Try neg_tf_from_sources_only=False or lower k_hops / panel size."
        )
    return sorted(neg_set)


def build_index_pairs_from_kegg_khop(
    gene_list: List[str],
    grn_matrix: torch.Tensor,
    grn_csv: str,
    tf_col: str,
    tg_col: str,
    tp_col: str,
    k_hops: int,
    neg_sample_ratio: float,
    seed: int,
    neg_tf_from_sources_only: bool = False,
) -> Tuple[List[Tuple[int, int, int]], Dict[str, int]]:
    """
    返回 (i, j, label) 列表，label 来自 grn_matrix[i,j]（无直接边则为 0）。
    """
    panel: Set[str] = set(gene_list)
    edges = kegg_edges_in_panel(grn_csv, tf_col, tg_col, tp_col, panel)
    sources = grn_sources_in_panel(grn_csv, tf_col, tg_col, tp_col, panel)
    adj = build_adj(edges)
    pos_sym = collect_khop_positive_pairs(sources, panel, adj, k_hops)
    n_pos = len(pos_sym)
    n_neg = int(n_pos * float(neg_sample_ratio))
    rng = np.random.RandomState(int(seed))
    g2i = {g: i for i, g in enumerate(gene_list)}
    sources_idx = {g2i[s] for s in sources if s in g2i}
    neg_sym = sample_khop_negatives(
        rng,
        gene_list,
        pos_sym,
        n_neg,
        neg_tf_from_sources_only=neg_tf_from_sources_only,
        sources_idx=sources_idx,
    )

    def _lab(i: int, j: int) -> int:
        v = int(grn_matrix[i, j].item())
        return v if v > 0 else 0

    all_pairs: List[Tuple[int, int, int]] = []
    for s, t in sorted(pos_sym):
        i, j = g2i[s], g2i[t]
        all_pairs.append((i, j, _lab(i, j)))
    for s, t in neg_sym:
        i, j = g2i[s], g2i[t]
        all_pairs.append((i, j, _lab(i, j)))

    stats = {
        "n_kegg_edges_undirected_in_panel": len(edges),
        "n_grn_sources_in_panel": len(sources),
        "n_positive_khop_pairs": n_pos,
        "n_negative_sampled": len(neg_sym),
        "n_total_pairs": len(all_pairs),
    }
    return all_pairs, stats
