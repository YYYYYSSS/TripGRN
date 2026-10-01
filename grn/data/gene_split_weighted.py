"""
在固定 train/val/test 基因数量的前提下，对基因做三划分，
最大化加权目标：w_repress * (块内抑制边数) + w_activate * (块内激活边数)。

用于缓解随机划分导致大量 GRN 边因跨块而无法进入任一分割的问题。

可选平衡项：在固定各块基因数的前提下，用 balance_gamma * min(各块内受监管边数)
抬高 val/test 中块内正样本规模，缓解 train 样本量远大于 val/test 的现象。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def regulated_edges_from_grn_matrix(GRN_matrix: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (tf_idx, tgt_idx, is_repress)；仅含 GRN>0 的受监管边。"""
    gm = GRN_matrix.detach().cpu().numpy()
    pos = np.where(gm > 0)
    tf_i = pos[0].astype(np.int32)
    tg_i = pos[1].astype(np.int32)
    lab = gm[pos]
    is_repress = (lab == 1)
    return tf_i, tg_i, is_repress


def _weighted_score(
    assignment: np.ndarray,
    tf_i: np.ndarray,
    tg_i: np.ndarray,
    is_repress: np.ndarray,
    w_repress: float,
    w_activate: float,
) -> Tuple[float, int, int]:
    same = assignment[tf_i] == assignment[tg_i]
    n_rep = int(np.count_nonzero(is_repress & same))
    n_act = int(np.count_nonzero((~is_repress) & same))
    return w_repress * n_rep + w_activate * n_act, n_rep, n_act


def _per_split_internal_regulated_counts(
    assignment: np.ndarray,
    tf_i: np.ndarray,
    tg_i: np.ndarray,
) -> np.ndarray:
    """各块内部受监管边条数 shape (3,)，边两端同块则计入该块。"""
    cnt = np.zeros(3, dtype=np.int64)
    m = assignment[tf_i] == assignment[tg_i]
    if not np.any(m):
        return cnt
    parts = assignment[tf_i[m]].astype(np.int64)
    np.add.at(cnt, parts, 1)
    return cnt


def _objective_with_balance(
    assignment: np.ndarray,
    tf_i: np.ndarray,
    tg_i: np.ndarray,
    is_repress: np.ndarray,
    w_repress: float,
    w_activate: float,
    balance_gamma: float,
) -> Tuple[float, float, int, int, np.ndarray]:
    core, n_rep, n_act = _weighted_score(assignment, tf_i, tg_i, is_repress, w_repress, w_activate)
    per = _per_split_internal_regulated_counts(assignment, tf_i, tg_i)
    bonus = float(balance_gamma) * float(per.min()) if balance_gamma else 0.0
    return core + bonus, core, n_rep, n_act, per


def optimize_gene_split_weighted(
    num_genes: int,
    sizes: Tuple[int, int, int],
    tf_i: np.ndarray,
    tg_i: np.ndarray,
    is_repress: np.ndarray,
    w_repress: float,
    w_activate: float,
    seed: int,
    steps: int,
    n_random_starts: int,
    balance_gamma: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    返回 assignment: shape (num_genes,), 取值 0=train, 1=val, 2=test。
    使用随机交换爬山：每次交换两个不同块中的基因，若目标分数不下降则接受。

    balance_gamma>0 时目标为：core + balance_gamma * min(各块内受监管边数)，
    core = w_repress*块内抑制边 + w_activate*块内激活边。
    """
    a, b, c = sizes
    if a + b + c != num_genes:
        raise ValueError(f"sizes sum {a+b+c} != num_genes {num_genes}")

    rng = np.random.RandomState(seed)

    if tf_i.size == 0:
        perm = rng.permutation(num_genes)
        assignment = np.empty(num_genes, dtype=np.int8)
        assignment[perm[:a]] = 0
        assignment[perm[a : a + b]] = 1
        assignment[perm[a + b :]] = 2
        meta = {
            "note": "no regulated edges; random split",
            "weighted_score": 0.0,
            "internal_repress": 0,
            "internal_activate": 0,
        }
        return assignment, meta

    best_assignment = None
    best_score = -1e100
    best_meta = {}

    for start in range(max(1, int(n_random_starts))):
        sub_seed = int(rng.randint(0, 2**31 - 1))
        perm = np.random.RandomState(sub_seed).permutation(num_genes)
        assignment = np.empty(num_genes, dtype=np.int8)
        assignment[perm[:a]] = 0
        assignment[perm[a : a + b]] = 1
        assignment[perm[a + b :]] = 2

        score, core, n_rep, n_act, per = _objective_with_balance(
            assignment, tf_i, tg_i, is_repress, w_repress, w_activate, balance_gamma
        )

        parts: List[set] = [set(np.where(assignment == p)[0].tolist()) for p in range(3)]

        for _ in range(max(0, int(steps))):
            p, q = rng.choice(3, size=2, replace=False)
            if len(parts[p]) == 0 or len(parts[q]) == 0:
                continue
            i = int(rng.choice(list(parts[p])))
            j = int(rng.choice(list(parts[q])))

            assignment[i], assignment[j] = assignment[j], assignment[i]

            new_score, new_core, new_nr, new_na, new_per = _objective_with_balance(
                assignment, tf_i, tg_i, is_repress, w_repress, w_activate, balance_gamma
            )

            if new_score >= score:
                score, core, n_rep, n_act, per = new_score, new_core, new_nr, new_na, new_per
                parts[p].remove(i)
                parts[q].remove(j)
                parts[p].add(j)
                parts[q].add(i)
            else:
                assignment[i], assignment[j] = assignment[j], assignment[i]

        if score > best_score:
            best_score = score
            best_assignment = assignment.copy()
            best_meta = {
                "start_index": start,
                "sub_seed": sub_seed,
                "objective_score": float(score),
                "weighted_core": float(core),
                "weighted_score": float(core),
                "internal_repress": n_rep,
                "internal_activate": n_act,
                "internal_regulated_per_split": per.astype(int).tolist(),
                "internal_regulated_min": int(per.min()),
            }

    assert best_assignment is not None
    best_meta["w_repress"] = float(w_repress)
    best_meta["w_activate"] = float(w_activate)
    best_meta["balance_gamma"] = float(balance_gamma)
    best_meta["opt_steps"] = int(steps)
    best_meta["n_random_starts"] = int(n_random_starts)
    best_meta["seed"] = int(seed)
    return best_assignment, best_meta


def assignment_to_gene_lists(assignment: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
    g_tr = np.where(assignment == 0)[0].tolist()
    g_va = np.where(assignment == 1)[0].tolist()
    g_te = np.where(assignment == 2)[0].tolist()
    return g_tr, g_va, g_te


def save_gene_split_json(
    path: str,
    gene_list: List[str],
    assignment: np.ndarray,
    sizes: Tuple[int, int, int],
    gene_split_ratio: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    a, b, c = sizes
    names_tr = [gene_list[i] for i in np.where(assignment == 0)[0]]
    names_va = [gene_list[i] for i in np.where(assignment == 1)[0]]
    names_te = [gene_list[i] for i in np.where(assignment == 2)[0]]
    payload: Dict[str, Any] = {
        "version": 1,
        "gene_split": gene_split_ratio,
        "split_sizes": [a, b, c],
        "genes_train": names_tr,
        "genes_val": names_va,
        "genes_test": names_te,
    }
    if extra:
        payload["meta"] = extra
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_gene_split_json(
    path: str,
    gene_list: List[str],
    expected_sizes: Tuple[int, int, int],
) -> Tuple[List[int], List[int], List[int], Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    exp_a, exp_b, exp_c = expected_sizes
    gmap = {g: i for i, g in enumerate(gene_list)}
    for key in ("genes_train", "genes_val", "genes_test"):
        if key not in data:
            raise ValueError(f"gene split json missing key: {key}")

    def to_indices(names: List[str], part: str) -> List[int]:
        out: List[int] = []
        for g in names:
            g = str(g).strip()
            if g not in gmap:
                raise ValueError(f"gene {g!r} in json {part} not in current gene_list")
            out.append(gmap[g])
        return out

    g_tr = to_indices(data["genes_train"], "genes_train")
    g_va = to_indices(data["genes_val"], "genes_val")
    g_te = to_indices(data["genes_test"], "genes_test")

    if len(g_tr) != exp_a or len(g_va) != exp_b or len(g_te) != exp_c:
        raise ValueError(
            f"json split sizes {len(g_tr)}/{len(g_va)}/{len(g_te)} "
            f"!= expected {exp_a}/{exp_b}/{exp_c}"
        )

    all_idx = set(g_tr) | set(g_va) | set(g_te)
    if len(all_idx) != len(gene_list):
        raise ValueError("gene lists in json do not partition full gene_list (size mismatch)")
    if len(g_tr) + len(g_va) + len(g_te) != len(gene_list):
        raise ValueError("duplicate genes across splits in json")

    meta = data.get("meta", {})
    return g_tr, g_va, g_te, meta
