#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 project_GRN 的 pancancer 九套 GSE 上运行 GENIE3（树模型特征重要性），与 GENELink / scMGATGRN
同一套 KEGG 二分类边与划分：

- 金标准与负采样、train/val/test 边划分：与 ``run_genelink_on_grn_pancancer_gse`` 一致；基因列表默认与其相同（``--max-genes 0`` = npy 全部基因）。
- GENIE3 使用 **全部细胞** 的表达矩阵估计 VIM（无监督，不使用边标签）；与深度方法一样，
  表达矩阵本身不按细胞划分。
- **AUC / AUPR**：测试边上取 VIM[TF, Target] 的原始分数（秩指标对单调变换稳健）。
- **Accuracy / F1 / MCC @0.5**：先用 **训练边** 上 VIM 分数的 min-max 将测试边分数线性压到 [0,1]，
  再取阈值 0.5，以便与 KEGG 基准 CSV 列语义对齐。

GENIE3 参考实现默认路径：``../project_GENIE3/GENIE3_python``（``--genie3-python-dir`` 可覆盖）。

用法::

  cd project_GRN
  python -m grn.scripts.run_genie3_on_grn_pancancer_gse --gse ALL --nthreads 8 --ntrees 300

说明：默认 **不限制基因数**（与 GENELink 基准脚本一致，``--max-genes 0``）。GENIE3 计算量随基因数与
``ntrees`` 增长很快，可按机器减小 ``--ntrees`` 或显式 ``--max-genes`` 做子集。

Windows 下 ``nthreads>1`` 时，官方 ``GENIE3.py`` 使用 ``multiprocessing.Pool``（spawn）。
须将 ``GENIE3_python`` 以 **可导入的包名** ``GENIE3`` 加入 ``sys.path``（本脚本通过环境变量
``GRN_GENIE3_PYTHON_DIR`` + ``import GENIE3`` 实现），否则子进程会报
``ModuleNotFoundError: No module named 'genie3_vendor_impl'``——与是否下载模型文件无关。
"""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import os
import sys
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from grn.data.preprocessing import (
    get_GRN,
    normalize_gene_symbol_series,
    resolve_grn_table_columns,
)

# 复用与 GENELink 基准相同的样本构造逻辑
from grn.scripts.run_genelink_on_grn_pancancer_gse import (
    _build_pair_samples,
    _pairs_to_binary_arrays,
    _select_gene_subset,
    _subset_expr_and_genes,
)


def _default_grn_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_genie3_python_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "project_GENIE3" / "GENIE3_python"


_ENV_GENIE3_DIR = "GRN_GENIE3_PYTHON_DIR"


def _prepend_pythonpath(entry: str) -> None:
    """Windows spawn 下 Pool worker 会重启解释器；PYTHONPATH 比父进程里改过的 sys.path 更可靠。"""
    prev = os.environ.get("PYTHONPATH", "")
    parts = [p for p in prev.split(os.pathsep) if p]
    if entry in parts:
        return
    os.environ["PYTHONPATH"] = entry if not parts else (entry + os.pathsep + prev)


def _ensure_genie3_on_sys_path(genie3_py_dir: Path) -> None:
    """
    将 GENIE3 官方脚本目录放到 sys.path 首位，并写入环境变量，供 Windows spawn 子进程
    在 unpickle ``GENIE3.wr_GENIE3_single`` 时能 ``import GENIE3``。
    """
    root = genie3_py_dir.resolve()
    py = root / "GENIE3.py"
    if not py.is_file():
        raise FileNotFoundError(f"未找到 GENIE3.py: {py}")
    s = str(root)
    os.environ[_ENV_GENIE3_DIR] = s
    if s not in sys.path:
        sys.path.insert(0, s)
    _prepend_pythonpath(s)


def _import_genie3_callable() -> Callable:
    mod = importlib.import_module("GENIE3")
    if not hasattr(mod, "GENIE3"):
        raise ImportError("模块 GENIE3 中未定义 GENIE3 函数")
    return mod.GENIE3


def _bootstrap_genie3_path_on_import() -> None:
    """模块加载时执行：子进程（spawn）继承环境变量后仍要把 GENIE3_python 加入 sys.path / PYTHONPATH。"""
    env = os.environ.get(_ENV_GENIE3_DIR)
    d = Path(env) if env else _default_genie3_python_dir()
    if d.is_dir() and (d / "GENIE3.py").is_file():
        s = str(d.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)
        _prepend_pythonpath(s)


_bootstrap_genie3_path_on_import()


def _edge_scores_from_vim(vim: np.ndarray, pairs_arr: np.ndarray) -> np.ndarray:
    tf_i = pairs_arr[:, 0].astype(np.int64, copy=False)
    tg_i = pairs_arr[:, 1].astype(np.int64, copy=False)
    return np.asarray(vim[tf_i, tg_i], dtype=np.float64)


def _minmax_from_train(train_scores: np.ndarray, query_scores: np.ndarray) -> np.ndarray:
    smin = float(np.min(train_scores))
    smax = float(np.max(train_scores))
    span = smax - smin + 1e-12
    return np.clip((query_scores - smin) / span, 0.0, 1.0)


def run_one_gse(
    *,
    genie3_py_dir: Path,
    gse: str,
    npy_dir: Path,
    kegg: Path,
    grn_tf_col: str,
    grn_target_col: str,
    grn_type_col: str,
    max_genes: int,
    neg_ratio: float,
    seed: int,
    tree_method: str,
    K: str | int,
    ntrees: int,
    nthreads: int,
    quiet: bool,
) -> Dict[str, float]:
    _ensure_genie3_on_sys_path(genie3_py_dir)
    GENIE3 = _import_genie3_callable()

    values_p = npy_dir / f"{gse}_values.npy"
    genes_p = npy_dir / f"{gse}_genes.npy"
    if not values_p.is_file() or not genes_p.is_file():
        raise FileNotFoundError(f"缺少 {values_p} 或 {genes_p}")

    g_arr = np.load(genes_p, allow_pickle=True)
    raw = np.load(values_p, mmap_mode="r")
    gene_list = normalize_gene_symbol_series(pd.Series([str(x) for x in g_arr.tolist()])).tolist()
    if len(gene_list) != raw.shape[0]:
        raise ValueError("genes.npy 与 values.npy 行数不一致")

    grn_df = pd.read_csv(kegg)
    tf_col, tg_col, tp_col = resolve_grn_table_columns(grn_df, grn_tf_col, grn_target_col, grn_type_col)
    if tf_col is None or tg_col is None or tp_col is None:
        raise ValueError(
            f"{gse}: GRN 列无法解析。columns={list(grn_df.columns)}；"
            f"传入 tf={grn_tf_col!r}, target={grn_target_col!r}, type={grn_type_col!r}"
        )
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
        raise ValueError(f"{gse}: KEGG 在表达基因上无可用边")

    sub_genes = _select_gene_subset(
        gene_list, np.asarray(raw[:, :], dtype=np.float32), grn_df, tf_col, tg_col, max_genes, seed
    )
    raw_sub, gene_list_sub = _subset_expr_and_genes(np.asarray(raw[:, :], dtype=np.float32), gene_list, sub_genes)

    sg = set(gene_list_sub)
    grn_sub = grn_df[grn_df[tf_col].isin(sg) & grn_df[tg_col].isin(sg)].copy()
    rel_sub = grn_sub[tp_col].astype(str).str.lower().str.strip()
    rel_sub = rel_sub.replace(
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
    grn_sub = grn_sub[rel_sub.isin(["activator", "repressor"])].copy()
    rel_sub = rel_sub[grn_sub.index]
    if grn_sub.empty:
        raise ValueError(
            f"{gse}: 所选基因子集上 KEGG 边为空。若为全基因仍失败，请检查 KEGG 与表达基因符号是否匹配；"
            "若使用了 --max-genes>0，可尝试增大上限。"
        )

    known_grn = (grn_sub[tf_col].values, grn_sub[tg_col].values, rel_sub.values)
    grn_matrix = get_GRN(known_grn, gene_list_sub)
    all_pairs = _build_pair_samples(grn_matrix, neg_ratio, seed)
    labels = np.array([lab for _, _, lab in all_pairs], dtype=np.int64)
    idx = np.arange(len(all_pairs))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels)
    y_tr = labels[tr_idx]
    tr_idx2, va_idx = train_test_split(tr_idx, test_size=0.2, random_state=seed, stratify=y_tr)

    pairs_train = [all_pairs[i] for i in tr_idx2]
    pairs_val = [all_pairs[i] for i in va_idx]
    pairs_test = [all_pairs[i] for i in te_idx]

    train_arr = _pairs_to_binary_arrays(pairs_train)
    val_arr = _pairs_to_binary_arrays(pairs_val)
    test_arr = _pairs_to_binary_arrays(pairs_test)

    # GENIE3: rows = cells, cols = genes
    expr = np.asarray(raw_sub.T, dtype=np.float64, order="C")
    buf = io.StringIO()
    ctx = redirect_stdout(buf) if quiet else nullcontext()
    with ctx:
        vim = GENIE3(
            expr,
            gene_names=gene_list_sub,
            regulators="all",
            tree_method=tree_method,
            K=K,
            ntrees=ntrees,
            nthreads=nthreads,
        )
    vim = np.asarray(vim, dtype=np.float64)

    s_train = _edge_scores_from_vim(vim, train_arr)
    s_val = _edge_scores_from_vim(vim, val_arr)
    s_test = _edge_scores_from_vim(vim, test_arr)

    y_t = test_arr[:, 2].astype(int, copy=False)
    auc_te = float(roc_auc_score(y_t, s_test))
    aupr_te = float(average_precision_score(y_t, s_test))

    s_test_01 = _minmax_from_train(s_train, s_test)
    s_val_01 = _minmax_from_train(s_train, s_val)
    # 若验证集上 0.5 退化（全一类），退回 0.5；否则可选优化阈值——此处保持 0.5 与基准一致
    _ = s_val_01  # 保留供将来做阈值调参
    y_bin = (s_test_01 >= 0.5).astype(int)
    acc_te = float(accuracy_score(y_t, y_bin))
    f1_te = float(f1_score(y_t, y_bin, zero_division=0))
    mcc_te = float(matthews_corrcoef(y_t, y_bin))

    return {
        "test_auc": auc_te,
        "test_aupr": aupr_te,
        "test_accuracy": acc_te,
        "test_f1": f1_te,
        "test_mcc": mcc_te,
        "n_genes": float(len(gene_list_sub)),
        "n_cells": float(raw_sub.shape[1]),
        "n_pairs_train": float(len(pairs_train)),
        "n_pairs_test": float(len(pairs_test)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="GENIE3 on GRN pancancer 9 GSE (KEGG-aligned binary edges).")
    ap.add_argument("--grn-root", type=Path, default=_default_grn_root())
    ap.add_argument("--genie3-python-dir", type=Path, default=_default_genie3_python_dir())
    ap.add_argument("--npy-dir", type=Path, default=None)
    ap.add_argument("--kegg", type=Path, default=None)
    ap.add_argument("--grn-tf-col", default="TF")
    ap.add_argument("--grn-target-col", default="target")
    ap.add_argument("--grn-type-col", default="Re_type")
    ap.add_argument("--gse", type=str, required=True, help="GSE ID 或 ALL")
    ap.add_argument(
        "--max-genes",
        type=int,
        default=0,
        help="<=0 使用 npy 全部基因；>0 时与 GENELink 基准相同子集逻辑",
    )
    ap.add_argument("--neg-sample-ratio", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tree-method", choices=("RF", "ET"), default="ET", help="Extra-Trees 通常更快")
    ap.add_argument(
        "--K",
        default="sqrt",
        help='RF 每节点候选特征数：sqrt / all / 正整数（与官方 GENIE3 一致）',
    )
    ap.add_argument("--ntrees", type=int, default=300, help="每靶基因的树棵数；越大越慢")
    ap.add_argument(
        "--nthreads",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 4))),
        help="并行靶基因数（官方实现的多进程）",
    )
    ap.add_argument("--quiet", action="store_true", help="抑制 GENIE3 逐基因打印")
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    K: str | int
    if isinstance(args.K, str) and args.K.isdigit():
        K = int(args.K)
    else:
        K = args.K  # type: ignore[assignment]

    grn_root = args.grn_root.resolve()
    npy_dir = (args.npy_dir or (grn_root / "data" / "exp" / "pancancer_scdata_step5" / "npy")).resolve()
    kegg = args.kegg or (grn_root / "data" / "network" / "grn" / "KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv")

    gses = sorted(p.stem.replace("_values", "") for p in npy_dir.glob("*_values.npy"))
    if args.gse.upper() == "ALL":
        run_list = gses
    else:
        if args.gse not in gses:
            raise SystemExit(f"未知 GSE {args.gse!r}；当前 npy 目录有: {gses}")
        run_list = [args.gse]

    out_rows: List[Dict[str, float]] = []
    for gse in run_list:
        print(f"[GENIE3] === {gse} ===")
        metrics = run_one_gse(
            genie3_py_dir=args.genie3_python_dir.resolve(),
            gse=gse,
            npy_dir=npy_dir,
            kegg=kegg.resolve(),
            grn_tf_col=args.grn_tf_col,
            grn_target_col=args.grn_target_col,
            grn_type_col=args.grn_type_col,
            max_genes=args.max_genes,
            neg_ratio=args.neg_sample_ratio,
            seed=args.seed,
            tree_method=args.tree_method,
            K=K,
            ntrees=args.ntrees,
            nthreads=args.nthreads,
            quiet=args.quiet,
        )
        metrics["gse_id"] = gse
        print(
            f"  [{gse}] TEST AUC={metrics['test_auc']:.4f} AUPR={metrics['test_aupr']:.4f} "
            f"ACC@0.5={metrics['test_accuracy']:.4f} F1={metrics['test_f1']:.4f} MCC={metrics['test_mcc']:.4f} "
            f"genes={int(metrics['n_genes'])} cells={int(metrics['n_cells'])}"
        )
        out_rows.append(metrics)

    out_csv = args.out_csv or (grn_root / "experiments" / "genie3_pancancer" / "genie3_kegg_binary_metrics.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        keys = list(out_rows[0].keys()) if out_rows else []
        w = csv.DictWriter(f, fieldnames=keys)
        if out_rows:
            w.writeheader()
            for r in out_rows:
                w.writerow({k: r[k] for k in keys})
    print(f"[GENIE3] wrote {out_csv}")


if __name__ == "__main__":
    main()
