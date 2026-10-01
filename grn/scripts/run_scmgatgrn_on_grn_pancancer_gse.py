#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 project_GRN 的 pancancer 九套 GSE 上复现 scMGATGRN（二分类 BCE），并与本仓库 GRN 基准对齐：

- 金标准：同一 KEGG CSV（KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv），
  激活/抑制在 scMGATGRN 侧合并为「正边」；负边采样逻辑与 train_unseen_samples_no_cv._build_pair_samples 一致。
- 边划分：与 train_unseen_samples_no_cv 相同的 stratified train/val/test（0.64 / 0.16 / 0.20）。
- 表达矩阵：基因×细胞（与 npy 一致）；默认 **max_genes=1500** 子集以适配图注意力与 ``adj.to_dense()`` 等显存开销；
  原始 scMGATGRN 实现未强制该数字；显存足够时可 ``--max-genes 0`` 使用 **全部基因**。
- 汇总 CSV：`experiments/scmgatgrn_pancancer/scmgatgrn_kegg_binary_metrics.csv`，含 test_auc / test_aupr / test_accuracy / test_f1 / test_mcc（F1、MCC 为测试集上 sigmoid≥0.5 的二分类预测）。

scMGATGRN 原始仓库路径默认：../project_scMGATGRN/scMGATGRN-main/Code（可用 --scmgat-code 覆盖）。

用法（在 project_GRN 根目录）::

  python -m grn.scripts.run_scmgatgrn_on_grn_pancancer_gse --gse ALL --epochs 20

单数据集::

  python -m grn.scripts.run_scmgatgrn_on_grn_pancancer_gse --gse GSE148071 --epochs 30
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from grn.data.preprocessing import (
    get_GRN,
    normalize_gene_symbol_series,
    resolve_grn_table_columns,
)
from grn.evaluation.confusion_matrix_io import export_confusion_matrix_artifacts
from grn.scripts.pancancer_step01_paths import step01_layout

BINARY_CLASS_NAMES = ("Negative", "Positive")


def _default_grn_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_scmgat_code() -> Path:
    return Path(__file__).resolve().parents[3] / "project_scMGATGRN" / "scMGATGRN-main" / "Code"


def _import_scmgat_modules(code_dir: Path):
    code_dir = code_dir.resolve()
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from utils import Evaluation, adj2saprse_tensor, scRNADataset  # type: ignore  # noqa: E402

    try:
        from scGNN import scMGATGRN  # type: ignore  # noqa: E402
    except Exception as e:  # pragma: no cover
        raise ImportError(
            f"无法从 {code_dir} 导入 scMGATGRN（需将 scMGATGRN 仓库置于 project 同级或传 --scmgat-code）。原始错误: {e}"
        ) from e
    return scMGATGRN, scRNADataset, adj2saprse_tensor, Evaluation


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


def _select_gene_subset(
    gene_list: List[str],
    raw_expr: np.ndarray,
    grn_df: pd.DataFrame,
    tf_col: str,
    tg_col: str,
    max_genes: int,
    seed: int,
) -> List[str]:
    """优先保留 KEGG 边上的基因，再按方差补足到 max_genes。max_genes<=0 表示使用全部基因。"""
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
        picked = [on_edge_list[i] for i in order]
        return picked

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
    # 打乱尾部非边基因顺序，避免总是按字母序
    rng.shuffle(extra)
    return on_edge_list + extra


def _subset_expr_and_genes(raw_expr: np.ndarray, gene_list: List[str], keep: List[str]) -> Tuple[np.ndarray, List[str]]:
    m = {g: i for i, g in enumerate(gene_list)}
    idx = [m[g] for g in keep]
    return np.asarray(raw_expr[idx, :], dtype=np.float32), keep


def _write_scmgat_dataset_layout(
    out_code_parent: Path,
    gse: str,
    folder_num: int,
    expr: np.ndarray,
    gene_list: List[str],
    tf_indices: np.ndarray,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    test_rows: np.ndarray,
) -> None:
    """写入 scMGATGRN main.py 所期望的相对路径（cwd = Code/）。"""
    bench = out_code_parent / "Dataset" / "Benchmark Dataset" / "Specific Dataset" / gse / f"TFs+{folder_num}"
    bench.mkdir(parents=True, exist_ok=True)
    split_dir = out_code_parent / "Specific" / f"{gse} {folder_num}"
    split_dir.mkdir(parents=True, exist_ok=True)

    cells = [f"C{i}" for i in range(expr.shape[1])]
    exp_df = pd.DataFrame(expr, index=gene_list, columns=cells)
    exp_df.to_csv(bench / "BL--ExpressionData.csv", encoding="utf-8")

    tf_df = pd.DataFrame({"TF": [gene_list[int(i)] for i in tf_indices], "index": tf_indices.astype(np.int64)})
    tf_df.to_csv(bench / "TF.csv", index_label="")

    tgt_df = pd.DataFrame({"Gene": gene_list, "index": np.arange(len(gene_list), dtype=np.int64)})
    tgt_df.to_csv(bench / "Target.csv", index_label="")

    # Label.csv / BL--network.csv：原版训练未读取，写最小占位便于人工检查
    pd.DataFrame({"note": ["synthetic_from_project_GRN_KEGG"]}).to_csv(bench / "Label.csv", index=False)
    pd.DataFrame({"note": ["synthetic"]}).to_csv(bench / "BL--network.csv", index=False)

    def _save_edges(name: str, arr: np.ndarray) -> None:
        df = pd.DataFrame(arr, columns=["TF", "Target", "Label"])
        df.to_csv(split_dir / name, index=True, index_label="")

    _save_edges("Train_set.csv", train_rows)
    _save_edges("Validation_set.csv", val_rows)
    _save_edges("Test_set.csv", test_rows)


def _pairs_to_scmgat_arrays(
    pairs: Sequence[Tuple[int, int, int]],
) -> np.ndarray:
    """三分类标签 -> 二分类：1/2 -> 1，0 -> 0。"""
    rows = []
    for i, j, lab in pairs:
        rows.append((i, j, 1 if lab in (1, 2) else 0))
    return np.asarray(rows, dtype=np.int64)


def run_one_gse(
    *,
    scmgat_code: Path,
    gse: str,
    npy_dir: Path,
    kegg: Path,
    grn_tf_col: str,
    grn_target_col: str,
    grn_type_col: str,
    max_genes: int,
    folder_num: int,
    neg_ratio: float,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device_s: str,
    write_layout: bool,
    artifact_dir: Path | None = None,
    skip_train: bool = False,
) -> Dict[str, float]:
    scMGATGRN, scRNADataset, adj2saprse_tensor, Evaluation = _import_scmgat_modules(scmgat_code)

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

    sub_genes = _select_gene_subset(gene_list, np.asarray(raw[:, :], dtype=np.float32), grn_df, tf_col, tg_col, max_genes, seed)
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

    train_arr = _pairs_to_scmgat_arrays(pairs_train)
    val_arr = _pairs_to_scmgat_arrays(pairs_val)
    test_arr = _pairs_to_scmgat_arrays(pairs_test)

    tf_syms = set(grn_sub[tf_col].astype(str))
    tf_indices = np.array([i for i, g in enumerate(gene_list_sub) if g in tf_syms], dtype=np.int64)
    if tf_indices.size == 0:
        raise RuntimeError(
            f"{gse}: 子集基因中无 KEGG TF。若为全基因仍失败请检查标注；若 --max-genes>0 可尝试增大上限。"
        )

    if write_layout:
        _write_scmgat_dataset_layout(
            scmgat_code.parent,
            gse,
            folder_num,
            raw_sub,
            gene_list_sub,
            tf_indices,
            train_arr,
            val_arr,
            test_arr,
        )

    device = torch.device("cuda:0" if device_s == "cuda" and torch.cuda.is_available() else "cpu")
    feature = torch.from_numpy(raw_sub.astype(np.float32))
    feature = feature.to(device)

    train_load = scRNADataset(train_arr, feature.shape[0], flag=False)
    adj = train_load.Adj_Generate(tf_indices, loop=False)
    adj_t = adj2saprse_tensor(adj).to(device)

    train_t = torch.from_numpy(train_arr).to(device)
    val_t = torch.from_numpy(val_arr).to(device)
    test_t = torch.from_numpy(test_arr).to(device)

    model = scMGATGRN(
        input_dim=feature.size(1),
        hidden1_dim=128,
        hidden2_dim=64,
        output_dim=32,
        num_head1=3,
        num_head2=3,
        alpha=0.2,
        device=device,
        type="dot",
        reduction="concate",
        num_nodes=feature.shape[0],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

    out_art = artifact_dir or (step01_layout(Path(__file__).resolve().parents[2])["out_scmgatgrn"] / gse)
    out_art.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_art / "best_model.pth"

    if skip_train:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"cm-only: missing {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"  [{gse}] cm-only loaded {ckpt_path.name}")
    else:
        for epoch in range(epochs):
            model.train()
            running = 0.0
            for train_x, train_y in DataLoader(train_load, batch_size=batch_size, shuffle=True):
                optimizer.zero_grad()
                train_x = train_x.to(device).long()
                train_y = train_y.to(device).view(-1, 1)
                pred = model(feature, adj_t, train_x)
                pred = torch.sigmoid(pred)
                loss = F.binary_cross_entropy(pred, train_y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                running += float(loss.item())
            model.eval()
            with torch.no_grad():
                score = model(feature, adj_t, val_t)
                score = torch.sigmoid(score)
            auc_v, aupr_v, _ = Evaluation(y_pred=score, y_true=val_t[:, -1], flag=False)
            print(f"  [{gse}] epoch {epoch+1}/{epochs} loss={running:.4f} val_AUC={auc_v:.4f} val_AUPR={aupr_v:.4f}")

    model.eval()
    with torch.no_grad():
        score_te = model(feature, adj_t, test_t)
        score_te = torch.sigmoid(score_te)
    y_t = test_t[:, -1].detach().cpu().numpy().astype(int)
    y_p = score_te.detach().cpu().numpy().flatten()
    y_bin = (y_p >= 0.5).astype(int)
    auc_te = float(roc_auc_score(y_t, y_p))
    aupr_te = float(average_precision_score(y_t, y_p))
    acc_te = float(accuracy_score(y_t, y_bin))
    f1_te = float(f1_score(y_t, y_bin, zero_division=0))
    mcc_te = float(matthews_corrcoef(y_t, y_bin))
    sens_te = float(recall_score(y_t, y_bin, zero_division=0))
    spec_te = float(recall_score(y_t, y_bin, pos_label=0, zero_division=0))
    cm = confusion_matrix(y_t, y_bin, labels=[0, 1])
    if not skip_train:
        torch.save(model.state_dict(), ckpt_path)
    export_confusion_matrix_artifacts(
        out_art,
        cm,
        title="scMGATGRN test confusion matrix (KEGG-binary)",
        class_names=BINARY_CLASS_NAMES,
        extra={"method": "scMGATGRN", "gse_id": gse, "threshold": 0.5},
    )
    return {
        "test_auc": auc_te,
        "test_aupr": aupr_te,
        "test_accuracy": acc_te,
        "test_f1": f1_te,
        "test_mcc": mcc_te,
        "test_sensitivity": sens_te,
        "test_specificity": spec_te,
        "n_genes": float(len(gene_list_sub)),
        "n_cells": float(raw_sub.shape[1]),
        "n_pairs_train": float(len(pairs_train)),
        "n_pairs_test": float(len(pairs_test)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="scMGATGRN on GRN pancancer 9 GSE (KEGG-aligned binary edges).")
    ap.add_argument("--grn-root", type=Path, default=_default_grn_root())
    ap.add_argument("--scmgat-code", type=Path, default=_default_scmgat_code())
    ap.add_argument("--npy-dir", type=Path, default=None, help="默认 grn-root/data/exp/pancancer_scdata_step5/npy")
    ap.add_argument("--kegg", type=Path, default=None)
    ap.add_argument(
        "--grn-tf-col",
        default="TF",
        help="与 train_unseen_samples_no_cv --grn-tf-col 一致（resolve_grn_table_columns 第一候选列）。",
    )
    ap.add_argument("--grn-target-col", default="target", help="GRN CSV 靶基因列名候选。")
    ap.add_argument("--grn-type-col", default="Re_type", help="GRN CSV 关系类型列名候选。")
    ap.add_argument("--gse", type=str, required=True, help="GSE ID 或 ALL")
    ap.add_argument(
        "--max-genes",
        type=int,
        default=1500,
        help=">0：优先 KEGG 边基因再按方差补足到该上限（默认 1500，适配图模型显存）；<=0：使用 npy 全部基因（显存风险高）",
    )
    ap.add_argument("--folder-num", type=int, default=1000, help="写入 Dataset/.../TFs+{folder_num} 与文件夹名后缀")
    ap.add_argument("--neg-sample-ratio", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--write-layout", action="store_true", help="写入 scMGATGRN 仓库下的 CSV（便于用原版 main.py 调试）")
    ap.add_argument("--out-csv", type=Path, default=None, help="汇总结果 CSV")
    ap.add_argument("--cm-only", action="store_true", help="加载 best_model.pth 仅导出 test 混淆矩阵")
    ap.add_argument("--force-train", action="store_true", help="忽略已有 checkpoint 重新训练")
    args = ap.parse_args()

    grn_root = args.grn_root.resolve()
    layout = step01_layout(grn_root)
    npy_dir = (args.npy_dir or layout["npy_dir"]).resolve()
    kegg = args.kegg or layout["kegg_csv"]
    kegg = Path(kegg).resolve()

    gses = sorted(p.stem.replace("_values", "") for p in npy_dir.glob("*_values.npy"))
    if args.gse.upper() == "ALL":
        run_list = gses
    else:
        if args.gse not in gses:
            raise SystemExit(f"未知 GSE {args.gse!r}；当前 npy 目录有: {gses}")
        run_list = [args.gse]

    out_rows = []
    for gse in run_list:
        print(f"[scMGATGRN] === {gse} ===")
        art_dir = layout["out_scmgatgrn"] / gse
        skip_train = (args.cm_only or art_dir.joinpath("best_model.pth").is_file()) and not args.force_train
        if args.cm_only and not art_dir.joinpath("best_model.pth").is_file():
            raise SystemExit(f"--cm-only: missing {art_dir / 'best_model.pth'}")
        metrics = run_one_gse(
            scmgat_code=args.scmgat_code.resolve(),
            gse=gse,
            npy_dir=npy_dir,
            kegg=kegg.resolve(),
            grn_tf_col=args.grn_tf_col,
            grn_target_col=args.grn_target_col,
            grn_type_col=args.grn_type_col,
            max_genes=args.max_genes,
            folder_num=args.folder_num,
            neg_ratio=args.neg_sample_ratio,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device_s=args.device,
            write_layout=args.write_layout,
            artifact_dir=art_dir,
            skip_train=skip_train,
        )
        metrics["gse_id"] = gse
        print(
            f"  [{gse}] TEST AUC={metrics['test_auc']:.4f} AUPR={metrics['test_aupr']:.4f} "
            f"ACC@0.5={metrics['test_accuracy']:.4f} F1={metrics['test_f1']:.4f} MCC={metrics['test_mcc']:.4f} "
            f"genes={int(metrics['n_genes'])} cells={int(metrics['n_cells'])}"
        )
        out_rows.append(metrics)

    out_csv = args.out_csv or (layout["out_scmgatgrn"] / "scmgatgrn_kegg_binary_metrics.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        if out_rows:
            w.writeheader()
            for r in out_rows:
                w.writerow({k: r[k] for k in w.fieldnames})
    print(f"[scMGATGRN] wrote {out_csv}")


if __name__ == "__main__":
    main()
