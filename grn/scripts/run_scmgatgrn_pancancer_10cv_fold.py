#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one scMGATGRN step01 fold (binary, unseen cells, KEGG-priority gene subset)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from grn.data.pancancer_paths import scmgatgrn_fold_dir, step01_grn_csv
from grn.evaluation.confusion_matrix_io import export_confusion_matrix_artifacts
from grn.scripts.pancancer_step01_binary import (
    binary_metrics_to_step01_json,
    build_grn_matrix,
    estimate_scmgat_dense_adj_gb,
    kegg_tf_indices,
    load_cv_column_splits,
    load_epithelial_expression,
    load_kegg_grn_dataframe,
    pairs_to_index_array,
    select_kegg_priority_gene_subset,
    split_binary_pairs,
    subsample_cols_to_common_length,
    subset_expression,
)

BINARY_CLASS_NAMES = ("No regulation", "Regulation")


def _default_scmgat_code() -> Path:
    return Path(__file__).resolve().parents[3] / "project_scMGATGRN" / "scMGATGRN-main" / "Code"


def _import_scmgat_modules(code_dir: Path):
    code_dir = code_dir.resolve()
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from utils import Evaluation, adj2saprse_tensor, scRNADataset  # type: ignore  # noqa: E402
    from scGNN import scMGATGRN  # type: ignore  # noqa: E402

    return scMGATGRN, scRNADataset, adj2saprse_tensor, Evaluation


def run_scmgatgrn_step01_fold(
    *,
    gse_id: str,
    fold_idx: int,
    grn_root: Path,
    kegg_path: Path,
    scmgat_code: Path,
    fold_dir: Path,
    max_genes: int = 1500,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 3e-3,
    device_s: str = "cuda",
) -> Dict[str, float]:
    scMGATGRN, scRNADataset, adj2saprse_tensor, Evaluation = _import_scmgat_modules(scmgat_code)

    gene_list, raw_expr = load_epithelial_expression(gse_id, grn_root)
    n_genes_full = len(gene_list)
    grn_df, tf_col, tg_col = load_kegg_grn_dataframe(gene_list, kegg_path)
    if max_genes <= 0 or max_genes >= n_genes_full:
        print(
            f"[scmgatgrn-10cv] WARNING: scMGATGRN uses dense N×N graph attention; "
            f"full genome ({n_genes_full} genes) will OOM on 16GB GPU. "
            f"Forcing max_genes=1500.",
            flush=True,
        )
        max_genes = 1500
    sub_genes = select_kegg_priority_gene_subset(
        gene_list, raw_expr, grn_df, tf_col, tg_col, max_genes, seed
    )
    gene_list, raw_expr = subset_expression(gene_list, raw_expr, sub_genes)
    est_gb = estimate_scmgat_dense_adj_gb(len(gene_list))
    print(
        f"[scmgatgrn-10cv] {gse_id} fold {fold_idx:02d}: genes {n_genes_full} -> {len(gene_list)} "
        f"(max_genes={max_genes}, est_dense_GB~{est_gb:.1f})",
        flush=True,
    )

    grn_matrix = build_grn_matrix(gene_list, kegg_path)
    pairs_train, pairs_val, pairs_test = split_binary_pairs(grn_matrix, seed=seed, neg_ratio=1.0)
    train_cols, val_cols, test_cols = load_cv_column_splits(gse_id, fold_idx, grn_root)
    col_rng = np.random.default_rng(int(seed) + int(fold_idx) - 1)
    train_cols, val_cols, test_cols, dim_cells = subsample_cols_to_common_length(
        train_cols, val_cols, test_cols, col_rng
    )
    print(
        f"[scmgatgrn-10cv] {gse_id} fold {fold_idx:02d}: aligned cell cols -> {dim_cells} "
        f"(train/val/test subsampled to min segment)",
        flush=True,
    )

    train_arr = pairs_to_index_array(pairs_train)
    val_arr = pairs_to_index_array(pairs_val)
    test_arr = pairs_to_index_array(pairs_test)
    tf_indices = kegg_tf_indices(gene_list, kegg_path)

    device = torch.device("cuda:0" if device_s == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[scmgatgrn-10cv] building tensors on {device} ...", flush=True)
    feature_tr = torch.from_numpy(raw_expr[:, train_cols].astype(np.float32)).to(device)
    feature_va = torch.from_numpy(raw_expr[:, val_cols].astype(np.float32)).to(device)
    feature_te = torch.from_numpy(raw_expr[:, test_cols].astype(np.float32)).to(device)

    train_load = scRNADataset(train_arr, feature_tr.shape[0], flag=False)
    print("[scmgatgrn-10cv] generating adjacency (train pairs) ...", flush=True)
    adj = train_load.Adj_Generate(tf_indices, loop=False)
    adj_t = adj2saprse_tensor(adj).to(device)

    train_t = torch.from_numpy(train_arr).to(device)
    val_t = torch.from_numpy(val_arr).to(device)
    test_t = torch.from_numpy(test_arr).to(device)

    print("[scmgatgrn-10cv] initializing scMGATGRN (first forward may take 1-2 min) ...", flush=True)
    model = scMGATGRN(
        input_dim=feature_tr.size(1),
        hidden1_dim=128,
        hidden2_dim=64,
        output_dim=32,
        num_head1=3,
        num_head2=3,
        alpha=0.2,
        device=device,
        type="dot",
        reduction="concate",
        num_nodes=feature_tr.shape[0],
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = fold_dir / "best_model.pth"

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for train_x, train_y in DataLoader(train_load, batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            train_x = train_x.to(device).long()
            train_y = train_y.to(device).view(-1, 1)
            pred = torch.sigmoid(model(feature_tr, adj_t, train_x))
            loss = F.binary_cross_entropy(pred, train_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += float(loss.item())
        model.eval()
        with torch.no_grad():
            score = torch.sigmoid(model(feature_va, adj_t, val_t))
        auc_v, aupr_v, _ = Evaluation(y_pred=score, y_true=val_t[:, -1], flag=False)
        print(
            f"[scmgatgrn-10cv] {gse_id} fold {fold_idx:02d} epoch {epoch+1}/{epochs} "
            f"loss={running:.4f} val_AUC={auc_v:.4f} val_AUPR={aupr_v:.4f}",
            flush=True,
        )

    torch.save(model.state_dict(), ckpt_path)

    model.eval()
    with torch.no_grad():
        score_te = torch.sigmoid(model(feature_te, adj_t, test_t))
    y_t = test_t[:, -1].detach().cpu().numpy().astype(int)
    y_p = score_te.detach().cpu().numpy().flatten()
    y_bin = (y_p >= 0.5).astype(int)
    cm = confusion_matrix(y_t, y_bin, labels=[0, 1])

    metrics = binary_metrics_to_step01_json(
        method="scMGATGRN",
        gse_id=gse_id,
        fold_idx=fold_idx,
        y_true=y_t,
        y_prob=y_p,
        y_pred=y_bin,
        cm=cm,
    )
    metrics["n_genes_full"] = int(n_genes_full)
    metrics["n_genes_used"] = int(len(gene_list))
    metrics["max_genes"] = int(max_genes)
    metrics["n_cells_aligned"] = int(dim_cells)
    (fold_dir / "unseen_samples_test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    export_confusion_matrix_artifacts(
        fold_dir,
        cm,
        title="scMGATGRN test confusion matrix (binary)",
        class_names=BINARY_CLASS_NAMES,
        extra={
            "method": "scMGATGRN",
            "fold": int(fold_idx),
            "gse_id": gse_id,
            "task": "binary",
            "n_genes_used": int(len(gene_list)),
        },
    )
    return {
        "test_auc": float(metrics["macro_auc"]),
        "test_aupr": float(metrics["auprc"]),
        "test_accuracy": float(metrics["accuracy"]),
        "test_f1": float(metrics["f1"]),
        "test_mcc": float(metrics["mcc"]),
        "n_genes": float(len(gene_list)),
        "n_cells_train": float(len(train_cols)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one scMGATGRN step01 fold")
    ap.add_argument("--gse-id", required=True)
    ap.add_argument("--fold-idx", type=int, required=True)
    ap.add_argument("--scmgat-code", type=Path, default=_default_scmgat_code())
    ap.add_argument(
        "--max-genes",
        type=int,
        default=1500,
        help="KEGG-priority gene cap (scMGATGRN cannot use full genome; default 1500).",
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    grn_root = Path(__file__).resolve().parents[2]
    kegg = step01_grn_csv(grn_root)
    print(f"[scmgatgrn-fold] grn_csv={kegg}", flush=True)
    fold_dir = scmgatgrn_fold_dir(args.gse_id, args.fold_idx, grn_root)
    run_scmgatgrn_step01_fold(
        gse_id=str(args.gse_id).strip(),
        fold_idx=int(args.fold_idx),
        grn_root=grn_root,
        kegg_path=kegg,
        scmgat_code=args.scmgat_code.resolve(),
        fold_dir=fold_dir,
        max_genes=int(args.max_genes),
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device_s=str(args.device),
    )
    print(f"[scmgatgrn-fold] done -> {fold_dir}")


if __name__ == "__main__":
    main()
