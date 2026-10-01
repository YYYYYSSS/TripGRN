#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infer TripGRN on fixed-split TEST cells; save full per-edge class probabilities.

Output columns: TF, Target, True_Label, True_Label_Name, Pred, Pred_Name,
P_NoReg, P_Repress, P_Activate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from grn.data.gse131907_celltype_paths import (
    celltype_genes_npy,
    celltype_values_npy,
    cv_splits_path,
    gse131907_celltype_layout,
    tripgrn_fixed_run_dir,
)
from grn.data.kegg_khop_pairs import build_index_pairs_from_kegg_khop
from grn.data.pancancer_cv import get_fold_split, load_cv_splits
from grn.data.preprocessing import (
    get_GRN,
    log2_minmax_per_gene_fit,
    log2_minmax_per_gene_transform,
    normalize_gene_symbol_series,
    resolve_grn_table_columns,
)
from grn.models.fusion_model import FusionModelAblation
from grn.scripts.train_unseen_samples_no_cv import (
    PairSubsetColumnDataset,
    _build_pair_samples,
    _forward_logits,
    _load_gene_feature_csv,
    _prepare_batch,
)


LABEL_NAMES = {0: "NoReg", 1: "Repress", 2: "Activate"}
PROB_COLS = ("P_NoReg", "P_Repress", "P_Activate")


def _autocast_bf16(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _predict_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp_bf16: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return y_true, y_pred, probs. Pair order matches loader (shuffle=False)."""
    model.eval()
    y_true: List[int] = []
    probs: List[np.ndarray] = []
    total = len(loader)
    t0 = time.time()
    with torch.inference_mode():
        for bi, batch in enumerate(loader, start=1):
            x = _prepare_batch(batch, device)
            with _autocast_bf16(device, use_amp_bf16):
                logits = _forward_logits(model, x)
            prob = torch.softmax(logits.float(), dim=1).cpu().numpy()
            y_true.extend(batch["label"].cpu().numpy().astype(np.int64).tolist())
            probs.append(prob)
            if bi == 1 or bi % max(1, total // 50) == 0 or bi == total:
                print(
                    f"[celltype-infer] batch {bi}/{total}, pairs={len(y_true)}, "
                    f"elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )
    return (
        np.asarray(y_true, dtype=np.int64),
        np.argmax(np.vstack(probs), axis=1).astype(np.int64),
        np.vstack(probs).astype(np.float32),
    )


def _run_one(args: argparse.Namespace, tid: str) -> None:
    layout = gse131907_celltype_layout()
    run_dir = Path(args.run_dir) if args.run_dir else tripgrn_fixed_run_dir(tid, layout["grn_root"])
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    best_path = run_dir / "best_model_fold0.pth"
    if not best_path.is_file():
        raise FileNotFoundError(best_path)

    out_dir = layout["grn_infer_base"] / tid
    out_csv = out_dir / "pairs_test_all_probs.csv"
    if out_csv.is_file() and not args.force:
        print(f"[celltype-infer] skip existing {out_csv}")
        return

    cfg_path = run_dir / "run_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}

    v_path = celltype_values_npy(tid, log_normalized=True, grn_root=layout["grn_root"])
    g_path = celltype_genes_npy(tid, grn_root=layout["grn_root"])
    splits_path = cv_splits_path(tid, fixed=True, grn_root=layout["grn_root"])
    fold = get_fold_split(load_cv_splits(splits_path), 1)
    train_cols = np.asarray(fold["train_cols"], dtype=np.int64)
    test_cols = np.asarray(fold["test_cols"], dtype=np.int64)

    g_arr = np.load(g_path, allow_pickle=True)
    raw_expr = np.load(v_path, mmap_mode="r")
    gene_list = normalize_gene_symbol_series(pd.Series([str(x) for x in g_arr.tolist()])).tolist()

    grn_csv = str(cfg.get("grn_csv") or layout["grn_csv"])
    cds_csv = str(cfg.get("cds_gene_feature_csv") or layout["cds_gene"])
    cds_gene_feat = _load_gene_feature_csv(cds_csv)
    feat_genes = set(gene_list) & set(cds_gene_feat.index.tolist())
    kept = [g for g in gene_list if g in feat_genes]
    if not kept:
        raise SystemExit("No genes after expression/CDS alignment")
    g_to_i = {g: i for i, g in enumerate(gene_list)}
    idx = np.array([g_to_i[g] for g in kept], dtype=np.int64)
    raw_expr = np.asarray(raw_expr[idx, :], dtype=np.float32)
    gene_list = kept

    grn_df = pd.read_csv(grn_csv)
    tf_col, tg_col, tp_col = resolve_grn_table_columns(
        grn_df, "TF", "target", "Re_type"
    )
    grn_df[tf_col] = normalize_gene_symbol_series(grn_df[tf_col])
    grn_df[tg_col] = normalize_gene_symbol_series(grn_df[tg_col])
    gene_set = set(gene_list)
    grn_df = grn_df[grn_df[tf_col].isin(gene_set) & grn_df[tg_col].isin(gene_set)].copy()
    rel = grn_df[tp_col].astype(str).str.lower().str.strip().replace(
        {
            "activation": "activator",
            "activate": "activator",
            "inhibition": "repressor",
            "inhibit": "repressor",
            "repression": "repressor",
        }
    )
    grn_df = grn_df[rel.isin(["activator", "repressor"])].copy()
    rel = rel[grn_df.index]
    grn_matrix = get_GRN((grn_df[tf_col].values, grn_df[tg_col].values, rel.values), gene_list)

    if args.pair_mode == "kegg_khop":
        all_pairs, kh_stats = build_index_pairs_from_kegg_khop(
            gene_list,
            grn_matrix,
            grn_csv,
            "TF",
            "target",
            "Re_type",
            int(args.kegg_k_hops),
            float(args.neg_sample_ratio),
            int(args.seed),
        )
        print(f"[celltype-infer] kegg_khop pairs={len(all_pairs)} stats={kh_stats}")
    else:
        all_pairs = _build_pair_samples(grn_matrix, float(args.neg_sample_ratio), int(args.seed))
        print(f"[celltype-infer] grn pairs={len(all_pairs)}")

    exp_tr_src = raw_expr[:, train_cols]
    exp_te_src = raw_expr[:, test_cols]
    x_min, x_max = log2_minmax_per_gene_fit(exp_tr_src, offset=1.0)
    exp_te = log2_minmax_per_gene_transform(exp_te_src, x_min, x_max, offset=1.0).contiguous()
    exp_tr = log2_minmax_per_gene_transform(exp_tr_src, x_min, x_max, offset=1.0).contiguous()

    # Match training-time expression length cap (read from run_config if present)
    max_expr_cols = int(cfg.get("max_expression_cols") or args.max_expression_cols or 0)
    reduce_mode = str(cfg.get("expression_col_reduce") or args.expression_col_reduce)
    if max_expr_cols > 0:
        for name, mat_attr in (("train", "exp_tr"), ("test", "exp_te")):
            mat = exp_tr if name == "train" else exp_te
            t = int(mat.shape[1])
            if t <= max_expr_cols:
                continue
            if reduce_mode == "avg_pool":
                mat = torch.nn.functional.adaptive_avg_pool1d(mat.unsqueeze(0), max_expr_cols).squeeze(0).contiguous()
            else:
                rng = np.random.default_rng(int(args.seed) + (0 if name == "train" else 2))
                pick = np.sort(rng.choice(t, size=max_expr_cols, replace=False))
                mat = mat[:, torch.as_tensor(pick, dtype=torch.long)].contiguous()
            if name == "train":
                exp_tr = mat
            else:
                exp_te = mat
            print(f"[celltype-infer] {name}: reduce cols {t} -> {max_expr_cols} ({reduce_mode})")
    _ = exp_tr  # fit-path already used; keep for future consistency checks

    use_ppi = not bool(cfg.get("no_ppi"))
    use_expr = not bool(cfg.get("no_expression"))
    use_cds = not bool(cfg.get("no_cds"))
    ppi_csv = str(cfg.get("ppi_csv") or layout["ppi_csv"])
    ppi_emb_dim = 160
    ppi_extractor = None
    ppi_node_features = None
    if use_ppi:
        from grn.features.ppi_features import PPIFeatureExtractor

        ppi_extractor = PPIFeatureExtractor(ppi_csv, gene_list=gene_list, emb_dim=ppi_emb_dim, device=None)
        ppi_node_features = ppi_extractor.extract_node_features().detach().cpu()

    seq_features = cds_gene_feat.loc[gene_list].values.astype(np.float32)

    ds_test = PairSubsetColumnDataset(
        exp_te,
        all_pairs,
        seq_features,
        ppi_extractor,
        ppi_node_features=ppi_node_features,
        device=None,
        use_ppi=use_ppi,
        use_expression=use_expr,
        use_cds=use_cds,
        ppi_feat_dim=ppi_emb_dim,
    )
    loader = DataLoader(ds_test, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _bf16_fn = getattr(torch.cuda, "is_bf16_supported", None)
    use_amp = bool(args.amp_bf16 and callable(_bf16_fn) and _bf16_fn() and device.type == "cuda")

    model = FusionModelAblation(
        seq_dim=int(seq_features.shape[1]),
        num_classes=3,
        use_expression=use_expr,
        use_cds=use_cds,
        use_ppi=use_ppi,
    ).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))

    yt, yp, pr = _predict_probs(model, loader, device, use_amp_bf16=use_amp)
    if len(yt) != len(all_pairs):
        raise RuntimeError(f"pair count mismatch: preds={len(yt)} pairs={len(all_pairs)}")
    tfn = [gene_list[int(i)] for i, _, _ in all_pairs]
    tgn = [gene_list[int(j)] for _, j, _ in all_pairs]

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "TF": tfn,
            "Target": tgn,
            "True_Label": yt,
            "True_Label_Name": [LABEL_NAMES[int(x)] for x in yt],
            "Pred": yp,
            "Pred_Name": [LABEL_NAMES[int(x)] for x in yp],
            PROB_COLS[0]: pr[:, 0],
            PROB_COLS[1]: pr[:, 1],
            PROB_COLS[2]: pr[:, 2],
        }
    )
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    meta = {
        "type_id": tid,
        "run_dir": str(run_dir.resolve()),
        "n_pairs": int(len(df)),
        "n_test_cells": int(len(test_cols)),
        "pair_mode": args.pair_mode,
        "prob_columns": list(PROB_COLS),
        "csv": out_csv.name,
    }
    (out_dir / "pairs_test_all_probs_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[celltype-infer] wrote {out_csv} ({len(df)} pairs)")


def main() -> None:
    ap = argparse.ArgumentParser(description="GSE131907 step06 test-set GRN inference with full probs")
    ap.add_argument("--type", required=True, help="type_id, annotation label, or ALL")
    ap.add_argument("--run-dir", type=Path, default=None, help="Fixed TripGRN run dir (single type only)")
    ap.add_argument("--pair-mode", choices=("grn", "kegg_khop"), default="grn")
    ap.add_argument("--kegg-k-hops", type=int, default=2)
    ap.add_argument("--neg-sample-ratio", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--amp-bf16", action="store_true")
    ap.add_argument("--max-expression-cols", type=int, default=10000)
    ap.add_argument("--expression-col-reduce", choices=("subsample", "avg_pool"), default="subsample")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from grn.data.gse131907_celltype_paths import resolve_type_ids

    types = resolve_type_ids(args.type)
    if args.run_dir is not None and len(types) != 1:
        raise SystemExit("--run-dir requires a single --type (not ALL)")
    for tid, label in types:
        print(f"[celltype-infer] === {tid} ({label}) ===", flush=True)
        _run_one(args, tid)


if __name__ == "__main__":
    main()
