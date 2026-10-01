"""
非脑转移样本上训练（列 5:2:3 + 与 train_unseen_samples_no_cv 相同的 pair 划分），
再在脑转移（mBrain）表达列上做独立测试（与非脑 test 使用同一 pairs_test，表达为全脑列）。

Excel: GSE131907_Lung_Cancer_Feature_Summary.xlsx
  第 3 行为表头，含 Samples、Tissue origins；mBrain 为脑转移。
表达矩阵列名需能匹配到 *_<Samples>（按最长 Samples 优先匹配）。
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import openpyxl
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from grn.data.preprocessing import (
    get_GRN,
    log_expression_tensor,
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
from grn.models.fusion_model import FusionModel, FusionModelAblation
from grn.reproducibility import set_seed

from grn.scripts.train_unseen_samples_no_cv import (
    PairSubsetColumnDataset,
    _build_pair_samples,
    _evaluate,
    _forward_logits,
    _load_gene_feature_csv,
    _prepare_batch,
    split3,
)


@contextmanager
def _openpyxl_workbook_skip_bad_custom_props(path: str) -> Iterator:
    """
    部分 Excel 文件的 docProps/custom.xml 会触发 openpyxl 解析错误；
    复制为临时 xlsx 并跳过该文件后再打开。
    """
    fd, tmp_path = tempfile.mkstemp(suffix="_metastasis_xlsx.xlsx")
    os.close(fd)
    try:
        with zipfile.ZipFile(path, "r") as zin:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                for info in zin.infolist():
                    if info.filename == "docProps/custom.xml":
                        continue
                    zout.writestr(info, zin.read(info.filename))
        wb = openpyxl.load_workbook(tmp_path, read_only=True, data_only=True)
        try:
            yield wb
        finally:
            wb.close()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _load_sample_tissue_from_feature_summary_xlsx(path: str) -> Tuple[List[Tuple[str, str]], Set[str]]:
    with _openpyxl_workbook_skip_bad_custom_props(path) as wb:
        ws = wb.active
        rows_iter = ws.iter_rows(min_row=3, max_row=3, values_only=True)
        header_row = next(rows_iter)
        if not header_row:
            raise ValueError("empty header row (row 3)")
        headers = [str(h).strip() if h is not None else "" for h in header_row]
        try:
            i_samples = headers.index("Samples")
            i_tissue = headers.index("Tissue origins")
        except ValueError as e:
            raise ValueError(f"need columns 'Samples' and 'Tissue origins', got {headers}") from e

        pairs: List[Tuple[str, str]] = []
        for row in ws.iter_rows(min_row=4, values_only=True):
            if not row or row[0] is None:
                break
            first = str(row[0]).strip()
            if first.startswith("*"):
                break
            sample = row[i_samples]
            tissue = row[i_tissue]
            if sample is None or tissue is None:
                continue
            sample_s = str(sample).strip()
            tissue_s = str(tissue).strip()
            if not sample_s:
                continue
            pairs.append((sample_s, tissue_s))

        brain_ids = {s for s, t in pairs if t == "mBrain"}
        return pairs, brain_ids


def _match_column_to_sample_id(col_name: str, sample_ids: Sequence[str]) -> Optional[str]:
    s = str(col_name).strip()
    for sid in sample_ids:
        suffix = "_" + sid
        if s.endswith(suffix):
            return sid
    return None


def _split_expression_columns_by_brain(
    column_names: List[str], brain_sample_ids: Set[str], all_sample_ids: Set[str]
) -> Tuple[List[str], List[str], List[str]]:
    ordered = sorted(all_sample_ids, key=len, reverse=True)
    non_brain: List[str] = []
    brain: List[str] = []
    unmatched: List[str] = []
    for c in column_names:
        sid = _match_column_to_sample_id(c, ordered)
        if sid is None:
            unmatched.append(str(c))
            continue
        if sid in brain_sample_ids:
            brain.append(c)
        else:
            non_brain.append(c)
    return non_brain, brain, unmatched


def _build_seq_features(
    gene_list: List[str],
    ref_indices: Optional[List[int]],
    args,
    use_cds_modal: bool,
    cds_gene_feat,
    tss_gene_feat,
) -> np.ndarray:
    if not use_cds_modal:
        return np.zeros((len(gene_list), 20), dtype=np.float32)
    if cds_gene_feat is not None or tss_gene_feat is not None:
        feat_blocks = []
        if cds_gene_feat is not None:
            feat_blocks.append(cds_gene_feat.loc[gene_list].values.astype(np.float32))
        elif args.cds_file:
            cds_df = pd.read_csv(args.cds_file, header=None, sep=",", skiprows=1)
            cds_all = cds_df.values.astype(np.float32)
            cds_features = cds_all[ref_indices] if ref_indices is not None else cds_all
            if len(cds_features) != len(gene_list):
                raise ValueError(f"CDS and gene_list mismatch: {len(cds_features)} vs {len(gene_list)}")
            feat_blocks.append(cds_features)
        if tss_gene_feat is not None:
            feat_blocks.append(tss_gene_feat.loc[gene_list].values.astype(np.float32))
        elif args.tss_file:
            tss_df = pd.read_csv(args.tss_file, header=None, sep=",", skiprows=1)
            tss_all = tss_df.values.astype(np.float32)
            tss_features = tss_all[ref_indices] if ref_indices is not None else tss_all
            if len(tss_features) != len(gene_list):
                raise ValueError(f"TSS and gene_list mismatch: {len(tss_features)} vs {len(gene_list)}")
            feat_blocks.append(tss_features)
        return np.concatenate(feat_blocks, axis=1)

    cds_df = pd.read_csv(args.cds_file, header=None, sep=",", skiprows=1)
    cds_all = cds_df.values.astype(np.float32)
    cds_features = cds_all[ref_indices] if ref_indices is not None else cds_all
    if len(cds_features) != len(gene_list):
        raise ValueError(f"CDS and gene_list mismatch: {len(cds_features)} vs {len(gene_list)}")
    if args.tss_file:
        tss_df = pd.read_csv(args.tss_file, header=None, sep=",", skiprows=1)
        tss_all = tss_df.values.astype(np.float32)
        tss_features = tss_all[ref_indices] if ref_indices is not None else tss_all
        if len(tss_features) != len(gene_list):
            raise ValueError(f"TSS and gene_list mismatch: {len(tss_features)} vs {len(gene_list)}")
        return np.concatenate([cds_features, tss_features], axis=1)
    return cds_features


CLASS_LABEL_NAMES = ("NoReg", "Repress", "Activate")


def _class_label_name(y: int) -> str:
    return CLASS_LABEL_NAMES[int(y)]


def _collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    gene_list: List[str],
):
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    prob_chunks: List[np.ndarray] = []
    tf_names: List[str] = []
    tg_names: List[str] = []
    with torch.no_grad():
        for batch in loader:
            x = _prepare_batch(batch, device)
            logits = _forward_logits(model, x)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            pred = np.argmax(prob, axis=1)
            y_true.extend(batch["label"].cpu().numpy().astype(np.int64).tolist())
            y_pred.extend(pred.astype(np.int64).tolist())
            prob_chunks.append(prob)
            ti = batch["tf_idx"].cpu().numpy()
            tj = batch["target_idx"].cpu().numpy()
            for i, j in zip(ti, tj):
                tf_names.append(gene_list[int(i)])
                tg_names.append(gene_list[int(j)])
    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        np.vstack(prob_chunks),
        tf_names,
        tg_names,
    )


def _export_nb_brain_pred_diff(
    run_dir: str,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    gene_list: List[str],
    pairs_test: List[Tuple[int, int, int]],
    exp_te: torch.Tensor,
    exp_brain_full: torch.Tensor,
    seq_features: np.ndarray,
    ppi_extractor,
    use_ppi: bool,
    use_expression: bool,
    use_cds_modal: bool,
    ppi_emb_dim: int,
    *,
    prob_diff_export_topk: int = 200,
) -> None:
    ds_nb = PairSubsetColumnDataset(
        exp_te,
        pairs_test,
        seq_features,
        ppi_extractor,
        device=None,
        use_ppi=use_ppi,
        use_expression=use_expression,
        use_cds=use_cds_modal,
        ppi_feat_dim=ppi_emb_dim,
    )
    ds_br = PairSubsetColumnDataset(
        exp_brain_full,
        pairs_test,
        seq_features,
        ppi_extractor,
        device=None,
        use_ppi=use_ppi,
        use_expression=use_expression,
        use_cds=use_cds_modal,
        ppi_feat_dim=ppi_emb_dim,
    )
    ln = DataLoader(ds_nb, batch_size=batch_size, shuffle=False)
    lb = DataLoader(ds_br, batch_size=batch_size, shuffle=False)

    yt, p_nb, pr_nb, tfn, tgn = _collect_predictions(model, ln, device, gene_list)
    yt2, p_br, pr_br, tfn2, tgn2 = _collect_predictions(model, lb, device, gene_list)
    if not (np.array_equal(yt, yt2) and tfn == tfn2 and tgn == tgn2):
        raise RuntimeError("misaligned pairs_test order between non-brain and brain loaders")

    diff_mask = p_nb != p_br
    n_diff = int(diff_mask.sum())
    n_total = int(len(yt))

    csv_path = os.path.join(run_dir, "pairs_test_nonbrain_vs_mBrain_pred_diff.csv")
    summary_path = os.path.join(run_dir, "pairs_test_nonbrain_vs_mBrain_pred_diff_summary.json")

    cols = {
        "TF": [tfn[i] for i in range(n_total) if diff_mask[i]],
        "Target": [tgn[i] for i in range(n_total) if diff_mask[i]],
        "True_Label": yt[diff_mask],
        "True_Label_Name": [_class_label_name(int(yt[i])) for i in range(n_total) if diff_mask[i]],
        "Pred_NonBrain_TestCols": p_nb[diff_mask],
        "Pred_NonBrain_Name": [_class_label_name(int(p_nb[i])) for i in range(n_total) if diff_mask[i]],
        "Pred_mBrain_AllCols": p_br[diff_mask],
        "Pred_mBrain_Name": [_class_label_name(int(p_br[i])) for i in range(n_total) if diff_mask[i]],
    }
    for k, name in enumerate(CLASS_LABEL_NAMES):
        cols[f"P_NonBrain_{name}"] = pr_nb[diff_mask, k]
        cols[f"P_mBrain_{name}"] = pr_br[diff_mask, k]

    pd.DataFrame(cols).to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_pairs_test": n_total,
                "n_pred_diff": n_diff,
                "fraction_pred_diff": (n_diff / n_total) if n_total else 0.0,
                "csv": os.path.basename(csv_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[metastasis] pred diff edges={n_diff}/{n_total} -> {csv_path}")

    # Also export "most different by probability" edges (even if argmax is identical).
    abs_diff = np.abs(pr_nb - pr_br)  # [N, 3]
    max_diff = abs_diff.max(axis=1)  # [N]
    topk = min(int(prob_diff_export_topk), n_total)
    top_idx = np.argsort(-max_diff)[:topk]

    prob_diff_csv = os.path.join(run_dir, f"pairs_test_nonbrain_vs_mBrain_prob_diff_top{topk}.csv")
    prob_diff_summary = os.path.join(run_dir, "pairs_test_nonbrain_vs_mBrain_prob_diff_summary.json")

    prob_cols = {
        "TF": [tfn[i] for i in top_idx],
        "Target": [tgn[i] for i in top_idx],
        "True_Label": [int(yt[i]) for i in top_idx],
        "True_Label_Name": [_class_label_name(int(yt[i])) for i in top_idx],
        "Pred_NonBrain": [int(p_nb[i]) for i in top_idx],
        "Pred_mBrain": [int(p_br[i]) for i in top_idx],
        "Pred_NonBrain_Name": [_class_label_name(int(p_nb[i])) for i in top_idx],
        "Pred_mBrain_Name": [_class_label_name(int(p_br[i])) for i in top_idx],
        "max_prob_diff": [float(max_diff[i]) for i in top_idx],
    }
    for k, name in enumerate(CLASS_LABEL_NAMES):
        prob_cols[f"P_NonBrain_{name}"] = [float(pr_nb[i, k]) for i in top_idx]
        prob_cols[f"P_mBrain_{name}"] = [float(pr_br[i, k]) for i in top_idx]

    pd.DataFrame(prob_cols).to_csv(prob_diff_csv, index=False, encoding="utf-8-sig")
    with open(prob_diff_summary, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_pairs_test": n_total,
                "n_pred_diff": n_diff,
                "max_prob_diff_all_edges": float(max_diff.max()) if n_total else 0.0,
                "mean_max_prob_diff_all_edges": float(max_diff.mean()) if n_total else 0.0,
                "edges_with_max_prob_diff_gt_1e-3": int((max_diff > 1e-3).sum()),
                "edges_with_max_prob_diff_gt_1e-2": int((max_diff > 1e-2).sum()),
                "edges_with_max_prob_diff_gt_5e-2": int((max_diff > 5e-2).sum()),
                "topk": topk,
                "csv": os.path.basename(prob_diff_csv),
                "prob_diff_export_topk_requested": int(prob_diff_export_topk),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[metastasis] prob diff top edges saved -> {prob_diff_csv}")


def _build_metastasis_eval_tensors(
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

    if write_subset_and_manifests:
        full_df = pd.read_csv(args.master_exp_csv)
        gene_col = full_df.columns[0]
        data_cols = [c for c in full_df.columns[1:]]
        non_brain_cols, brain_cols, unmatched = _split_expression_columns_by_brain(
            data_cols, brain_ids, all_sample_ids
        )
        if unmatched:
            print(f"[metastasis] warning: {len(unmatched)} expression columns could not be matched to any Excel Samples")
        if not non_brain_cols:
            raise ValueError("No non-brain expression columns after filtering.")
        if not brain_cols:
            raise ValueError("No brain expression columns after filtering.")
        full_df[[gene_col] + non_brain_cols].to_csv(non_brain_path, index=False)
        full_df[[gene_col] + brain_cols].to_csv(brain_path, index=False)
        manifest = {
            "brain_sample_ids": brain_sample_ids_sorted,
            "non_brain_sample_count": len(non_brain_sample_ids),
            "expression_columns_non_brain": len(non_brain_cols),
            "expression_columns_brain": len(brain_cols),
            "unmatched_expression_columns_count": len(unmatched),
            "pair_split_note": "pairs_train/val/test identical for non-brain train and brain external test (same seed, same GRN).",
        }
        with open(os.path.join(run_dir, "metastasis_split_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    else:
        if not os.path.isfile(non_brain_path) or not os.path.isfile(brain_path):
            raise FileNotFoundError(
                f"subset CSV missing under {run_dir}; run full training once or use write_subset_and_manifests=True"
            )
        peek = pd.read_csv(non_brain_path, nrows=0)
        gene_col = peek.columns[0]

    expr_df = pd.read_csv(non_brain_path)
    expr_df[gene_col] = normalize_gene_symbol_series(expr_df[gene_col])
    expr_df = expr_df.drop_duplicates(subset=[gene_col], keep="first").set_index(gene_col)
    expr_df = expr_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    ref_indices = None
    if args.reference_gene_list_csv:
        ref = normalize_gene_symbol_series(
            pd.read_csv(args.reference_gene_list_csv, usecols=[0]).iloc[:, 0]
        ).tolist()
        ref_map = {g: i for i, g in enumerate(ref)}
        kept = [g for g in expr_df.index.tolist() if g in ref_map]
        expr_df = expr_df.loc[kept]
        ref_indices = [ref_map[g] for g in kept]
        print(f"[metastasis] anchor genes={len(kept)} (no fill-zero)")

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
        print(f"[metastasis] genes after expression and feature alignment: {len(kept)}")

    gene_list = expr_df.index.tolist()
    raw_expr_nonbrain = expr_df.values.astype(np.float32)
    num_samples = raw_expr_nonbrain.shape[1]

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
    print(f"[metastasis] non-brain columns train/val/test = {c_tr}/{c_va}/{c_te}")

    ppi_emb_dim = 160
    ppi_extractor = None
    if use_ppi:
        from grn.features.ppi_features import PPIFeatureExtractor

        ppi_extractor = PPIFeatureExtractor(args.ppi_csv, gene_list=gene_list, emb_dim=ppi_emb_dim, device=None)

    seq_features = _build_seq_features(gene_list, ref_indices, args, use_cds_modal, cds_gene_feat, tss_gene_feat)
    print(f"[metastasis] seq_features shape={seq_features.shape}")

    # 表达归一化：用 non-brain train 列拟合“按基因”的 log2(min-max) 归一化，
    # 再变换 non-brain val/test 与 brain 外验，避免列分布不一致导致表达坍缩。
    x_min, x_max = log2_minmax_per_gene_fit(raw_expr_nonbrain[:, sl_tr], offset=1.0)
    exp_tr = log2_minmax_per_gene_transform(raw_expr_nonbrain[:, sl_tr], x_min, x_max, offset=1.0)
    exp_va = log2_minmax_per_gene_transform(raw_expr_nonbrain[:, sl_va], x_min, x_max, offset=1.0)
    exp_te = log2_minmax_per_gene_transform(raw_expr_nonbrain[:, sl_te], x_min, x_max, offset=1.0)

    brain_df = pd.read_csv(brain_path)
    brain_df[gene_col] = normalize_gene_symbol_series(brain_df[gene_col])
    brain_df = brain_df.drop_duplicates(subset=[gene_col], keep="first").set_index(gene_col)
    brain_df = brain_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    missing = [g for g in gene_list if g not in brain_df.index]
    if missing:
        raise ValueError(f"Brain matrix missing {len(missing)} genes present in non-brain training list (e.g. {missing[:5]})")
    brain_sub = brain_df.loc[gene_list]
    raw_brain = brain_sub.values.astype(np.float32)
    exp_brain_full = log2_minmax_per_gene_transform(raw_brain, x_min, x_max, offset=1.0)

    seq_dim = int(seq_features.shape[1])
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
        "seq_features": seq_features,
        "seq_dim": seq_dim,
        "ppi_extractor": ppi_extractor,
        "ppi_emb_dim": ppi_emb_dim,
        "use_ppi": use_ppi,
        "use_expr": use_expr,
        "use_cds_modal": use_cds_modal,
        "non_brain_path": non_brain_path,
        "brain_path": brain_path,
    }


def _safe_experiment_subdir(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = str(name).strip().replace("\\", "/")
    base = os.path.basename(s)
    if not base or base in {".", ".."}:
        return None
    return base


def _resolve_metastasis_run_dir(
    output_base_dir: str,
    experiment_name: str,
    experiment_subdir: Optional[str],
    no_timestamp: bool,
) -> str:
    parts: List[str] = [output_base_dir, experiment_name]
    sub = _safe_experiment_subdir(experiment_subdir)
    if sub:
        parts.append(sub)
    if no_timestamp:
        return os.path.join(*parts)
    return os.path.join(*parts, datetime.now().strftime("%Y%m%d-%H%M%S"))


def _run_export_pred_diff_only(run_dir: str) -> None:
    run_dir = os.path.abspath(run_dir)
    cfg_path = os.path.join(run_dir, "run_config_master.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(cfg_path)
    with open(cfg_path, encoding="utf-8") as f:
        j = json.load(f)
    j.pop("run_dir", None)
    args = argparse.Namespace(**j)
    set_seed(args.seed)
    ctx = _build_metastasis_eval_tensors(args, run_dir, write_subset_and_manifests=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 用 FusionModelAblation：当 CDS/PPI 被禁用时，不参与融合（避免常量分支污染表达-only）。
    model = FusionModelAblation(
        seq_dim=ctx["seq_dim"],
        num_classes=3,
        use_expression=ctx["use_expr"],
        use_cds=ctx["use_cds_modal"],
        use_ppi=ctx["use_ppi"],
    ).to(device)
    ckpt = os.path.join(run_dir, "best_model_nonbrain_train_brain_extval_fold0.pth")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    _export_nb_brain_pred_diff(
        run_dir=run_dir,
        model=model,
        device=device,
        batch_size=args.batch_size,
        gene_list=ctx["gene_list"],
        pairs_test=ctx["pairs_test"],
        exp_te=ctx["exp_te"],
        exp_brain_full=ctx["exp_brain_full"],
        seq_features=ctx["seq_features"],
        ppi_extractor=ctx["ppi_extractor"],
        use_ppi=ctx["use_ppi"],
        use_expression=ctx["use_expr"],
        use_cds_modal=ctx["use_cds_modal"],
        ppi_emb_dim=ctx["ppi_emb_dim"],
        prob_diff_export_topk=int(getattr(args, "prob_diff_export_topk", 200)),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train on non-mBrain expression columns; test on held-out columns; "
        "then evaluate on mBrain columns with same pairs_test."
    )
    parser.add_argument(
        "--export-pred-diff-only",
        action="store_true",
        help="Only load run_dir config + checkpoint and write pred-diff CSV; requires --run-dir.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Experiment directory with run_config_master.json and subset CSVs; use with --export-pred-diff-only.",
    )
    parser.add_argument("--master-exp-csv", default=None, help="Full scRNA matrix (gene + cell columns).")
    parser.add_argument(
        "--feature-summary-xlsx",
        default=None,
        help="GSE131907_Lung_Cancer_Feature_Summary.xlsx (Samples + Tissue origins).",
    )
    parser.add_argument("--reference-gene-list-csv", default=None)
    parser.add_argument("--grn-csv", default=None)
    parser.add_argument("--cds-file", default=None)
    parser.add_argument("--tss-file", default=None)
    parser.add_argument("--cds-gene-feature-csv", default=None)
    parser.add_argument("--tss-gene-feature-csv", default=None)
    parser.add_argument(
        "--ppi-csv",
        default=None,
        help="PPI 边表 CSV（from,to[,combined_score]）；推荐普适 ppi_string_v12_human_score900.csv，按 gene_list 自动取子图。",
    )
    parser.add_argument("--grn-tf-col", default="TF")
    parser.add_argument("--grn-target-col", default="target")
    parser.add_argument("--grn-type-col", default="Re_type")

    parser.add_argument("--output-base-dir", default="experiments")
    parser.add_argument("--experiment-name", default="repro_metastasis_nonbrain_train_brain_extval")
    parser.add_argument(
        "--experiment-subdir",
        default=None,
        help="Optional single-level subfolder under output-base-dir/experiment-name so several runs share one experiment tree.",
    )
    parser.add_argument("--no-timestamp", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--neg-sample-ratio", type=float, default=1.0)
    parser.add_argument("--sample-split", default="5,2,3")
    parser.add_argument(
        "--prob-diff-export-topk",
        type=int,
        default=200,
        help="导出「非脑 test 列 vs 脑 mBrain 全列」三分类概率 L1 差最大的前 K 条边（pairs_test_nonbrain_vs_mBrain_prob_diff_topK.csv）。",
    )
    parser.add_argument("--no-ppi", action="store_true")
    parser.add_argument("--no-cds", action="store_true")
    parser.add_argument("--no-expression", action="store_true")
    args = parser.parse_args()

    if args.export_pred_diff_only:
        if not args.run_dir:
            raise SystemExit("--run-dir is required with --export-pred-diff-only")
        _run_export_pred_diff_only(args.run_dir)
        return

    if not args.master_exp_csv or not args.feature_summary_xlsx or not args.grn_csv:
        raise SystemExit("Training requires --master-exp-csv, --feature-summary-xlsx, and --grn-csv")

    use_expr = not args.no_expression
    use_cds_modal = not args.no_cds
    use_ppi = not args.no_ppi
    if not (use_expr or use_cds_modal or use_ppi):
        raise ValueError("At least one modality must be enabled (expression, CDS, or PPI).")
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

    ctx = _build_metastasis_eval_tensors(args, run_dir, write_subset_and_manifests=True)
    gene_list = ctx["gene_list"]
    pairs_train = ctx["pairs_train"]
    pairs_val = ctx["pairs_val"]
    pairs_test = ctx["pairs_test"]
    exp_tr = ctx["exp_tr"]
    exp_va = ctx["exp_va"]
    exp_te = ctx["exp_te"]
    exp_brain_full = ctx["exp_brain_full"]
    seq_features = ctx["seq_features"]
    ppi_extractor = ctx["ppi_extractor"]
    ppi_emb_dim = ctx["ppi_emb_dim"]

    ds_train = PairSubsetColumnDataset(
        exp_tr,
        pairs_train,
        seq_features,
        ppi_extractor,
        device=None,
        use_ppi=use_ppi,
        use_expression=use_expr,
        use_cds=use_cds_modal,
        ppi_feat_dim=ppi_emb_dim,
    )
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
    seq_dim = ctx["seq_dim"]
    # 与训练阶段架构保持一致
    model = FusionModelAblation(
        seq_dim=seq_dim,
        num_classes=3,
        use_expression=ctx["use_expr"],
        use_cds=ctx["use_cds_modal"],
        use_ppi=ctx["use_ppi"],
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

    best_path = os.path.join(run_dir, "best_model_nonbrain_train_brain_extval_fold0.pth")
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
            logits = _forward_logits(model, x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_metrics_ep = _evaluate(model, val_loader, device)
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
    nonbrain_val_metrics = _evaluate(model, val_loader, device)
    nonbrain_test_metrics = _evaluate(model, test_loader, device)

    with open(os.path.join(run_dir, "nonbrain_unseen_samples_val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in nonbrain_val_metrics.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(os.path.join(run_dir, "nonbrain_unseen_samples_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in nonbrain_test_metrics.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[metastasis] non-brain val_acc={nonbrain_val_metrics['accuracy']:.4f}, "
        f"test_acc={nonbrain_test_metrics['accuracy']:.4f}"
    )

    try:
        save_learning_curve(plot_dir, epoch_hist, train_loss_hist, val_acc_hist, val_f1_hist)
        save_accuracy_bar_val_test(
            plot_dir,
            float(nonbrain_val_metrics["accuracy"]),
            float(nonbrain_test_metrics["accuracy"]),
            title="非脑列：验证集 vs 测试集 Accuracy",
        )
        save_grouped_metrics_val_test(
            plot_dir,
            nonbrain_val_metrics,
            nonbrain_test_metrics,
            title="非脑列：验证集 vs 测试集 主要指标",
        )
        save_confusion_matrix_fig(
            plot_dir,
            np.asarray(nonbrain_val_metrics["confusion_matrix"]),
            "非脑列·验证集混淆矩阵",
            "confusion_matrix_nonbrain_val.png",
        )
        save_confusion_matrix_fig(
            plot_dir,
            np.asarray(nonbrain_test_metrics["confusion_matrix"]),
            "非脑列·测试集混淆矩阵",
            "confusion_matrix_nonbrain_test.png",
        )
    except Exception as ex:
        print(f"[metastasis] warning: non-brain plot export failed: {ex}")

    # ----- Brain: same genes, all brain columns, same pairs_test -----
    ds_brain_test = PairSubsetColumnDataset(
        exp_brain_full,
        pairs_test,
        seq_features,
        ppi_extractor,
        device=None,
        use_ppi=use_ppi,
        use_expression=use_expr,
        use_cds=use_cds_modal,
        ppi_feat_dim=ppi_emb_dim,
    )
    brain_loader = DataLoader(ds_brain_test, batch_size=args.batch_size, shuffle=False)
    brain_metrics = _evaluate(model, brain_loader, device)
    with open(os.path.join(run_dir, "brain_mBrain_external_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in brain_metrics.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[metastasis] brain mBrain external test_acc={brain_metrics['accuracy']:.4f} (same pairs_test as non-brain)")

    try:
        save_confusion_matrix_fig(
            plot_dir,
            np.asarray(brain_metrics["confusion_matrix"]),
            "脑转移列·外验混淆矩阵（同 pairs_test）",
            "confusion_matrix_brain_mbrain_external.png",
        )
        save_multi_accuracy_bar(
            plot_dir,
            ("非脑 Val", "非脑 Test", "脑转移外验"),
            (
                float(nonbrain_val_metrics["accuracy"]),
                float(nonbrain_test_metrics["accuracy"]),
                float(brain_metrics["accuracy"]),
            ),
            "accuracy_nonbrain_vs_brain_external.png",
            "Accuracy：非脑划分 vs 脑转移外验",
        )
    except Exception as ex:
        print(f"[metastasis] warning: brain / summary plot export failed: {ex}")

    _export_nb_brain_pred_diff(
        run_dir=run_dir,
        model=model,
        device=device,
        batch_size=args.batch_size,
        gene_list=gene_list,
        pairs_test=pairs_test,
        exp_te=exp_te,
        exp_brain_full=exp_brain_full,
        seq_features=seq_features,
        ppi_extractor=ppi_extractor,
        use_ppi=use_ppi,
        use_expression=use_expr,
        use_cds_modal=use_cds_modal,
        ppi_emb_dim=ppi_emb_dim,
        prob_diff_export_topk=int(args.prob_diff_export_topk),
    )

    print(f"[metastasis] saved run_dir={run_dir}")
    print(f"[metastasis] model={best_path}")


if __name__ == "__main__":
    main()
