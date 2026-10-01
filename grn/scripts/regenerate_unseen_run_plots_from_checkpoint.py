"""
仅用已有 best_model_fold0.pth + run_config.json 重建 val/test 推理并写出 plots/
（不重新训练）。数据管线与 train_unseen_samples_no_cv 一致（含 npy 表达、列划分、pair 划分）。

无法重建：learning_curve.png（无 epoch 历史文件）。

用法：
  python -m grn.scripts.regenerate_unseen_run_plots_from_checkpoint --parent-dir experiments/gse253718_ablation_step1_seed42
  python -m grn.scripts.regenerate_unseen_run_plots_from_checkpoint --run-dir experiments/gse253718_ablation_step1_seed42/01_expr_only
"""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from grn.data.preprocessing import (
    get_GRN,
    log2_minmax_per_gene_fit,
    log2_minmax_per_gene_transform,
    normalize_gene_symbol_series,
    resolve_grn_table_columns,
)
from grn.data.tf_source_filter import filter_grn_by_source_symbols, load_regulator_symbol_set
from grn.evaluation.training_plots import (
    per_class_ovr_accuracy_precision_recall_from_cm,
    save_accuracy_bar_val_test,
    save_confusion_matrix_fig,
    save_grouped_metrics_val_test,
    save_ovr_roc_pr_curves,
    save_per_class_auprc_bar,
    save_per_class_metric_bars,
)
from grn.models.fusion_model import FusionModelAblation
from grn.reproducibility import set_seed
from grn.scripts.train_unseen_samples_no_cv import (
    PairSubsetColumnDataset,
    _build_pair_samples,
    _evaluate,
    _load_gene_feature_csv,
    _predict_arrays,
    split3,
)


def _validate_args_like_train(args: Namespace) -> None:
    use_expr = not args.no_expression
    use_cds_modal = not args.no_cds
    use_ppi = not args.no_ppi
    if not (use_expr or use_cds_modal or use_ppi):
        raise ValueError("At least one modality must be enabled (expression, CDS, or PPI).")
    if use_ppi and not args.ppi_csv:
        raise ValueError("--ppi-csv is required unless --no-ppi is set.")
    if use_cds_modal and args.cds_file is None and args.cds_gene_feature_csv is None:
        raise ValueError("Either --cds-file or --cds-gene-feature-csv must be provided when CDS is enabled.")
    if bool(args.exp_npy_values) ^ bool(args.exp_npy_genes):
        raise ValueError("--exp-npy-values 与 --exp-npy-genes 须同时指定或同时省略")
    if args.exp_csv and args.exp_npy_values:
        raise ValueError("请勿同时使用 --exp-csv 与 --exp-npy-values/--exp-npy-genes")
    if not args.exp_csv and not args.exp_npy_values:
        raise ValueError("须指定 --exp-csv，或同时指定 --exp-npy-values 与 --exp-npy-genes")


def _build_pipeline(
    args: Namespace,
    *,
    norm_fit_all_columns: bool = False,
    include_train_loader: bool = False,
    column_splits: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[FusionModelAblation, Optional[DataLoader], DataLoader, DataLoader, torch.device, bool]:
    """与 train_unseen_samples_no_cv.main 中数据构建一致。

    返回 model, train_loader（仅当 include_train_loader 时非 None）, val_loader, test_loader, device, use_amp_bf16。

    norm_fit_all_columns:
      False（默认）：仅在「训练列」上拟合 log2-minmax，与 GSE253718 Step1 一致。
      True：在 raw_expr 的全部列上拟合，再按 train/val/test 列切片变换。
            用于「非 mBrain | mBrain」水平拼接矩阵，否则训练列可能全部落在单侧条件上，
            与 case_study 的 norm_fit_domain=concat 不一致，会导致表达尺度失真、预测塌缩到无调控。

    column_splits:
      可选 (train_cols, val_cols, test_cols) 显式细胞列索引（用于 10-fold CV）。
      未提供时回退为 sample_split 比例的连续切分 split3。
    """
    use_expr = not args.no_expression
    use_cds_modal = not args.no_cds
    use_ppi = not args.no_ppi

    if args.exp_npy_values:
        g_arr = np.load(args.exp_npy_genes, allow_pickle=True)
        raw_expr = np.load(args.exp_npy_values, mmap_mode="r")
        gene_list = normalize_gene_symbol_series(pd.Series([str(x) for x in g_arr.tolist()])).tolist()
        if len(gene_list) != raw_expr.shape[0]:
            raise ValueError(f"基因数 {len(gene_list)} 与矩阵行数 {raw_expr.shape[0]} 不一致")
    else:
        expr_df = pd.read_csv(args.exp_csv)
        gene_col = expr_df.columns[0]
        expr_df[gene_col] = normalize_gene_symbol_series(expr_df[gene_col])
        expr_df = expr_df.drop_duplicates(subset=[gene_col], keep="first").set_index(gene_col)
        expr_df = expr_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        gene_list = expr_df.index.tolist()
        raw_expr = expr_df.values.astype(np.float32)
        del expr_df

    def _subset_gene_rows(kept_genes: List[str]) -> None:
        nonlocal gene_list, raw_expr
        g_to_i = {g: i for i, g in enumerate(gene_list)}
        idx = [g_to_i[g] for g in kept_genes]
        raw_expr = np.asarray(raw_expr[idx, :], dtype=np.float32)
        gene_list = kept_genes

    ref_indices = None
    if args.reference_gene_list_csv:
        ref = normalize_gene_symbol_series(
            pd.read_csv(args.reference_gene_list_csv, usecols=[0]).iloc[:, 0]
        ).tolist()
        ref_map = {g: i for i, g in enumerate(ref)}
        kept = [g for g in gene_list if g in ref_map]
        if not kept:
            raise ValueError("参考基因列表与表达矩阵无交集")
        _subset_gene_rows(kept)
        ref_indices = [ref_map[g] for g in kept]

    cds_gene_feat = None
    tss_gene_feat = None
    if args.cds_gene_feature_csv:
        cds_gene_feat = _load_gene_feature_csv(args.cds_gene_feature_csv)
    if args.tss_gene_feature_csv:
        tss_gene_feat = _load_gene_feature_csv(args.tss_gene_feature_csv)
    if use_cds_modal and (cds_gene_feat is not None or tss_gene_feat is not None):
        feat_genes = set(gene_list)
        if cds_gene_feat is not None:
            feat_genes &= set(cds_gene_feat.index.tolist())
        if tss_gene_feat is not None:
            feat_genes &= set(tss_gene_feat.index.tolist())
        kept = [g for g in gene_list if g in feat_genes]
        if not kept:
            raise ValueError("表达与 CDS/TSS 特征无基因交集")
        _subset_gene_rows(kept)

    num_samples = raw_expr.shape[1]

    grn_df = pd.read_csv(args.grn_csv)
    tf_col, tg_col, tp_col = resolve_grn_table_columns(
        grn_df, args.grn_tf_col, args.grn_target_col, args.grn_type_col
    )
    if tf_col is None or tg_col is None or tp_col is None:
        raise ValueError(f"GRN columns not found. got columns={list(grn_df.columns)}")
    grn_df[tf_col] = normalize_gene_symbol_series(grn_df[tf_col])
    grn_df[tg_col] = normalize_gene_symbol_series(grn_df[tg_col])
    if args.grn_tf_symbols_file or args.grn_exclude_source_symbols_file:
        include_tf = (
            load_regulator_symbol_set(args.grn_tf_symbols_file)
            if args.grn_tf_symbols_file
            else None
        )
        exclude_src = (
            load_regulator_symbol_set(args.grn_exclude_source_symbols_file)
            if args.grn_exclude_source_symbols_file
            else None
        )
        grn_df, _st = filter_grn_by_source_symbols(
            grn_df, tf_col, include_tf=include_tf, exclude_sources=exclude_src
        )
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
        raise ValueError("GRN 在表达基因交集上未得到任何正样本边")
    labels = np.array([lab for _, _, lab in all_pairs], dtype=np.int64)
    idx = np.arange(len(all_pairs))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=args.seed, stratify=labels)
    y_tr = labels[tr_idx]
    tr_idx2, va_idx = train_test_split(tr_idx, test_size=0.2, random_state=args.seed, stratify=y_tr)

    pairs_train = [all_pairs[i] for i in tr_idx2]
    pairs_val = [all_pairs[i] for i in va_idx]
    pairs_test = [all_pairs[i] for i in te_idx]

    ppi_emb_dim = 160
    ppi_extractor = None
    if use_ppi:
        from grn.features.ppi_features import PPIFeatureExtractor

        ppi_extractor = PPIFeatureExtractor(args.ppi_csv, gene_list=gene_list, emb_dim=ppi_emb_dim, device=None)

    if not use_cds_modal:
        seq_features = np.zeros((len(gene_list), 20), dtype=np.float32)
    elif cds_gene_feat is not None or tss_gene_feat is not None:
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

        seq_features = np.concatenate(feat_blocks, axis=1)
    else:
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
            seq_features = np.concatenate([cds_features, tss_features], axis=1)
        else:
            seq_features = cds_features

    w1, w2, w3 = [int(x.strip()) for x in args.sample_split.split(",")]
    if column_splits is not None:
        train_cols = np.asarray(column_splits[0], dtype=np.int64)
        val_cols = np.asarray(column_splits[1], dtype=np.int64)
        test_cols = np.asarray(column_splits[2], dtype=np.int64)
        print(
            f"[rebuild-pipeline] explicit column splits train/val/test = "
            f"{len(train_cols)}/{len(val_cols)}/{len(test_cols)}",
            flush=True,
        )
    else:
        c_tr, c_va, c_te = split3(num_samples, w1, w2, w3)
        train_cols = np.arange(0, c_tr, dtype=np.int64)
        val_cols = np.arange(c_tr, c_tr + c_va, dtype=np.int64)
        test_cols = np.arange(c_tr + c_va, num_samples, dtype=np.int64)

    def _fit_col_chunk(n_samples: int):
        # 极宽矩阵（如 GSE131907 non-mBrain）下避免一次性向量化 fit 占满内存
        return 512 if int(n_samples) > 4096 else None

    def _fit_progress(msg: str) -> None:
        print(f"[rebuild-pipeline] {msg}", flush=True)

    if norm_fit_all_columns:
        fc = _fit_col_chunk(raw_expr.shape[1])
        x_min, x_max = log2_minmax_per_gene_fit(
            raw_expr,
            offset=1.0,
            col_chunk=fc,
            progress_callback=_fit_progress if fc else None,
        )
    else:
        fc = _fit_col_chunk(int(len(train_cols)))
        x_min, x_max = log2_minmax_per_gene_fit(
            raw_expr[:, train_cols],
            offset=1.0,
            col_chunk=fc,
            progress_callback=_fit_progress if fc else None,
        )

    tc_tr = _fit_col_chunk(int(len(train_cols)))
    tc_va = _fit_col_chunk(int(len(val_cols)))
    tc_te = _fit_col_chunk(int(len(test_cols)))
    exp_tr = log2_minmax_per_gene_transform(
        raw_expr[:, train_cols], x_min, x_max, offset=1.0, col_chunk=tc_tr
    )
    exp_va = log2_minmax_per_gene_transform(
        raw_expr[:, val_cols], x_min, x_max, offset=1.0, col_chunk=tc_va
    )
    exp_te = log2_minmax_per_gene_transform(
        raw_expr[:, test_cols], x_min, x_max, offset=1.0, col_chunk=tc_te
    )

    # Match train_unseen_samples_no_cv: optionally reduce wide expression matrices.
    max_expr_cols = int(getattr(args, "max_expression_cols", 0) or 0)
    if max_expr_cols > 0:

        def _reduce_cols(mat: torch.Tensor, *, split_name: str) -> torch.Tensor:
            t = int(mat.shape[1])
            if t <= max_expr_cols:
                return mat
            mode = str(getattr(args, "expression_col_reduce", "subsample") or "subsample")
            if mode == "avg_pool":
                packed = mat.unsqueeze(0)
                pooled = torch.nn.functional.adaptive_avg_pool1d(packed, max_expr_cols).squeeze(0)
                print(
                    f"[rebuild-pipeline] {split_name}: avg_pool expression cols {t} -> {max_expr_cols}",
                    flush=True,
                )
                return pooled.contiguous()
            seed = int(getattr(args, "seed", 42))
            rng = np.random.default_rng(
                seed + (0 if split_name == "train" else (1 if split_name == "val" else 2))
            )
            pick = np.sort(rng.choice(t, size=max_expr_cols, replace=False))
            print(
                f"[rebuild-pipeline] {split_name}: subsample expression cols {t} -> {max_expr_cols} "
                f"(seed={seed})",
                flush=True,
            )
            return mat[:, torch.as_tensor(pick, dtype=torch.long)].contiguous()

        exp_tr = _reduce_cols(exp_tr, split_name="train")
        exp_va = _reduce_cols(exp_va, split_name="val")
        exp_te = _reduce_cols(exp_te, split_name="test")

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

    train_loader: Optional[DataLoader] = None
    if include_train_loader:
        y_train = np.array([lab for _, _, lab in pairs_train], dtype=np.int64)
        counts = np.bincount(y_train, minlength=3)
        w = 1.0 / (counts + 1e-6)
        sample_w = w[y_train]
        sampler = WeightedRandomSampler(
            torch.tensor(sample_w, dtype=torch.double),
            num_samples=len(sample_w),
            replacement=True,
        )
        train_loader = DataLoader(ds_train, batch_size=int(args.batch_size), sampler=sampler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _bf16_fn = getattr(torch.cuda, "is_bf16_supported", None)
    _bf16_ok = bool(_bf16_fn()) if callable(_bf16_fn) and device.type == "cuda" else False
    use_amp_bf16 = bool(args.amp_bf16 and _bf16_ok)

    seq_dim = int(seq_features.shape[1])
    model = FusionModelAblation(
        seq_dim=seq_dim,
        num_classes=3,
        use_expression=use_expr,
        use_cds=use_cds_modal,
        use_ppi=use_ppi,
        expression_gradient_checkpointing=args.expression_gradient_checkpointing,
    ).to(device)

    val_loader = DataLoader(ds_val, batch_size=int(args.batch_size), shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=int(args.batch_size), shuffle=False)

    return model, train_loader, val_loader, test_loader, device, use_amp_bf16


def regenerate_one_run(run_dir: Path, write_metrics_json: bool = True) -> None:
    run_dir = run_dir.resolve()
    cfg_path = run_dir / "run_config.json"
    ckpt = run_dir / "best_model_fold0.pth"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing run_config.json: {cfg_path}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing best_model_fold0.pth: {ckpt}")

    with cfg_path.open(encoding="utf-8") as f:
        cfg: Dict[str, Any] = json.load(f)
    args = Namespace(**cfg)
    _validate_args_like_train(args)
    set_seed(int(args.seed))

    model, _, val_loader, test_loader, device, use_amp_bf16 = _build_pipeline(args)
    plot_dir = str(run_dir / "plots")
    os.makedirs(plot_dir, exist_ok=True)

    state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state)
    model.eval()

    val_metrics = _evaluate(model, val_loader, device, use_amp_bf16=use_amp_bf16)
    test_metrics = _evaluate(model, test_loader, device, use_amp_bf16=use_amp_bf16)
    y_test_true, y_test_prob = _predict_arrays(model, test_loader, device, use_amp_bf16=use_amp_bf16)

    if write_metrics_json:
        with open(run_dir / "unseen_samples_val_metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in val_metrics.items()},
                f,
                ensure_ascii=False,
                indent=2,
            )
        with open(run_dir / "unseen_samples_test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in test_metrics.items()},
                f,
                ensure_ascii=False,
                indent=2,
            )

    save_accuracy_bar_val_test(
        plot_dir, float(val_metrics["accuracy"]), float(test_metrics["accuracy"])
    )
    save_grouped_metrics_val_test(plot_dir, val_metrics, test_metrics)
    cm_v = np.asarray(val_metrics["confusion_matrix"])
    cm_t = np.asarray(test_metrics["confusion_matrix"])
    save_confusion_matrix_fig(plot_dir, cm_v, "Validation confusion matrix", "confusion_matrix_val.png")
    save_confusion_matrix_fig(plot_dir, cm_t, "Test confusion matrix", "confusion_matrix_test.png")

    disp_names = ["No regulation", "Activation", "Repression"]
    disp_idx = [0, 2, 1]
    acc_ovr, prec_pc, rec_pc = per_class_ovr_accuracy_precision_recall_from_cm(cm_t)
    save_per_class_metric_bars(
        plot_dir,
        disp_names,
        [float(acc_ovr[i]) for i in disp_idx],
        [float(prec_pc[i]) for i in disp_idx],
        [float(rec_pc[i]) for i in disp_idx],
    )
    save_per_class_auprc_bar(
        plot_dir,
        disp_names,
        [float(x) for x in test_metrics["auprc"]],
    )
    save_ovr_roc_pr_curves(
        plot_dir,
        y_test_true,
        y_test_prob,
        disp_names,
        disp_idx,
        legend_roc_ovr=[float(x) for x in test_metrics["auc_ovr"]],
        legend_auprc=[float(x) for x in test_metrics["auprc"]],
    )
    print(f"[regen-unseen-plots] OK plots -> {plot_dir}")
    print(
        "[regen-unseen-plots] skip learning_curve.png (no epoch history on disk). "
        "Re-train to regenerate it."
    )


def _discover_ablation_runs(parent: Path) -> List[Path]:
    out: List[Path] = []
    for name in (
        "01_expr_only",
        "02_expr_cds",
        "03_expr_ppi",
        "04_expr_cds_ppi",
        "05_cds_only",
        "06_ppi_only",
        "07_cds_ppi",
    ):
        p = parent / name
        if (p / "best_model_fold0.pth").is_file():
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, action="append", default=[], metavar="DIR")
    ap.add_argument(
        "--parent-dir",
        type=Path,
        default=None,
        help="GSE253718 ablation 父目录（将处理 01..07 子目录）。",
    )
    ap.add_argument(
        "--no-write-metrics-json",
        action="store_true",
        help="不写回 unseen_samples_*_metrics.json（默认会写，与当前权重一致）。",
    )
    args = ap.parse_args()

    todo: List[Path] = [p.resolve() for p in args.run_dir]
    if args.parent_dir is not None:
        parent = args.parent_dir.resolve()
        if not parent.is_dir():
            raise SystemExit(f"--parent-dir 不是目录: {parent}")
        todo.extend(_discover_ablation_runs(parent))

    if not todo:
        raise SystemExit("请指定 --run-dir 和/或 --parent-dir。")

    seen = set()
    write_m = not args.no_write_metrics_json
    for rd in todo:
        if rd in seen:
            continue
        seen.add(rd)
        print(f"[regen-unseen-plots] --- {rd}")
        regenerate_one_run(rd, write_metrics_json=write_m)


if __name__ == "__main__":
    main()
