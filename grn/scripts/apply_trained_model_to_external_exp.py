from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from grn.data.preprocessing import (
    get_GRN,
    log2_minmax_per_gene_fit,
    log2_minmax_per_gene_transform,
    normalize_gene_symbol_series,
    resolve_grn_table_columns,
)
from grn.evaluation.metrics import calculate_metrics
from grn.evaluation.training_plots import save_confusion_matrix_fig
from grn.models.fusion_model import FusionModelAblation
from grn.scripts.train_unseen_samples_no_cv import PairSubsetColumnDataset, _build_pair_samples


@torch.no_grad()
def _predict_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs: List[np.ndarray] = []
    all_preds: List[int] = []
    all_labels: List[int] = []
    for batch in loader:
        x = {
            "exp_tf": batch["tf"].to(device),
            "exp_target": batch["target"].to(device),
            "seq_tf": batch["seq_tf"].to(device),
            "seq_target": batch["seq_target"].to(device),
            "ppi_tf": batch["ppi_tf"].to(device),
            "ppi_target": batch["ppi_target"].to(device),
        }
        y = batch["label"].cpu().numpy().astype(np.int64)
        logits = model(
            x["exp_tf"],
            x["exp_target"],
            x["seq_tf"],
            x["seq_target"],
            x["ppi_tf"],
            x["ppi_target"],
        )
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        pred = np.argmax(prob, axis=1).astype(np.int64)
        all_probs.append(prob)
        all_preds.extend(pred.tolist())
        all_labels.extend(y.tolist())
    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = np.asarray(all_preds, dtype=np.int64)
    y_prob = np.vstack(all_probs) if all_probs else np.zeros((0, 3), dtype=np.float32)
    return y_true, y_pred, y_prob


def _load_gene_feature_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"feature csv needs >=2 columns (gene + features): {path}")
    gene_col = df.columns[0]
    df[gene_col] = normalize_gene_symbol_series(df[gene_col])
    df = df.drop_duplicates(subset=[gene_col], keep="first").set_index(gene_col)
    feat = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return feat


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step2: apply trained model (best_model_fold0.pth) to an external expression matrix and compute accuracy."
    )
    parser.add_argument(
        "--step1-run-dir",
        required=True,
        help="Run dir from step1. Must contain run_config.json and best_model_fold0.pth",
    )
    parser.add_argument(
        "--external-exp-csv",
        required=True,
        help="External expression CSV: first col gene symbol, remaining cols samples.",
    )
    parser.add_argument("--output-base-dir", default="experiments")
    parser.add_argument("--experiment-name", default="benchmark_luad_step2_external_apply")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Override seed (default: use step1 run_config seed).")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size (default: use step1 run_config).")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    args = parser.parse_args()

    # output dir
    if args.no_timestamp:
        run_dir = os.path.join(args.output_base_dir, args.experiment_name)
    else:
        run_dir = os.path.join(args.output_base_dir, args.experiment_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # load step1 config + model
    cfg_path = os.path.join(args.step1_run_dir, "run_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"run_config.json not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg: Dict = json.load(f)

    model_path = os.path.join(args.step1_run_dir, "best_model_fold0.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"best_model_fold0.pth not found: {model_path}")

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 32))
    neg_ratio = float(cfg.get("neg_sample_ratio", 1.0))

    use_ppi = not bool(cfg.get("no_ppi", False))
    use_cds = not bool(cfg.get("no_cds", False))
    use_expr = not bool(cfg.get("no_expression", False))

    grn_csv = cfg["grn_csv"]
    ppi_csv = cfg.get("ppi_csv")
    cds_gene_feature_csv = cfg.get("cds_gene_feature_csv")
    tss_gene_feature_csv = cfg.get("tss_gene_feature_csv")

    if use_ppi and not ppi_csv:
        raise ValueError("Step1 config indicates PPI enabled but ppi_csv missing.")
    if use_cds and not (cds_gene_feature_csv or tss_gene_feature_csv):
        raise ValueError("Step1 config indicates CDS enabled but cds_gene_feature_csv/tss_gene_feature_csv missing.")

    # load external expression
    ext_df = pd.read_csv(args.external_exp_csv)
    gene_col = ext_df.columns[0]
    ext_df[gene_col] = normalize_gene_symbol_series(ext_df[gene_col])
    ext_df = ext_df.drop_duplicates(subset=[gene_col], keep="first").set_index(gene_col)
    ext_df = ext_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # gene-level feature alignment (same as step1)
    cds_gene_feat = _load_gene_feature_csv(cds_gene_feature_csv) if cds_gene_feature_csv else None
    tss_gene_feat = _load_gene_feature_csv(tss_gene_feature_csv) if tss_gene_feature_csv else None
    if use_cds and (cds_gene_feat is not None or tss_gene_feat is not None):
        feat_genes = set(ext_df.index.tolist())
        if cds_gene_feat is not None:
            feat_genes &= set(cds_gene_feat.index.tolist())
        if tss_gene_feat is not None:
            feat_genes &= set(tss_gene_feat.index.tolist())
        kept = [g for g in ext_df.index.tolist() if g in feat_genes]
        ext_df = ext_df.loc[kept]

    gene_list = ext_df.index.tolist()
    raw_expr = ext_df.values.astype(np.float32)  # [genes, samples]

    # build GRN matrix on this gene list
    grn_df = pd.read_csv(grn_csv)
    tf_col, tg_col, tp_col = resolve_grn_table_columns(
        grn_df, str(cfg.get("grn_tf_col", "TF")), str(cfg.get("grn_target_col", "target")), str(cfg.get("grn_type_col", "Re_type"))
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

    all_pairs = _build_pair_samples(grn_matrix, neg_ratio, seed)
    if len(all_pairs) == 0:
        raise ValueError("No GRN pairs after filtering. Cannot evaluate on external expression.")

    # sequence features
    if not use_cds:
        seq_features = np.zeros((len(gene_list), 20), dtype=np.float32)
        seq_dim = 20
    else:
        blocks = []
        if cds_gene_feat is not None:
            blocks.append(cds_gene_feat.loc[gene_list].values.astype(np.float32))
        if tss_gene_feat is not None:
            blocks.append(tss_gene_feat.loc[gene_list].values.astype(np.float32))
        if not blocks:
            raise ValueError("CDS enabled but no gene-aligned feature blocks available.")
        seq_features = np.concatenate(blocks, axis=1)
        seq_dim = int(seq_features.shape[1])

    # PPI extractor
    ppi_extractor = None
    ppi_emb_dim = 160
    if use_ppi:
        from grn.features.ppi_features import PPIFeatureExtractor

        ppi_extractor = PPIFeatureExtractor(ppi_csv, gene_list=gene_list, emb_dim=ppi_emb_dim, device=None)

    # expression transform for external data:
    # fit per-gene min/max on the external expression itself (log2 + minmax per gene).
    x_min, x_max = log2_minmax_per_gene_fit(raw_expr, offset=1.0)
    exp_all = log2_minmax_per_gene_transform(raw_expr, x_min, x_max, offset=1.0)  # torch.Tensor [G, S]

    ds = PairSubsetColumnDataset(
        exp_all,
        all_pairs,
        seq_features,
        ppi_extractor,
        device=None,
        use_ppi=use_ppi,
        use_expression=use_expr,
        use_cds=use_cds,
        ppi_feat_dim=ppi_emb_dim,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

    model = FusionModelAblation(
        seq_dim=seq_dim,
        num_classes=3,
        use_expression=use_expr,
        use_cds=use_cds,
        use_ppi=use_ppi,
    ).to(dev)
    model.load_state_dict(torch.load(model_path, map_location=dev))

    y_true, y_pred, y_prob = _predict_probs(model, loader, dev)
    metrics = calculate_metrics(y_true, y_pred, y_prob)

    # write outputs
    with open(os.path.join(run_dir, "step2_apply_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "step2_apply_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "step1_run_dir": args.step1_run_dir,
                "external_exp_csv": args.external_exp_csv,
                "n_genes": len(gene_list),
                "n_samples": int(raw_expr.shape[1]),
                "n_pairs": int(len(all_pairs)),
                "neg_sample_ratio": float(neg_ratio),
                "seed": seed,
                "batch_size": batch_size,
                "device": str(dev),
                "label_order_internal": {0: "NoReg", 1: "Repress", 2: "Activate"},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    try:
        cm = np.asarray(metrics["confusion_matrix"])
        save_confusion_matrix_fig(plot_dir, cm, "External apply confusion matrix", "confusion_matrix_external_apply.png")
    except Exception as ex:
        print(f"[step2] warning: plot export failed: {ex}")

    print(f"[step2] external apply done. accuracy={float(metrics['accuracy']):.4f}")
    print(f"[step2] outputs -> {run_dir}")


if __name__ == "__main__":
    main()

