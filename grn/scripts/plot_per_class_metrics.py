from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
import torch
from torch.utils.data import DataLoader

from grn.models.fusion_model import FusionModelAblation
from grn.scripts.train_unseen_samples_no_cv import (
    PairSubsetColumnDataset,
    _build_pair_samples,
    log2_minmax_per_gene_fit,
    log2_minmax_per_gene_transform,
    split3,
)
from grn.data.preprocessing import get_GRN, normalize_gene_symbol_series, resolve_grn_table_columns
from grn.evaluation.training_plots import per_class_ovr_accuracy_precision_recall_from_cm


def _pyplot():
    # Keep consistent with grn.evaluation.training_plots font behavior
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "DengXian",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _as_numpy_cm(cm_like) -> np.ndarray:
    cm = np.asarray(cm_like, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"confusion_matrix must be square 2D array, got shape={cm.shape}")
    return cm


def _per_class_from_cm(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ovr_accuracy, precision, recall) per class in the *cm index order*."""
    return per_class_ovr_accuracy_precision_recall_from_cm(cm)


def _load_run_config(run_dir: str) -> Dict:
    cfg_path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"run_config.json not found in run_dir: {run_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_gene_feature_csv(path: str):
    import pandas as pd

    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"feature csv needs >=2 columns (gene + features): {path}")
    gene_col = df.columns[0]
    df[gene_col] = normalize_gene_symbol_series(df[gene_col])
    df = df.drop_duplicates(subset=[gene_col], keep="first").set_index(gene_col)
    feat = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return feat


@torch.no_grad()
def _predict_probs(model, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    for batch in loader:
        x = {
            "exp_tf": batch["tf"].to(device),
            "exp_target": batch["target"].to(device),
            "seq_tf": batch["seq_tf"].to(device),
            "seq_target": batch["seq_target"].to(device),
            "ppi_tf": batch["ppi_tf"].to(device),
            "ppi_target": batch["ppi_target"].to(device),
        }
        y = batch["label"].cpu().numpy()
        logits = model(
            x["exp_tf"],
            x["exp_target"],
            x["seq_tf"],
            x["seq_target"],
            x["ppi_tf"],
            x["ppi_target"],
        )
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(prob)
        all_labels.append(y)
    return np.concatenate(all_labels, axis=0), np.vstack(all_probs)


def export_test_scores_from_run_dir(
    run_dir: str,
    out_dir: str,
    device: str = "auto",
) -> Tuple[str, str]:
    """
    Re-run inference on the test split and export y_true/y_prob.
    This does NOT retrain; it uses best_model_fold0.pth.
    """
    cfg = _load_run_config(run_dir)
    model_path = os.path.join(run_dir, "best_model_fold0.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"best_model_fold0.pth not found in run_dir: {run_dir}")

    import pandas as pd

    exp_csv = cfg["exp_csv"]
    grn_csv = cfg["grn_csv"]
    cds_gene_feature_csv = cfg.get("cds_gene_feature_csv")
    tss_gene_feature_csv = cfg.get("tss_gene_feature_csv")
    ppi_csv = cfg.get("ppi_csv")
    seed = int(cfg.get("seed", 42))
    batch_size = int(cfg.get("batch_size", 32))
    neg_ratio = float(cfg.get("neg_sample_ratio", 1.0))
    sample_split = str(cfg.get("sample_split", "5,2,3"))
    use_ppi = not bool(cfg.get("no_ppi", False))
    use_cds = not bool(cfg.get("no_cds", False))
    use_expr = not bool(cfg.get("no_expression", False))

    if use_cds and not (cds_gene_feature_csv or tss_gene_feature_csv):
        raise ValueError("CDS enabled but cds_gene_feature_csv/tss_gene_feature_csv missing in run_config.json")
    if use_ppi and not ppi_csv:
        raise ValueError("PPI enabled but ppi_csv missing in run_config.json")

    expr_df = pd.read_csv(exp_csv)
    gene_col = expr_df.columns[0]
    expr_df[gene_col] = normalize_gene_symbol_series(expr_df[gene_col])
    expr_df = expr_df.drop_duplicates(subset=[gene_col], keep="first").set_index(gene_col)
    expr_df = expr_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    cds_gene_feat = _load_gene_feature_csv(cds_gene_feature_csv) if cds_gene_feature_csv else None
    tss_gene_feat = _load_gene_feature_csv(tss_gene_feature_csv) if tss_gene_feature_csv else None
    if use_cds and (cds_gene_feat is not None or tss_gene_feat is not None):
        feat_genes = set(expr_df.index.tolist())
        if cds_gene_feat is not None:
            feat_genes &= set(cds_gene_feat.index.tolist())
        if tss_gene_feat is not None:
            feat_genes &= set(tss_gene_feat.index.tolist())
        kept = [g for g in expr_df.index.tolist() if g in feat_genes]
        expr_df = expr_df.loc[kept]

    gene_list = expr_df.index.tolist()
    raw_expr = expr_df.values.astype(np.float32)
    num_samples = raw_expr.shape[1]

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
        raise ValueError("No GRN pairs available after filtering; cannot export curves.")
    labels = np.array([lab for _, _, lab in all_pairs], dtype=np.int64)
    idx = np.arange(len(all_pairs))
    # Reproduce exact splits used by training
    from sklearn.model_selection import train_test_split

    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels)
    y_tr = labels[tr_idx]
    tr_idx2, va_idx = train_test_split(tr_idx, test_size=0.2, random_state=seed, stratify=y_tr)
    pairs_test = [all_pairs[i] for i in te_idx]

    w1, w2, w3 = [int(x.strip()) for x in sample_split.split(",")]
    c_tr, c_va, c_te = split3(num_samples, w1, w2, w3)
    sl_te = slice(c_tr + c_va, num_samples)

    # Build features
    if not use_cds:
        seq_features = np.zeros((len(gene_list), 20), dtype=np.float32)
    else:
        blocks = []
        if cds_gene_feat is not None:
            blocks.append(cds_gene_feat.loc[gene_list].values.astype(np.float32))
        if tss_gene_feat is not None:
            blocks.append(tss_gene_feat.loc[gene_list].values.astype(np.float32))
        seq_features = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(gene_list), 20), dtype=np.float32)

    ppi_extractor = None
    ppi_emb_dim = 160
    if use_ppi:
        from grn.features.ppi_features import PPIFeatureExtractor

        ppi_extractor = PPIFeatureExtractor(ppi_csv, gene_list=gene_list, emb_dim=ppi_emb_dim, device=None)

    # Match training pipeline exactly: fit per-gene log2(min-max) on train columns,
    # then transform test columns before inference.
    sl_tr = slice(0, c_tr)
    x_min, x_max = log2_minmax_per_gene_fit(raw_expr[:, sl_tr], offset=1.0)
    exp_te = log2_minmax_per_gene_transform(raw_expr[:, sl_te], x_min, x_max, offset=1.0)

    ds_test = PairSubsetColumnDataset(
        exp_te,
        pairs_test,
        seq_features,
        ppi_extractor,
        device=None,
        use_ppi=use_ppi,
        use_expression=use_expr,
        use_cds=use_cds,
        ppi_feat_dim=ppi_emb_dim,
    )
    loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    model = FusionModelAblation(
        seq_dim=int(seq_features.shape[1]),
        num_classes=3,
        use_expression=use_expr,
        use_cds=use_cds,
        use_ppi=use_ppi,
    ).to(dev)
    state = torch.load(model_path, map_location=dev)
    model.load_state_dict(state)

    y_true, y_prob = _predict_probs(model, loader, dev)

    os.makedirs(out_dir, exist_ok=True)
    y_true_path = os.path.join(out_dir, "test_y_true.npy")
    y_prob_path = os.path.join(out_dir, "test_y_prob.npy")
    np.save(y_true_path, y_true.astype(np.int64))
    np.save(y_prob_path, y_prob.astype(np.float32))
    return y_true_path, y_prob_path


def plot_roc_pr_curves(
    out_dir: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names_display: Sequence[str],
    display_idx: Sequence[int],
    legend_roc_ovr: Sequence[float] | None = None,
    legend_auprc: Sequence[float] | None = None,
) -> Tuple[str, str]:
    """
    Plot OvR ROC and PR curves for each class (display order).
    """
    plt = _pyplot()
    os.makedirs(out_dir, exist_ok=True)

    roc_path = os.path.join(out_dir, "per_class_roc_curves_ovr.png")
    pr_path = os.path.join(out_dir, "per_class_pr_curves_ovr.png")

    fig1, ax1 = plt.subplots(figsize=(6.6, 5.2))
    fig2, ax2 = plt.subplots(figsize=(6.6, 5.2))

    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for k, (name, cls_idx, color) in enumerate(zip(class_names_display, display_idx, colors)):
        y_bin = (y_true == cls_idx).astype(int)
        scores = y_prob[:, cls_idx]

        fpr, tpr, _ = roc_curve(y_bin, scores)
        if legend_roc_ovr is not None and k < len(legend_roc_ovr):
            roc_auc = float(legend_roc_ovr[k])
        else:
            try:
                roc_auc = float(roc_auc_score(y_bin, scores))
            except ValueError:
                roc_auc = float("nan")
        ax1.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")

        prec, rec, _ = precision_recall_curve(y_bin, scores)
        if legend_auprc is not None and k < len(legend_auprc):
            auprc = float(legend_auprc[k])
        else:
            try:
                auprc = float(average_precision_score(y_bin, scores))
            except ValueError:
                auprc = float("nan")
        ax2.plot(rec, prec, color=color, linewidth=2, label=f"{name} (AUPRC={auprc:.3f})")

    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("One-vs-Rest ROC curves")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)
    fig1.tight_layout()
    fig1.savefig(roc_path, dpi=150)
    plt.close(fig1)

    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("One-vs-Rest PR curves")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(pr_path, dpi=150)
    plt.close(fig2)

    return roc_path, pr_path


def plot_per_class_pr_re_acc(
    out_dir: str,
    class_names_display: Sequence[str],
    accuracy_ovr: Sequence[float],
    precision: Sequence[float],
    recall: Sequence[float],
    fname: str = "per_class_accuracy_precision_recall.png",
    title: str = "Per-class Accuracy (OvR) / Precision / Recall",
) -> str:
    plt = _pyplot()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)

    x = np.arange(len(class_names_display))
    w = 0.24
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    b1 = ax.bar(x - w, list(accuracy_ovr), width=w, label="Accuracy (OvR)", color="#4C78A8")
    b2 = ax.bar(x, list(precision), width=w, label="Precision", color="#F58518")
    b3 = ax.bar(x + w, list(recall), width=w, label="Recall", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(list(class_names_display))
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.text(
        0.5,
        0.02,
        "OvR accuracy per class k: (TP_k+TN_k)/N on the k-vs-rest binary partition.",
        ha="center",
        fontsize=7,
        color="dimgray",
    )

    for bars in (b1, b2, b3):
        for bar in bars:
            v = float(bar.get_height())
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.02,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.18)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_per_class_auprc(
    out_dir: str,
    class_names_display: Sequence[str],
    auprc: Sequence[float],
    fname: str = "per_class_auprc.png",
    title: str = "Per-class AUPRC",
) -> str:
    plt = _pyplot()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)

    x = np.arange(len(class_names_display))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(x, list(auprc), color=["#4C78A8", "#F58518", "#54A24B"], width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(list(class_names_display))
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("AUPRC")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    for i, v in enumerate(auprc):
        if np.isfinite(v):
            ax.text(i, float(v) + 0.02, f"{float(v):.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-class metrics from *_metrics.json (no retraining needed).")
    parser.add_argument("--metrics-json", required=True, help="Path to unseen_samples_*_metrics.json")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="If provided, load best_model_fold0.pth in this run_dir, re-run test inference and plot ROC/PR curves.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for inference when using --run-dir. auto/cpu/cuda",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for plots. Default: <experiment_dir>/plots (sibling of metrics json).",
    )
    args = parser.parse_args()

    with open(args.metrics_json, "r", encoding="utf-8") as f:
        metrics: Dict = json.load(f)

    if "confusion_matrix" not in metrics:
        raise ValueError("metrics json missing key: confusion_matrix")
    cm = _as_numpy_cm(metrics["confusion_matrix"])

    # Internal class index order in this project is:
    # 0: No regulation, 1: Repression, 2: Activation
    # Display order requested by user:
    # No regulation, Activation, Repression  => indices [0,2,1]
    display_names = ["No regulation", "Activation", "Repression"]
    display_idx = [0, 2, 1]

    acc_i, precision_i, recall_i = _per_class_from_cm(cm)
    accuracy_ovr = acc_i[display_idx]
    precision = precision_i[display_idx]
    recall = recall_i[display_idx]

    if "auprc" not in metrics:
        raise ValueError("metrics json missing key: auprc")
    auprc_i = np.asarray(metrics["auprc"], dtype=np.float64)
    if auprc_i.ndim != 1 or auprc_i.shape[0] != cm.shape[0]:
        raise ValueError(f"auprc must have length={cm.shape[0]}, got shape={auprc_i.shape}")
    if metrics.get("per_class_order") == "no_regulation_activation_repression":
        auprc = [float(x) for x in auprc_i]
    else:
        # 旧版 JSON：auprc 为内部下标顺序 [0,1,2]=无调控、抑制、激活
        auprc = [float(auprc_i[i]) for i in display_idx]

    if args.out_dir is None:
        exp_dir = os.path.dirname(os.path.abspath(args.metrics_json))
        out_dir = os.path.join(exp_dir, "plots")
    else:
        out_dir = args.out_dir

    p1 = plot_per_class_pr_re_acc(
        out_dir,
        display_names,
        accuracy_ovr=accuracy_ovr,
        precision=precision,
        recall=recall,
        title="Per-class Accuracy (OvR) / Precision / Recall",
    )
    p2 = plot_per_class_auprc(out_dir, display_names, auprc=auprc, title="Per-class AUPRC")

    print(f"[per-class-plots] saved: {p1}")
    print(f"[per-class-plots] saved: {p2}")

    if args.run_dir:
        y_true_path, y_prob_path = export_test_scores_from_run_dir(args.run_dir, out_dir, device=args.device)
        y_true = np.load(y_true_path)
        y_prob = np.load(y_prob_path)
        # Sanity check: if metrics json has accuracy, compare with re-inference accuracy.
        if "accuracy" in metrics:
            acc_reinfer = float((np.argmax(y_prob, axis=1) == y_true).mean())
            acc_json = float(metrics["accuracy"])
            if abs(acc_reinfer - acc_json) > 0.05:
                print(
                    "[per-class-plots] warning: model/metrics mismatch detected. "
                    f"re-infer acc={acc_reinfer:.4f}, metrics-json acc={acc_json:.4f}. "
                    "Please ensure run-dir and metrics-json come from the same training run."
                )
        leg_roc = [float(x) for x in metrics["auc_ovr"]] if "auc_ovr" in metrics else None
        if "auprc" not in metrics:
            leg_ap = None
        elif metrics.get("per_class_order") == "no_regulation_activation_repression":
            leg_ap = [float(x) for x in metrics["auprc"]]
        else:
            leg_ap = [float(metrics["auprc"][i]) for i in display_idx]
        roc_path, pr_path = plot_roc_pr_curves(
            out_dir,
            y_true,
            y_prob,
            display_names,
            display_idx,
            legend_roc_ovr=leg_roc,
            legend_auprc=leg_ap,
        )
        print(f"[per-class-plots] exported: {y_true_path}")
        print(f"[per-class-plots] exported: {y_prob_path}")
        print(f"[per-class-plots] saved: {roc_path}")
        print(f"[per-class-plots] saved: {pr_path}")


if __name__ == "__main__":
    main()

