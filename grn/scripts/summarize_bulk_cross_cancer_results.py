from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _extract_cancer_code(run_dir_name: str) -> str:
    m = re.search(r"(TCGA-[A-Z0-9]+)", run_dir_name)
    return m.group(1) if m else run_dir_name


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _per_class_pr_from_cm(cm: np.ndarray) -> Dict[str, float]:
    # confusion_matrix 行/列顺序: 0=NoReg, 1=Repress, 2=Activate
    out: Dict[str, float] = {}
    class_names = [("no_reg", 0), ("repress", 1), ("activate", 2)]
    for name, k in class_names:
        tp = float(cm[k, k])
        row_sum = float(cm[k, :].sum())
        col_sum = float(cm[:, k].sum())
        out[f"precision_{name}"] = tp / col_sum if col_sum > 0 else np.nan
        out[f"recall_{name}"] = tp / row_sum if row_sum > 0 else np.nan
    return out


def _collect_rows(bulk_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for d in sorted([p for p in bulk_dir.iterdir() if p.is_dir()]):
        test_json = d / "unseen_samples_test_metrics.json"
        val_json = d / "unseen_samples_val_metrics.json"
        if not test_json.is_file():
            continue
        t = _read_json(test_json)
        v = _read_json(val_json) if val_json.is_file() else {}
        row = {
            "run_dir": d.name,
            "cancer": _extract_cancer_code(d.name),
            "val_accuracy": float(v.get("accuracy", np.nan)),
            "test_accuracy": float(t.get("accuracy", np.nan)),
            "test_weighted_precision": float(t.get("precision", np.nan)),
            "test_weighted_recall": float(t.get("recall", np.nan)),
            "test_weighted_f1": float(t.get("f1", np.nan)),
            "test_macro_auc": float(t.get("macro_auc", np.nan)),
            "test_auc_ovo": float(t.get("auc_ovo", np.nan)),
            "test_mcc": float(t.get("mcc", np.nan)),
            "acc_no_reg": float(t.get("无调控_accuracy", np.nan)),
            "acc_repress": float(t.get("抑制_accuracy", np.nan)),
            "acc_activate": float(t.get("激活_accuracy", np.nan)),
        }
        cm = t.get("confusion_matrix", None)
        if isinstance(cm, list):
            try:
                cm_arr = np.asarray(cm, dtype=float)
                row["n_test_pairs"] = int(cm_arr.sum())
                row.update(_per_class_pr_from_cm(cm_arr))
            except Exception:
                row["n_test_pairs"] = np.nan
                row["precision_no_reg"] = np.nan
                row["precision_repress"] = np.nan
                row["precision_activate"] = np.nan
                row["recall_no_reg"] = np.nan
                row["recall_repress"] = np.nan
                row["recall_activate"] = np.nan
        else:
            row["n_test_pairs"] = np.nan
            row["precision_no_reg"] = np.nan
            row["precision_repress"] = np.nan
            row["precision_activate"] = np.nan
            row["recall_no_reg"] = np.nan
            row["recall_repress"] = np.nan
            row["recall_activate"] = np.nan
        rows.append(row)

    if not rows:
        raise SystemExit(f"No valid run dirs found under: {bulk_dir}")
    df = pd.DataFrame(rows).sort_values("cancer").reset_index(drop=True)
    return df


def _plot_bar_core(df: pd.DataFrame, out_dir: Path) -> None:
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(df)), 5))
    ax.bar(x - width / 2, df["test_accuracy"], width=width, label="Test Accuracy")
    ax.bar(x + width / 2, df["test_weighted_f1"], width=width, label="Test Weighted F1")
    ax.set_xticks(x)
    ax.set_xticklabels(df["cancer"], rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Cross-cancer generalization: Accuracy vs Weighted F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "bulk_overview_accuracy_weightedf1.png", dpi=200)
    plt.close(fig)


def _plot_bar_auc_mcc(df: pd.DataFrame, out_dir: Path) -> None:
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(df)), 5))
    ax.bar(x - width / 2, df["test_macro_auc"], width=width, label="Test Macro AUC")
    ax.bar(x + width / 2, df["test_mcc"], width=width, label="Test MCC")
    ax.set_xticks(x)
    ax.set_xticklabels(df["cancer"], rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Cross-cancer generalization: Macro AUC vs MCC")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "bulk_overview_macroauc_mcc.png", dpi=200)
    plt.close(fig)


def _plot_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    metric_cols = [
        "test_accuracy",
        "test_weighted_precision",
        "test_weighted_recall",
        "test_weighted_f1",
        "test_macro_auc",
        "test_auc_ovo",
        "test_mcc",
        "acc_no_reg",
        "acc_repress",
        "acc_activate",
    ]
    mat = df[metric_cols].to_numpy(dtype=float).T

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(df)), 6))
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["cancer"], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(metric_cols)))
    ax.set_yticklabels(metric_cols)
    ax.set_title("Cross-cancer metrics heatmap (test)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Metric value")
    fig.tight_layout()
    fig.savefig(out_dir / "bulk_overview_metrics_heatmap.png", dpi=200)
    plt.close(fig)


def _plot_perclass_grouped(df: pd.DataFrame, out_dir: Path, prefix: str, title: str, cols: List[str]) -> None:
    x = np.arange(len(df))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(df)), 5))
    ax.bar(x - width, df[cols[0]], width=width, label="No regulation")
    ax.bar(x, df[cols[1]], width=width, label="Activation")
    ax.bar(x + width, df[cols[2]], width=width, label="Repression")
    ax.set_xticks(x)
    ax.set_xticklabels(df["cancer"], rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / prefix, dpi=200)
    plt.close(fig)


def _plot_radar(df: pd.DataFrame, out_dir: Path) -> None:
    metric_cols = [
        "test_accuracy",
        "test_weighted_f1",
        "test_macro_auc",
        "test_mcc",
        "acc_no_reg",
        "acc_activate",
        "acc_repress",
    ]
    labels = ["Acc", "wF1", "MacroAUC", "MCC", "Acc_NoReg", "Acc_Act", "Acc_Rep"]
    values = df[metric_cols].to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0)

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, cancer in enumerate(df["cancer"].tolist()):
        v = values[i].tolist()
        v += v[:1]
        ax.plot(angles, v, linewidth=1.2, label=cancer)
        ax.fill(angles, v, alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Cross-cancer radar (core metrics)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "bulk_overview_radar_core_metrics.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize cross-cancer bulk results and draw overview figures.")
    ap.add_argument("--bulk-dir", default="experiments/Bulk", help="Directory containing per-cancer run dirs.")
    ap.add_argument("--out-dir", default=None, help="Output directory. Default: <bulk-dir>/summary")
    args = ap.parse_args()

    bulk_dir = Path(args.bulk_dir).resolve()
    if not bulk_dir.is_dir():
        raise SystemExit(f"bulk dir not found: {bulk_dir}")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (bulk_dir / "summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _collect_rows(bulk_dir)
    df.to_csv(out_dir / "bulk_cross_cancer_metrics_summary.csv", index=False, encoding="utf-8-sig")

    rank_cols = ["cancer", "test_accuracy", "test_weighted_f1", "test_macro_auc", "test_mcc", "n_test_pairs"]
    df.sort_values(["test_weighted_f1", "test_accuracy"], ascending=False)[rank_cols].to_csv(
        out_dir / "bulk_cross_cancer_rank_by_f1.csv", index=False, encoding="utf-8-sig"
    )

    _plot_bar_core(df, out_dir)
    _plot_bar_auc_mcc(df, out_dir)
    _plot_heatmap(df, out_dir)
    _plot_perclass_grouped(
        df,
        out_dir,
        "bulk_perclass_accuracy_grouped.png",
        "Per-class OvR Accuracy across cancers (test)",
        ["acc_no_reg", "acc_activate", "acc_repress"],
    )
    _plot_perclass_grouped(
        df,
        out_dir,
        "bulk_perclass_precision_grouped.png",
        "Per-class Precision across cancers (test)",
        ["precision_no_reg", "precision_activate", "precision_repress"],
    )
    _plot_perclass_grouped(
        df,
        out_dir,
        "bulk_perclass_recall_grouped.png",
        "Per-class Recall across cancers (test)",
        ["recall_no_reg", "recall_activate", "recall_repress"],
    )
    _plot_radar(df, out_dir)

    print(f"[bulk-summary] done. out_dir={out_dir}")
    print(f"[bulk-summary] runs={len(df)}")


if __name__ == "__main__":
    main()
