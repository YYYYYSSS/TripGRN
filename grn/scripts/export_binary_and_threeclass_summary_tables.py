#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export binary (5 methods) and 3-class (TripGRN/DeepFGRN) summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from grn.data.pancancer_paths import PANCANCER_GSE_IDS, grn_root_from_here
from grn.scripts.plot_step01_pancancer_method_comparison import (
    CLASS_LABELS,
    _binary_auc_auprc_from_3class_json,
    _parse_cm,
    _per_class_from_3class_json,
)

METHODS_BIN = ("TripGRN", "DeepFGRN", "DeepTGI", "scMGATGRN", "GENELink")
METHODS_3 = ("TripGRN", "DeepFGRN")
BIN_NATIVE = {
    "DeepTGI": "deeptgi_10cv_fold_metrics.csv",
    "scMGATGRN": "scmgatgrn_10cv_fold_metrics.csv",
    "GENELink": "genelink_10cv_fold_metrics.csv",
}
THREE_FILES = {
    "TripGRN": "tripgrn_10cv_fold_metrics.csv",
    "DeepFGRN": "deepfgrn_10cv_fold_metrics.csv",
}
METRIC_BIN = ["accuracy", "precision", "recall", "f1", "mcc", "auc", "auprc"]
METRIC_3_OVERALL = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "mcc",
    "macro_auc",
    "auc_ovo",
    "auprc_macro",
]
METRIC_3_CLASS = ["accuracy", "precision", "recall", "auc", "auprc"]
CLASS_PREFIX = {
    "No regulation": "noreg",
    "Activation": "activation",
    "Repression": "repression",
}
# Display order used when writing auc_ovr_*/auprc_* columns in fold CSVs.
CLASS_ORDER = ("No regulation", "Activation", "Repression")


def _resolve_auc_auprc_cols(df: pd.DataFrame) -> Dict[str, Dict[str, Optional[str]]]:
    """Map English class -> {auc, auprc} columns.

    Prefer keyword match; fall back to positional order among auc_ovr_* / auprc_*
    (NoReg, Activation, Repression), which matches existing fold-metric exporters.
    """
    needles = {
        "No regulation": ("NoReg", "noreg", "no_reg"),
        "Activation": ("Activate", "activation", "Act"),
        "Repression": ("Repress", "repression", "Inh", "inh"),
    }
    # Also match common UTF-8 Chinese suffixes if present in column names.
    chinese = {
        "No regulation": ("\u65e0\u8c03\u63a7",),  # ???
        "Activation": ("\u6fc0\u6d3b",),  # ??
        "Repression": ("\u6291\u5236",),  # ??
    }
    out: Dict[str, Dict[str, Optional[str]]] = {
        lab: {"auc": None, "auprc": None} for lab in CLASS_ORDER
    }
    for lab in CLASS_ORDER:
        keys = needles[lab] + chinese[lab]
        for c in df.columns:
            cs = str(c)
            if not any(k in cs for k in keys):
                continue
            if cs.startswith("auc_ovr_") and out[lab]["auc"] is None:
                out[lab]["auc"] = cs
            if cs.startswith("auprc_") and out[lab]["auprc"] is None:
                out[lab]["auprc"] = cs
    auc_cols = [c for c in df.columns if str(c).startswith("auc_ovr_")]
    auprc_cols = [c for c in df.columns if str(c).startswith("auprc_")]
    for i, lab in enumerate(CLASS_ORDER):
        if out[lab]["auc"] is None and i < len(auc_cols):
            out[lab]["auc"] = str(auc_cols[i])
        if out[lab]["auprc"] is None and i < len(auprc_cols):
            out[lab]["auprc"] = str(auprc_cols[i])
    return out


def _cm_to_y(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt: List[int] = []
    yp: List[int] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            n = int(cm[i, j])
            yt.extend([i] * n)
            yp.extend([j] * n)
    return np.asarray(yt), np.asarray(yp)


def _binary_metrics_from_3class_cm(cm: np.ndarray) -> Dict[str, float]:
    """Collapse native CM order: class0=NoReg vs class1+=regulated."""
    yt, yp = _cm_to_y(cm)
    yt_b = (yt != 0).astype(int)
    yp_b = (yp != 0).astype(int)
    mcc = float("nan")
    if len(np.unique(yt_b)) > 1 and len(np.unique(yp_b)) > 1:
        mcc = float(matthews_corrcoef(yt_b, yp_b))
    return {
        "accuracy": float(accuracy_score(yt_b, yp_b)),
        "precision": float(precision_score(yt_b, yp_b, average="binary", zero_division=0, pos_label=1)),
        "recall": float(recall_score(yt_b, yp_b, average="binary", zero_division=0, pos_label=1)),
        "f1": float(f1_score(yt_b, yp_b, average="binary", zero_division=0, pos_label=1)),
        "mcc": mcc,
    }


def _mean_auprc_3class(r: pd.Series) -> float:
    cols = [c for c in r.index if str(c).startswith("auprc_")]
    vals = [float(r[c]) for c in cols if pd.notna(r[c])]
    return float(np.nanmean(vals)) if vals else float("nan")


def build_binary_fold(summary: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for method, fname in BIN_NATIVE.items():
        path = summary / fname
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append(
                {
                    "gse_id": str(r["gse_id"]),
                    "method": method,
                    "task": "binary",
                    "fold": int(r["fold"]),
                    "accuracy": float(r["accuracy"]),
                    "precision": float(r["precision"]),
                    "recall": float(r["recall"]),
                    "f1": float(r["f1"]),
                    "mcc": float(r["mcc"]),
                    "auc": float(r["macro_auc"]),
                    "auprc": float(r["auprc"]),
                }
            )

    for method, fname in THREE_FILES.items():
        path = summary / fname
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        colmap = _resolve_auc_auprc_cols(df)
        for _, r in df.iterrows():
            cm = _parse_cm(r["confusion_matrix"])
            bin_m = _binary_metrics_from_3class_cm(cm)
            # Strict binary ROC-AUC (regulated vs not) == NoReg OvR AUC (train-time probs).
            # Strict binary AUPRC needs P(reg) scores; filled from checkpoint cache when present.
            auc_noreg_col = colmap["No regulation"]["auc"]
            auc_bin = float(r[auc_noreg_col]) if auc_noreg_col else float("nan")
            auprc_bin = float("nan")
            rows.append(
                {
                    "gse_id": str(r["gse_id"]),
                    "method": method,
                    "task": "binary_from_3class",
                    "fold": int(r["fold"]),
                    "accuracy": bin_m["accuracy"],
                    "precision": bin_m["precision"],
                    "recall": bin_m["recall"],
                    "f1": bin_m["f1"],
                    "mcc": bin_m["mcc"],
                    "auc": auc_bin,
                    "auprc": auprc_bin,
                }
            )

    out = pd.DataFrame(rows)
    # Fill Trip/Deep AUPRC (and optionally overwrite AUC) from strict-binary cache.
    cache = summary / "trip_deep_strict_binary_ranking_cache.csv"
    if cache.is_file() and not out.empty:
        c = pd.read_csv(cache)
        c = c[c["status"].astype(str) == "ok"].copy()
        if not c.empty:
            c["fold"] = c["fold"].astype(int)
            key = ["method", "gse_id", "fold"]
            out = out.merge(
                c[key + ["auc", "auprc"]].rename(columns={"auc": "auc_cache", "auprc": "auprc_cache"}),
                on=key,
                how="left",
            )
            # Prefer cache AUPRC always; keep stored NoReg OvR as AUC unless cache AUC present.
            m = out["auprc_cache"].notna()
            out.loc[m, "auprc"] = out.loc[m, "auprc_cache"]
            m2 = out["auc_cache"].notna()
            out.loc[m2, "auc"] = out.loc[m2, "auc_cache"]
            out = out.drop(columns=["auc_cache", "auprc_cache"])

    out["method"] = pd.Categorical(out["method"], categories=list(METHODS_BIN), ordered=True)
    gse_order = [g for g in PANCANCER_GSE_IDS if g in set(out["gse_id"])]
    out["gse_id"] = pd.Categorical(out["gse_id"], categories=gse_order, ordered=True)
    return out.sort_values(["gse_id", "method", "fold"]).reset_index(drop=True)


def build_threeclass_fold(summary: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for method, fname in THREE_FILES.items():
        path = summary / fname
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        colmap = _resolve_auc_auprc_cols(df)
        for _, r in df.iterrows():
            cm = _parse_cm(r["confusion_matrix"])
            j = {
                "confusion_matrix": cm.tolist(),
                "auc_ovr": [
                    float(r[colmap[lab]["auc"]]) if colmap[lab]["auc"] else np.nan
                    for lab in CLASS_ORDER
                ],
                "auprc": [
                    float(r[colmap[lab]["auprc"]]) if colmap[lab]["auprc"] else np.nan
                    for lab in CLASS_ORDER
                ],
            }
            per = _per_class_from_3class_json(j, method)
            base: Dict[str, Any] = {
                "gse_id": str(r["gse_id"]),
                "method": method,
                "task": "3-class",
                "fold": int(r["fold"]),
                "accuracy": float(r["accuracy"]),
                "precision_macro": float(r["precision_macro"]),
                "recall_macro": float(r["recall_macro"]),
                "f1_macro": float(r["f1_macro"]),
                "mcc": float(r["mcc"]),
                "macro_auc": float(r["macro_auc"]),
                "auc_ovo": float(r["auc_ovo"]),
                "auprc_macro": _mean_auprc_3class(r),
            }
            for cls in CLASS_LABELS:
                prefix = CLASS_PREFIX[cls]
                base[f"{prefix}_accuracy"] = per[cls]["accuracy"]
                base[f"{prefix}_precision"] = per[cls]["precision"]
                base[f"{prefix}_recall"] = per[cls]["recall"]
                base[f"{prefix}_auc"] = per[cls]["auc"]
                base[f"{prefix}_auprc"] = per[cls]["auprc"]
            rows.append(base)

    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], categories=list(METHODS_3), ordered=True)
    gse_order = [g for g in PANCANCER_GSE_IDS if g in set(out["gse_id"])]
    out["gse_id"] = pd.Categorical(out["gse_id"], categories=gse_order, ordered=True)
    return out.sort_values(["gse_id", "method", "fold"]).reset_index(drop=True)


def gse_mean_std(fold_df: pd.DataFrame, value_cols: Sequence[str]) -> pd.DataFrame:
    group = ["gse_id", "method"]
    task_map = (
        fold_df.groupby("method", observed=True)["task"].first().to_dict()
        if "task" in fold_df.columns
        else {}
    )
    rows: List[Dict[str, Any]] = []
    for keys, sub in fold_df.groupby(group, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, Any] = {g: keys[i] for i, g in enumerate(group)}
        row["n_folds"] = int(len(sub))
        if task_map:
            row["task"] = task_map.get(row["method"], "")
        for c in value_cols:
            if c not in sub.columns:
                continue
            vals = pd.to_numeric(sub[c], errors="coerce")
            row[f"{c}_mean"] = float(vals.mean())
            row[f"{c}_std"] = float(vals.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group).reset_index(drop=True)


def pan_mean_of_gse_means(gse_df: pd.DataFrame, value_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for method, sub in gse_df.groupby("method", observed=True):
        row: Dict[str, Any] = {"method": method, "n_gse": int(sub["gse_id"].nunique())}
        if "task" in sub.columns:
            row["task"] = sub["task"].iloc[0]
        for c in value_cols:
            mc = f"{c}_mean"
            if mc not in sub.columns:
                continue
            vals = pd.to_numeric(sub[mc], errors="coerce")
            row[f"{c}_mean"] = float(vals.mean())
            row[f"{c}_std_across_gse"] = float(vals.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def export_one(summary: Path) -> None:
    summary.mkdir(parents=True, exist_ok=True)
    bin_fold = build_binary_fold(summary)
    bin_gse = gse_mean_std(bin_fold, METRIC_BIN)
    bin_pan = pan_mean_of_gse_means(bin_gse, METRIC_BIN)

    three_fold = build_threeclass_fold(summary)
    three_value_cols = METRIC_3_OVERALL + [
        f"{p}_{m}" for p in ("noreg", "activation", "repression") for m in METRIC_3_CLASS
    ]
    three_gse = gse_mean_std(three_fold, three_value_cols)
    three_pan = pan_mean_of_gse_means(three_gse, three_value_cols)

    outputs = {
        "binary_five_method_fold_metrics.csv": bin_fold,
        "binary_five_method_gse_mean_std.csv": bin_gse,
        "binary_five_method_pan_mean.csv": bin_pan,
        "threeclass_trip_deep_fold_metrics.csv": three_fold,
        "threeclass_trip_deep_gse_mean_std.csv": three_gse,
        "threeclass_trip_deep_pan_mean.csv": three_pan,
    }
    for name, df in outputs.items():
        out = summary / name
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[export] {out} shape={df.shape}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export binary / 3-class summary tables")
    ap.add_argument(
        "--experiment-name",
        action="append",
        default=None,
        help="Experiment folder under experiments/. Repeatable.",
    )
    args = ap.parse_args()
    root = grn_root_from_here()
    exps = args.experiment_name or [
        "step01_pancancer_epithelial_10cv",
        "step01_pancancer_epithelial_10cv_dorothea_abc",
    ]
    for exp in exps:
        print(f"==== {exp} ====")
        export_one(root / "experiments" / exp / "summary")


if __name__ == "__main__":
    main()
