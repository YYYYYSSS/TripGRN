#!/usr/bin/env python3
"""Restore DoRothEA DeepFGRN fold metrics JSON from pre-corruption summary CSV."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from grn.data.pancancer_paths import PANCANCER_GSE_IDS

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "fig01_pancancer_benchmark" / "experiments" / "dorothea_abc_epithelial_10cv"
SUMMARY = EXP / "summary" / "deepfgrn_10cv_fold_metrics.csv"
DEEP = EXP / "DeepFGRN"


def row_to_metrics(r: pd.Series) -> dict:
    cm = json.loads(r["confusion_matrix"]) if isinstance(r["confusion_matrix"], str) else r["confusion_matrix"]
    return {
        "method": "DeepFGRN",
        "gse_id": str(r["gse_id"]),
        "fold_idx": int(r["fold"]),
        "accuracy": float(r["accuracy"]),
        "precision": float(r["precision"]),
        "precision_macro": float(r["precision_macro"]),
        "recall": float(r["recall"]),
        "recall_macro": float(r["recall_macro"]),
        "f1": float(r["f1"]),
        "f1_macro": float(r["f1_macro"]),
        "mcc": float(r["mcc"]),
        "macro_auc": float(r["macro_auc"]),
        "auc_ovo": float(r["auc_ovo"]),
        "auprc_macro": float(r["auprc_macro"]),
        "auc_ovr": [
            float(r["auc_ovr_无调控"]),
            float(r["auc_ovr_激活"]),
            float(r["auc_ovr_抑制"]),
        ],
        "auprc": [
            float(r["auprc_无调控"]),
            float(r["auprc_激活"]),
            float(r["auprc_抑制"]),
        ],
        "specificity_ovr": [
            float(r["specificity_无调控"]),
            float(r["specificity_激活"]),
            float(r["specificity_抑制"]),
        ],
        "无调控_accuracy": float(r["acc_class_无调控"]),
        "激活_accuracy": float(r["acc_class_激活"]),
        "抑制_accuracy": float(r["acc_class_抑制"]),
        "confusion_matrix": cm,
        "per_class_order": "no_regulation_activation_repression",
    }


def main() -> None:
    df = pd.read_csv(SUMMARY)
    n = 0
    for _, r in df.iterrows():
        gse = str(r["gse_id"])
        fold = int(r["fold"])
        if gse not in PANCANCER_GSE_IDS:
            continue
        out = DEEP / gse / f"fold_{fold:02d}" / "unseen_samples_test_metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = row_to_metrics(r)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        # Also restore CM artifact if present
        cm_path = out.parent / "confusion_matrix_test.json"
        cm_path.write_text(
            json.dumps(
                {
                    "confusion_matrix": payload["confusion_matrix"],
                    "class_names": ["No regulation", "Repression", "Activation"],
                    "method": "DeepFGRN",
                    "fold": fold,
                    "gse_id": gse,
                    "restored_from": "deepfgrn_10cv_fold_metrics.csv",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        n += 1
    print(f"[restore] wrote {n} fold metrics under {DEEP}")


if __name__ == "__main__":
    main()
