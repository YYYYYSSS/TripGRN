# -*- coding: utf-8 -*-
"""Convert DeepFGRN compute_metrics_bundle output to step01 JSON schema."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def deepfgrn_metrics_to_step01_json(
    metrics: Dict[str, Any],
    cm: np.ndarray,
    *,
    gse_id: str = "",
    fold_idx: int = 0,
) -> Dict[str, Any]:
    cm_arr = np.asarray(cm, dtype=np.int64)
    total = int(cm_arr.sum())
    accuracy = float(cm_arr.trace() / total) if total > 0 else float("nan")

    auc_ovr = [
        float(metrics.get("AUC_OVR_无调控", np.nan)),
        float(metrics.get("AUC_OVR_激活", np.nan)),
        float(metrics.get("AUC_OVR_抑制", np.nan)),
    ]
    auprc_macro = float(metrics.get("AUPR", np.nan))

    return {
        "method": "DeepFGRN",
        "gse_id": gse_id,
        "fold_idx": int(fold_idx),
        "accuracy": accuracy,
        "precision": float(metrics.get("Precision", np.nan)),
        "precision_macro": float(metrics.get("Precision", np.nan)),
        "recall": float(metrics.get("Recall", np.nan)),
        "recall_macro": float(metrics.get("Recall", np.nan)),
        "f1": float(metrics.get("F1", np.nan)),
        "f1_macro": float(metrics.get("F1", np.nan)),
        "mcc": float(metrics.get("MCC", np.nan)),
        "macro_auc": float(metrics.get("Macro_AUC_OVR", np.nan)),
        "auc_ovo": float(metrics.get("AUC", np.nan)),
        "auc_ovr": auc_ovr,
        "auprc": [auprc_macro, auprc_macro, auprc_macro],
        "无调控_accuracy": float(metrics.get("ACC_无调控", np.nan)),
        "激活_accuracy": float(metrics.get("ACC_激活", np.nan)),
        "抑制_accuracy": float(metrics.get("ACC_抑制", np.nan)),
        "confusion_matrix": cm_arr.tolist(),
    }
