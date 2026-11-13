# -*- coding: utf-8 -*-
"""Convert DeepFGRN compute_metrics_bundle (+ probs) to step01 JSON schema."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


# DeepFGRN label ids: 0=NoReg, 1=Repression, 2=Activation
# Display / TripGRN JSON order: NoReg, Activation, Repression
_DISP_FROM_NATIVE = (0, 2, 1)
_DISP_NAMES = ("无调控", "激活", "抑制")


def deepfgrn_metrics_to_step01_json(
    metrics: Dict[str, Any],
    cm: np.ndarray,
    *,
    gse_id: str = "",
    fold_idx: int = 0,
    y_true: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    cm_arr = np.asarray(cm, dtype=np.int64)
    total = int(cm_arr.sum())
    accuracy = float(cm_arr.trace() / total) if total > 0 else float("nan")

    auc_ovr = [
        float(metrics.get("AUC_OVR_无调控", np.nan)),
        float(metrics.get("AUC_OVR_激活", np.nan)),
        float(metrics.get("AUC_OVR_抑制", np.nan)),
    ]

    # Prefer true per-class AUPRC from probabilities when available.
    auprc_disp = [float("nan"), float("nan"), float("nan")]
    auprc_macro = float(metrics.get("AUPR", np.nan))
    precision_macro = float(metrics.get("Precision", np.nan))
    recall_macro = float(metrics.get("Recall", np.nan))
    f1_macro = float(metrics.get("F1", np.nan))
    specificity_ovr = [float("nan"), float("nan"), float("nan")]

    yt = None
    if y_true is not None:
        yt = np.asarray(y_true)
        if yt.ndim > 1:
            yt = np.argmax(yt, axis=1)
        yt = yt.astype(np.int64)
    if y_prob is not None and yt is not None:
        yp = np.asarray(y_prob, dtype=np.float64)
        per = []
        for native_cls in range(3):
            try:
                per.append(
                    float(
                        average_precision_score(
                            (yt == native_cls).astype(int), yp[:, native_cls]
                        )
                    )
                )
            except Exception:
                per.append(float("nan"))
        auprc_disp = [per[i] for i in _DISP_FROM_NATIVE]
        auprc_macro = float(np.nanmean(np.asarray(auprc_disp, dtype=float)))
        yhat = np.argmax(yp, axis=1)
        precision_macro = float(
            precision_score(yt, yhat, average="macro", zero_division=0)
        )
        recall_macro = float(recall_score(yt, yhat, average="macro", zero_division=0))
        f1_macro = float(f1_score(yt, yhat, average="macro", zero_division=0))

    # Specificity OvR from CM in native label order, then display reorder.
    spec_native = []
    for cls in range(3):
        tp = float(cm_arr[cls, cls])
        fn = float(cm_arr[cls, :].sum() - tp)
        fp = float(cm_arr[:, cls].sum() - tp)
        tn = float(cm_arr.sum() - tp - fn - fp)
        denom = tn + fp
        spec_native.append(float(tn / denom) if denom > 0 else float("nan"))
    specificity_ovr = [spec_native[i] for i in _DISP_FROM_NATIVE]

    out: Dict[str, Any] = {
        "method": "DeepFGRN",
        "gse_id": gse_id,
        "fold_idx": int(fold_idx),
        "accuracy": accuracy,
        "precision": float(metrics.get("Precision", np.nan)),
        "precision_macro": precision_macro,
        "recall": float(metrics.get("Recall", np.nan)),
        "recall_macro": recall_macro,
        "f1": float(metrics.get("F1", np.nan)),
        "f1_macro": f1_macro,
        "mcc": float(metrics.get("MCC", np.nan)),
        "macro_auc": float(metrics.get("Macro_AUC_OVR", np.nan)),
        "auc_ovo": float(metrics.get("AUC", np.nan)),
        "auprc_macro": auprc_macro,
        "auc_ovr": auc_ovr,
        "auprc": auprc_disp,
        "specificity_ovr": specificity_ovr,
        "无调控_accuracy": float(metrics.get("ACC_无调控", np.nan)),
        "激活_accuracy": float(metrics.get("ACC_激活", np.nan)),
        "抑制_accuracy": float(metrics.get("ACC_抑制", np.nan)),
        "confusion_matrix": cm_arr.tolist(),
        "per_class_order": "no_regulation_activation_repression",
    }
    return out
