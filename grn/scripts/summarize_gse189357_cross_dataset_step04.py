"""
Summarize GSE189357 cross-dataset step04 runs into one CSV for R plotting.

Rows (test set, top-to-bottom order in R):
  - frozen_step1: optional frozen LUAD step1 on harmonized GSE189357
  - finetune_pre: fusion-head finetune, pre post-hoc calibration
  - finetune_cal: fusion-head finetune + post-hoc calibration (main unseen_samples_test_metrics.json)

Frozen metrics path (first hit):
  <exp-dir>/frozen_step1_zeroshot/unseen_samples_test_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from grn.scripts.summarize_gse253718_ablation import (
    _float,
    _macro_metrics,
    _per_class_pr_from_cm,
)


def _load_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _vec3(mj: Dict[str, Any], key: str) -> Tuple[float, float, float]:
    v = mj.get(key)
    if not isinstance(v, (list, tuple)) or len(v) < 3:
        return float("nan"), float("nan"), float("nan")
    try:
        return float(v[0]), float(v[1]), float(v[2])
    except (TypeError, ValueError):
        return float("nan"), float("nan"), float("nan")


def _row_from_metrics(run_id: str, label: str, mj: Dict[str, Any]) -> Dict[str, Any]:
    cm = mj.get("confusion_matrix")
    rec_raw, pre_raw = _per_class_pr_from_cm(cm)
    auc_no, auc_act, auc_rep = _vec3(mj, "auc_ovr")
    if np.isnan(auc_no):
        auc_no, auc_act, auc_rep = _vec3(mj, "auc_ovo_per_class")
    auprc_no, auprc_act, auprc_rep = _vec3(mj, "auprc")
    r_macro, p_macro, f1_macro = _macro_metrics(mj)
    return {
        "run_id": run_id,
        "scenario_label": label,
        "test_accuracy": _float(mj, "accuracy"),
        "test_weighted_f1": _float(mj, "f1"),
        "test_macro_auc": _float(mj, "macro_auc"),
        "test_mcc": _float(mj, "mcc"),
        "test_precision": _float(mj, "precision"),
        "test_recall": _float(mj, "recall"),
        "test_precision_macro": p_macro,
        "test_recall_macro": r_macro,
        "test_f1_macro": f1_macro,
        "test_pc_noreg_accuracy": _float(mj, "无调控_accuracy"),
        "test_pc_activation_accuracy": _float(mj, "激活_accuracy"),
        "test_pc_repression_accuracy": _float(mj, "抑制_accuracy"),
        "test_pc_noreg_recall": float(rec_raw[0]),
        "test_pc_activation_recall": float(rec_raw[2]),
        "test_pc_repression_recall": float(rec_raw[1]),
        "test_pc_noreg_precision": float(pre_raw[0]),
        "test_pc_activation_precision": float(pre_raw[2]),
        "test_pc_repression_precision": float(pre_raw[1]),
        "test_pc_noreg_auc_ovr": auc_no,
        "test_pc_activation_auc_ovr": auc_act,
        "test_pc_repression_auc_ovr": auc_rep,
        "test_pc_noreg_auprc": auprc_no,
        "test_pc_activation_auprc": auprc_act,
        "test_pc_repression_auprc": auprc_rep,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize GSE189357 step04 cross-dataset metrics CSV.")
    p.add_argument(
        "--exp-dir",
        type=Path,
        default=Path("experiments/benchmark_luad_gse253718_step3_cross_dataset_gse189357"),
    )
    args = p.parse_args()
    exp = args.exp_dir.resolve()
    if not exp.is_dir():
        raise SystemExit(f"Missing experiment dir: {exp}")

    specs: List[Tuple[str, str, Path]] = [
        (
            "frozen_step1",
            "Frozen Step1 (zero-shot)",
            exp / "frozen_step1_zeroshot" / "unseen_samples_test_metrics.json",
        ),
        (
            "finetune_pre",
            "Fusion finetune",
            exp / "unseen_samples_test_metrics_pre_calibration.json",
        ),
        (
            "finetune_cal",
            "Fusion finetune + calib.",
            exp / "unseen_samples_test_metrics.json",
        ),
    ]

    rows: List[Dict[str, Any]] = []
    for run_id, label, json_path in specs:
        mj = _load_json(json_path)
        if mj is None:
            print(f"[step04-summary] skip (no json): {json_path}")
            continue
        rows.append(_row_from_metrics(run_id, label, mj))

    if not rows:
        raise SystemExit(
            f"No metrics under {exp}. Run step04 bat first; optional frozen eval:\n"
            f"  see step04 bat frozen_step1_zeroshot block."
        )

    csv_path = exp / "gse189357_cross_dataset_metrics_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[step04-summary] wrote {csv_path} ({len(rows)} scenarios)")


if __name__ == "__main__":
    main()
