"""
Summarize GSE131907 step05 application finetune accuracy (mBrain vs non-mBrain) for R plots.

Writes:
  <case-dir>/gse131907_step05_domain_accuracy_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _load_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _float(d: Dict[str, Any], key: str) -> float:
    v = d.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _row_from_json(
    run_id: str,
    label: str,
    test_mj: Dict[str, Any],
    val_mj: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        "run_id": run_id,
        "domain_label": label,
        "val_accuracy": _float(val_mj, "accuracy") if val_mj else float("nan"),
        "test_accuracy": _float(test_mj, "accuracy"),
        "test_weighted_f1": _float(test_mj, "f1"),
        "test_macro_auc": _float(test_mj, "macro_auc"),
        "test_mcc": _float(test_mj, "mcc"),
        "test_f1_macro": _float(test_mj, "f1_macro"),
        "test_pc_noreg_accuracy": _float(test_mj, "无调控_accuracy"),
        "test_pc_activation_accuracy": _float(test_mj, "激活_accuracy"),
        "test_pc_repression_accuracy": _float(test_mj, "抑制_accuracy"),
    }
    if val_mj:
        row["val_weighted_f1"] = _float(val_mj, "f1")
        row["val_macro_auc"] = _float(val_mj, "macro_auc")
        row["val_mcc"] = _float(val_mj, "mcc")
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="GSE131907 step05 domain accuracy summary CSV.")
    p.add_argument(
        "--case-dir",
        type=Path,
        default=Path("experiments/gse131907_application_step4_case_nonmbrain_vs_mbrain"),
        help="Case-study output dir; finetune metrics read from sibling dirs under experiments/",
    )
    p.add_argument(
        "--nonbrain-finetune-dir",
        type=Path,
        default=None,
        help="default: <repo>/experiments/gse131907_application_step4_finetune_nonbrain",
    )
    p.add_argument(
        "--mbrain-finetune-dir",
        type=Path,
        default=None,
        help="default: <repo>/experiments/gse131907_application_step4_finetune_mbrain",
    )
    args = p.parse_args()
    case_dir = args.case_dir.resolve()
    exp_dir = case_dir.parent  # experiments/
    nb_dir = (
        args.nonbrain_finetune_dir.resolve()
        if args.nonbrain_finetune_dir
        else exp_dir / "gse131907_application_step4_finetune_nonbrain"
    )
    mb_dir = (
        args.mbrain_finetune_dir.resolve()
        if args.mbrain_finetune_dir
        else exp_dir / "gse131907_application_step4_finetune_mbrain"
    )

    runs = [
        ("nonbrain_app", "non-mBrain", nb_dir),
        ("mbrain_app", "mBrain", mb_dir),
    ]
    rows: List[Dict[str, Any]] = []
    for run_id, label, run_dir in runs:
        test_mj = _load_json(run_dir / "unseen_samples_test_metrics.json")
        if test_mj is None:
            print(f"[step05-summary] skip {run_dir}")
            continue
        val_mj = _load_json(run_dir / "unseen_samples_val_metrics.json")
        rows.append(_row_from_json(run_id, label, test_mj, val_mj))

    if not rows:
        raise SystemExit("No finetune metrics found for application nonbrain/mbrain.")

    out_csv = case_dir / "gse131907_step05_domain_accuracy_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[step05-summary] wrote {out_csv}")


if __name__ == "__main__":
    main()
