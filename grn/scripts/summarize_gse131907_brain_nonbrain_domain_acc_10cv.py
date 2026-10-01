#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize GSE131907 brain/nonbrain domain-accuracy 10CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

METRIC_KEYS = [
    "accuracy",
    "precision",
    "precision_macro",
    "recall",
    "recall_macro",
    "f1",
    "f1_macro",
    "mcc",
    "macro_auc",
    "auc_ovo",
]

DOMAIN_LABEL = {"mbrain": "mBrain", "nonbrain": "non-mBrain"}


def _load_metrics(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "unseen_samples_test_metrics.json"
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp-root",
        type=Path,
        default=None,
        help="Default: experiments/gse131907_brain_nonbrain_domain_acc_10cv_seed42",
    )
    args = ap.parse_args()
    grn_root = Path(__file__).resolve().parents[2]
    root = Path(args.exp_root or (grn_root / "experiments" / "gse131907_brain_nonbrain_domain_acc_10cv_seed42"))
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for dom in ("mbrain", "nonbrain"):
        for fold_idx in range(1, 11):
            run_dir = root / dom / f"fold_{fold_idx:02d}"
            if not (run_dir / "unseen_samples_test_metrics.json").is_file():
                continue
            m = _load_metrics(run_dir)
            row: Dict[str, Any] = {
                "domain_id": dom,
                "domain_label": DOMAIN_LABEL[dom],
                "fold": fold_idx,
            }
            for k in METRIC_KEYS:
                row[k] = m.get(k, np.nan)
            for i, name in enumerate(["NoReg", "Activation", "Repression"]):
                row[f"auc_ovr_{name}"] = (m.get("auc_ovr") or [np.nan] * 3)[i] if m.get("auc_ovr") else np.nan
                row[f"acc_class_{name}"] = m.get(f"{name}_accuracy", np.nan)
            rows.append(row)

    if not rows:
        raise SystemExit(f"No fold metrics under {root}")

    fold_df = pd.DataFrame(rows)
    fold_csv = summary_dir / "domain_acc_10cv_fold_metrics.csv"
    fold_df.to_csv(fold_csv, index=False)

    agg_rows = []
    for dom, g in fold_df.groupby("domain_id"):
        rec: Dict[str, Any] = {
            "domain_id": dom,
            "domain_label": DOMAIN_LABEL[str(dom)],
            "n_folds": int(len(g)),
        }
        for k in METRIC_KEYS:
            rec[f"{k}_mean"] = float(g[k].mean())
            rec[f"{k}_std"] = float(g[k].std(ddof=1)) if len(g) > 1 else 0.0
        agg_rows.append(rec)
    mean_df = pd.DataFrame(agg_rows)
    mean_csv = summary_dir / "domain_acc_10cv_domain_mean_std.csv"
    mean_df.to_csv(mean_csv, index=False)

    print(mean_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"[OK] wrote {fold_csv}")
    print(f"[OK] wrote {mean_csv}")


if __name__ == "__main__":
    main()
