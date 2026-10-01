"""
Aggregate unseen_samples_*_metrics.json from multiple fixed run directories (ablation summary).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Tuple


def _load_metrics(run_dir: str) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None, str | None]:
    val_path = os.path.join(run_dir, "unseen_samples_val_metrics.json")
    te_path = os.path.join(run_dir, "unseen_samples_test_metrics.json")
    err = None
    if not os.path.isdir(run_dir):
        return None, None, f"not a directory: {run_dir}"
    val_m, te_m = None, None
    if os.path.isfile(val_path):
        with open(val_path, encoding="utf-8") as f:
            val_m = json.load(f)
    else:
        err = f"missing {val_path}"
    if os.path.isfile(te_path):
        with open(te_path, encoding="utf-8") as f:
            te_m = json.load(f)
    else:
        err = (err + "; " if err else "") + f"missing {te_path}"
    return val_m, te_m, err


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect unseen-samples ablation metrics into one JSON (and optional CSV).")
    parser.add_argument(
        "--run",
        action="append",
        metavar="NAME=DIR",
        required=True,
        help="Experiment key and run directory, e.g. baseline=experiments/repro_tcga_unseen_samples_no_cv",
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--csv-output", default=None, help="Optional CSV path (test metrics summary).")
    args = parser.parse_args()

    runs: List[Tuple[str, str]] = []
    for item in args.run:
        if "=" not in item:
            raise SystemExit(f"Bad --run (expect NAME=DIR): {item}")
        name, _, path = item.partition("=")
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise SystemExit(f"Bad --run: {item}")
        runs.append((name, path))

    summary: Dict[str, Any] = {"runs": {}}
    rows_for_csv: List[Dict[str, Any]] = []

    for name, run_dir in runs:
        val_m, te_m, err = _load_metrics(run_dir)
        cfg_path = os.path.join(run_dir, "run_config.json")
        run_cfg = None
        if os.path.isfile(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                run_cfg = json.load(f)
        entry: Dict[str, Any] = {
            "run_dir": os.path.abspath(run_dir),
            "error": err,
            "run_config": run_cfg,
            "val_metrics": val_m,
            "test_metrics": te_m,
        }
        summary["runs"][name] = entry

        row = {"name": name, "run_dir": os.path.abspath(run_dir), "error": err or ""}
        if te_m:
            row["test_accuracy"] = te_m.get("accuracy")
            row["test_f1_weighted"] = te_m.get("f1")
            row["test_macro_auc"] = te_m.get("macro_auc")
            row["test_auc_ovo"] = te_m.get("auc_ovo")
        rows_for_csv.append(row)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.csv_output:
        cdir = os.path.dirname(os.path.abspath(args.csv_output))
        if cdir:
            os.makedirs(cdir, exist_ok=True)
        fieldnames = ["name", "run_dir", "error", "test_accuracy", "test_f1_weighted", "test_macro_auc", "test_auc_ovo"]
        with open(args.csv_output, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows_for_csv:
                w.writerow(row)

    print(f"[collect-ablation] wrote {args.output}")
    if args.csv_output:
        print(f"[collect-ablation] wrote {args.csv_output}")


if __name__ == "__main__":
    main()
