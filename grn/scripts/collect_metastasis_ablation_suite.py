"""
Scan a metastasis ablation suite directory (parent with baseline/, no_ppi/, …)
and write one wide CSV row per subrun plus optional full JSON.

Expected per subrun:
  nonbrain_unseen_samples_val_metrics.json
  nonbrain_unseen_samples_test_metrics.json
  brain_mBrain_external_test_metrics.json
  run_config_master.json (optional)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

# Preferred row order when these names exist; others sorted alphabetically after.
_SUBRUN_ORDER = ["baseline", "no_ppi", "no_cds", "cds_only", "ppi_only"]

_VAL = "nonbrain_unseen_samples_val_metrics.json"
_TEST = "nonbrain_unseen_samples_test_metrics.json"
_BRAIN = "brain_mBrain_external_test_metrics.json"
_CFG = "run_config_master.json"

_CFG_KEYS = (
    "experiment_subdir",
    "no_ppi",
    "no_cds",
    "no_expression",
    "seed",
    "batch_size",
    "epochs",
    "lr",
    "patience",
    "neg_sample_ratio",
    "sample_split",
)


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _flatten_metrics(prefix: str, m: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not m:
        return {}
    out: Dict[str, Any] = {}
    scalars = ("accuracy", "recall", "precision", "f1", "mcc", "macro_auc", "auc_ovo")
    for k in scalars:
        if k in m:
            out[f"{prefix}_{k}"] = m[k]

    ov = m.get("auc_ovr")
    if isinstance(ov, list):
        for i, v in enumerate(ov):
            out[f"{prefix}_auc_ovr_class{i}"] = v

    ap = m.get("auprc")
    if isinstance(ap, list):
        for i, v in enumerate(ap):
            out[f"{prefix}_auprc_class{i}"] = v

    ovo = m.get("auc_ovo_per_class")
    if isinstance(ovo, list):
        for i, v in enumerate(ovo):
            out[f"{prefix}_auc_ovo_per_class_{i}"] = v

    cm = m.get("confusion_matrix")
    if isinstance(cm, list):
        for ri, row in enumerate(cm):
            if not isinstance(row, list):
                continue
            for ci, cell in enumerate(row):
                out[f"{prefix}_confusion_r{ri}c{ci}"] = cell

    for k, v in m.items():
        if k == "accuracy":
            continue
        if k.endswith("_accuracy") and isinstance(v, (int, float)):
            safe = k.replace(" ", "_")
            out[f"{prefix}_perclass_{safe}"] = v

    return out


def _flatten_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not cfg:
        return {}
    out: Dict[str, Any] = {}
    for k in _CFG_KEYS:
        if k in cfg:
            out[f"cfg_{k}"] = cfg[k]
    return out


def _discover_subruns(suite_dir: str) -> List[str]:
    out: List[str] = []
    for name in sorted(os.listdir(suite_dir)):
        sub = os.path.join(suite_dir, name)
        if not os.path.isdir(sub):
            continue
        if os.path.isfile(os.path.join(sub, _VAL)) and os.path.isfile(os.path.join(sub, _TEST)):
            out.append(name)
    return out


def _sort_subruns(names: List[str]) -> List[str]:
    rank = {n: i for i, n in enumerate(_SUBRUN_ORDER)}

    def key(x: str) -> Tuple[int, str]:
        return (rank.get(x, 10_000), x)

    return sorted(names, key=key)


def _build_row(suite_dir: str, subrun: str) -> Tuple[Dict[str, Any], Optional[str]]:
    sub_path = os.path.join(suite_dir, subrun)
    err: Optional[str] = None

    val_m = _load_json(os.path.join(sub_path, _VAL))
    te_m = _load_json(os.path.join(sub_path, _TEST))
    br_m = _load_json(os.path.join(sub_path, _BRAIN))
    cfg = _load_json(os.path.join(sub_path, _CFG))

    if val_m is None:
        err = (err + "; " if err else "") + f"missing {_VAL}"
    if te_m is None:
        err = (err + "; " if err else "") + f"missing {_TEST}"
    if br_m is None:
        err = (err + "; " if err else "") + f"missing {_BRAIN}"

    row: Dict[str, Any] = {
        "subrun": subrun,
        "run_dir": os.path.abspath(sub_path),
        "error": err or "",
    }
    row.update(_flatten_cfg(cfg))
    row.update(_flatten_metrics("val", val_m))
    row.update(_flatten_metrics("test", te_m))
    row.update(_flatten_metrics("brain", br_m))

    ta = te_m.get("accuracy") if te_m else None
    ba = br_m.get("accuracy") if br_m else None
    if isinstance(ta, (int, float)) and isinstance(ba, (int, float)):
        row["test_brain_acc_equal"] = abs(float(ta) - float(ba)) < 1e-12
    else:
        row["test_brain_acc_equal"] = ""

    return row, err


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect metastasis ablation subruns into one CSV (and optional JSON)."
    )
    parser.add_argument(
        "--suite-dir",
        required=True,
        help="Parent experiment dir containing baseline/, no_ppi/, …",
    )
    parser.add_argument(
        "--subruns",
        default=None,
        help="Comma-separated subfolder names to include, in order. Default: auto-discover.",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Output CSV path (UTF-8 with BOM for Excel).",
    )
    parser.add_argument("--json", default=None, help="Optional full JSON path (nested metrics + rows).")
    args = parser.parse_args()

    suite_dir = os.path.abspath(args.suite_dir)
    if not os.path.isdir(suite_dir):
        raise SystemExit(f"not a directory: {suite_dir}")

    if args.subruns:
        subruns = [s.strip() for s in args.subruns.split(",") if s.strip()]
    else:
        subruns = _sort_subruns(_discover_subruns(suite_dir))

    if not subruns:
        raise SystemExit(f"no subruns with metrics under {suite_dir}")

    rows: List[Dict[str, Any]] = []
    nested: Dict[str, Any] = {"suite_dir": suite_dir, "subruns": {}}
    for name in subruns:
        sub_path = os.path.join(suite_dir, name)
        val_m = _load_json(os.path.join(sub_path, _VAL))
        te_m = _load_json(os.path.join(sub_path, _TEST))
        br_m = _load_json(os.path.join(sub_path, _BRAIN))
        cfg = _load_json(os.path.join(sub_path, _CFG))
        row, err = _build_row(suite_dir, name)
        rows.append(row)
        nested["subruns"][name] = {
            "run_dir": row["run_dir"],
            "error": err,
            "run_config_master": cfg,
            "val_metrics": val_m,
            "test_nonbrain_metrics": te_m,
            "brain_external_metrics": br_m,
        }

    all_keys_set: set = set()
    for row in rows:
        all_keys_set.update(row.keys())

    def _col_order(k: str) -> Tuple[int, str]:
        if k == "subrun":
            return (0, k)
        if k == "run_dir":
            return (1, k)
        if k == "error":
            return (2, k)
        if k == "test_brain_acc_equal":
            return (3, k)
        if k.startswith("cfg_"):
            return (4, k)
        if k.startswith("val_"):
            return (5, k)
        if k.startswith("test_"):
            return (6, k)
        if k.startswith("brain_"):
            return (7, k)
        return (8, k)

    all_keys = sorted(all_keys_set, key=_col_order)

    csv_abspath = os.path.abspath(args.csv)
    os.makedirs(os.path.dirname(csv_abspath) or ".", exist_ok=True)
    with open(csv_abspath, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"[collect-metastasis-ablation] subruns={len(rows)} -> {csv_abspath}")

    if args.json:
        nested["csv_columns"] = all_keys
        nested["flat_rows"] = rows
        jp = os.path.abspath(args.json)
        os.makedirs(os.path.dirname(jp) or ".", exist_ok=True)
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(nested, f, ensure_ascii=False, indent=2)
        print(f"[collect-metastasis-ablation] wrote {jp}")


if __name__ == "__main__":
    main()
