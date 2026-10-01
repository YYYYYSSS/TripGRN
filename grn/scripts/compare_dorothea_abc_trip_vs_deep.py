#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare TripGRN vs DeepFGRN on DoRothEA A-B-C pan-cancer 10CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from grn.data.pancancer_paths import PANCANCER_GSE_IDS, grn_root_from_here

EXP_NAME = "step01_pancancer_epithelial_10cv_dorothea_abc"
METRICS = ["accuracy", "macro_auc", "mcc", "macro_auprc", "macro_f1"]


def _fold_metrics_from_dirs(exp_root: Path, method: str) -> pd.DataFrame:
    mdir = exp_root / method
    rows: List[Dict[str, Any]] = []
    for gse in PANCANCER_GSE_IDS:
        for fold_idx in range(1, 11):
            p = mdir / gse / f"fold_{fold_idx:02d}" / "unseen_samples_test_metrics.json"
            if not p.is_file():
                continue
            m = json.loads(p.read_text(encoding="utf-8"))
            row = {"gse_id": gse, "fold": fold_idx}
            for k in METRICS:
                row[k] = m.get(k)
            rows.append(row)
    return pd.DataFrame(rows)


def _load_method(exp_root: Path, method: str) -> Optional[pd.DataFrame]:
    key = "tripgrn" if method == "TripGRN" else "deepfgrn"
    summary = exp_root / "summary" / f"{key}_10cv_fold_metrics.csv"
    if summary.is_file():
        return pd.read_csv(summary)
    df = _fold_metrics_from_dirs(exp_root, method)
    return df if len(df) else None


def _stats(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {"n_folds": float(len(df))}
    for k in METRICS:
        if k not in df.columns:
            out[f"{k}_mean"] = float("nan")
            out[f"{k}_std"] = float("nan")
            continue
        vals = pd.to_numeric(df[k], errors="coerce")
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=0))
    if "gse_id" in df.columns:
        present = [k for k in METRICS if k in df.columns]
        g = df.groupby("gse_id")[present].mean(numeric_only=True)
        for k in present:
            out[f"{k}_gse_mean"] = float(g[k].mean())
            out[f"{k}_gse_std"] = float(g[k].std(ddof=0))
            out[f"{k}_trip_wins_gse"] = float("nan")  # filled later for deltas
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Trip vs Deep on DoRothEA ABC 10CV")
    ap.add_argument("--exp-name", default=EXP_NAME)
    args = ap.parse_args()

    root = grn_root_from_here()
    exp_root = root / "experiments" / args.exp_name
    out_dir = exp_root / "compare_trip_vs_deep"
    out_dir.mkdir(parents=True, exist_ok=True)

    tdf = _load_method(exp_root, "TripGRN")
    ddf = _load_method(exp_root, "DeepFGRN")
    n_t = 0 if tdf is None else len(tdf)
    n_d = 0 if ddf is None else len(ddf)

    report: Dict[str, Any] = {
        "experiments_name": args.exp_name,
        "gold_standard": "DoRothEA human A+B+C",
        "trip_folds_done": n_t,
        "deep_folds_done": n_d,
        "complete": bool(n_t >= 100 and n_d >= 100),
    }

    lines = [
        "# DoRothEA A-B-C: TripGRN vs DeepFGRN",
        "",
        f"- Experiments: `{args.exp_name}`",
        f"- Gold standard: DoRothEA human confidence **A+B+C** (~13.2k signed edges)",
        f"- Progress: TripGRN {n_t}/100 folds, DeepFGRN {n_d}/100 folds",
        "",
    ]

    if tdf is None or ddf is None or n_t == 0 or n_d == 0:
        lines.append("Results incomplete; re-run after training finishes.")
        (out_dir / "COMPARE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (out_dir / "compare_stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    ts = _stats(tdf)
    ds = _stats(ddf)
    report["trip"] = ts
    report["deep"] = ds
    deltas = {}
    for m in METRICS:
        if f"{m}_mean" in ts and f"{m}_mean" in ds:
            deltas[m] = float(ts[f"{m}_mean"] - ds[f"{m}_mean"])
    report["delta_trip_minus_deep"] = deltas

    # per-GSE win counts on available metrics
    gse_rows = []
    for gse in PANCANCER_GSE_IDS:
        tr = tdf[tdf["gse_id"] == gse] if "gse_id" in tdf.columns else tdf.iloc[0:0]
        de = ddf[ddf["gse_id"] == gse] if "gse_id" in ddf.columns else ddf.iloc[0:0]
        if len(tr) == 0 or len(de) == 0:
            continue
        row = {"gse_id": gse, "n_trip_folds": len(tr), "n_deep_folds": len(de)}
        for m in METRICS:
            if m not in tr.columns or m not in de.columns:
                continue
            tm = float(pd.to_numeric(tr[m], errors="coerce").mean())
            dm = float(pd.to_numeric(de[m], errors="coerce").mean())
            row[f"trip_{m}"] = tm
            row[f"deep_{m}"] = dm
            row[f"delta_{m}"] = tm - dm
            row[f"trip_wins_{m}"] = bool(tm > dm)
        gse_rows.append(row)
    gse_df = pd.DataFrame(gse_rows)
    gse_df.to_csv(out_dir / "per_gse_trip_vs_deep.csv", index=False)

    lines.append("## Pan-cancer mean (all folds)")
    lines.append("")
    lines.append("| Metric | TripGRN | DeepFGRN | Delta (Trip-Deep) |")
    lines.append("|--------|---------|----------|-------------------|")
    for m in METRICS:
        if f"{m}_mean" not in ts:
            continue
        lines.append(
            f"| {m} | {ts[f'{m}_mean']:.4f} +/- {ts[f'{m}_std']:.4f} | "
            f"{ds[f'{m}_mean']:.4f} +/- {ds[f'{m}_std']:.4f} | {deltas.get(m, float('nan')):+.4f} |"
        )
    lines.append("")
    if len(gse_df):
        lines.append("## Per-GSE wins (Trip > Deep on cohort mean)")
        lines.append("")
        for m in ("accuracy", "macro_auc", "mcc"):
            col = f"trip_wins_{m}"
            if col in gse_df.columns:
                wins = int(gse_df[col].sum())
                lines.append(f"- {m}: Trip wins **{wins}/{len(gse_df)}** cohorts")
        lines.append("")
        # verdict
        acc_d = deltas.get("accuracy", float("nan"))
        auc_d = deltas.get("macro_auc", float("nan"))
        mcc_d = deltas.get("mcc", float("nan"))
        lines.append("## Verdict")
        lines.append("")
        lines.append(
            f"- Acc delta={acc_d:+.4f}, Macro-AUC delta={auc_d:+.4f}, MCC delta={mcc_d:+.4f} "
            "(positive => TripGRN higher)."
        )

    (out_dir / "COMPARE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "compare_stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))
    print(f"[compare] wrote {out_dir}")


if __name__ == "__main__":
    main()
