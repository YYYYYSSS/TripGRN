#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score completed TRRUST seed-scan runs: TripGRN vs DeepFGRN pan-cancer means/stds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from grn.data.pancancer_paths import PANCANCER_GSE_IDS, grn_root_from_here

HUB_REL = Path("experiments") / "step01_trrust_trip_vs_deep_seedscan"
METRICS = ["accuracy", "macro_auc", "mcc"]


def _load_manifest(hub: Path) -> dict:
    return json.loads((hub / "SEED_MANIFEST.json").read_text(encoding="utf-8"))


def _fold_csv(exp_root: Path, method: str) -> Path:
    name = "tripgrn_10cv_fold_metrics.csv" if method == "TripGRN" else "deepfgrn_10cv_fold_metrics.csv"
    return exp_root / "summary" / name


def _collect_from_fold_dirs(exp_root: Path, method: str) -> Optional[pd.DataFrame]:
    """Fallback: read per-fold metrics JSON if summary CSV missing."""
    mdir = exp_root / method
    if not mdir.is_dir():
        return None
    rows: List[Dict[str, Any]] = []
    for gse in PANCANCER_GSE_IDS:
        for fold_idx in range(1, 11):
            p = mdir / gse / f"fold_{fold_idx:02d}" / "unseen_samples_test_metrics.json"
            if not p.is_file():
                continue
            m = json.loads(p.read_text(encoding="utf-8"))
            rows.append(
                {
                    "gse_id": gse,
                    "fold": fold_idx,
                    "accuracy": m.get("accuracy"),
                    "macro_auc": m.get("macro_auc"),
                    "mcc": m.get("mcc"),
                }
            )
    if not rows:
        return None
    return pd.DataFrame(rows)


def _load_method_folds(exp_root: Path, method: str) -> Optional[pd.DataFrame]:
    csv_path = _fold_csv(exp_root, method)
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        return df
    return _collect_from_fold_dirs(exp_root, method)


def _pan_stats(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {"n_folds": float(len(df))}
    for k in METRICS:
        if k not in df.columns:
            out[f"{k}_mean"] = float("nan")
            out[f"{k}_std"] = float("nan")
            continue
        vals = pd.to_numeric(df[k], errors="coerce")
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=0))
    # GSE-level means then cross-GSE std (stability across cohorts)
    if "gse_id" in df.columns:
        g = df.groupby("gse_id")[METRICS].mean(numeric_only=True)
        for k in METRICS:
            if k in g.columns:
                out[f"{k}_gse_mean"] = float(g[k].mean())
                out[f"{k}_gse_std"] = float(g[k].std(ddof=0))
    return out


def _completion(exp_root: Path) -> Tuple[int, int]:
    n_t = n_d = 0
    for gse in PANCANCER_GSE_IDS:
        for fold_idx in range(1, 11):
            if (exp_root / "TripGRN" / gse / f"fold_{fold_idx:02d}" / "unseen_samples_test_metrics.json").is_file():
                n_t += 1
            if (exp_root / "DeepFGRN" / gse / f"fold_{fold_idx:02d}" / "unseen_samples_test_metrics.json").is_file():
                n_d += 1
    return n_t, n_d


def score_seed(grn_root: Path, seed_entry: dict) -> Dict[str, Any]:
    seed = int(seed_entry["seed"])
    exp_name = seed_entry["experiments_name"]
    exp_root = grn_root / "experiments" / exp_name
    n_t, n_d = _completion(exp_root)
    row: Dict[str, Any] = {
        "seed": seed,
        "role": seed_entry.get("role"),
        "experiments_name": exp_name,
        "trip_folds_done": n_t,
        "deep_folds_done": n_d,
        "complete": bool(n_t >= 100 and n_d >= 100),
        "status": "complete" if (n_t >= 100 and n_d >= 100) else ("partial" if (n_t + n_d) > 0 else seed_entry.get("status", "pending")),
    }
    tdf = _load_method_folds(exp_root, "TripGRN")
    ddf = _load_method_folds(exp_root, "DeepFGRN")
    if tdf is None or ddf is None or len(tdf) == 0 or len(ddf) == 0:
        return row
    ts = _pan_stats(tdf)
    ds = _pan_stats(ddf)
    for k, v in ts.items():
        row[f"trip_{k}"] = v
    for k, v in ds.items():
        row[f"deep_{k}"] = v
    for m in METRICS:
        row[f"delta_{m}_mean"] = float(ts.get(f"{m}_mean", np.nan) - ds.get(f"{m}_mean", np.nan))
    # Composite: sum of deltas (higher better for Trip) minus Trip fold-std penalty
    deltas = [row.get(f"delta_{m}_mean", 0.0) for m in METRICS]
    trip_stds = [ts.get(f"{m}_std", 0.0) for m in METRICS]
    row["score_mean_advantage"] = float(np.nansum(deltas))
    row["score_stability_penalty"] = float(np.nansum(trip_stds))
    row["score_composite"] = float(row["score_mean_advantage"] - 0.5 * row["score_stability_penalty"])
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize / rank TRRUST seed-scan")
    args = ap.parse_args()
    _ = args

    grn_root = grn_root_from_here()
    hub = grn_root / HUB_REL
    manifest = _load_manifest(hub)
    ranking_dir = hub / "ranking"
    ranking_dir.mkdir(parents=True, exist_ok=True)

    rows = [score_seed(grn_root, s) for s in manifest["seeds"]]
    df = pd.DataFrame(rows)
    if "score_composite" in df.columns:
        df = df.sort_values(
            by=["complete", "score_composite"],
            ascending=[False, False],
            na_position="last",
        )
    out_csv = ranking_dir / "seed_comparison.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # Update manifest statuses for complete seeds
    by_seed = {int(r["seed"]): r for r in rows}
    for s in manifest["seeds"]:
        r = by_seed.get(int(s["seed"]))
        if r and r.get("complete"):
            if s.get("role") == "baseline_already_run":
                s["status"] = "done_baseline"
            else:
                s["status"] = "done"
        elif r and r.get("status") == "partial":
            s["status"] = "partial"
    (hub / "SEED_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# TRRUST seed-scan ranking",
        "",
        f"Source: `{out_csv.as_posix()}`",
        "",
        "| seed | complete | ΔAcc | ΔMacroAUC | ΔMCC | TripAcc±std | composite |",
        "|------|----------|------|-----------|------|-------------|-----------|",
    ]
    for _, r in df.iterrows():
        if not bool(r.get("complete")) and pd.isna(r.get("score_composite")):
            lines.append(
                f"| {int(r['seed'])} | {r.get('trip_folds_done', 0)}/{r.get('deep_folds_done', 0)} | — | — | — | — | pending |"
            )
            continue
        lines.append(
            "| {seed} | {ok} | {da:.4f} | {du:.4f} | {dm:.4f} | {ta:.4f}±{ts:.4f} | {sc:.4f} |".format(
                seed=int(r["seed"]),
                ok="yes" if r.get("complete") else f"{int(r.get('trip_folds_done', 0))}/{int(r.get('deep_folds_done', 0))}",
                da=float(r.get("delta_accuracy_mean", np.nan)),
                du=float(r.get("delta_macro_auc_mean", np.nan)),
                dm=float(r.get("delta_mcc_mean", np.nan)),
                ta=float(r.get("trip_accuracy_mean", np.nan)),
                ts=float(r.get("trip_accuracy_std", np.nan)),
                sc=float(r.get("score_composite", np.nan)),
            )
        )
    lines.extend(
        [
            "",
            "Δ = TripGRN − DeepFGRN（泛癌全部 fold 均值）。composite ≈ ΣΔ − 0.5·Σ(Trip fold-std)。",
            "仅作内部筛选；正式写作需披露筛选过程或改用预注册种子。",
        ]
    )
    md_path = ranking_dir / "seed_ranking.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    status_path = hub / "status.json"
    prev = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    prev["score"] = {
        "n_seeds": len(rows),
        "n_complete": int(sum(1 for r in rows if r.get("complete"))),
        "comparison_csv": str(out_csv.resolve()),
        "ranking_md": str(md_path.resolve()),
    }
    status_path.write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[seedscan-score] wrote {out_csv}")
    print(f"[seedscan-score] wrote {md_path}")
    print(df[["seed", "complete", "trip_folds_done", "deep_folds_done"] + [c for c in df.columns if c.startswith("delta_") or c.startswith("score_")]].to_string(index=False))


if __name__ == "__main__":
    main()
