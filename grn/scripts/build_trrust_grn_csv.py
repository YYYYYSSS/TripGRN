#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download (optional) and convert TRRUST v2 human TF–target edges into a
KEGG-compatible GRN CSV for step01 (source_symbol / target_symbol / relation_subtype).

Rules:
  - Keep Activation -> activation, Repression -> inhibition
  - Drop Unknown mode (no typed label for 3-class)
  - Drop TF–target pairs with conflicting Activation+Repression evidence
  - Deduplicate by (TF, target); keep PMIDs in pmid column
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

TRRUST_HUMAN_URL = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        dest.write_bytes(resp.read())


def build_trrust_csv(
    raw_tsv: Path,
    out_csv: Path,
    *,
    drop_conflicts: bool = True,
) -> Dict[str, Any]:
    df = pd.read_csv(
        raw_tsv,
        sep="\t",
        header=None,
        names=["tf", "target", "mode", "pmid"],
        dtype=str,
    )
    df["tf"] = df["tf"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df["mode"] = df["mode"].astype(str).str.strip()
    df["pmid"] = df["pmid"].astype(str).str.strip()
    df = df[(df["tf"] != "") & (df["target"] != "") & (df["tf"] != "nan") & (df["target"] != "nan")]

    mode_counts = df["mode"].value_counts().to_dict()
    typed = df[df["mode"].isin(["Activation", "Repression"])].copy()

    rows: List[Dict[str, str]] = []
    n_conflict = 0
    n_act = 0
    n_inh = 0
    for (tf, tg), sub in typed.groupby(["tf", "target"], sort=True):
        modes = set(sub["mode"].tolist())
        pmids = ";".join(sorted(set(sub["pmid"].tolist())))
        if modes == {"Activation", "Repression"}:
            n_conflict += 1
            if drop_conflicts:
                continue
        if modes == {"Activation"} or ("Activation" in modes and not drop_conflicts and "Repression" not in modes):
            rel = "activation"
            n_act += 1
        elif modes == {"Repression"} or ("Repression" in modes and not drop_conflicts and "Activation" not in modes):
            rel = "inhibition"
            n_inh += 1
        elif "Activation" in modes and "Repression" not in modes:
            rel = "activation"
            n_act += 1
        elif "Repression" in modes and "Activation" not in modes:
            rel = "inhibition"
            n_inh += 1
        else:
            # conflicting and not dropped: skip (ambiguous for 3-class)
            continue
        rows.append(
            {
                "source": "",
                "source_symbol": str(tf),
                "target": "",
                "target_symbol": str(tg),
                "relation_subtype": rel,
                "pathway_id": "TRRUST_v2",
                "pathway_title": "TRRUST v2 human TF-target",
                "pmid": pmids,
            }
        )

    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    stats: Dict[str, Any] = {
        "source_url": TRRUST_HUMAN_URL,
        "raw_tsv": str(raw_tsv),
        "out_csv": str(out_csv),
        "raw_rows": int(len(df)),
        "raw_mode_counts": mode_counts,
        "unique_tf": int(out["source_symbol"].nunique()) if len(out) else 0,
        "unique_target": int(out["target_symbol"].nunique()) if len(out) else 0,
        "edges_activation": int(n_act),
        "edges_inhibition": int(n_inh),
        "edges_total": int(len(out)),
        "conflicts_dropped": int(n_conflict) if drop_conflicts else 0,
        "drop_conflicts": bool(drop_conflicts),
    }
    stats_path = out_csv.with_name(out_csv.stem + "_build_stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Build TRRUST human GRN CSV for step01")
    ap.add_argument(
        "--raw-tsv",
        type=Path,
        default=None,
        help="Local TRRUST human TSV (download if missing)",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output KEGG-compatible CSV",
    )
    ap.add_argument("--download", action="store_true", help="Force re-download raw TSV")
    ap.add_argument(
        "--keep-conflicts",
        action="store_true",
        help="Do not drop Activation+Repression pairs (still skipped as ambiguous)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    raw = Path(args.raw_tsv or (root / "data" / "network" / "grn" / "raw" / "trrust_rawdata.human.tsv"))
    out = Path(
        args.out_csv
        or (root / "data" / "network" / "grn" / "TRRUST_human_activation_inhibition_with_symbols.csv")
    )

    if args.download or not raw.is_file():
        print(f"[trrust] downloading {TRRUST_HUMAN_URL} -> {raw}")
        _download(TRRUST_HUMAN_URL, raw)
    else:
        print(f"[trrust] using existing raw: {raw}")

    stats = build_trrust_csv(raw, out, drop_conflicts=not args.keep_conflicts)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[trrust] wrote {out}")


if __name__ == "__main__":
    main()
