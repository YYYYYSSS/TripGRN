#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download DoRothEA human regulons (dorothea_hs, confidence A-E) and convert to a
KEGG/TRRUST-compatible GRN CSV for step01.

Source: saezlab/dorothea package data (dorothea_hs.rda)
  https://github.com/saezlab/dorothea/blob/master/data/dorothea_hs.rda

Rules:
  - Keep all confidence levels A-E by default
  - mor > 0 -> activation; mor < 0 -> inhibition; mor == 0 dropped
  - Deduplicate by (tf, target); if conflicting signs, drop the pair
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

DOROTHEA_HS_URL = "https://cdn.jsdelivr.net/gh/saezlab/dorothea@master/data/dorothea_hs.rda"
DOROTHEA_HS_URL_FALLBACKS = (
    DOROTHEA_HS_URL,
    "https://github.com/saezlab/dorothea/raw/master/data/dorothea_hs.rda",
    "https://raw.githubusercontent.com/saezlab/dorothea/master/data/dorothea_hs.rda",
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for u in ((url,) + tuple(x for x in DOROTHEA_HS_URL_FALLBACKS if x != url)):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()
            if len(data) < 10_000:
                raise RuntimeError(f"download too small ({len(data)} bytes) from {u}")
            dest.write_bytes(data)
            return
        except Exception as e:  # noqa: BLE001 - try next mirror
            last_err = e
            continue
    raise RuntimeError(f"failed to download DoRothEA rda: {last_err}")


def _load_rda(path: Path) -> pd.DataFrame:
    try:
        import pyreadr
    except ImportError as e:
        raise ImportError("Need pyreadr: pip install pyreadr") from e
    res = pyreadr.read_r(str(path))
    if not res:
        raise ValueError(f"empty RData: {path}")
    df = list(res.values())[0]
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return df


def build_dorothea_csv(
    raw_rda: Path,
    out_csv: Path,
    *,
    levels: Optional[List[str]] = None,
    drop_conflicts: bool = True,
) -> Dict[str, Any]:
    df = _load_rda(raw_rda)
    need = {"tf", "target", "mor", "confidence"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"dorothea_hs missing columns: {sorted(missing)}; got {list(df.columns)}")

    df = df.copy()
    df["tf"] = df["tf"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df["confidence"] = df["confidence"].astype(str).str.strip().str.upper()
    df["mor"] = pd.to_numeric(df["mor"], errors="coerce")
    df = df[(df["tf"] != "") & (df["target"] != "") & df["mor"].notna()]
    df = df[df["mor"] != 0]

    raw_level_counts = df["confidence"].value_counts().sort_index().to_dict()
    raw_mor_sign = {
        "positive": int((df["mor"] > 0).sum()),
        "negative": int((df["mor"] < 0).sum()),
    }

    if levels:
        levels_u = [x.strip().upper() for x in levels]
        df = df[df["confidence"].isin(levels_u)]

    conf_rank = {c: i for i, c in enumerate(["A", "B", "C", "D", "E"])}
    df["_rank"] = df["confidence"].map(lambda x: conf_rank.get(x, 99))
    df = df.sort_values(["tf", "target", "_rank"])

    rows: List[Dict[str, str]] = []
    n_conflict = 0
    n_act = 0
    n_inh = 0
    for (tf, tg), sub in df.groupby(["tf", "target"], sort=True):
        signs = set(1 if m > 0 else -1 for m in sub["mor"].tolist())
        if len(signs) > 1:
            n_conflict += 1
            if drop_conflicts:
                continue
        r0 = sub.iloc[0]
        mor = float(r0["mor"])
        if mor > 0:
            rel = "activation"
            n_act += 1
        else:
            rel = "inhibition"
            n_inh += 1
        rows.append(
            {
                "source": "",
                "source_symbol": str(tf),
                "target": "",
                "target_symbol": str(tg),
                "relation_subtype": rel,
                "pathway_id": f"DoRothEA_{r0['confidence']}",
                "pathway_title": "DoRothEA human TF-target (dorothea_hs)",
                "confidence": str(r0["confidence"]),
                "mor": f"{mor:g}",
            }
        )

    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    # Keep flat table next to out_csv; name follows levels (e.g. ABC / AE).
    lvl_tag = "".join(levels) if levels else "AE"
    raw_tsv = out_csv.parent / f"DoRothEA_human_{lvl_tag}_raw_table.tsv"
    df.drop(columns=["_rank"], errors="ignore").to_csv(raw_tsv, sep="\t", index=False)

    level_out = out["confidence"].value_counts().sort_index().to_dict() if len(out) else {}
    stats: Dict[str, Any] = {
        "source_url": DOROTHEA_HS_URL,
        "raw_rda": str(raw_rda),
        "raw_tsv": str(raw_tsv),
        "out_csv": str(out_csv),
        "levels_requested": levels or ["A", "B", "C", "D", "E"],
        "raw_rows_in_rda_after_nonzero_mor": int(sum(raw_level_counts.values())),
        "raw_level_counts": {str(k): int(v) for k, v in raw_level_counts.items()},
        "raw_mor_sign_counts": raw_mor_sign,
        "unique_tf": int(out["source_symbol"].nunique()) if len(out) else 0,
        "unique_target": int(out["target_symbol"].nunique()) if len(out) else 0,
        "edges_activation": int(n_act),
        "edges_inhibition": int(n_inh),
        "edges_total": int(len(out)),
        "level_counts_out": {str(k): int(v) for k, v in level_out.items()},
        "conflicts_dropped": int(n_conflict) if drop_conflicts else 0,
        "drop_conflicts": bool(drop_conflicts),
    }
    stats_path = out_csv.with_name(out_csv.stem + "_build_stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Download/build DoRothEA human GRN CSV (A-E)")
    ap.add_argument(
        "--raw-rda",
        type=Path,
        default=root / "data" / "network" / "grn" / "raw" / "dorothea_hs.rda",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=root
        / "data"
        / "network"
        / "grn"
        / "DoRothEA_human_AE_activation_inhibition_with_symbols.csv",
    )
    ap.add_argument(
        "--levels",
        default="A,B,C,D,E",
        help="Comma-separated confidence levels to keep (default A-E).",
    )
    ap.add_argument("--download", action="store_true", help="Force re-download dorothea_hs.rda")
    ap.add_argument("--no-download", action="store_true", help="Do not download even if missing")
    ap.add_argument("--keep-conflicts", action="store_true")
    args = ap.parse_args()

    if args.download or (not args.raw_rda.exists() and not args.no_download):
        print(f"[DoRothEA] downloading {DOROTHEA_HS_URL}")
        _download(DOROTHEA_HS_URL, args.raw_rda)
        print(f"[DoRothEA] saved {args.raw_rda} ({args.raw_rda.stat().st_size} bytes)")
    elif not args.raw_rda.exists():
        raise FileNotFoundError(args.raw_rda)

    levels = [x.strip() for x in str(args.levels).split(",") if x.strip()]
    stats = build_dorothea_csv(
        args.raw_rda,
        args.out_csv,
        levels=levels,
        drop_conflicts=not args.keep_conflicts,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
