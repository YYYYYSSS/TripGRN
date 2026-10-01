# -*- coding: utf-8 -*-
"""Extract GSE123902_RAW.tar, merge dense CSVs, QC, write genes x cells counts npy."""
from __future__ import annotations

import argparse
import gzip
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from grn.data.gse123902_paths import SAMPLE_CATALOG, gse123902_layout


def _parse_gsm_from_name(name: str) -> str | None:
    m = re.match(r"(GSM\d+)_", Path(name).name)
    return m.group(1) if m else None


def extract_tar(tar_path: Path, raw_dir: Path, *, force: bool) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(raw_dir.glob("GSM*_dense.csv.gz"))
    if existing and not force:
        print(f"[extract] skip ({len(existing)} files in {raw_dir})", flush=True)
        return existing
    if not tar_path.is_file():
        raise FileNotFoundError(tar_path)
    print(f"[extract] {tar_path} -> {raw_dir}", flush=True)
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(path=raw_dir)
    files = sorted(raw_dir.glob("GSM*_dense.csv.gz"))
    if not files:
        # some tars extract into a nested folder
        files = sorted(raw_dir.rglob("GSM*_dense.csv.gz"))
        if files and files[0].parent != raw_dir:
            for f in files:
                dest = raw_dir / f.name
                if not dest.exists():
                    f.replace(dest)
            files = sorted(raw_dir.glob("GSM*_dense.csv.gz"))
    if len(files) < 17:
        raise RuntimeError(f"Expected 17 dense CSV.gz, found {len(files)} under {raw_dir}")
    return files


def _read_dense_csv_gz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return genes (n,), cell_ids (m,), counts genes x cells float32."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        df = pd.read_csv(f, index_col=0)
    # cells x genes in file
    genes = np.asarray([str(g).strip() for g in df.columns], dtype=object)
    cells = np.asarray([str(c).strip() for c in df.index], dtype=object)
    mat = np.asarray(df.to_numpy(), dtype=np.float32).T  # genes x cells
    return genes, cells, mat


def _mt_mask(genes: np.ndarray) -> np.ndarray:
    g = np.asarray(genes, dtype=object)
    return np.asarray([str(x).upper().startswith("MT-") for x in g], dtype=bool)


def merge_and_qc(
    files: list[Path],
    *,
    out_dir: Path,
    min_features: int,
    max_features: int,
    max_mt_pct: float,
    sites_keep: set[str] | None,
    force: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "merge_qc_report.json"
    values_path = out_dir / "counts_values.npy"
    genes_path = out_dir / "counts_genes.npy"
    cells_path = out_dir / "counts_cells.npy"
    meta_path = out_dir / "cell_metadata.csv"
    if values_path.is_file() and report_path.is_file() and not force:
        print(f"[merge] skip exists {values_path}", flush=True)
        return json.loads(report_path.read_text(encoding="utf-8"))

    loaded = []
    for fp in files:
        gsm = _parse_gsm_from_name(fp.name)
        if gsm is None or gsm not in SAMPLE_CATALOG:
            print(f"[merge] WARN skip unrecognized {fp.name}", flush=True)
            continue
        info = SAMPLE_CATALOG[gsm]
        if sites_keep is not None and info["site"] not in sites_keep:
            print(f"[merge] skip site={info['site']} {fp.name}", flush=True)
            continue
        genes, cells, mat = _read_dense_csv_gz(fp)
        cell_ids = np.asarray([f"{gsm}_{c}" for c in cells], dtype=object)
        loaded.append(
            {
                "gsm": gsm,
                "patient": info["patient"],
                "site": info["site"],
                "file": fp.name,
                "genes": genes,
                "cells": cell_ids,
                "mat": mat,
            }
        )
        print(
            f"[merge] loaded {fp.name}: site={info['site']} genes={genes.size} cells={cells.size}",
            flush=True,
        )

    if not loaded:
        raise RuntimeError("No samples loaded")

    # intersect genes across samples
    gene_sets = [set(x["genes"].tolist()) for x in loaded]
    common = set.intersection(*gene_sets)
    # drop empty / NA-like
    common = {g for g in common if g and g.lower() not in {"nan", "none", "null"}}
    genes = np.asarray(sorted(common), dtype=object)
    if genes.size < 1000:
        raise RuntimeError(f"Too few shared genes after intersect: {genes.size}")

    blocks = []
    meta_rows = []
    for item in loaded:
        g2i = {g: i for i, g in enumerate(item["genes"].tolist())}
        idx = np.asarray([g2i[g] for g in genes.tolist()], dtype=np.int64)
        sub = np.asarray(item["mat"][idx, :], dtype=np.float32)
        # QC per sample on shared genes
        nfeat = (sub > 0).sum(axis=0)
        mt = _mt_mask(genes)
        mt_pct = np.zeros(sub.shape[1], dtype=np.float64)
        col_sum = sub.sum(axis=0)
        pos = col_sum > 0
        if mt.any():
            mt_pct[pos] = 100.0 * sub[mt][:, pos].sum(axis=0) / col_sum[pos]
        keep = (nfeat >= int(min_features)) & (nfeat <= int(max_features)) & (mt_pct < float(max_mt_pct))
        n_before = int(sub.shape[1])
        sub = sub[:, keep]
        cells_k = item["cells"][keep]
        nfeat_k = nfeat[keep]
        mt_k = mt_pct[keep]
        print(
            f"[qc] {item['gsm']} {item['site']}: keep {sub.shape[1]}/{n_before}",
            flush=True,
        )
        blocks.append(sub)
        for j, cid in enumerate(cells_k.tolist()):
            meta_rows.append(
                {
                    "cell_id": cid,
                    "gsm": item["gsm"],
                    "patient": item["patient"],
                    "site": item["site"],
                    "source_file": item["file"],
                    "n_features": int(nfeat_k[j]),
                    "percent_mt": float(mt_k[j]),
                }
            )

    values = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    cells = np.asarray([r["cell_id"] for r in meta_rows], dtype=object)
    assert values.shape[1] == cells.size

    np.save(values_path, values)
    np.save(genes_path, genes)
    np.save(cells_path, cells)
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)

    site_counts = {}
    for r in meta_rows:
        site_counts[r["site"]] = site_counts.get(r["site"], 0) + 1

    report = {
        "n_genes": int(genes.size),
        "n_cells": int(cells.size),
        "n_samples_loaded": len(loaded),
        "site_cell_counts": site_counts,
        "qc": {
            "min_features": int(min_features),
            "max_features": int(max_features),
            "max_mt_pct": float(max_mt_pct),
        },
        "paths": {
            "values": str(values_path),
            "genes": str(genes_path),
            "cells": str(cells_path),
            "meta": str(meta_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare GSE123902 merged counts for SingleR")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-features", type=int, default=200)
    ap.add_argument("--max-features", type=int, default=6000)
    ap.add_argument("--max-mt-pct", type=float, default=20.0)
    ap.add_argument(
        "--sites",
        default="primary,brain",
        help="Comma sites to keep for merge (default primary,brain). Use ALL for every sample.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    layout = gse123902_layout(root)
    layout["root"].mkdir(parents=True, exist_ok=True)
    # persist catalog
    layout["sample_catalog_json"].write_text(
        json.dumps(SAMPLE_CATALOG, indent=2), encoding="utf-8"
    )

    files = extract_tar(layout["raw_tar"], layout["raw_dir"], force=bool(args.force))
    sites = None
    if str(args.sites).strip().upper() != "ALL":
        sites = {s.strip().lower() for s in str(args.sites).split(",") if s.strip()}
    merge_and_qc(
        files,
        out_dir=layout["merged_dir"],
        min_features=int(args.min_features),
        max_features=int(args.max_features),
        max_mt_pct=float(args.max_mt_pct),
        sites_keep=sites,
        force=bool(args.force),
    )
    print("[prepare] done", flush=True)


if __name__ == "__main__":
    main()
