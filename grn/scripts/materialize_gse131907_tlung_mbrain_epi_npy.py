# -*- coding: utf-8 -*-
"""Materialize GSE131907 tLung / mBrain epithelial-only log-normalized npy.

Filters full tLung / mBrain matrices to Epithelial cells barcodes
(from celltype_step06_bytype overall split).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _extract(src: Path, col_names: np.ndarray, want: list[str], out_vals: Path, out_cols: Path) -> int:
    col_to_i = {str(c): i for i, c in enumerate(col_names.astype(str))}
    missing = [c for c in want if c not in col_to_i]
    if missing:
        raise SystemExit(f"{len(missing)} barcodes missing in {src.name}; e.g. {missing[:3]}")
    idx = np.asarray([col_to_i[c] for c in want], dtype=np.int64)
    mm_in = np.load(src, mmap_mode="r")
    n_genes = int(mm_in.shape[0])
    out_vals.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(str(out_vals), mode="w+", dtype=np.float32, shape=(n_genes, int(len(idx))))
    chunk = 2048
    for s in range(0, len(idx), chunk):
        e = min(s + chunk, len(idx))
        mm[:, s:e] = np.asarray(mm_in[:, idx[s:e]], dtype=np.float32)
        if s == 0 or e == len(idx) or (s // chunk) % 5 == 0:
            print(f"[epi-mat] {out_vals.name} cols {e}/{len(idx)}", flush=True)
    mm.flush()
    np.save(out_cols, np.asarray(want, dtype=object))
    return int(len(idx))


def main() -> None:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    ap.add_argument(
        "--tlung-dir",
        type=Path,
        default=root / "data/exp/GSE131907/precomputed_tlung_mbrain_npy",
    )
    ap.add_argument(
        "--brain-dir",
        type=Path,
        default=root / "data/exp/GSE131907/precomputed_brain_nonbrain_npy",
    )
    ap.add_argument(
        "--epithelial-barcodes-csv",
        type=Path,
        default=root
        / "data/exp/GSE131907/celltype_step06_bytype/cv_splits_barcodes/epithelial_cells_overall_split.csv",
    )
    ap.add_argument(
        "--tlung-barcodes-csv",
        type=Path,
        default=root
        / "data/other/GSE131907/cell_barcodes_for_virtual_ko/GSE131907_tLung_primary_cell_barcodes.csv",
    )
    ap.add_argument(
        "--mbrain-barcodes-csv",
        type=Path,
        default=root
        / "data/other/GSE131907/cell_barcodes_for_virtual_ko/GSE131907_mBrain_met_cell_barcodes.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root / "data/exp/GSE131907/precomputed_tlung_mbrain_epi_npy",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    man = out / "manifest.json"
    if man.is_file() and not args.force:
        print(f"[epi-mat] skip exists {man}", flush=True)
        return

    tlung_dir = Path(args.tlung_dir).resolve()
    brain_dir = Path(args.brain_dir).resolve()
    # ensure full tLung exists
    tlung_vals = tlung_dir / "tlung_values_log_normalized.npy"
    if not tlung_vals.is_file():
        raise SystemExit(f"missing {tlung_vals}; run materialize_gse131907_tlung_mbrain_npy first")
    mbrain_vals = brain_dir / "brain_values_log_normalized.npy"
    if not mbrain_vals.is_file():
        raise SystemExit(f"missing {mbrain_vals}")

    epi = set(pd.read_csv(args.epithelial_barcodes_csv)["barcode"].astype(str))
    tl = [b for b in pd.read_csv(args.tlung_barcodes_csv)["cell_barcode"].astype(str).tolist() if b in epi]
    mb = [b for b in pd.read_csv(args.mbrain_barcodes_csv)["cell_barcode"].astype(str).tolist() if b in epi]
    if not tl or not mb:
        raise SystemExit(f"empty epithelial intersection: tLung={len(tl)} mBrain={len(mb)}")

    import shutil

    shutil.copy2(tlung_dir / "genes.npy", out / "genes.npy")
    n_tl = _extract(
        tlung_vals,
        np.load(tlung_dir / "tlung_column_names.npy", allow_pickle=True),
        tl,
        out / "tlung_values_log_normalized.npy",
        out / "tlung_column_names.npy",
    )
    n_mb = _extract(
        mbrain_vals,
        np.load(brain_dir / "brain_column_names.npy", allow_pickle=True),
        mb,
        out / "mbrain_values_log_normalized.npy",
        out / "mbrain_column_names.npy",
    )
    payload = {
        "n_genes": int(np.load(out / "genes.npy", allow_pickle=True).shape[0]),
        "n_tlung_epi": n_tl,
        "n_mbrain_epi": n_mb,
        "tlung_values": str(out / "tlung_values_log_normalized.npy"),
        "mbrain_values": str(out / "mbrain_values_log_normalized.npy"),
        "genes": str(out / "genes.npy"),
        "epithelial_barcodes_csv": str(Path(args.epithelial_barcodes_csv).resolve()),
        "note": "epithelial-only tLung primary vs mBrain; for transfer seed screen",
    }
    man.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print("[epi-mat] done", flush=True)


if __name__ == "__main__":
    main()
