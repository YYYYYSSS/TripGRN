# -*- coding: utf-8 -*-
"""Materialize GSE131907 tLung-primary expression npy from nonbrain matrix + barcodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    ap.add_argument(
        "--precomputed-dir",
        type=Path,
        default=root / "data" / "exp" / "GSE131907" / "precomputed_brain_nonbrain_npy",
    )
    ap.add_argument(
        "--tlung-barcodes-csv",
        type=Path,
        default=root
        / "data"
        / "other"
        / "GSE131907"
        / "cell_barcodes_for_virtual_ko"
        / "GSE131907_tLung_primary_cell_barcodes.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root / "data" / "exp" / "GSE131907" / "precomputed_tlung_mbrain_npy",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    pre = Path(args.precomputed_dir).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    man = out / "manifest.json"
    if man.is_file() and not args.force:
        print(f"[tlung-mat] skip exists {man}", flush=True)
        return

    nb_cols = np.load(pre / "nonbrain_column_names.npy", allow_pickle=True).astype(str)
    genes = np.load(pre / "genes.npy", allow_pickle=True)
    bc = pd.read_csv(args.tlung_barcodes_csv)
    want = bc["cell_barcode"].astype(str).tolist()
    col_to_i = {c: i for i, c in enumerate(nb_cols)}
    missing = [c for c in want if c not in col_to_i]
    if missing:
        raise SystemExit(f"{len(missing)} tLung barcodes not in nonbrain columns; e.g. {missing[:3]}")
    idx = np.asarray([col_to_i[c] for c in want], dtype=np.int64)
    print(f"[tlung-mat] extracting {len(idx)} tLung cols from nonbrain {nb_cols.shape[0]}", flush=True)

    src = np.load(pre / "nonbrain_values_log_normalized.npy", mmap_mode="r")
    # copy selected columns (gene x cells)
    n_genes = int(src.shape[0])
    out_vals = out / "tlung_values_log_normalized.npy"
    # write in column chunks to limit RAM
    mm = np.lib.format.open_memmap(
        str(out_vals), mode="w+", dtype=np.float32, shape=(n_genes, int(len(idx)))
    )
    chunk = 2048
    for s in range(0, len(idx), chunk):
        e = min(s + chunk, len(idx))
        mm[:, s:e] = np.asarray(src[:, idx[s:e]], dtype=np.float32)
        if s == 0 or e == len(idx) or (s // chunk) % 10 == 0:
            print(f"[tlung-mat] cols {e}/{len(idx)}", flush=True)
    mm.flush()
    del mm

    import shutil

    # genes + column names only; mBrain matrix stays in precomputed (avoid 3GB duplicate)
    shutil.copy2(pre / "genes.npy", out / "genes.npy")
    shutil.copy2(pre / "brain_column_names.npy", out / "mbrain_column_names.npy")
    np.save(out / "tlung_column_names.npy", np.asarray(want, dtype=object))
    mbrain_vals = pre / "brain_values_log_normalized.npy"
    # pointer file for orchestrators
    (out / "mbrain_values_log_normalized.path.txt").write_text(str(mbrain_vals.resolve()) + "\n", encoding="utf-8")
    payload = {
        "n_genes": int(n_genes),
        "n_tlung_cols": int(len(idx)),
        "n_mbrain_cols": int(np.load(mbrain_vals, mmap_mode="r").shape[1]),
        "tlung_values": str(out_vals),
        "mbrain_values": str(mbrain_vals.resolve()),
        "genes": str(out / "genes.npy"),
        "source_nonbrain": str(pre / "nonbrain_values_log_normalized.npy"),
        "tlung_barcodes_csv": str(Path(args.tlung_barcodes_csv).resolve()),
        "note": "tLung extracted from nonbrain; mBrain path points to precomputed brain matrix (no copy)",
    }
    man.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print("[tlung-mat] done", flush=True)


if __name__ == "__main__":
    main()
