# -*- coding: utf-8 -*-
"""Export epithelial tLung/mBrain log-normalized matrix to HDF5 for Seurat UMAP."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NPY = ROOT / "data" / "exp" / "GSE131907" / "precomputed_tlung_mbrain_epi_npy"
OUT = (
    ROOT
    / "experiments"
    / "gse131907_transfer_from_gse253718_epi_seed2"
    / "umap_input_tlung_mbrain_epi.h5"
)


def main() -> None:
    genes = np.load(NPY / "genes.npy", allow_pickle=True).astype(str)
    xt = np.load(NPY / "tlung_values_log_normalized.npy", mmap_mode="r")  # genes x cells
    xb = np.load(NPY / "mbrain_values_log_normalized.npy", mmap_mode="r")
    nt, nb = int(xt.shape[1]), int(xb.shape[1])
    n_genes = int(genes.shape[0])
    n_cells = nt + nb
    print(f"[export] genes={n_genes} tLung={nt} mBrain={nb}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()

    with h5py.File(OUT, "w") as h5:
        h5.create_dataset(
            "genes",
            data=np.asarray(genes, dtype=h5py.string_dtype(encoding="utf-8")),
        )
        comps = np.array(["tLung"] * nt + ["mBrain"] * nb, dtype=object)
        h5.create_dataset(
            "compartment",
            data=np.asarray(comps, dtype=h5py.string_dtype(encoding="utf-8")),
        )
        # Store genes x cells float32
        dset = h5.create_dataset(
            "lognorm",
            shape=(n_genes, n_cells),
            dtype="float32",
            chunks=(min(512, n_genes), min(512, n_cells)),
            compression="gzip",
            compression_opts=1,
        )
        dset[:, :nt] = np.asarray(xt, dtype=np.float32)
        dset[:, nt:] = np.asarray(xb, dtype=np.float32)
        h5.attrs["n_tLung"] = nt
        h5.attrs["n_mBrain"] = nb
        h5.attrs["note"] = "log-normalized expression; genes x cells"

    print(f"[export] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
