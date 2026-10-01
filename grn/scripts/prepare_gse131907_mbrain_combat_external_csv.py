from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from grn.data.preprocessing import normalize_gene_symbol_series


def _load_expr_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"expression csv has no sample columns: {path}")
    gcol = df.columns[0]
    df[gcol] = normalize_gene_symbol_series(df[gcol])
    df = df.drop_duplicates(subset=[gcol], keep="first").set_index(gcol)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


def _subsample_target(df: pd.DataFrame, max_samples: int, seed: int) -> Tuple[pd.DataFrame, list[str]]:
    cols = df.columns.tolist()
    if max_samples <= 0 or len(cols) <= max_samples:
        return df, cols
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(cols), size=max_samples, replace=False).tolist())
    picked_cols = [cols[i] for i in pick]
    return df[picked_cols], picked_cols


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare GSE131907 mBrain small-sample CSV via ComBat with GSE253718 as reference."
    )
    ap.add_argument("--train-exp-csv", required=True, help="Reference expression CSV (gene x sample), e.g. GSE253718_exp.csv")
    ap.add_argument("--target-exp-csv", required=True, help="Target expression CSV (gene x sample), e.g. GSE131907_mBrain_expr.csv")
    ap.add_argument("--target-max-samples", type=int, default=800, help="Subsample target columns to this size (<=0 means no subsample)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", required=True, help="Output corrected target CSV (gene x sample)")
    ap.add_argument("--meta-json", default=None, help="Optional metadata JSON path")
    args = ap.parse_args()

    train_path = Path(args.train_exp_csv).resolve()
    target_path = Path(args.target_exp_csv).resolve()
    out_csv = Path(args.out_csv).resolve()
    meta_json = Path(args.meta_json).resolve() if args.meta_json else out_csv.with_suffix(".meta.json")

    if not train_path.is_file():
        raise FileNotFoundError(f"train csv not found: {train_path}")
    if not target_path.is_file():
        raise FileNotFoundError(f"target csv not found: {target_path}")

    print(f"[combat-prep] load train: {train_path}")
    train_df = _load_expr_csv(train_path)
    print(f"[combat-prep] load target: {target_path}")
    target_df = _load_expr_csv(target_path)

    target_df, picked_cols = _subsample_target(target_df, int(args.target_max_samples), int(args.seed))
    print(f"[combat-prep] target samples after subsample: {target_df.shape[1]}")

    shared_genes = [g for g in train_df.index.tolist() if g in set(target_df.index.tolist())]
    if not shared_genes:
        raise ValueError("no shared genes between train and target")
    train_sub = train_df.loc[shared_genes]
    target_sub = target_df.loc[shared_genes]
    print(f"[combat-prep] shared genes: {len(shared_genes)}")

    # AnnData shape: cells x genes
    X_train = train_sub.values.T.astype(np.float32, copy=False)
    X_target = target_sub.values.T.astype(np.float32, copy=False)
    X_all = np.vstack([X_train, X_target])
    batch = np.array(["train"] * X_train.shape[0] + ["target"] * X_target.shape[0], dtype=object)

    adata = ad.AnnData(X=X_all)
    adata.obs["batch"] = pd.Categorical(batch)
    adata.var_names = shared_genes
    adata.obs_names = (
        [f"train::{c}" for c in train_sub.columns.tolist()]
        + [f"target::{c}" for c in target_sub.columns.tolist()]
    )

    # ComBat on expression matrix
    print("[combat-prep] run ComBat ...")
    sc.pp.combat(adata, key="batch")
    print("[combat-prep] ComBat done.")

    is_target = (adata.obs["batch"].astype(str).values == "target")
    X_target_corr = np.asarray(adata.X[is_target, :], dtype=np.float32)  # cells x genes
    out_df = pd.DataFrame(X_target_corr.T, index=shared_genes, columns=target_sub.columns.tolist())
    out_df = out_df.reset_index().rename(columns={"index": "gene"})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[combat-prep] wrote corrected target csv: {out_csv}")

    meta = {
        "train_exp_csv": str(train_path),
        "target_exp_csv": str(target_path),
        "out_csv": str(out_csv),
        "seed": int(args.seed),
        "target_max_samples": int(args.target_max_samples),
        "target_samples_after_subsample": int(target_sub.shape[1]),
        "target_sample_names_after_subsample": picked_cols,
        "n_shared_genes": int(len(shared_genes)),
        "n_train_samples": int(train_sub.shape[1]),
        "n_target_samples": int(target_sub.shape[1]),
        "method": "scanpy.pp.combat",
    }
    meta_json.parent.mkdir(parents=True, exist_ok=True)
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[combat-prep] wrote meta json: {meta_json}")


if __name__ == "__main__":
    main()
