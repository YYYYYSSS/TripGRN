from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from grn.data.preprocessing import normalize_gene_symbol_series


def _load_expr(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    gcol = df.columns[0]
    df[gcol] = normalize_gene_symbol_series(df[gcol])
    df = df.drop_duplicates(subset=[gcol], keep="first").set_index(gcol)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Concat two gene-by-sample expression CSVs by shared genes (horizontal columns).")
    ap.add_argument("--left-csv", required=True)
    ap.add_argument("--right-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    left = _load_expr(args.left_csv)
    right = _load_expr(args.right_csv)

    shared = [g for g in left.index.tolist() if g in set(right.index.tolist())]
    if not shared:
        raise SystemExit("No shared genes between left/right CSV.")

    left_sub = left.loc[shared]
    right_sub = right.loc[shared]
    out = pd.concat([left_sub, right_sub], axis=1)
    out = out.reset_index().rename(columns={"index": "gene"})

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(
        f"[concat-expr-csv] genes={len(shared)} "
        f"left_cols={left_sub.shape[1]} right_cols={right_sub.shape[1]} "
        f"-> {out_path}"
    )


if __name__ == "__main__":
    main()
