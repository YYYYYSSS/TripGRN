from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert gzipped TSV matrix to CSV (streaming, low memory).")
    ap.add_argument("--in-gz-tsv", required=True, help="Input .txt.gz/.tsv.gz matrix path")
    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    ap.add_argument(
        "--rename-first-col-to-gene",
        action="store_true",
        help="Rename first header field to 'gene' (recommended for expression matrix).",
    )
    args = ap.parse_args()

    in_path = Path(args.in_gz_tsv).resolve()
    out_path = Path(args.out_csv).resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"input file not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    with gzip.open(in_path, "rt", encoding="utf-8", errors="replace", newline="") as fin, open(
        out_path, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"empty input: {in_path}")
        if args.rename_first_col_to_gene and len(header) > 0:
            header[0] = "gene"
        writer.writerow(header)
        for row in reader:
            writer.writerow(row)
            n_rows += 1

    print(f"[convert-gz-tsv-to-csv] done: {out_path} rows={n_rows}")


if __name__ == "__main__":
    main()
