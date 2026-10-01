import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch

from grn.scripts.extract_cds_features import extract_features_for_one_seq


def _read_table_auto(path: str, sep: Optional[str]) -> pd.DataFrame:
    if sep is not None:
        return pd.read_csv(path, sep=sep)
    # 自动分隔符：支持 csv/tsv/混合文本
    return pd.read_csv(path, sep=None, engine="python")


def _read_fasta(path: str):
    records = []
    header = None
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Extract TSS upstream/downstream sequence features (20-dim per gene, kmer+biGRU)."
    )
    parser.add_argument("--expr-csv", required=True, help="Expression CSV, first column is gene symbol.")
    parser.add_argument("--tss-raw-file", required=True, help="Raw TSS sequence table path.")
    parser.add_argument("--output-csv", required=True, help="Output feature CSV path.")
    parser.add_argument(
        "--tss-gene-col",
        default=None,
        help="Gene column name in raw TSS file; default uses first column.",
    )
    parser.add_argument(
        "--tss-seq-col",
        default=None,
        help="Sequence column name in raw TSS file; default uses second column if exists, else first.",
    )
    parser.add_argument(
        "--tss-sep",
        default=None,
        help=r"Optional separator override, e.g. ',' or '\t'. Default auto-detect.",
    )
    parser.add_argument(
        "--tss-format",
        choices=["auto", "table", "fasta"],
        default="auto",
        help="Input format for TSS raw file.",
    )
    parser.add_argument(
        "--fasta-map-by-order",
        action="store_true",
        help="For FASTA input: ignore header gene parsing and align by expression gene order.",
    )
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    expr_df = pd.read_csv(args.expr_csv)
    expr_gene_col = expr_df.columns[0]
    gene_list = expr_df.iloc[:, 0].astype(str).str.strip().tolist()

    fmt = args.tss_format
    if fmt == "auto":
        lower = args.tss_raw_file.lower()
        fmt = "fasta" if lower.endswith((".fa", ".fasta", ".fna")) else "table"

    if fmt == "fasta":
        recs = _read_fasta(args.tss_raw_file)
        if not recs:
            raise ValueError("TSS FASTA is empty.")
        if args.fasta_map_by_order:
            seqs = [seq for _, seq in recs]
            if len(seqs) < len(gene_list):
                raise ValueError(
                    f"FASTA records({len(seqs)}) < expression genes({len(gene_list)}), cannot map by order."
                )
            seq_data = np.array(seqs[: len(gene_list)], dtype=object)
            print("[extract_tss_features] FASTA mapped by order.")
        else:
            rows = []
            for h, seq in recs:
                # default: gene symbol before first "::"
                g = h.split("::")[0].strip()
                rows.append((g, seq))
            tss_df = pd.DataFrame(rows, columns=["gene", "seq"])
            tss_df["gene"] = tss_df["gene"].astype(str).str.strip()
            tss_df["seq"] = tss_df["seq"].astype(str).str.strip()
            tss_df = tss_df.drop_duplicates(subset=["gene"], keep="first").set_index("gene")
            tss_df = tss_df.reindex(gene_list)
            seq_data = tss_df["seq"].fillna("").values
    else:
        tss_df = _read_table_auto(args.tss_raw_file, args.tss_sep)
        if tss_df.empty:
            raise ValueError("TSS raw file is empty.")

        gene_col = args.tss_gene_col or tss_df.columns[0]
        if gene_col not in tss_df.columns:
            raise ValueError(f"TSS gene column not found: {gene_col}")

        if args.tss_seq_col is not None:
            seq_col = args.tss_seq_col
        else:
            seq_col = tss_df.columns[1] if len(tss_df.columns) > 1 else tss_df.columns[0]
        if seq_col not in tss_df.columns:
            raise ValueError(f"TSS sequence column not found: {seq_col}")

        tss_df[gene_col] = tss_df[gene_col].astype(str).str.strip()
        tss_df[seq_col] = tss_df[seq_col].astype(str).str.strip()
        tss_df = tss_df.drop_duplicates(subset=[gene_col], keep="first").set_index(gene_col)
        tss_df = tss_df.reindex(gene_list)
        seq_data = tss_df[seq_col].fillna("").values

    features = []
    for s in seq_data:
        # 与 CDS 特征器保持一致：长度不足会走内部兜底返回 20 维极小值
        features.append(
            extract_features_for_one_seq(
                str(s),
                hidden_size=args.hidden_size,
                device=device,
            )
        )

    out = np.array(features, dtype=np.float32)
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(out).to_csv(args.output_csv, index=False)
    print(f"[extract_tss_features] genes={len(gene_list)} dim={out.shape[1]} output={args.output_csv}")


if __name__ == "__main__":
    main()
