import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from grn.scripts.extract_cds_features import extract_features_for_one_seq


def read_fasta(path: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header = None
    chunks: List[str] = []
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


def header_to_gene(header: str) -> str:
    # 形如 OR4F5::chr1:65564-65573(+)
    return header.split("::", 1)[0].strip() if "::" in header else header.strip().split()[0]


def build_gene_to_seqs(records: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    g2s: Dict[str, List[str]] = {}
    for h, seq in records:
        g = header_to_gene(h)
        g2s.setdefault(g, []).append(seq)
    return g2s


def mean_feature_for_gene(seqs: List[str], hidden_size: int, device: torch.device) -> np.ndarray:
    feats = [
        extract_features_for_one_seq(str(s), hidden_size=hidden_size, device=device)
        for s in seqs
    ]
    return np.mean(np.asarray(feats, dtype=np.float32), axis=0)


def feature_df_from_gene_map(
    g2s: Dict[str, List[str]],
    genes: List[str],
    hidden_size: int,
    device: torch.device,
) -> pd.DataFrame:
    out = []
    for g in genes:
        vec = mean_feature_for_gene(g2s[g], hidden_size=hidden_size, device=device)
        out.append(vec)
    arr = np.asarray(out, dtype=np.float32)
    df = pd.DataFrame(arr)
    df.insert(0, "gene", genes)
    return df


def main():
    parser = argparse.ArgumentParser(description="Extract CDS/TSS gene-level features on shared genes.")
    parser.add_argument("--cds-fasta", required=True)
    parser.add_argument("--tss-fasta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    cds_records = read_fasta(args.cds_fasta)
    tss_records = read_fasta(args.tss_fasta)
    cds_g2s = build_gene_to_seqs(cds_records)
    tss_g2s = build_gene_to_seqs(tss_records)

    cds_genes = set(cds_g2s.keys())
    tss_genes = set(tss_g2s.keys())
    shared_genes = sorted(cds_genes & tss_genes)

    report = {
        "cds_records": len(cds_records),
        "tss_records": len(tss_records),
        "cds_genes": len(cds_genes),
        "tss_genes": len(tss_genes),
        "shared_genes": len(shared_genes),
        "shared_ratio_vs_cds": float(len(shared_genes) / max(len(cds_genes), 1)),
        "shared_ratio_vs_tss": float(len(shared_genes) / max(len(tss_genes), 1)),
    }
    with open(os.path.join(args.output_dir, "cds_tss_gene_overlap_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "shared_genes.txt"), "w", encoding="utf-8") as f:
        for g in shared_genes:
            f.write(g + "\n")

    cds_df = feature_df_from_gene_map(cds_g2s, shared_genes, args.hidden_size, device)
    tss_df = feature_df_from_gene_map(tss_g2s, shared_genes, args.hidden_size, device)
    seq_df = pd.DataFrame(
        np.concatenate([cds_df.iloc[:, 1:].values, tss_df.iloc[:, 1:].values], axis=1)
    )
    seq_df.insert(0, "gene", shared_genes)

    cds_path = os.path.join(args.output_dir, "cds_gene_feature_k4.csv")
    tss_path = os.path.join(args.output_dir, "tss_gene_feature_k4.csv")
    seq_path = os.path.join(args.output_dir, "cds_tss_gene_feature_k4.csv")
    cds_df.to_csv(cds_path, index=False)
    tss_df.to_csv(tss_path, index=False)
    seq_df.to_csv(seq_path, index=False)

    print(f"[done] shared_genes={len(shared_genes)}")
    print(f"[done] cds={cds_path}")
    print(f"[done] tss={tss_path}")
    print(f"[done] concat={seq_path}")


if __name__ == "__main__":
    main()

