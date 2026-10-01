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


def header_to_key(header: str) -> str:
    # header 示例: gene_name::chr1:65564-65573(+)
    if "::" in header:
        return header.split("::", 1)[1].strip()
    return header.strip()


def build_key_to_seq(records: List[Tuple[str, str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for h, seq in records:
        k = header_to_key(h)
        if k not in out:
            out[k] = seq
    return out


def extract_feature_matrix(keys: List[str], key_to_seq: Dict[str, str], hidden_size: int, device: torch.device) -> np.ndarray:
    feat = []
    for k in keys:
        s = key_to_seq.get(k, "")
        feat.append(extract_features_for_one_seq(str(s), hidden_size=hidden_size, device=device))
    return np.asarray(feat, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Count overlap between CDS/TSS FASTA and extract overlap-only features."
    )
    parser.add_argument("--cds-fasta", required=True, help="CDS FASTA path, e.g. all_cds_sequences.fa")
    parser.add_argument("--tss-fasta", required=True, help="TSS FASTA path, e.g. all_genes_upstream_downstream.fa")
    parser.add_argument("--output-dir", required=True, help="Output directory for report/features")
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--match-mode",
        choices=["key", "order"],
        default="key",
        help="key: 按 header 解析键匹配；order: 按 FASTA 行记录顺序对齐。",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only output overlap report, skip feature extraction.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    cds_records = read_fasta(args.cds_fasta)
    tss_records = read_fasta(args.tss_fasta)
    if args.match_mode == "key":
        cds_map = build_key_to_seq(cds_records)
        tss_map = build_key_to_seq(tss_records)
        cds_keys = set(cds_map.keys())
        tss_keys = set(tss_map.keys())
        overlap_keys = sorted(cds_keys & tss_keys)
        report = {
            "match_mode": "key",
            "cds_records_raw": len(cds_records),
            "tss_records_raw": len(tss_records),
            "cds_keys_unique": len(cds_keys),
            "tss_keys_unique": len(tss_keys),
            "overlap_keys": len(overlap_keys),
            "overlap_ratio_vs_cds": float(len(overlap_keys) / max(len(cds_keys), 1)),
            "overlap_ratio_vs_tss": float(len(overlap_keys) / max(len(tss_keys), 1)),
        }
    else:
        n = min(len(cds_records), len(tss_records))
        overlap_keys = [f"order_{i}" for i in range(n)]
        cds_map = {overlap_keys[i]: cds_records[i][1] for i in range(n)}
        tss_map = {overlap_keys[i]: tss_records[i][1] for i in range(n)}
        report = {
            "match_mode": "order",
            "cds_records_raw": len(cds_records),
            "tss_records_raw": len(tss_records),
            "overlap_keys": n,
            "overlap_ratio_vs_cds": float(n / max(len(cds_records), 1)),
            "overlap_ratio_vs_tss": float(n / max(len(tss_records), 1)),
            "warning": "Order mode assumes two FASTA files are already in the same gene order.",
        }

    report_path = os.path.join(args.output_dir, "cds_tss_overlap_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    overlap_key_path = os.path.join(args.output_dir, "overlap_keys.txt")
    with open(overlap_key_path, "w", encoding="utf-8") as f:
        for k in overlap_keys:
            f.write(k + "\n")

    if args.match_mode == "key":
        print(
            f"[overlap] mode=key cds_unique={report['cds_keys_unique']} "
            f"tss_unique={report['tss_keys_unique']} overlap={report['overlap_keys']}"
        )
    else:
        print(
            f"[overlap] mode=order cds_records={report['cds_records_raw']} "
            f"tss_records={report['tss_records_raw']} overlap={report['overlap_keys']}"
        )
    print(f"[overlap] report={report_path}")

    if args.count_only:
        return

    cds_feat = extract_feature_matrix(overlap_keys, cds_map, hidden_size=args.hidden_size, device=device)
    tss_feat = extract_feature_matrix(overlap_keys, tss_map, hidden_size=args.hidden_size, device=device)
    fused_feat = np.concatenate([cds_feat, tss_feat], axis=1)

    pd.DataFrame(cds_feat).to_csv(os.path.join(args.output_dir, "cds_overlap_feature_k4.csv"), index=False)
    pd.DataFrame(tss_feat).to_csv(os.path.join(args.output_dir, "tss_overlap_feature_k4.csv"), index=False)
    pd.DataFrame(fused_feat).to_csv(os.path.join(args.output_dir, "cds_tss_overlap_feature_k4.csv"), index=False)
    print(
        "[feature] saved: cds_overlap_feature_k4.csv, tss_overlap_feature_k4.csv, "
        "cds_tss_overlap_feature_k4.csv"
    )


if __name__ == "__main__":
    main()

