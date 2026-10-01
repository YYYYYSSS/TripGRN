import argparse
import math
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def kmer_func(seq: List[str], k: int = 4) -> np.ndarray:
    """从序列中提取 k-mer（与原脚本逻辑保持一致，默认 k=4）。"""
    X = [None] * len(seq)
    for i in range(len(seq)):
        a = seq[i]
        l = []
        for index in range(len(a)):
            t = a[index : index + k]
            if len(t) == k:
                l.append(t)
        X[i] = l
    return np.array(X, dtype=object)


def encode_matrix(seq_matrix, alphabet) -> np.ndarray:
    """
    把 k-mer 序列转成 one-hot（保持原脚本思想：每个 k-mer 对应一个 vocab index）。
    返回形状大致为 [num_windows, num_tokens, vocab_size]（原脚本里 windows=1）。
    """
    seq_X = [None] * len(seq_matrix)
    ind_to_char = alphabet
    char_to_int = dict((c, i) for i, c in enumerate(ind_to_char))

    for j in range(len(seq_matrix)):
        data = seq_matrix[j]
        integer_encoded = [char_to_int[char] for char in data]
        onehot_encoded = []
        for value in integer_encoded:
            letter = [0 for _ in range(len(ind_to_char))]
            letter[value] = 1
            onehot_encoded.append(letter)
        seq_X[j] = onehot_encoded

    return np.array(seq_X, dtype=np.float32)


def extract_features_for_one_seq(
    seq_str: str,
    hidden_size: int = 10,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    将单条 CDS 序列映射到固定 20 维向量（bi-GRU，bidirectional=True，hidden_size=10 -> 20维）。
    """
    seq = [seq_str]

    if len(seq[0]) > 1:
        alphabet = np.unique(kmer_func(seq, k=4))
        k_mers = [alphabet]
        onehot = encode_matrix(k_mers, alphabet=alphabet)  # [1, num_tokens, vocab_size]

        # CDS 特征提取是计算密集部分之一：确保 GRU 与输入张量在同一设备（GPU/CPU）上
        onehot_tensor = torch.tensor(onehot, dtype=torch.float32, device=device)
        # 稳健处理不同维度，统一到 GRU 需要的 [seq_len, batch, input_size]
        if onehot_tensor.dim() == 3:
            # [1, seq_len, vocab] -> [seq_len, 1, vocab]
            input_tensor = onehot_tensor.squeeze(0).unsqueeze(1)
        elif onehot_tensor.dim() == 2:
            # [seq_len, vocab] -> [seq_len, 1, vocab]
            input_tensor = onehot_tensor.unsqueeze(1)
        elif onehot_tensor.dim() == 1:
            # [vocab] -> [1, 1, vocab]
            input_tensor = onehot_tensor.unsqueeze(0).unsqueeze(1)
        else:
            raise RuntimeError(f"Unexpected onehot dim: {onehot_tensor.dim()}")

        vocab_size = input_tensor.shape[-1]
        gru = nn.GRU(
            input_size=int(vocab_size),
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=False,
        ).to(device)

        batch_size = int(input_tensor.shape[1])
        h0 = torch.randn(2, batch_size, hidden_size, device=device)  # (num_directions, batch, hidden_size)
        with torch.no_grad():
            output, _ = gru(input_tensor, h0)

        bi_gru = output[-1]  # [batch=1, 2*hidden_size] -> [1, 20]
        bi_gru_np = bi_gru.detach().cpu().numpy()
        bi_gru_avg = bi_gru_np.mean(axis=0)  # [20]
        return bi_gru_avg.astype(np.float32)

    # 与原脚本保持一致：不足时填充特殊小值（长度 20）
    non_geng = np.zeros(20, dtype=np.float32)
    for j in range(len(non_geng)):
        non_geng[j] = math.pow(10, -99)
    return non_geng


def main():
    parser = argparse.ArgumentParser(description="Extract CDS k-mer + biGRU features (20-dim per gene).")
    parser.add_argument("--expr-csv", required=True, help="Expression CSV (first column is gene name).")
    parser.add_argument("--cds-raw-txt", required=True, help="CDS raw file (tab-separated; first column is gene name).")
    parser.add_argument("--output-csv", required=True, help="Output CDS feature CSV path.")
    parser.add_argument("--hidden-size", type=int, default=10)
    # 默认优先使用 CUDA（如不可用会在后续回退到 CPU）
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    expr_data = pd.read_csv(args.expr_csv)
    gene_list = expr_data.iloc[:, :1].values.reshape(-1).tolist()

    original_data = pd.read_table(args.cds_raw_txt, header=None, sep="\t", skiprows=1)
    original_data.set_index(0, inplace=True)
    original_data = original_data.reindex(gene_list)
    original_data = np.array(original_data)

    seq_data = original_data[:, 0]
    seq_feature = []
    for i in range(len(seq_data)):
        seq_feature.append(
            extract_features_for_one_seq(
                str(seq_data[i]),
                hidden_size=args.hidden_size,
                device=device,
            )
        )

    seq_feature = np.array(seq_feature, dtype=np.float32)
    pd.DataFrame(seq_feature).to_csv(args.output_csv, index=False)
    print(f"已生成 CDS 特征：{args.output_csv}")


if __name__ == "__main__":
    main()

