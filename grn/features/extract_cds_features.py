#%%
import os

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import pandas as pd

def kmer_func(seq):
    X = [None] * len(seq)
    for i in range(len(seq)):
        a = seq[i]
        t = 0
        l = []
        for index in range(len(a)):
            t = a[index:index + 4]  # # 从当前位置index开始，提取一个长度为4的子串（k-mer）。
            if (len(t)) == 4:       # # 检查当前提取的子串是否为4个字符（保证k-mer的长度为4）。
                l.append(t)
        X[i] = l
    return np.array(X)

def encode_matrix(seq_matrix):
    seq_X = [None] * len(seq_matrix)
    for j in range(len(seq_matrix)):
        data = seq_matrix[j]
        ind_to_char = k_mer
        char_to_int = dict((c, i) for i, c in enumerate(ind_to_char))
        integer_encoded = [char_to_int[char] for char in data]
        onthot_encoded = list()
        for value in integer_encoded:
            letter = [0 for _ in range(len(ind_to_char))]
            letter[value] = 1
            onthot_encoded.append(letter)
        seq_X[j] = onthot_encoded
    return seq_X

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

#%%
dir = "D:\\R\\workspace\\work\\WZH毕设重构\\project8\\data\\"
expr_data = pd.read_csv(dir + "exp\\testexp_scep.csv")
gene = expr_data.iloc[:, :1]
gene = np.array(gene)
gene_list_A = gene.T[0].tolist()

original_data = pd.read_table(dir + "CDS\\cds_test_gene_filtered.txt",header=None, sep="\t",skiprows=1)
original_data.set_index(0, inplace=True)
original_data = original_data.reindex(gene_list_A)
original_data = np.array(original_data)
seq_data = original_data[:, 0]
seq_feature = []
for i in range(len(seq_data)):
    seq = [seq_data[i]]
    ad = len(seq[0])
    if len(seq[0]) > 1:
        k_mer = kmer_func(seq)
        k_mer = np.unique(k_mer)
        k_mers = []
        k_mers.append(k_mer)
        onehot = encode_matrix(k_mers)
        onehot = np.array(onehot)
        onehot = onehot.astype('float32')

        onehot = torch.tensor(onehot)
        rnn3 = nn.GRU(onehot.shape[2], 10, 1, bidirectional=True)

        h1 = torch.randn(2, onehot.shape[2], 10)
        c1 = torch.randn(2, onehot.shape[2], 10)

        output3, hn3 = rnn3(onehot, h1)
        bi_gru = output3[-1]
        bi_gru1 = bi_gru.detach().numpy()
        bi_gru_avg = bi_gru1.mean(axis=0)
        seq_feature.append(bi_gru_avg)
    else:
        non_geng = np.zeros(20)
        for j in range(len(non_geng)):
            non_geng[j] = math.pow(10, -99)
        seq_feature.append(non_geng)
seq_feature = np.array(seq_feature)
print(os.getcwd())
pd.DataFrame(seq_feature).to_csv( "D:\\R\\workspace\\work\\WZH毕设重构\\project8\\data\\CDS\\scRNA_cdstestscep_seq_feature_k4.csv", index=False)