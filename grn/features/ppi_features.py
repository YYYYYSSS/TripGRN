import os
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def read_ppi_gene_symbol_universe(ppi_file: str) -> Set[str]:
    """用于数据统计：PPI CSV 中 from / to 出现的基因符号并集。"""
    try:
        df = pd.read_csv(ppi_file, usecols=["from", "to"])
    except ValueError:
        df = pd.read_csv(ppi_file)[["from", "to"]]
    genes = set(df["from"].astype(str).str.strip()) | set(df["to"].astype(str).str.strip())
    genes.discard("")
    genes.discard("nan")
    return genes


def _edges_from_symbol_csv(ppi_file: str, gene_to_idx: Dict[str, int]) -> List[List[int]]:
    try:
        df = pd.read_csv(ppi_file, usecols=["from", "to"])
    except ValueError:
        df = pd.read_csv(ppi_file)[["from", "to"]]
    fg = df["from"].astype(str)
    tg = df["to"].astype(str)
    i0 = fg.map(gene_to_idx)
    i1 = tg.map(gene_to_idx)
    valid = i0.notna() & i1.notna()
    if not valid.any():
        return []
    e0 = i0[valid].astype(np.int64).to_numpy()
    e1 = i1[valid].astype(np.int64).to_numpy()
    return np.column_stack([e0, e1]).tolist()


class PPIGraphBuilder:
    """
    PPI 图结构（仅拓扑，忽略 combined_score）。
    `ppi_file` 宜为「普适」边表（如全人类 STRING 子集）；仅当 `from`/`to` 均落在当前
    `gene_list` 中的行会参与构图，因此换表达谱时无需换 PPI 文件。
    """

    def __init__(self, ppi_file: str, gene_list: List[str]):
        self.gene_list = gene_list
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(gene_list)}

        if not os.path.isfile(ppi_file):
            raise FileNotFoundError(f"PPI CSV 不存在: {ppi_file}")
        edges = _edges_from_symbol_csv(ppi_file, self.gene_to_idx)

        self.num_nodes = len(gene_list)
        if len(edges) > 0:
            self.edge_index = torch.tensor(edges, dtype=torch.long).t()

            adj = torch.zeros(self.num_nodes, self.num_nodes)
            adj[self.edge_index[0], self.edge_index[1]] = 1
            self.adj_matrix = (adj + adj.t()).clamp(0, 1)
        else:
            self.adj_matrix = torch.eye(self.num_nodes)
            self.edge_index = torch.tensor([], dtype=torch.long).t()

        self.norm_adj = self._normalize_adjacency(self.adj_matrix)

    def _normalize_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        """对称归一化，用于GNN消息传递"""
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        return D_inv_sqrt @ adj @ D_inv_sqrt


class GraphConvolutionLayer(nn.Module):
    """可学习的图卷积层 - 极简设计，专注拓扑"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = adj @ x
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        return self.dropout(x)


class PPINet(nn.Module):
    """轻量GNN - 3层结构，最后一层无激活保留更多信息"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 160):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GraphConvolutionLayer(input_dim, hidden_dim),
                GraphConvolutionLayer(hidden_dim, hidden_dim),
                GraphConvolutionLayer(hidden_dim, output_dim, dropout=0.0),
            ]
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x, adj)

        x = self.layers[-1].norm(self.layers[-1].linear(adj @ x))
        return x


class PPIFeatureExtractor:
    """PPI特征提取主接口 - 支持Inductive模式（修复数据泄露）"""

    def __init__(self, ppi_file: str, gene_list: List[str], emb_dim: int = 160, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 使用 ASCII 日志，避免 Windows 默认代码页下 UTF-8 源码被误读为乱码
        print(f"[PPI] loading graph: {ppi_file}")
        self.graph = PPIGraphBuilder(ppi_file, gene_list)
        print(
            f"[PPI] graph ready: nodes={self.graph.num_nodes}, edges={self.graph.edge_index.size(1)}"
        )

        self.node_features = torch.eye(len(gene_list)).to(self.device)

        self.ppi_net = PPINet(input_dim=len(gene_list), hidden_dim=emb_dim, output_dim=emb_dim).to(self.device)

        self.graph.norm_adj = self.graph.norm_adj.to(self.device)
        self.graph.adj_matrix = self.graph.adj_matrix.to(self.device)

        self._cached_features = None
        self._train_mask = None
        self._inductive_mode = False

    def set_train_mask(self, train_indices):
        """启用Inductive模式：设置训练集掩码，隔离测试集节点"""
        self._train_mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool, device=self.device)

        if not isinstance(train_indices, torch.Tensor):
            train_indices = torch.tensor(train_indices, dtype=torch.long)
        else:
            train_indices = train_indices.long()

        train_indices = train_indices.cpu()

        if train_indices.numel() > 0 and train_indices.max() >= self.graph.num_nodes:
            raise ValueError(f"训练集索引 {train_indices.max()} 超出基因总数 {self.graph.num_nodes}")

        self._train_mask[train_indices] = True
        self._inductive_mode = True
        self._cached_features = None
        print(f"[PPI Inductive] train mask on: {len(train_indices)}/{self.graph.num_nodes} genes")

    def clear_train_mask(self):
        """清除训练掩码，恢复Transductive模式（用于全量预测或交叉验证结束后）"""
        self._train_mask = None
        self._inductive_mode = False
        self._cached_features = None
        print("[PPI Inductive] train mask cleared (full graph)")

    def extract_node_features(self) -> torch.Tensor:
        """提取节点特征（默认 transductive 或已设置 inductive 训练模式）"""
        if self._cached_features is not None:
            return self._cached_features

        self.ppi_net.eval()
        with torch.no_grad():
            if self._inductive_mode and self._train_mask is not None:
                features = self._extract_inductive_train()
            else:
                features = self.ppi_net(self.node_features, self.graph.norm_adj)

            self._cached_features = features

        return self._cached_features

    def _extract_inductive_train(self) -> torch.Tensor:
        train_mask = self._train_mask
        full_adj = self.graph.adj_matrix

        row_mask = train_mask.unsqueeze(1).expand_as(full_adj)
        col_mask = train_mask.unsqueeze(0).expand_as(full_adj)
        sub_adj = torch.zeros_like(full_adj)
        sub_adj[row_mask & col_mask] = full_adj[row_mask & col_mask]

        sub_norm_adj = self._normalize_adjacency(sub_adj)
        all_features = self.ppi_net(self.node_features, sub_norm_adj)

        # 测试集节点特征置零（强制隔离）
        all_features[~train_mask] = 0.0
        return all_features

    def extract_inductive_inference(self, train_indices, test_indices) -> torch.Tensor:
        """
        Inductive推理特征提取：
        - 允许 train->train, train->test
        - 禁止 test->train, test->test（防止测试集互相泄露）
        """
        self.ppi_net.eval()

        train_indices = torch.tensor(train_indices, dtype=torch.long, device=self.device)
        test_indices = torch.tensor(test_indices, dtype=torch.long, device=self.device)

        train_mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool, device=self.device)
        test_mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool, device=self.device)
        train_mask[train_indices] = True
        test_mask[test_indices] = True

        full_adj = self.graph.adj_matrix
        inductive_adj = torch.zeros_like(full_adj)

        # 注意矩阵乘法语义：features = adj @ node_features
        # 因此 adj[i, j] 表示“i 聚合来自 j 的信息”。
        # 为了允许 train->test：应保留 adj[test, train]；
        # 为了禁止 test->train：应置零 adj[train, test]。
        row_train = train_mask.unsqueeze(1).expand_as(full_adj)
        col_train = train_mask.unsqueeze(0).expand_as(full_adj)
        inductive_adj[row_train & col_train] = full_adj[row_train & col_train]

        # 训练集 -> 测试集（test 行、train 列）
        inductive_adj[test_mask.unsqueeze(1) & train_mask.unsqueeze(0)] = full_adj[
            test_mask.unsqueeze(1) & train_mask.unsqueeze(0)
        ]

        norm_adj = self._normalize_adjacency(inductive_adj)
        with torch.no_grad():
            features = self.ppi_net(self.node_features, norm_adj)
        return features

    def _normalize_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        return D_inv_sqrt @ adj @ D_inv_sqrt

    def get_pair_features(self, tf_idx: int, target_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        all_features = self.extract_node_features()
        return all_features[tf_idx], all_features[target_idx]

    def refresh_cache(self):
        self._cached_features = None
