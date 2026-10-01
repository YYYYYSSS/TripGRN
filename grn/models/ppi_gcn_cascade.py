import torch
import torch.nn as nn
import torch.nn.functional as F

from grn.models.expression_set_encoder import ExpressionSampleSetEncoder
from grn.models.resnet import ResNetBranch


def build_inductive_message_edges(
    base_edge_index: torch.Tensor,
    num_nodes: int,
    train_nodes: torch.Tensor,
    eval_nodes: torch.Tensor = None,
    mode: str = "train",
    add_self_loops: bool = True,
):
    """
    构建稀疏消息边（receiver <- sender）。
    - mode=train: 仅 train<-train
    - mode=eval: train<-train + eval<-train
    返回: src(接收端), dst(发送端), norm(对称归一化权重)
    """
    if mode not in ["train", "eval"]:
        raise ValueError(f"unknown mode={mode}")

    device = base_edge_index.device
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[train_nodes.long()] = True

    src = base_edge_index[0].long()
    dst = base_edge_index[1].long()

    if mode == "train":
        keep = train_mask[src] & train_mask[dst]
    else:
        if eval_nodes is None:
            raise ValueError("eval mode needs eval_nodes")
        eval_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        eval_mask[eval_nodes.long()] = True
        keep = (train_mask[src] & train_mask[dst]) | (eval_mask[src] & train_mask[dst])

    src = src[keep]
    dst = dst[keep]

    if add_self_loops:
        if mode == "train":
            loop_nodes = torch.where(train_mask)[0]
        else:
            eval_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            eval_mask[eval_nodes.long()] = True
            loop_nodes = torch.where(train_mask | eval_mask)[0]
        src = torch.cat([src, loop_nodes], dim=0)
        dst = torch.cat([dst, loop_nodes], dim=0)

    deg_src = torch.bincount(src, minlength=num_nodes).float().clamp_min(1.0)
    deg_dst = torch.bincount(dst, minlength=num_nodes).float().clamp_min(1.0)
    norm = (deg_src[src] * deg_dst[dst]).rsqrt()
    return src, dst, norm


class SparseGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, edge_norm: torch.Tensor):
        msg = x[dst] * edge_norm.unsqueeze(1)
        out = torch.zeros_like(x)
        out.index_add_(0, src, msg)
        out = self.linear(out)
        out = self.norm(out)
        out = F.relu(out)
        return self.dropout(out)


class NodeFeatureEncoder(nn.Module):
    """
    将 gene 的表达向量+CDS 向量编码为节点初始表征。
    """

    def __init__(
        self,
        seq_dim: int = 20,
        emb_dim: int = 160,
        expr_d_model: int = 128,
        expr_pooled_len: int = 512,
        expr_nhead: int = 4,
        expr_num_layers: int = 2,
        expr_dim_ff: int = 256,
        expr_dropout: float = 0.1,
    ):
        super().__init__()
        self.exp_encoder = ExpressionSampleSetEncoder(
            out_dim=emb_dim,
            d_model=expr_d_model,
            pooled_len=expr_pooled_len,
            nhead=expr_nhead,
            num_layers=expr_num_layers,
            dim_ff=expr_dim_ff,
            dropout=expr_dropout,
        )
        self.seq_encoder = ResNetBranch(seq_dim, output_dim=emb_dim)
        self.fuse = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, exp_all: torch.Tensor, cds_all: torch.Tensor, chunk_size: int = 512):
        n = exp_all.size(0)
        exp_chunks = []
        seq_chunks = []
        for st in range(0, n, chunk_size):
            ed = min(n, st + chunk_size)
            exp_chunks.append(self.exp_encoder(exp_all[st:ed]))
            seq_chunks.append(self.seq_encoder(cds_all[st:ed]))
        exp_feat = torch.cat(exp_chunks, dim=0)
        seq_feat = torch.cat(seq_chunks, dim=0)
        return self.fuse(torch.cat([exp_feat, seq_feat], dim=1))


class StageBHeadCascade(nn.Module):
    def __init__(self, edge_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(edge_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, edge_feat: torch.Tensor, logits_stage_a: torch.Tensor):
        p_a = torch.softmax(logits_stage_a, dim=1)
        g = self.gate(torch.cat([edge_feat, p_a], dim=1))
        z = edge_feat * g + edge_feat
        return self.classifier(z)


class PpiGcnTwoStageCascade(nn.Module):
    """
    全图 GCN + 级联分类：
    - 共享节点编码器（表达+CDS -> GCN）
    - Stage-A: NoReg vs Reg
    - Stage-B: Repress vs Activate（接收 Stage-A logits 门控）
    """

    def __init__(
        self,
        seq_dim: int = 20,
        emb_dim: int = 160,
        expr_d_model: int = 128,
        expr_pooled_len: int = 512,
        expr_nhead: int = 4,
        expr_num_layers: int = 2,
        expr_dim_ff: int = 256,
        expr_dropout: float = 0.1,
        gcn_layers: int = 2,
        gcn_dropout: float = 0.1,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.node_encoder = NodeFeatureEncoder(
            seq_dim=seq_dim,
            emb_dim=emb_dim,
            expr_d_model=expr_d_model,
            expr_pooled_len=expr_pooled_len,
            expr_nhead=expr_nhead,
            expr_num_layers=expr_num_layers,
            expr_dim_ff=expr_dim_ff,
            expr_dropout=expr_dropout,
        )
        self.gcn_layers = nn.ModuleList(
            [SparseGCNLayer(emb_dim, emb_dim, dropout=gcn_dropout) for _ in range(gcn_layers)]
        )
        edge_dim = 4 * emb_dim
        self.stage_a_head = nn.Sequential(
            nn.Linear(edge_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 2),
        )
        self.stage_b_head = StageBHeadCascade(edge_dim=edge_dim, hidden_dim=head_hidden_dim, dropout=head_dropout)

    @staticmethod
    def _edge_features(h_tf: torch.Tensor, h_tg: torch.Tensor):
        return torch.cat([h_tf, h_tg, torch.abs(h_tf - h_tg), h_tf * h_tg], dim=1)

    def encode_nodes(
        self,
        exp_all: torch.Tensor,
        cds_all: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_norm: torch.Tensor,
        chunk_size: int = 512,
    ):
        x = self.node_encoder(exp_all, cds_all, chunk_size=chunk_size)
        for layer in self.gcn_layers:
            x = layer(x, src=src, dst=dst, edge_norm=edge_norm)
        return x

    def stage_a_logits(self, h_nodes: torch.Tensor, tf_idx: torch.Tensor, target_idx: torch.Tensor):
        e = self._edge_features(h_nodes[tf_idx], h_nodes[target_idx])
        return self.stage_a_head(e)

    def stage_b_logits(
        self,
        h_nodes: torch.Tensor,
        tf_idx: torch.Tensor,
        target_idx: torch.Tensor,
        logits_stage_a: torch.Tensor,
    ):
        e = self._edge_features(h_nodes[tf_idx], h_nodes[target_idx])
        return self.stage_b_head(e, logits_stage_a=logits_stage_a)

