"""
E7: 无 PPI，仅表达 + CDS；Stage-B 在融合表征上注入 Stage-A 概率并做门控，利于 Repress/Activate 分界。
"""

import torch
import torch.nn as nn

from grn.models.expression_set_encoder import ExpressionSampleSetEncoder
from grn.models.resnet import ResNetBranch


class NoPIBackbone(nn.Module):
    """表达×2 + CDS×2 + 成对交互特征 -> fc_fusion -> [B, emb_dim]"""

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
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.exp_encoder_tf = ExpressionSampleSetEncoder(
            out_dim=emb_dim,
            d_model=expr_d_model,
            pooled_len=expr_pooled_len,
            nhead=expr_nhead,
            num_layers=expr_num_layers,
            dim_ff=expr_dim_ff,
            dropout=expr_dropout,
        )
        self.exp_encoder_target = ExpressionSampleSetEncoder(
            out_dim=emb_dim,
            d_model=expr_d_model,
            pooled_len=expr_pooled_len,
            nhead=expr_nhead,
            num_layers=expr_num_layers,
            dim_ff=expr_dim_ff,
            dropout=expr_dropout,
        )
        self.seq_branch_tf = ResNetBranch(seq_dim, output_dim=emb_dim)
        self.seq_branch_target = ResNetBranch(seq_dim, output_dim=emb_dim)
        # 额外加入 pair-level 交互：|tf-target| 与 tf*target（表达分支 + CDS 分支）
        fusion_input_dim = 8 * emb_dim
        self.fc_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, emb_dim),
        )

    def forward(self, exp_tf: torch.Tensor, exp_target: torch.Tensor, seq_tf: torch.Tensor, seq_target: torch.Tensor):
        exp_tf_feat = self.exp_encoder_tf(exp_tf)
        exp_target_feat = self.exp_encoder_target(exp_target)
        seq_tf_feat = self.seq_branch_tf(seq_tf)
        seq_target_feat = self.seq_branch_target(seq_target)
        exp_diff = torch.abs(exp_tf_feat - exp_target_feat)
        exp_mul = exp_tf_feat * exp_target_feat
        seq_diff = torch.abs(seq_tf_feat - seq_target_feat)
        seq_mul = seq_tf_feat * seq_target_feat
        features = torch.cat(
            [
                exp_tf_feat,
                exp_target_feat,
                seq_tf_feat,
                seq_target_feat,
                exp_diff,
                exp_mul,
                seq_diff,
                seq_mul,
            ],
            dim=1,
        )
        return self.fc_fusion(features)


def _classifier_head(emb_dim: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=16, padding=8),
        nn.BatchNorm1d(16),
        nn.MaxPool1d(5, padding=2),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(16, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )


class FusionModelNoPPI(nn.Module):
    """NoReg vs Reg 或独立二分类；无 PPI。"""

    def __init__(
        self,
        seq_dim: int = 20,
        emb_dim: int = 160,
        num_classes: int = 2,
        expr_d_model: int = 128,
        expr_pooled_len: int = 512,
        expr_nhead: int = 4,
        expr_num_layers: int = 2,
        expr_dim_ff: int = 256,
        expr_dropout: float = 0.1,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = NoPIBackbone(
            seq_dim=seq_dim,
            emb_dim=emb_dim,
            expr_d_model=expr_d_model,
            expr_pooled_len=expr_pooled_len,
            expr_nhead=expr_nhead,
            expr_num_layers=expr_num_layers,
            expr_dim_ff=expr_dim_ff,
            expr_dropout=expr_dropout,
            fusion_hidden_dim=fusion_hidden_dim,
            fusion_dropout=fusion_dropout,
        )
        self.classifier = _classifier_head(emb_dim, num_classes)

    def forward(
        self,
        exp_tf: torch.Tensor,
        exp_target: torch.Tensor,
        seq_tf: torch.Tensor,
        seq_target: torch.Tensor,
        return_embedding: bool = False,
    ):
        h = self.backbone(exp_tf, exp_target, seq_tf, seq_target)
        if return_embedding:
            return h
        return self.classifier(h.unsqueeze(1))


class CascadeStageBModel(nn.Module):
    """
    Repress(0) vs Activate(1)。独立 backbone + Stage-A 概率门控残差。
    forward 需传入 Stage-A 的 logits（训练时建议 detach）。
    """

    def __init__(
        self,
        seq_dim: int = 20,
        emb_dim: int = 160,
        num_classes: int = 2,
        expr_d_model: int = 128,
        expr_pooled_len: int = 512,
        expr_nhead: int = 4,
        expr_num_layers: int = 2,
        expr_dim_ff: int = 256,
        expr_dropout: float = 0.1,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = NoPIBackbone(
            seq_dim=seq_dim,
            emb_dim=emb_dim,
            expr_d_model=expr_d_model,
            expr_pooled_len=expr_pooled_len,
            expr_nhead=expr_nhead,
            expr_num_layers=expr_num_layers,
            expr_dim_ff=expr_dim_ff,
            expr_dropout=expr_dropout,
            fusion_hidden_dim=fusion_hidden_dim,
            fusion_dropout=fusion_dropout,
        )
        self.gate_net = nn.Sequential(
            nn.Linear(emb_dim + 2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, emb_dim),
            nn.Sigmoid(),
        )
        self.classifier = _classifier_head(emb_dim, num_classes)

    def forward(
        self,
        exp_tf: torch.Tensor,
        exp_target: torch.Tensor,
        seq_tf: torch.Tensor,
        seq_target: torch.Tensor,
        logits_stage_a: torch.Tensor,
    ):
        h = self.backbone(exp_tf, exp_target, seq_tf, seq_target)
        p_a = torch.softmax(logits_stage_a, dim=1)
        z = torch.cat([h, p_a], dim=1)
        g = self.gate_net(z)
        h2 = h * g + h
        return self.classifier(h2.unsqueeze(1))


class PPIBackbone(nn.Module):
    """表达+CDS+PPI(节点级) -> [B, emb_dim]"""

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
        ppi_emb_dim: int = 160,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.3,
    ):
        super().__init__()
        self.exp_encoder_tf = ExpressionSampleSetEncoder(
            out_dim=emb_dim,
            d_model=expr_d_model,
            pooled_len=expr_pooled_len,
            nhead=expr_nhead,
            num_layers=expr_num_layers,
            dim_ff=expr_dim_ff,
            dropout=expr_dropout,
        )
        self.exp_encoder_target = ExpressionSampleSetEncoder(
            out_dim=emb_dim,
            d_model=expr_d_model,
            pooled_len=expr_pooled_len,
            nhead=expr_nhead,
            num_layers=expr_num_layers,
            dim_ff=expr_dim_ff,
            dropout=expr_dropout,
        )

        self.seq_branch_tf = ResNetBranch(seq_dim, output_dim=emb_dim)
        self.seq_branch_target = ResNetBranch(seq_dim, output_dim=emb_dim)
        self.ppi_branch_tf = ResNetBranch(ppi_emb_dim, output_dim=emb_dim)
        self.ppi_branch_target = ResNetBranch(ppi_emb_dim, output_dim=emb_dim)

        fusion_input_dim = 6 * emb_dim
        self.fc_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, emb_dim),
        )

    def forward(
        self,
        exp_tf: torch.Tensor,
        exp_target: torch.Tensor,
        seq_tf: torch.Tensor,
        seq_target: torch.Tensor,
        ppi_tf: torch.Tensor,
        ppi_target: torch.Tensor,
    ) -> torch.Tensor:
        exp_tf_feat = self.exp_encoder_tf(exp_tf)
        exp_target_feat = self.exp_encoder_target(exp_target)
        seq_tf_feat = self.seq_branch_tf(seq_tf)
        seq_target_feat = self.seq_branch_target(seq_target)
        ppi_tf_feat = self.ppi_branch_tf(ppi_tf)
        ppi_target_feat = self.ppi_branch_target(ppi_target)

        features = torch.cat(
            [
                exp_tf_feat,
                exp_target_feat,
                seq_tf_feat,
                seq_target_feat,
                ppi_tf_feat,
                ppi_target_feat,
            ],
            dim=1,
        )
        return self.fc_fusion(features)


class CascadeStageBModelWithPPI(nn.Module):
    """
    Repress(0) vs Activate(1)：
    backbone 加回 PPI（node-level）并在融合表征上注入 Stage-A 概率门控。
    """

    def __init__(
        self,
        seq_dim: int = 20,
        emb_dim: int = 160,
        num_classes: int = 2,
        expr_d_model: int = 128,
        expr_pooled_len: int = 512,
        expr_nhead: int = 4,
        expr_num_layers: int = 2,
        expr_dim_ff: int = 256,
        expr_dropout: float = 0.1,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.3,
        ppi_emb_dim: int = 160,
    ):
        super().__init__()
        self.backbone = PPIBackbone(
            seq_dim=seq_dim,
            emb_dim=emb_dim,
            expr_d_model=expr_d_model,
            expr_pooled_len=expr_pooled_len,
            expr_nhead=expr_nhead,
            expr_num_layers=expr_num_layers,
            expr_dim_ff=expr_dim_ff,
            expr_dropout=expr_dropout,
            ppi_emb_dim=ppi_emb_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            fusion_dropout=fusion_dropout,
        )
        self.gate_net = nn.Sequential(
            nn.Linear(emb_dim + 2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, emb_dim),
            nn.Sigmoid(),
        )
        self.classifier = _classifier_head(emb_dim, num_classes)

    def forward(
        self,
        exp_tf: torch.Tensor,
        exp_target: torch.Tensor,
        seq_tf: torch.Tensor,
        seq_target: torch.Tensor,
        ppi_tf: torch.Tensor,
        ppi_target: torch.Tensor,
        logits_stage_a: torch.Tensor,
    ) -> torch.Tensor:
        h = self.backbone(
            exp_tf=exp_tf,
            exp_target=exp_target,
            seq_tf=seq_tf,
            seq_target=seq_target,
            ppi_tf=ppi_tf,
            ppi_target=ppi_target,
        )
        p_a = torch.softmax(logits_stage_a, dim=1)
        z = torch.cat([h, p_a], dim=1)
        g = self.gate_net(z)
        h2 = h * g + h
        return self.classifier(h2.unsqueeze(1))
