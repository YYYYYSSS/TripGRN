import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ExpressionSampleSetEncoder(nn.Module):
    """
    变长表达输入编码器：
    - 输入：exp 表达向量，形状为 [B, T]（T = bulk 列数，可变）
    - 使用 1D 卷积在可变长度上提取局部模式
    - 通过全局池化得到固定维度向量

    说明：
    之前实现依赖 Transformer + 固定 pooled_len，并在 ablation（表达-only）场景下容易出现表达坍缩。
    Conv/Pooling 方案不依赖固定序列长度的 learned positional embedding，更利于泛化。
    """

    def __init__(
        self,
        out_dim: int = 160,
        d_model: int = 128,
        pooled_len: int = 512,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.d_model = d_model
        self.pooled_len = pooled_len
        self.gradient_checkpointing = bool(gradient_checkpointing)

        # 1D 卷积对可变长度表达向量建模
        self.stem = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res1 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.out_proj = nn.Sequential(
            nn.Linear(2 * d_model, out_dim),
        )

    def _conv_trunk(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, 1, T]"""
        h = self.stem(h)
        h = h + self.res1(h)
        h = h + self.res2(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T]
        return: [B, out_dim]
        """
        if x.dim() != 2:
            raise ValueError(f"ExpressionSampleSetEncoder expects x with shape [B, T], got {tuple(x.shape)}")

        # [B, T] -> [B, 1, T]
        h = x.unsqueeze(1)
        if self.gradient_checkpointing and self.training:
            h = checkpoint(self._conv_trunk, h, use_reentrant=False)
        else:
            h = self._conv_trunk(h)

        # 全局池化到固定向量
        avg = self.avg_pool(h).squeeze(-1)  # [B, d_model]
        mx = self.max_pool(h).squeeze(-1)   # [B, d_model]
        feat = torch.cat([avg, mx], dim=1)  # [B, 2*d_model]
        return self.out_proj(feat)

