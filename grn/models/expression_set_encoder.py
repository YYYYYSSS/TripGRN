"""Expression encoders for TripGRN.

Default front-end is the permutation-invariant VAE (`ExpressionVAEEncoder`).
The previous Conv1d encoder is kept as `ExpressionConvEncoder` for ablation /
rollback (see backups/20261010_pre_vae_gene_disjoint/).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from grn.models.expression_vae_encoder import ExpressionVAEEncoder

# Public name used by FusionModel / schematics: now the VAE.
ExpressionSampleSetEncoder = ExpressionVAEEncoder

__all__ = [
    "ExpressionSampleSetEncoder",
    "ExpressionVAEEncoder",
    "ExpressionConvEncoder",
]


class ExpressionConvEncoder(nn.Module):
    """
    Legacy 1D-CNN expression encoder (pre-VAE).
    Input: [B, T]; output: [B, out_dim].
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
        del pooled_len, nhead, num_layers, dim_ff
        self.out_dim = out_dim
        self.d_model = d_model
        self.gradient_checkpointing = bool(gradient_checkpointing)

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
        h = self.stem(h)
        h = h + self.res1(h)
        h = h + self.res2(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"ExpressionConvEncoder expects x with shape [B, T], got {tuple(x.shape)}")

        h = x.unsqueeze(1)
        if self.gradient_checkpointing and self.training:
            h = checkpoint(self._conv_trunk, h, use_reentrant=False)
        else:
            h = self._conv_trunk(h)

        avg = self.avg_pool(h).squeeze(-1)
        mx = self.max_pool(h).squeeze(-1)
        feat = torch.cat([avg, mx], dim=1)
        return self.out_proj(feat)
