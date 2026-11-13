"""Permutation-invariant VAE encoder for per-gene expression across cells/samples."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ExpressionVAEEncoder(nn.Module):
    """
    DeepSets-style VAE over the cell/sample set of one gene.

    Input x: [B, T] (T = number of cells/samples, variable).
    Each cell is an element of a set → shared MLP → mean pool (permutation-invariant)
    → μ, logσ² → latent z → out_dim embedding.

    Also produces a light reconstruction of set statistics (mean/std of x) for the
    VAE training signal. Callers can read ``last_kl`` / ``last_recon`` after forward.
    """

    def __init__(
        self,
        out_dim: int = 160,
        d_model: int = 128,
        latent_dim: int = 64,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        # Accepted for FusionModel API compatibility with the old CNN encoder:
        pooled_len: int = 512,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        **_ignored,
    ):
        super().__init__()
        del pooled_len, nhead, num_layers, dim_ff  # unused (compat only)
        self.out_dim = int(out_dim)
        self.d_model = int(d_model)
        self.latent_dim = int(latent_dim)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        # Per-cell encoder (scalar expression → d_model)
        self.cell_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)
        self.z_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )
        # Reconstruct empirical (mean, std) of the set from z
        self.decoder_stats = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

        self.last_kl: Optional[torch.Tensor] = None
        self.last_recon: Optional[torch.Tensor] = None
        self.last_mu: Optional[torch.Tensor] = None
        self.last_logvar: Optional[torch.Tensor] = None

    def _encode_pool(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T] → pooled [B, d_model]"""
        h = self.cell_mlp(x.unsqueeze(-1))  # [B, T, d_model]
        return h.mean(dim=1)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar.clamp(-20.0, 20.0))
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T]
        return: [B, out_dim]
        """
        if x.dim() != 2:
            raise ValueError(f"ExpressionVAEEncoder expects x with shape [B, T], got {tuple(x.shape)}")

        if self.gradient_checkpointing and self.training:
            pooled = checkpoint(self._encode_pool, x, use_reentrant=False)
        else:
            pooled = self._encode_pool(x)

        mu = self.mu_head(pooled)
        logvar = self.logvar_head(pooled).clamp(-20.0, 20.0)
        z = self._reparameterize(mu, logvar) if self.training else mu

        # KL(q(z|x) || N(0,I)), mean over batch
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

        # Reconstruction of set statistics (permutation-invariant targets)
        emp_mean = x.mean(dim=1)
        emp_std = x.std(dim=1, unbiased=False).clamp_min(1e-6)
        pred_stats = self.decoder_stats(z)
        recon = F.mse_loss(pred_stats[:, 0], emp_mean) + F.mse_loss(pred_stats[:, 1], emp_std)

        self.last_kl = kl
        self.last_recon = recon
        self.last_mu = mu
        self.last_logvar = logvar
        return self.z_proj(z)

    def vae_losses(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (kl, recon) from the last forward; zeros if never called."""
        zero = torch.tensor(0.0)
        kl = self.last_kl if self.last_kl is not None else zero
        recon = self.last_recon if self.last_recon is not None else zero
        return kl, recon
