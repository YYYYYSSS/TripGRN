import torch
import torch.nn as nn

from grn.models.expression_set_encoder import ExpressionConvEncoder
from grn.models.expression_vae_encoder import ExpressionVAEEncoder
from grn.models.resnet import ResNetBranch


def _build_expression_encoder(
    kind: str,
    *,
    emb_dim: int,
    expr_d_model: int,
    expr_pooled_len: int,
    expr_nhead: int,
    expr_num_layers: int,
    expr_dim_ff: int,
    expr_dropout: float,
    expr_latent_dim: int,
    gradient_checkpointing: bool = False,
):
    """kind: 'vae' (default / Fig.1–2) or 'conv' (legacy CNN / Fig.3 original)."""
    k = str(kind).strip().lower()
    if k in ("conv", "cnn", "legacy"):
        return ExpressionConvEncoder(
            out_dim=emb_dim,
            d_model=expr_d_model,
            pooled_len=expr_pooled_len,
            nhead=expr_nhead,
            num_layers=expr_num_layers,
            dim_ff=expr_dim_ff,
            dropout=expr_dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
    if k not in ("vae", "expressionvae", "expression_vae"):
        raise ValueError(f"Unknown expression_encoder={kind!r}; use 'vae' or 'conv'")
    return ExpressionVAEEncoder(
        out_dim=emb_dim,
        d_model=expr_d_model,
        latent_dim=expr_latent_dim,
        pooled_len=expr_pooled_len,
        nhead=expr_nhead,
        num_layers=expr_num_layers,
        dim_ff=expr_dim_ff,
        dropout=expr_dropout,
        gradient_checkpointing=gradient_checkpointing,
    )


class FusionModel(nn.Module):
    def __init__(
        self,
        seq_dim: int = 20,
        emb_dim: int = 160,
        num_classes: int = 3,
        expr_d_model: int = 128,
        expr_pooled_len: int = 512,
        expr_nhead: int = 4,
        expr_num_layers: int = 2,
        expr_dim_ff: int = 256,
        expr_dropout: float = 0.1,
        expr_latent_dim: int = 64,
        ppi_emb_dim: int = 160,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.3,
        expression_encoder: str = "conv",
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.expression_encoder = str(expression_encoder).strip().lower()

        # 表达分支：vae（Fig.1–2）或 conv（原始 CNN）
        self.exp_encoder_tf = _build_expression_encoder(
            self.expression_encoder,
            emb_dim=emb_dim,
            expr_d_model=expr_d_model,
            expr_pooled_len=expr_pooled_len,
            expr_nhead=expr_nhead,
            expr_num_layers=expr_num_layers,
            expr_dim_ff=expr_dim_ff,
            expr_dropout=expr_dropout,
            expr_latent_dim=expr_latent_dim,
        )
        self.exp_encoder_target = _build_expression_encoder(
            self.expression_encoder,
            emb_dim=emb_dim,
            expr_d_model=expr_d_model,
            expr_pooled_len=expr_pooled_len,
            expr_nhead=expr_nhead,
            expr_num_layers=expr_num_layers,
            expr_dim_ff=expr_dim_ff,
            expr_dropout=expr_dropout,
            expr_latent_dim=expr_latent_dim,
        )

        # CDS分支（定长 20-dim）
        self.seq_branch_tf = ResNetBranch(seq_dim, output_dim=emb_dim)
        self.seq_branch_target = ResNetBranch(seq_dim, output_dim=emb_dim)

        # PPI分支（定长 160-dim）
        self.ppi_branch_tf = ResNetBranch(ppi_emb_dim, output_dim=emb_dim)
        self.ppi_branch_target = ResNetBranch(ppi_emb_dim, output_dim=emb_dim)

        # 融合 + 分类头
        fusion_input_dim = 6 * emb_dim
        self.fc_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, emb_dim),
        )

        self.classifier = nn.Sequential(
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

    def forward(self, exp_tf, exp_target, seq_tf, seq_target, ppi_tf, ppi_target):
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
        combined = self.fc_fusion(features)
        combined = combined.unsqueeze(1)  # [B, 1, emb_dim]
        return self.classifier(combined)

    def vae_aux_losses(self):
        """Mean KL / recon from TF+target expression VAEs (zeros if unavailable / conv)."""
        if not hasattr(self.exp_encoder_tf, "vae_losses"):
            z = torch.tensor(0.0)
            return z, z
        kl_t, rec_t = self.exp_encoder_tf.vae_losses()
        kl_g, rec_g = self.exp_encoder_target.vae_losses()
        return 0.5 * (kl_t + kl_g), 0.5 * (rec_t + rec_g)


class FusionModelAblation(nn.Module):
    """
    Same fusion head as FusionModel, but each modality (expression / CDS / PPI) can be disabled.
    Disabled branches are not instantiated; fusion input dim shrinks accordingly (2 * emb_dim per enabled modality).
    """

    def __init__(
        self,
        seq_dim: int = 20,
        emb_dim: int = 160,
        num_classes: int = 3,
        use_expression: bool = True,
        use_cds: bool = True,
        use_ppi: bool = True,
        expr_d_model: int = 128,
        expr_pooled_len: int = 512,
        expr_nhead: int = 4,
        expr_num_layers: int = 2,
        expr_dim_ff: int = 256,
        expr_dropout: float = 0.1,
        expr_latent_dim: int = 64,
        ppi_emb_dim: int = 160,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.3,
        expression_gradient_checkpointing: bool = False,
        expression_encoder: str = "conv",
    ):
        super().__init__()
        if not (use_expression or use_cds or use_ppi):
            raise ValueError("At least one of use_expression, use_cds, use_ppi must be True.")
        self.emb_dim = emb_dim
        self.use_expression = use_expression
        self.use_cds = use_cds
        self.use_ppi = use_ppi
        self.expression_encoder = str(expression_encoder).strip().lower()

        if use_expression:
            self.exp_encoder_tf = _build_expression_encoder(
                self.expression_encoder,
                emb_dim=emb_dim,
                expr_d_model=expr_d_model,
                expr_pooled_len=expr_pooled_len,
                expr_nhead=expr_nhead,
                expr_num_layers=expr_num_layers,
                expr_dim_ff=expr_dim_ff,
                expr_dropout=expr_dropout,
                expr_latent_dim=expr_latent_dim,
                gradient_checkpointing=expression_gradient_checkpointing,
            )
            self.exp_encoder_target = _build_expression_encoder(
                self.expression_encoder,
                emb_dim=emb_dim,
                expr_d_model=expr_d_model,
                expr_pooled_len=expr_pooled_len,
                expr_nhead=expr_nhead,
                expr_num_layers=expr_num_layers,
                expr_dim_ff=expr_dim_ff,
                expr_dropout=expr_dropout,
                expr_latent_dim=expr_latent_dim,
                gradient_checkpointing=expression_gradient_checkpointing,
            )
        else:
            self.exp_encoder_tf = None
            self.exp_encoder_target = None

        if use_cds:
            self.seq_branch_tf = ResNetBranch(seq_dim, output_dim=emb_dim)
            self.seq_branch_target = ResNetBranch(seq_dim, output_dim=emb_dim)
        else:
            self.seq_branch_tf = None
            self.seq_branch_target = None

        if use_ppi:
            self.ppi_branch_tf = ResNetBranch(ppi_emb_dim, output_dim=emb_dim)
            self.ppi_branch_target = ResNetBranch(ppi_emb_dim, output_dim=emb_dim)
        else:
            self.ppi_branch_tf = None
            self.ppi_branch_target = None

        n_blocks = 2 * (int(use_expression) + int(use_cds) + int(use_ppi))
        fusion_input_dim = n_blocks * emb_dim
        self.fc_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, emb_dim),
        )

        self.classifier = nn.Sequential(
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

    def forward(self, exp_tf, exp_target, seq_tf, seq_target, ppi_tf, ppi_target):
        parts = []
        if self.use_expression:
            parts.append(self.exp_encoder_tf(exp_tf))
            parts.append(self.exp_encoder_target(exp_target))
        if self.use_cds:
            parts.append(self.seq_branch_tf(seq_tf))
            parts.append(self.seq_branch_target(seq_target))
        if self.use_ppi:
            parts.append(self.ppi_branch_tf(ppi_tf))
            parts.append(self.ppi_branch_target(ppi_target))
        features = torch.cat(parts, dim=1)
        combined = self.fc_fusion(features)
        combined = combined.unsqueeze(1)
        return self.classifier(combined)

    def vae_aux_losses(self):
        """Mean KL / recon from enabled expression VAEs; (0,0) if expression off / conv."""
        if not self.use_expression or self.exp_encoder_tf is None:
            z = torch.tensor(0.0)
            return z, z
        if not hasattr(self.exp_encoder_tf, "vae_losses"):
            z = torch.tensor(0.0)
            return z, z
        kl_t, rec_t = self.exp_encoder_tf.vae_losses()
        kl_g, rec_g = self.exp_encoder_target.vae_losses()
        return 0.5 * (kl_t + kl_g), 0.5 * (rec_t + rec_g)

