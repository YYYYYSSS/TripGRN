from dataclasses import dataclass


@dataclass
class ClassifierConfig:
    batch_size: int = 32
    epochs: int = 150
    lr: float = 1e-3
    patience: int = 5
    emb_dim: int = 160
    num_classes: int = 3

    # Expression encoder (变长表达编码)
    expr_d_model: int = 128
    expr_pooled_len: int = 512
    expr_nhead: int = 4
    expr_num_layers: int = 2
    expr_dim_ff: int = 256
    expr_dropout: float = 0.1

    # Dataset
    neg_sample_ratio: float = 1.0

    # CV
    n_splits: int = 5
    random_state: int = 42

    # Validation split inside training fold
    val_ratio: float = 0.2

