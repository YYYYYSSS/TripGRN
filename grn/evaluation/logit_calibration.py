"""
多分类事后校准：在验证集 logits 上拟合标量温度 T 与每类偏置 b，使 softmax(logits/T + b) 的 NLL 最小。
用于跨域评估时缓解「几乎全预测多数类」而 AUC 仍尚可的现象。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature_and_bias(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int = 600,
    lr: float = 0.08,
    class_weights: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> Tuple[float, np.ndarray]:
    """
    返回 (T, bias)，T>0，bias shape (C,)。预测时使用 softmax(logits / T + bias)。

    class_weights: shape (C,) 传入 CrossEntropy 的 weight（例如按验证集类别频率的逆权重）。
    """
    dev = device or torch.device("cpu")
    Z = torch.from_numpy(np.asarray(logits, dtype=np.float32)).to(dev)
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(dev)
    n_cls = int(Z.shape[1])
    log_T = torch.nn.Parameter(torch.zeros(1, device=dev))
    bias = torch.nn.Parameter(torch.zeros(n_cls, device=dev))
    w_tensor = None
    if class_weights is not None:
        w_tensor = torch.from_numpy(np.asarray(class_weights, dtype=np.float32)).to(dev)

    opt = torch.optim.Adam([log_T, bias], lr=lr)
    for _ in range(int(steps)):
        opt.zero_grad()
        T = torch.exp(log_T).clamp(min=1e-4, max=1e4)
        z = Z / T + bias
        loss = F.cross_entropy(z, y, weight=w_tensor)
        loss.backward()
        opt.step()

    T_final = float(torch.exp(log_T).clamp(min=1e-4, max=1e4).item())
    b_final = bias.detach().cpu().numpy().astype(np.float32)
    return T_final, b_final


def apply_temperature_and_bias(logits: np.ndarray, T: float, bias: np.ndarray) -> np.ndarray:
    """logits: (N, C)；bias: (C,)"""
    z = np.asarray(logits, dtype=np.float32) / float(T) + np.asarray(bias, dtype=np.float32).reshape(1, -1)
    ex = np.exp(z - z.max(axis=1, keepdims=True))
    return ex / ex.sum(axis=1, keepdims=True)


def val_class_inverse_weights(labels: np.ndarray, n_classes: int = 3) -> np.ndarray:
    cnt = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes).astype(np.float64)
    w = 1.0 / (cnt + 1e-6)
    w = w * (n_classes / w.sum())
    return w.astype(np.float32)
