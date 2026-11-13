# -*- coding: utf-8 -*-
"""DataLoader helpers for TripGRN pair training (no model changes)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from grn.training.batch_size import suggest_batch_size, suggest_keras_batch_size

__all__ = [
    "pairs_to_ndarray",
    "suggest_batch_size",
    "suggest_keras_batch_size",
    "collate_pair_batch",
    "PairSubsetColumnDataset",
    "build_pair_dataloader",
]


def pairs_to_ndarray(
    pairs: Union[Sequence[Tuple[int, int, int]], np.ndarray],
) -> np.ndarray:
    if isinstance(pairs, np.ndarray):
        arr = np.ascontiguousarray(pairs.astype(np.int32, copy=False))
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("pairs ndarray must be (N, 3)")
        return arr
    return np.asarray(pairs, dtype=np.int32)


def collate_pair_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "tf": torch.stack([b["tf"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
        "seq_tf": torch.stack([b["seq_tf"] for b in batch], dim=0),
        "seq_target": torch.stack([b["seq_target"] for b in batch], dim=0),
        "ppi_tf": torch.stack([b["ppi_tf"] for b in batch], dim=0),
        "ppi_target": torch.stack([b["ppi_target"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
    }


class PairSubsetColumnDataset(Dataset):
    """Column-subset expression + cached seq/PPI lookups for TF-target pairs."""

    def __init__(
        self,
        exp_data: torch.Tensor,
        samples: Union[Sequence[Tuple[int, int, int]], np.ndarray],
        seq_features: np.ndarray,
        ppi_extractor=None,
        *,
        ppi_node_features: Optional[torch.Tensor] = None,
        device=None,
        use_ppi: bool = True,
        use_expression: bool = True,
        use_cds: bool = True,
        ppi_feat_dim: int = 160,
    ):
        self.exp_data = exp_data
        self._pair_array = pairs_to_ndarray(samples)
        self.seq_features_t = torch.as_tensor(seq_features, dtype=torch.float32)
        self.seq_dim = int(self.seq_features_t.shape[1]) if self.seq_features_t.ndim == 2 else 20
        self.ppi_extractor = ppi_extractor
        self.ppi_node_features = ppi_node_features
        self.ppi_feat_dim = int(ppi_feat_dim)
        self.device = device
        self.use_ppi = use_ppi
        self.use_expression = use_expression
        self.use_cds = use_cds
        self._zero_seq = torch.zeros(self.seq_dim, dtype=torch.float32)
        self._zero_ppi = torch.zeros(self.ppi_feat_dim, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self._pair_array.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        i = int(self._pair_array[idx, 0])
        j = int(self._pair_array[idx, 1])
        label = int(self._pair_array[idx, 2])

        if self.use_expression:
            tf_e = self.exp_data[i]
            tg_e = self.exp_data[j]
        else:
            tf_e = torch.zeros_like(self.exp_data[i])
            tg_e = torch.zeros_like(self.exp_data[j])

        if self.use_cds:
            seq_tf = self.seq_features_t[i]
            seq_target = self.seq_features_t[j]
        else:
            seq_tf = self._zero_seq
            seq_target = self._zero_seq

        if self.use_ppi:
            if self.ppi_node_features is not None:
                ppi_tf = self.ppi_node_features[i]
                ppi_target = self.ppi_node_features[j]
            else:
                ppi_tf, ppi_target = self.ppi_extractor.get_pair_features(i, j)
        else:
            ppi_tf = self._zero_ppi
            ppi_target = self._zero_ppi

        sample = {
            "tf": tf_e,
            "target": tg_e,
            "seq_tf": seq_tf,
            "seq_target": seq_target,
            "ppi_tf": ppi_tf,
            "ppi_target": ppi_target,
            "label": torch.tensor(label, dtype=torch.long),
            # gene indices for case-study name recovery (ignored by training collate)
            "tf_idx": torch.tensor(i, dtype=torch.long),
            "target_idx": torch.tensor(j, dtype=torch.long),
        }
        if self.device:
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    sample[k] = v.to(self.device, non_blocking=True)
        return sample


def build_pair_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    sampler: Optional[WeightedRandomSampler] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "collate_fn": collate_pair_batch,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
        kwargs["shuffle"] = False
    else:
        kwargs["shuffle"] = shuffle
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)
