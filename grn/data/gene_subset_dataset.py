import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from grn.features.ppi_features import PPIFeatureExtractor


class GRNGeneSubsetDataset(Dataset):
    """
    基因不相交约束的数据集：
    - 仅允许 (TF, Target) 对同时落在 allowed_gene_indices 内
    - 正样本来自 GRN_matrix[i, j] in {1, 2}
    - 负样本来自 GRN_matrix[i, j] == 0（在 allowed_gene_indices 内采样）
    """

    def __init__(
        self,
        exp_data: torch.Tensor,
        GRN_matrix: torch.Tensor,
        cds_file: str,
        gene_list,
        ppi_file: str,
        allowed_gene_indices,
        neg_sample_ratio: float = 1.0,
        device=None,
        seed: int = 42,
        cds_features=None,
        ppi_extractor: PPIFeatureExtractor = None,
        use_ppi: bool = True,
    ):
        super().__init__()
        np.random.seed(seed)

        self.exp_data = exp_data
        self.GRN = GRN_matrix
        self.gene_list = gene_list
        self.device = device
        self.use_ppi = use_ppi

        if use_ppi:
            self.ppi_extractor = ppi_extractor or PPIFeatureExtractor(
                ppi_file=ppi_file,
                gene_list=gene_list,
                emb_dim=160,
                device=device,
            )
        else:
            self.ppi_extractor = None

        if cds_features is None:
            df = pd.read_csv(cds_file, header=None, sep=",", skiprows=1)
            cds_features = df.values.astype(np.float32)
        self.cds_features = cds_features

        if len(self.cds_features) != len(gene_list):
            raise ValueError(
                f"CDS特征数量({len(self.cds_features)})与基因列表长度({len(gene_list)})不一致"
            )

        self.allowed_gene_indices = list(allowed_gene_indices)
        self.allowed_set = set(self.allowed_gene_indices)

        self.positive_pairs_full = self._build_positive_pairs_full()
        self.positive_pairs = [p for p in self.positive_pairs_full if p[0] in self.allowed_set and p[1] in self.allowed_set]
        self.negative_pairs = self._sample_negative_pairs(neg_sample_ratio)
        self.samples = self.positive_pairs + self.negative_pairs

        print(
            f"[GRNGeneSubsetDataset] allowed_genes={len(self.allowed_gene_indices)} "
            f"| samples={len(self.samples)} | pos={len(self.positive_pairs)} | neg={len(self.negative_pairs)}"
        )

    def _build_positive_pairs_full(self):
        pos = torch.where(self.GRN > 0)
        pairs = []
        for i, j in zip(pos[0], pos[1]):
            label = int(self.GRN[i, j].item())
            pairs.append((int(i.item()), int(j.item()), label))
        return pairs

    def _sample_negative_pairs(self, ratio: float):
        num_pos = len(self.positive_pairs)
        num_neg = int(num_pos * ratio)
        if num_pos == 0:
            return []

        existing_pos = set((i, j) for i, j, _ in self.positive_pairs)
        allowed = self.allowed_gene_indices

        neg_pairs = []
        attempts = 0
        # 给一个上限，避免极端情况下死循环（可根据需要调大）
        max_attempts = max(1000000, num_neg * 50)
        while len(neg_pairs) < num_neg and attempts < max_attempts:
            attempts += 1
            i = int(np.random.choice(allowed))
            j = int(np.random.choice(allowed))
            if (i, j) in existing_pos:
                continue
            if int(self.GRN[i, j].item()) != 0:
                continue
            neg_pairs.append((i, j, 0))
            existing_pos.add((i, j))

        if len(neg_pairs) < num_neg:
            print(
                f"[GRNGeneSubsetDataset] warning: neg sampling truncated "
                f"{len(neg_pairs)}/{num_neg} (allowed={len(allowed)})"
            )
        return neg_pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i, j, label = self.samples[idx]

        # CDS TF
        seq_tf = self.cds_features[i]
        if np.all(seq_tf == 0):
            seq_tf = np.full_like(seq_tf, 1e-99, dtype=np.float32)

        # CDS Target
        seq_target = self.cds_features[j]
        if np.all(seq_target == 0):
            seq_target = np.full_like(seq_target, 1e-99, dtype=np.float32)

        if self.use_ppi:
            ppi_tf, ppi_target = self.ppi_extractor.get_pair_features(i, j)
        else:
            ppi_tf = torch.zeros(160, dtype=torch.float32)
            ppi_target = torch.zeros(160, dtype=torch.float32)

        sample = {
            "tf": self.exp_data[i].clone().detach(),
            "target": self.exp_data[j].clone().detach(),
            "seq_tf": torch.tensor(seq_tf, dtype=torch.float32),
            "seq_target": torch.tensor(seq_target, dtype=torch.float32),
            "ppi_tf": ppi_tf,
            "ppi_target": ppi_target,
            "label": torch.tensor(label, dtype=torch.long),
            "tf_idx": i,
            "target_idx": j,
        }

        if self.device:
            for key in sample:
                if isinstance(sample[key], torch.Tensor):
                    sample[key] = sample[key].to(self.device)
        return sample

