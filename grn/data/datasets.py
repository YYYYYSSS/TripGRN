import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from grn.features.ppi_features import PPIFeatureExtractor


class GRNDataset(Dataset):
    """
    基因调控关系数据集：
    - 正样本：由 GRN 矩阵中非零元素构造（激活/抑制）
    - 负样本：在无关系位置随机采样
    - 每个样本包含表达 TF/Target、CDS 序列特征、PPI 网络特征与标签
    """

    def __init__(
        self,
        exp_data: torch.Tensor,
        GRN_matrix: torch.Tensor,
        cds_file: str,
        gene_list,
        ppi_file: str = "PPI_bulk",
        neg_sample_ratio: float = 1.0,
        device=None,
        transform=None,
        cds_features=None,
    ):
        # 加载 CDS 特征（支持外部传入已对子集对齐的特征）
        if cds_features is None:
            cds_dataset = CDSDataset(cds_file, gene_list)
            self.cds_features = cds_dataset.features
        else:
            self.cds_features = cds_features
        assert len(self.cds_features) == len(gene_list), "CDS特征数量与基因列表长度不一致"

        # 加载 PPI 特征提取器
        self.ppi_extractor = PPIFeatureExtractor(
            ppi_file=ppi_file,
            gene_list=gene_list,
            emb_dim=160,  # 与原模型融合维度一致
            device=device,
        )

        # 保存数据
        self.exp_data = exp_data
        self.GRN = GRN_matrix
        self.gene_list = gene_list
        self.device = device
        self.transform = transform

        # 获取样本对
        self.positive_pairs = self._get_positive_pairs()
        self.negative_pairs = self._sample_negative_pairs(neg_sample_ratio)
        self.samples = self.positive_pairs + self.negative_pairs

        print(
            f"数据集统计: 总样本={len(self.samples)}, "
            f"正样本={len(self.positive_pairs)}, "
            f"负样本={len(self.negative_pairs)}"
        )

    def _get_positive_pairs(self):
        """获取正样本 (激活/抑制关系)"""
        pos_indices = torch.where(self.GRN > 0)
        return [(i.item(), j.item(), self.GRN[i, j].item()) for i, j in zip(pos_indices[0], pos_indices[1])]

    def _sample_negative_pairs(self, ratio: float):
        """随机采样负样本 (无关系)"""
        num_pos = len(self.positive_pairs)
        num_neg = int(num_pos * ratio)
        num_genes = self.GRN.shape[0]
        neg_pairs = []

        existing_pairs = set((i, j) for i, j, _ in self.positive_pairs)

        while len(neg_pairs) < num_neg:
            i = np.random.randint(0, num_genes)
            j = np.random.randint(0, num_genes)

            # 跳过已有关系
            if self.GRN[i, j] > 0 or (i, j) in existing_pairs:
                continue

            neg_pairs.append((i, j, 0))
            existing_pairs.add((i, j))

        return neg_pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i, j, label = self.samples[idx]

        # CDS TF
        seq_tf = self.cds_features[i]
        if isinstance(seq_tf, pd.Series):
            seq_tf = seq_tf.values
        if np.all(seq_tf == 0):
            seq_tf = np.full(20, 1e-99, dtype=np.float32)

        # CDS Target
        seq_target = self.cds_features[j]
        if isinstance(seq_target, pd.Series):
            seq_target = seq_target.values
        if np.all(seq_target == 0):
            seq_target = np.full(20, 1e-99, dtype=np.float32)

        # PPI pair features
        ppi_tf, ppi_target = self.ppi_extractor.get_pair_features(i, j)

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

        if self.transform:
            sample = self.transform(sample)

        if self.device:
            for key in sample:
                if isinstance(sample[key], torch.Tensor):
                    sample[key] = sample[key].to(self.device)

        return sample

    def refresh_ppi_cache(self):
        """强制刷新 PPI 特征缓存（切换折时备用接口）"""
        if hasattr(self, "ppi_extractor"):
            self.ppi_extractor.refresh_cache()


class CDSDataset(Dataset):
    """加载 CDS 特征数据（每基因 20 维）"""

    def __init__(self, cds_file: str, gene_list):
        try:
            df = pd.read_csv(cds_file, header=None, sep=",", skiprows=1)
            self.features = df.values.astype(np.float32)

            if self.features.shape[1] != 20:
                raise ValueError(f"CDS特征维度应为20, 实际为{self.features.shape[1]}")

            if np.isnan(self.features).any():
                print("警告: CDS特征包含NaN值, 填充为0")
                self.features = np.nan_to_num(self.features)

            zero_count = np.sum(np.all(self.features == 0, axis=1))
            if zero_count > 0:
                print(f"警告: 发现{zero_count}个全零序列, 将在使用时填充为特殊值")

        except Exception as e:
            raise RuntimeError(f"加载CDS特征失败: {str(e)}")

        if len(self.features) != len(gene_list):
            raise ValueError(f"CDS特征数量({len(self.features)})与基因列表长度({len(gene_list)})不一致")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

