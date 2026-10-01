import torch
from torch.utils.data import Dataset

from grn.models.fusion_model import FusionModel
from grn.models.fusion_model_noppi_cascade import CascadeStageBModelWithPPI
from grn.scripts.train_gene_disjoint_unseen_bulk_twostage_tcga import predict_combined_probs


class DummyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.g = 8
        self.T = 6
        self.seq_dim = 20
        self.ppi_dim = 160

        labels = [0, 1, 2, 1, 2, 0, 1, 2, 0, 2]
        pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0), (1, 3), (2, 5)]
        self.samples = [(i, j, lab) for (i, j), lab in zip(pairs, labels)]

        self.tf = torch.randn(self.g, self.T)
        self.target = torch.randn(self.g, self.T)
        self.seq = torch.randn(self.g, self.seq_dim)
        self.ppi_tf = torch.randn(self.g, self.ppi_dim)
        self.ppi_target = torch.randn(self.g, self.ppi_dim)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i, j, lab = self.samples[idx]
        return {
            "tf": self.tf[i],
            "target": self.target[j],
            "seq_tf": self.seq[i],
            "seq_target": self.seq[j],
            "ppi_tf": self.ppi_tf[i],
            "ppi_target": self.ppi_target[j],
            "label": torch.tensor(lab, dtype=torch.long),
            "tf_idx": i,
            "target_idx": j,
        }


def main():
    gene_list = [f"G{i}" for i in range(8)]
    d = DummyDataset()

    modelA = FusionModel(
        seq_dim=20,
        emb_dim=32,
        num_classes=2,
        expr_d_model=32,
        expr_pooled_len=16,
        expr_nhead=4,
        expr_num_layers=1,
        expr_dim_ff=64,
        expr_dropout=0.1,
        ppi_emb_dim=160,
        fusion_hidden_dim=64,
        fusion_dropout=0.1,
    )
    modelB = CascadeStageBModelWithPPI(
        seq_dim=20,
        emb_dim=32,
        num_classes=2,
        expr_d_model=32,
        expr_pooled_len=16,
        expr_nhead=4,
        expr_num_layers=1,
        expr_dim_ff=64,
        expr_dropout=0.1,
        fusion_hidden_dim=64,
        fusion_dropout=0.1,
        ppi_emb_dim=160,
    )

    device = torch.device("cpu")
    modelA.to(device)
    modelB.to(device)

    gene_pairs, y_true, prob = predict_combined_probs(
        modelA=modelA,
        modelB=modelB,
        dataset=d,
        gene_list=gene_list,
        device=device,
        train_genes=None,
        eval_genes=None,
        stageB_noppi=False,
        stageB_cascade=True,
        stageA_noppi=False,
    )

    assert prob.shape == (len(d), 3)
    assert len(gene_pairs) == len(d)
    assert len(y_true) == len(d)
    print("e7_stageB_ppi_cascade smoke OK", prob.shape)


if __name__ == "__main__":
    main()

