import argparse
import json
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from grn.data.preprocessing import get_GRN, log_expression_tensor
from grn.evaluation.metrics import calculate_metrics
from grn.features.ppi_features import PPIFeatureExtractor
from grn.models.fusion_model import FusionModel
from grn.reproducibility import set_seed


def split3(n: int, w1: int, w2: int, w3: int) -> Tuple[int, int, int]:
    w = w1 + w2 + w3
    a = n * w1 // w
    b = n * w2 // w
    c = n - a - b
    return a, b, c


class GeneSubsetDataset(Dataset):
    def __init__(
        self,
        exp_data: torch.Tensor,
        grn_matrix: torch.Tensor,
        allowed_genes: List[int],
        cds_features: np.ndarray,
        ppi_extractor: PPIFeatureExtractor,
        neg_sample_ratio: float = 1.0,
        seed: int = 42,
        device=None,
    ):
        self.exp_data = exp_data
        self.grn = grn_matrix
        self.allowed = list(allowed_genes)
        self.allowed_set = set(self.allowed)
        self.cds_features = cds_features
        self.ppi_extractor = ppi_extractor
        self.device = device
        self.rng = np.random.RandomState(seed)

        pos = torch.where(self.grn > 0)
        pos_pairs_all = [(int(i.item()), int(j.item()), int(self.grn[i, j].item())) for i, j in zip(pos[0], pos[1])]
        self.positive_pairs = [p for p in pos_pairs_all if p[0] in self.allowed_set and p[1] in self.allowed_set]

        n_pos = len(self.positive_pairs)
        n_neg = int(n_pos * neg_sample_ratio)
        existing = set((i, j) for i, j, _ in self.positive_pairs)
        self.negative_pairs = []
        max_attempts = max(1000000, n_neg * 50)
        attempts = 0
        while len(self.negative_pairs) < n_neg and attempts < max_attempts:
            attempts += 1
            i = int(self.rng.choice(self.allowed))
            j = int(self.rng.choice(self.allowed))
            if (i, j) in existing:
                continue
            if int(self.grn[i, j].item()) != 0:
                continue
            self.negative_pairs.append((i, j, 0))
            existing.add((i, j))

        self.samples = self.positive_pairs + self.negative_pairs
        print(
            f"[GeneSubsetDataset] genes={len(self.allowed)} | samples={len(self.samples)} "
            f"| pos={len(self.positive_pairs)} | neg={len(self.negative_pairs)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i, j, label = self.samples[idx]
        seq_tf = self.cds_features[i]
        seq_target = self.cds_features[j]
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
        if self.device:
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    sample[k] = v.to(self.device)
        return sample


def _prepare_batch(batch, device):
    return {
        "exp_tf": batch["tf"].to(device),
        "exp_target": batch["target"].to(device),
        "seq_tf": batch["seq_tf"].to(device),
        "seq_target": batch["seq_target"].to(device),
        "ppi_tf": batch["ppi_tf"].to(device),
        "ppi_target": batch["ppi_target"].to(device),
    }


@torch.no_grad()
def _evaluate(model, loader, device, ppi_extractor=None, train_genes=None, eval_genes=None):
    # 对验证/测试使用 inductive inference 特征，避免复用错误缓存导致评估退化
    if ppi_extractor is not None and train_genes is not None and eval_genes is not None:
        ppi_extractor.refresh_cache()
        ppi_extractor._cached_features = ppi_extractor.extract_inductive_inference(train_genes, eval_genes)

    model.eval()
    ys, yps, probs = [], [], []
    for batch in loader:
        x = _prepare_batch(batch, device)
        y = batch["label"].cpu().numpy()
        logits = model(**x)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        yp = np.argmax(p, axis=1)
        ys.extend(y)
        yps.extend(yp)
        probs.append(p)
    probs = np.vstack(probs)
    if ppi_extractor is not None:
        ppi_extractor.refresh_cache()
    return calculate_metrics(np.array(ys), np.array(yps), probs)


def _to_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="No-CV gene-disjoint + unseen-samples (single-stage, no class balancing).")
    parser.add_argument("--exp-csv", required=True)
    parser.add_argument("--reference-gene-list-csv", default=None)
    parser.add_argument("--grn-csv", required=True)
    parser.add_argument("--cds-file", required=True)
    parser.add_argument("--ppi-csv", required=True)
    parser.add_argument("--output-base-dir", default="experiments")
    parser.add_argument("--experiment-name", default="repro_gene_disjoint_unseen_samples_no_cv")
    parser.add_argument("--no-timestamp", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--neg-sample-ratio", type=float, default=1.0)
    parser.add_argument("--gene-split", default="6,2,2")
    parser.add_argument("--sample-split", default="5,2,3")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.no_timestamp:
        run_dir = os.path.join(args.output_base_dir, args.experiment_name)
    else:
        run_dir = os.path.join(args.output_base_dir, args.experiment_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # Expression anchor
    expr_df = pd.read_csv(args.exp_csv)
    gene_col = expr_df.columns[0]
    expr_df[gene_col] = expr_df[gene_col].astype(str).str.strip()
    expr_df = expr_df.drop_duplicates(subset=[gene_col], keep="first").set_index(gene_col)
    expr_df = expr_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    ref_indices = None
    if args.reference_gene_list_csv:
        ref = pd.read_csv(args.reference_gene_list_csv, usecols=[0]).iloc[:, 0].astype(str).str.strip().tolist()
        ref_map = {g: i for i, g in enumerate(ref)}
        kept = [g for g in expr_df.index.tolist() if g in ref_map]
        expr_df = expr_df.loc[kept]
        ref_indices = [ref_map[g] for g in kept]
        print(f"[gene-disjoint] anchor genes={len(kept)} (no fill-zero)")

    gene_list = expr_df.index.tolist()
    raw_expr = expr_df.values.astype(np.float32)
    n_gene, n_sample = raw_expr.shape

    # GRN filter
    grn_df = pd.read_csv(args.grn_csv)
    gs = set(gene_list)
    grn_df = grn_df[grn_df["TF"].astype(str).isin(gs) & grn_df["target"].astype(str).isin(gs)].copy()
    grn = get_GRN((grn_df["TF"].values, grn_df["target"].values, grn_df["Re_type"].values), gene_list)

    # splits
    g1, g2, g3 = [int(x.strip()) for x in args.gene_split.split(",")]
    a, b, c = split3(n_gene, g1, g2, g3)
    perm = np.random.RandomState(args.seed).permutation(n_gene)
    g_tr = perm[:a].tolist()
    g_va = perm[a : a + b].tolist()
    g_te = perm[a + b :].tolist()

    s1, s2, s3 = [int(x.strip()) for x in args.sample_split.split(",")]
    c1, c2, c3 = split3(n_sample, s1, s2, s3)
    sl_tr = slice(0, c1)
    sl_va = slice(c1, c1 + c2)
    sl_te = slice(c1 + c2, n_sample)
    print(f"[gene-disjoint] sample columns train/val/test = {c1}/{c2}/{c3}")

    exp_tr = log_expression_tensor(raw_expr[:, sl_tr], offset=1.0)
    exp_va = log_expression_tensor(raw_expr[:, sl_va], offset=1.0)
    exp_te = log_expression_tensor(raw_expr[:, sl_te], offset=1.0)

    cds_all = pd.read_csv(args.cds_file, header=None, sep=",", skiprows=1).values.astype(np.float32)
    cds_features = cds_all[ref_indices] if ref_indices is not None else cds_all
    if len(cds_features) != len(gene_list):
        raise ValueError(f"CDS/gene mismatch: {len(cds_features)} vs {len(gene_list)}")

    ppi = PPIFeatureExtractor(args.ppi_csv, gene_list=gene_list, emb_dim=160, device=None)
    # inductive mode: train genes visible, val/test through train->test only
    ppi.set_train_mask(g_tr)
    ds_train = GeneSubsetDataset(exp_tr, grn, g_tr, cds_features, ppi, args.neg_sample_ratio, seed=args.seed)
    ds_val = GeneSubsetDataset(exp_va, grn, g_va, cds_features, ppi, args.neg_sample_ratio, seed=args.seed + 1)
    ds_test = GeneSubsetDataset(exp_te, grn, g_te, cds_features, ppi, args.neg_sample_ratio, seed=args.seed + 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(seq_dim=20, num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # 不做类别平衡：直接 shuffle
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    best_val = -1.0
    patience_counter = 0
    best_path = os.path.join(run_dir, "best_model_fold0.pth")
    for ep in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {ep+1}"):
            x = _prepare_batch(batch, device)
            y = batch["label"].long().to(device)
            logits = model(**x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_metrics = _evaluate(
            model,
            val_loader,
            device,
            ppi_extractor=ppi,
            train_genes=g_tr,
            eval_genes=g_va,
        )
        val_acc = float(val_metrics["accuracy"])
        print(f"Epoch {ep+1}/{args.epochs} | Train Loss: {epoch_loss/max(len(train_loader),1):.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[Early Stop] patience={args.patience}, stop at epoch {ep+1}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    val_metrics = _evaluate(
        model,
        val_loader,
        device,
        ppi_extractor=ppi,
        train_genes=g_tr,
        eval_genes=g_va,
    )
    test_metrics = _evaluate(
        model,
        test_loader,
        device,
        ppi_extractor=ppi,
        train_genes=g_tr,
        eval_genes=g_te,
    )
    with open(os.path.join(run_dir, "gene_disjoint_unseen_samples_val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(val_metrics), f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "gene_disjoint_unseen_samples_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(test_metrics), f, ensure_ascii=False, indent=2)

    print(f"[gene-disjoint] best_model: {best_path}")
    print(f"[gene-disjoint] val_acc={val_metrics['accuracy']:.4f}, test_acc={test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()

