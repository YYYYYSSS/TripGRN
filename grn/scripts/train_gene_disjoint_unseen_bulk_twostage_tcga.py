import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from grn.data.gene_split_weighted import (
    load_gene_split_json,
    optimize_gene_split_weighted,
    regulated_edges_from_grn_matrix,
    save_gene_split_json,
    assignment_to_gene_lists,
)
from grn.data.gene_subset_dataset import GRNGeneSubsetDataset
from grn.data.preprocessing import get_GRN, log_expression_tensor, resolve_grn_table_columns
from grn.evaluation.metrics import calculate_metrics
from grn.models.fusion_model import FusionModel
from grn.models.fusion_model_noppi_cascade import (
    CascadeStageBModel,
    CascadeStageBModelWithPPI,
    FusionModelNoPPI,
)
from grn.models.ppi_gcn_cascade import PpiGcnTwoStageCascade, build_inductive_message_edges
from grn.reproducibility import set_seed
from grn.scripts.tune_repress_threshold import run_threshold_tune


def split3(n: int, w1: int, w2: int, w3: int) -> Tuple[int, int, int]:
    w = w1 + w2 + w3
    a = n * w1 // w
    b = n * w2 // w
    c = n - a - b
    return a, b, c


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


def _load_gene_feature_csv(path: str):
    import pandas as pd

    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"feature csv needs >=2 columns (gene + features): {path}")
    gene_col = df.columns[0]
    df[gene_col] = df[gene_col].astype(str).str.strip()
    df = df.drop_duplicates(subset=[gene_col], keep="first").set_index(gene_col)
    feat = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return feat


@dataclass
class TrainStageConfig:
    epochs: int
    lr: float
    patience: int
    batch_size: int
    neg_sample_ratio: float


def _build_binary_labels_stageA(labels_012: Sequence[int]) -> np.ndarray:
    # 0: NoReg, 1: Reg
    return np.array([0 if x == 0 else 1 for x in labels_012], dtype=np.int64)


def _build_binary_labels_stageB(labels_012: Sequence[int]) -> np.ndarray:
    # Stage-B: Repress vs Activate
    # 1 (repress) -> 0, 2 (activate) -> 1
    return np.array([0 if x == 1 else 1 for x in labels_012], dtype=np.int64)


def _batch_exp_seq(inputs: dict) -> tuple:
    exp_tf = inputs["exp_tf"] if "exp_tf" in inputs else inputs["tf"]
    exp_target = inputs.get("exp_target") if "exp_target" in inputs else inputs["target"]
    return exp_tf, exp_target, inputs["seq_tf"], inputs["seq_target"]


def _pairs_to_tensors(samples, device: torch.device):
    tf_idx = torch.tensor([i for i, _, _ in samples], dtype=torch.long, device=device)
    tg_idx = torch.tensor([j for _, j, _ in samples], dtype=torch.long, device=device)
    y012 = torch.tensor([lab for _, _, lab in samples], dtype=torch.long, device=device)
    return tf_idx, tg_idx, y012


def _build_binary_labels_stageA_tensor(y012: torch.Tensor) -> torch.Tensor:
    return (y012 > 0).long()


def _build_binary_labels_stageB_tensor(y012: torch.Tensor) -> torch.Tensor:
    # 1->Repress(0), 2->Activate(1)
    return torch.where(y012 == 1, torch.zeros_like(y012), torch.ones_like(y012))


def _weighted_sampler_from_binary_labels(bin_labels: np.ndarray, seed: int) -> WeightedRandomSampler:
    counts = np.bincount(bin_labels, minlength=2)
    weights_per_class = 1.0 / (counts + 1e-6)
    sample_weights = weights_per_class[bin_labels]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return sampler


def train_one_epoch_stageA(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ohem_frac: float = 1.0,
    stageA_noppi: bool = False,
):
    model.train()
    criterion = nn.CrossEntropyLoss(reduction="none")
    total_loss = 0.0
    total_samples = 0

    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items() if k not in ["label", "tf_idx", "target_idx"]}

        y_012 = batch["label"].cpu().numpy().tolist()
        y = torch.tensor(_build_binary_labels_stageA(y_012), dtype=torch.long, device=device)

        exp_tf, exp_target, seq_tf, seq_target = _batch_exp_seq(inputs)
        if stageA_noppi:
            logits = model(exp_tf, exp_target, seq_tf, seq_target)
        else:
            logits = model(
                exp_tf=exp_tf,
                exp_target=exp_target,
                seq_tf=seq_tf,
                seq_target=seq_target,
                ppi_tf=inputs["ppi_tf"],
                ppi_target=inputs["ppi_target"],
            )

        per_sample_loss = criterion(logits, y)
        if ohem_frac < 1.0:
            k = max(1, int(len(per_sample_loss) * ohem_frac))
            topk = torch.topk(per_sample_loss, k=k, largest=True).values
            loss = topk.mean()
        else:
            loss = per_sample_loss.mean()

        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_stageA(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ppi_extractor=None,
    train_genes=None,
    eval_genes=None,
    stageA_noppi: bool = False,
) -> Dict[str, float]:
    if (
        not stageA_noppi
        and ppi_extractor is not None
        and train_genes is not None
        and eval_genes is not None
    ):
        ppi_extractor.refresh_cache()
        ppi_extractor._cached_features = ppi_extractor.extract_inductive_inference(train_genes, eval_genes)

    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k not in ["label", "tf_idx", "target_idx"]}

        y_012 = batch["label"].cpu().numpy().tolist()
        y = _build_binary_labels_stageA(y_012)

        exp_tf, exp_target, seq_tf, seq_target = _batch_exp_seq(inputs)
        if stageA_noppi:
            logits = model(exp_tf, exp_target, seq_tf, seq_target)
        else:
            logits = model(
                exp_tf=exp_tf,
                exp_target=exp_target,
                seq_tf=seq_tf,
                seq_target=seq_target,
                ppi_tf=inputs["ppi_tf"],
                ppi_target=inputs["ppi_target"],
            )
        preds = logits.argmax(dim=1).cpu().numpy().tolist()

        y_true.extend(y)
        y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)
    if not stageA_noppi and ppi_extractor is not None:
        ppi_extractor.refresh_cache()
    return {"acc": float(acc)}


def train_one_epoch_stageB(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ohem_frac: float = 1.0,
    stageB_noppi: bool = False,
    stageB_cascade: bool = False,
    stageA_noppi: bool = False,
    model_a: nn.Module = None,
    stage_b_loss: str = "ce",
    focal_gamma: float = 2.0,
    w_repress: float = 1.0,
    w_activate: float = 1.0,
):
    model.train()
    if model_a is not None:
        model_a.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    total_loss = 0.0
    total_samples = 0

    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items() if k not in ["label", "tf_idx", "target_idx"]}

        y_012 = batch["label"].cpu().numpy().tolist()
        y = torch.tensor(_build_binary_labels_stageB(y_012), dtype=torch.long, device=device)

        exp_tf, exp_target, seq_tf, seq_target = _batch_exp_seq(inputs)
        if stageB_cascade:
            with torch.no_grad():
                if stageA_noppi:
                    logits_a = model_a(exp_tf, exp_target, seq_tf, seq_target)
                else:
                    logits_a = model_a(
                        exp_tf=exp_tf,
                        exp_target=exp_target,
                        seq_tf=seq_tf,
                        seq_target=seq_target,
                        ppi_tf=inputs["ppi_tf"],
                        ppi_target=inputs["ppi_target"],
                    )
            if stageB_noppi:
                logits = model(
                    exp_tf,
                    exp_target,
                    seq_tf,
                    seq_target,
                    logits_stage_a=logits_a.detach(),
                )
            else:
                logits = model(
                    exp_tf=exp_tf,
                    exp_target=exp_target,
                    seq_tf=seq_tf,
                    seq_target=seq_target,
                    ppi_tf=inputs["ppi_tf"],
                    ppi_target=inputs["ppi_target"],
                    logits_stage_a=logits_a.detach(),
                )
            ce = F.cross_entropy(logits, y, reduction="none")
            wt = torch.where(
                y == 0,
                torch.full_like(y, w_repress, dtype=torch.float),
                torch.full_like(y, w_activate, dtype=torch.float),
            )
            if stage_b_loss == "focal":
                pt = torch.exp(-ce)
                per_sample_loss = ((1.0 - pt) ** focal_gamma) * ce * wt
            else:
                per_sample_loss = ce * wt
            if ohem_frac < 1.0:
                k = max(1, int(len(per_sample_loss) * ohem_frac))
                loss = torch.topk(per_sample_loss, k=k, largest=True).values.mean()
            else:
                loss = per_sample_loss.mean()
        else:
            logits = model(
                exp_tf=exp_tf,
                exp_target=exp_target,
                seq_tf=seq_tf,
                seq_target=seq_target,
                ppi_tf=inputs["ppi_tf"],
                ppi_target=inputs["ppi_target"],
            )
            per_sample_loss = criterion(logits, y)
            if ohem_frac < 1.0:
                k = max(1, int(len(per_sample_loss) * ohem_frac))
                topk = torch.topk(per_sample_loss, k=k, largest=True).values
                loss = topk.mean()
            else:
                loss = per_sample_loss.mean()

        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_stageB(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ppi_extractor=None,
    train_genes=None,
    eval_genes=None,
    stageB_noppi: bool = False,
    stageB_cascade: bool = False,
    stageA_noppi: bool = False,
    model_a: nn.Module = None,
) -> Dict[str, float]:
    if (
        not stageA_noppi
        and ppi_extractor is not None
        and train_genes is not None
        and eval_genes is not None
    ):
        # 无论 Stage-B 是否级联，只要 Stage-A 需要 PPI，就必须先注入归纳特征。
        ppi_extractor.refresh_cache()
        ppi_extractor._cached_features = ppi_extractor.extract_inductive_inference(train_genes, eval_genes)

    model.eval()
    if model_a is not None:
        model_a.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k not in ["label", "tf_idx", "target_idx"]}
        y_012 = batch["label"].cpu().numpy().tolist()
        y = _build_binary_labels_stageB(y_012)

        exp_tf, exp_target, seq_tf, seq_target = _batch_exp_seq(inputs)
        if stageB_cascade:
            with torch.no_grad():
                if stageA_noppi:
                    logits_a = model_a(exp_tf, exp_target, seq_tf, seq_target)
                else:
                    logits_a = model_a(
                        exp_tf=exp_tf,
                        exp_target=exp_target,
                        seq_tf=seq_tf,
                        seq_target=seq_target,
                        ppi_tf=inputs["ppi_tf"],
                        ppi_target=inputs["ppi_target"],
                    )
            if stageB_noppi:
                logits = model(exp_tf, exp_target, seq_tf, seq_target, logits_stage_a=logits_a)
            else:
                logits = model(
                    exp_tf=exp_tf,
                    exp_target=exp_target,
                    seq_tf=seq_tf,
                    seq_target=seq_target,
                    ppi_tf=inputs["ppi_tf"],
                    ppi_target=inputs["ppi_target"],
                    logits_stage_a=logits_a,
                )
        else:
            logits = model(
                exp_tf=exp_tf,
                exp_target=exp_target,
                seq_tf=seq_tf,
                seq_target=seq_target,
                ppi_tf=inputs["ppi_tf"],
                ppi_target=inputs["ppi_target"],
            )
        preds = logits.argmax(dim=1).cpu().numpy().tolist()

        y_true.extend(y)
        y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)
    if ppi_extractor is not None:
        ppi_extractor.refresh_cache()
    return {"acc": float(acc)}


@torch.no_grad()
def predict_combined_probs(
    modelA: nn.Module,
    modelB: nn.Module,
    dataset: GRNGeneSubsetDataset,
    gene_list: List[str],
    device: torch.device,
    train_genes=None,
    eval_genes=None,
    stageB_noppi: bool = False,
    stageB_cascade: bool = False,
    stageA_noppi: bool = False,
) -> Tuple[List[Tuple[str, str]], np.ndarray, np.ndarray]:
    """
    return:
      gene_pairs: [(TF, Target), ...] length N
      y_true: [N] in {0,1,2}
      prob_3: [N,3] columns [NoReg, Repress, Activate]
    """
    modelA.eval()
    modelB.eval()
    if (
        not stageA_noppi
        and hasattr(dataset, "ppi_extractor")
        and dataset.ppi_extractor is not None
        and train_genes is not None
        and eval_genes is not None
    ):
        dataset.ppi_extractor.refresh_cache()
        dataset.ppi_extractor._cached_features = dataset.ppi_extractor.extract_inductive_inference(
            train_genes, eval_genes
        )

    loader_all = DataLoader(dataset, batch_size=256, shuffle=False)
    n = len(dataset)
    probA = np.zeros((n, 2), dtype=np.float32)
    # stageB only for regulated pairs, others set to 0
    probB = np.zeros((n, 2), dtype=np.float32)

    # For gene_pairs and y_true we reuse dataset.samples ordering
    gene_pairs = []
    y_true = np.zeros((n,), dtype=np.int64)
    for idx, (i, j, lab) in enumerate(dataset.samples):
        gene_pairs.append((gene_list[i], gene_list[j]))
        y_true[idx] = lab

    # Stage A inference for all pairs
    for batch_start, batch in enumerate(loader_all):
        bs = len(batch["label"])
        # loader_all order == dataset.samples order
        batch_indices_start = batch_start * loader_all.batch_size
        inputs = {k: v.to(device) for k, v in batch.items() if k not in ["label", "tf_idx", "target_idx"]}

        exp_tf, exp_target, seq_tf, seq_target = _batch_exp_seq(inputs)
        if stageA_noppi:
            logitsA = modelA(exp_tf, exp_target, seq_tf, seq_target)
        else:
            logitsA = modelA(
                exp_tf=exp_tf,
                exp_target=exp_target,
                seq_tf=seq_tf,
                seq_target=seq_target,
                ppi_tf=inputs["ppi_tf"],
                ppi_target=inputs["ppi_target"],
            )
        pA = torch.softmax(logitsA, dim=1).cpu().numpy()
        probA[batch_indices_start : batch_indices_start + bs] = pA

    # Stage B inference for regulated subset only
    regulated_indices = [idx for idx, (_, _, lab) in enumerate(dataset.samples) if lab > 0]
    if regulated_indices:
        ds_reg = Subset(dataset, regulated_indices)
        loaderB = DataLoader(ds_reg, batch_size=256, shuffle=False)
        for k, batch in enumerate(loaderB):
            # k*batch_size offset within ds_reg, map back to regulated_indices
            bs = len(batch["label"])
            inputs = {kk: vv.to(device) for kk, vv in batch.items() if kk not in ["label", "tf_idx", "target_idx"]}
            exp_tf, exp_target, seq_tf, seq_target = _batch_exp_seq(inputs)
            if stageB_cascade:
                with torch.no_grad():
                    if stageA_noppi:
                        la = modelA(exp_tf, exp_target, seq_tf, seq_target)
                    else:
                        la = modelA(
                            exp_tf=exp_tf,
                            exp_target=exp_target,
                            seq_tf=seq_tf,
                            seq_target=seq_target,
                            ppi_tf=inputs["ppi_tf"],
                            ppi_target=inputs["ppi_target"],
                        )
                if stageB_noppi:
                    logitsB = modelB(exp_tf, exp_target, seq_tf, seq_target, logits_stage_a=la)
                else:
                    logitsB = modelB(
                        exp_tf=exp_tf,
                        exp_target=exp_target,
                        seq_tf=seq_tf,
                        seq_target=seq_target,
                        ppi_tf=inputs["ppi_tf"],
                        ppi_target=inputs["ppi_target"],
                        logits_stage_a=la,
                    )
            else:
                logitsB = modelB(
                    exp_tf=exp_tf,
                    exp_target=exp_target,
                    seq_tf=seq_tf,
                    seq_target=seq_target,
                    ppi_tf=inputs["ppi_tf"],
                    ppi_target=inputs["ppi_target"],
                )
            pB = torch.softmax(logitsB, dim=1).cpu().numpy()  # [Repress, Activate]
            for bi in range(bs):
                true_idx = k * loaderB.batch_size + bi
                global_idx = regulated_indices[true_idx]
                probB[global_idx] = pB[bi]

    prob_3 = np.zeros((n, 3), dtype=np.float32)
    prob_3[:, 0] = probA[:, 0]  # NoReg
    prob_3[:, 1] = probA[:, 1] * probB[:, 0]  # Repress
    prob_3[:, 2] = probA[:, 1] * probB[:, 1]  # Activate
    if hasattr(dataset, "ppi_extractor") and dataset.ppi_extractor is not None:
        dataset.ppi_extractor.refresh_cache()
    return gene_pairs, y_true, prob_3


def _safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """避免验证子集仅含单一类别时 sklearn 行为不稳定；否则与 balanced_accuracy_score 一致。"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0:
        return 0.0
    if np.unique(y_true).size < 2:
        return float((y_true == y_pred).mean())
    return float(balanced_accuracy_score(y_true, y_pred))


def _binary_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """二类 macro-F1；在「全判 0 / 全判 1」两种塌缩下取值不同，可区分 balanced_acc 恒为 0.5 的情况。"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0:
        return 0.0
    if np.unique(y_true).size < 2:
        return float((y_true == y_pred).mean())
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _safe_binary_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    return float(mcc) if np.isfinite(mcc) else 0.0


def _gcn_maybe_clip_grad(model: nn.Module, max_norm: float) -> None:
    if max_norm is not None and max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def _apply_gcn_stageb_requires_grad(
    model: nn.Module,
    *,
    freeze_node: bool,
    train_gcn_stageb: bool,
    head_only_warmup: bool,
) -> None:
    """Stage-B：stage_a 恒冻结；可选仅训 stage_b_head，使 h 固定时先学二类头再联合微调。"""
    for p in model.stage_a_head.parameters():
        p.requires_grad = False
    for p in model.stage_b_head.parameters():
        p.requires_grad = True
    if head_only_warmup:
        for p in model.node_encoder.parameters():
            p.requires_grad = False
        for p in model.gcn_layers.parameters():
            p.requires_grad = False
        return
    if freeze_node:
        for p in model.node_encoder.parameters():
            p.requires_grad = False
    else:
        for p in model.node_encoder.parameters():
            p.requires_grad = True
    if train_gcn_stageb:
        for p in model.gcn_layers.parameters():
            p.requires_grad = True
    else:
        for p in model.gcn_layers.parameters():
            p.requires_grad = False


def run_twostage_gcn_shared(
    args,
    run_dir: str,
    device: torch.device,
    gene_list: List[str],
    ds_train: GRNGeneSubsetDataset,
    ds_val: GRNGeneSubsetDataset,
    ds_test: GRNGeneSubsetDataset,
    g_tr: List[int],
    g_va: List[int],
    g_te: List[int],
):
    freeze_node = not args.gcn_unfreeze_node_encoder
    train_gcn_stageb = args.gcn_stageb_train_gcn
    enc_lr_mult = args.gcn_node_encoder_lr_mult

    # full-graph tensors
    cds_tensor = torch.tensor(ds_train.cds_features, dtype=torch.float32, device=device)
    exp_tr = ds_train.exp_data.to(device)
    exp_va = ds_val.exp_data.to(device)
    exp_te = ds_test.exp_data.to(device)

    edge_index = ds_train.ppi_extractor.graph.edge_index.to(device)
    # undirected message edges
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    train_nodes = torch.tensor(g_tr, dtype=torch.long, device=device)
    val_nodes = torch.tensor(g_va, dtype=torch.long, device=device)
    test_nodes = torch.tensor(g_te, dtype=torch.long, device=device)

    tr_src, tr_dst, tr_norm = build_inductive_message_edges(
        edge_index, len(gene_list), train_nodes=train_nodes, mode="train"
    )
    va_src, va_dst, va_norm = build_inductive_message_edges(
        edge_index, len(gene_list), train_nodes=train_nodes, eval_nodes=val_nodes, mode="eval"
    )
    te_src, te_dst, te_norm = build_inductive_message_edges(
        edge_index, len(gene_list), train_nodes=train_nodes, eval_nodes=test_nodes, mode="eval"
    )

    tf_tr, tg_tr, y012_tr = _pairs_to_tensors(ds_train.samples, device)
    tf_va, tg_va, y012_va = _pairs_to_tensors(ds_val.samples, device)
    tf_te, tg_te, y012_te = _pairs_to_tensors(ds_test.samples, device)

    yA_tr = _build_binary_labels_stageA_tensor(y012_tr)
    yA_va = _build_binary_labels_stageA_tensor(y012_va)

    cntA = torch.bincount(yA_tr.cpu(), minlength=2).float()
    nA_tot = float(cntA.sum().item())
    wA = (nA_tot / (2.0 * cntA.clamp_min(1.0))).to(device)
    print(
        f"[StageA] train binary label counts NoReg={int(cntA[0].item())} Reg={int(cntA[1].item())} "
        f"(weighted CE; early_stop=max macro-F1, tie MCC, tie min val_CE)"
    )

    model = PpiGcnTwoStageCascade(
        seq_dim=args.seq_dim,
        emb_dim=args.emb_dim,
        expr_d_model=args.expr_d_model,
        expr_pooled_len=args.expr_pooled_len,
        expr_nhead=args.expr_nhead,
        expr_num_layers=args.expr_num_layers,
        expr_dim_ff=args.expr_dim_ff,
        expr_dropout=args.expr_dropout,
    ).to(device)

    # Stage A：默认冻结节点编码；--gcn-unfreeze-node-encoder 时训练节点初值（显存显著增加）
    if freeze_node:
        for p in model.node_encoder.parameters():
            p.requires_grad = False
        model.node_encoder.eval()
    else:
        for p in model.node_encoder.parameters():
            p.requires_grad = True

    if freeze_node:
        optA = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr_stageA)
    else:
        enc_p = list(model.node_encoder.parameters())
        rest_p = [p for n, p in model.named_parameters() if not n.startswith("node_encoder")]
        optA = torch.optim.Adam(
            [{"params": enc_p, "lr": args.lr_stageA * enc_lr_mult}, {"params": rest_p, "lr": args.lr_stageA}]
        )

    bestA_key: Tuple[float, float, float] = (-1.0, -1.0, float("-inf"))
    patience_counter = 0
    for epoch in range(args.epochs_stageA):
        model.train()
        optA.zero_grad()

        if freeze_node:
            model.node_encoder.eval()
            with torch.no_grad():
                h0_tr = model.node_encoder(exp_tr, cds_tensor, chunk_size=64)
        else:
            model.node_encoder.train()
            h0_tr = model.node_encoder(exp_tr, cds_tensor, chunk_size=64)

        h_tr = h0_tr
        for layer in model.gcn_layers:
            h_tr = layer(h_tr, src=tr_src, dst=tr_dst, edge_norm=tr_norm)

        edge_bs = max(256, args.batch_size * 4)
        lossA_val = 0.0
        nA = len(yA_tr)
        for st in range(0, nA, edge_bs):
            ed = min(nA, st + edge_bs)
            logitsA_chunk = model.stage_a_logits(h_tr, tf_tr[st:ed], tg_tr[st:ed])
            yA_chunk = yA_tr[st:ed]
            loss_chunk = F.cross_entropy(logitsA_chunk, yA_chunk, weight=wA, reduction="mean")
            retain = ed < nA
            loss_chunk.backward(retain_graph=retain)
            lossA_val += float(loss_chunk.item()) * (ed - st)
        _gcn_maybe_clip_grad(model, args.gcn_max_grad_norm)
        optA.step()

        lossA_val = lossA_val / max(nA, 1)

        model.eval()
        with torch.no_grad():
            h0_va = model.node_encoder(exp_va, cds_tensor, chunk_size=64)
            h_va = h0_va
            for layer in model.gcn_layers:
                h_va = layer(h_va, src=va_src, dst=va_dst, edge_norm=va_norm)

            # val：分块 logits，与 train 同权的 mean CE + 分类指标
            edge_bs = max(256, args.batch_size * 4)
            pred_chunks = []
            val_ce_sum = 0.0
            n_va = len(yA_va)
            for st in range(0, n_va, edge_bs):
                ed = min(n_va, st + edge_bs)
                logitsA_chunk = model.stage_a_logits(h_va, tf_va[st:ed], tg_va[st:ed])
                pred_chunks.append(logitsA_chunk.argmax(dim=1))
                val_ce_sum += float(
                    F.cross_entropy(logitsA_chunk, yA_va[st:ed], weight=wA, reduction="sum").item()
                )
            pred_va = torch.cat(pred_chunks, dim=0)
            val_acc = (pred_va == yA_va).float().mean().item()
            yva_np = yA_va.detach().cpu().numpy()
            pva_np = pred_va.detach().cpu().numpy()
            val_bal = _safe_balanced_accuracy(yva_np, pva_np)
            val_mf1 = _binary_macro_f1(yva_np, pva_np)
            val_mcc = _safe_binary_mcc(yva_np, pva_np)
            val_ce = val_ce_sum / max(n_va, 1)
        keyA = (val_mf1, val_mcc, -val_ce)
        print(
            f"[StageA] Epoch {epoch+1}/{args.epochs_stageA} | val_acc={val_acc:.4f} | "
            f"val_balanced_acc={val_bal:.4f} | val_macro_f1={val_mf1:.4f} | val_mcc={val_mcc:.4f} | "
            f"val_ce={val_ce:.4f} | train_loss={lossA_val:.4f}"
        )
        if keyA > bestA_key:
            bestA_key = keyA
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "stageA_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("[StageA] early stop")
                break

    model.load_state_dict(torch.load(os.path.join(run_dir, "stageA_best.pth"), map_location=device))

    # Stage B：默认冻结 GCN；--gcn-stageb-train-gcn 时训练 GCN（并可与节点编码器联合，见 --gcn-unfreeze-node-encoder）
    if not train_gcn_stageb:
        for p in model.gcn_layers.parameters():
            p.requires_grad = False
    else:
        for p in model.gcn_layers.parameters():
            p.requires_grad = True
    for p in model.stage_a_head.parameters():
        p.requires_grad = False

    b_groups = []
    if not freeze_node:
        b_groups.append({"params": list(model.node_encoder.parameters()), "lr": args.lr_stageB * enc_lr_mult})
    if train_gcn_stageb:
        b_groups.append({"params": list(model.gcn_layers.parameters()), "lr": args.lr_stageB})
    b_groups.append({"params": list(model.stage_b_head.parameters()), "lr": args.lr_stageB})
    optB = torch.optim.Adam(b_groups)

    reg_tr = y012_tr > 0
    reg_va = y012_va > 0
    yB_tr = _build_binary_labels_stageB_tensor(y012_tr[reg_tr])
    yB_va = _build_binary_labels_stageB_tensor(y012_va[reg_va])
    cntB = torch.bincount(yB_tr.cpu(), minlength=2).float()
    cntB_va = torch.bincount(yB_va.cpu(), minlength=2).float()
    print(
        f"[StageB] train Repress={int(cntB[0].item())} Activate={int(cntB[1].item())} | "
        f"val Repress={int(cntB_va[0].item())} Activate={int(cntB_va[1].item())} "
        f"(early_stop=max val_macro_f1, tie MCC, tie min weighted val_CE; "
        f"OHEM warmup={args.gcn_ohem_warmup_epochs} ep)"
    )

    bestB_key: Tuple[float, float, float] = (-1.0, -1.0, float("-inf"))
    patience_counter = 0
    hw = int(getattr(args, "gcn_stageb_head_warmup_epochs", 0) or 0)
    if hw > 0:
        print(f"[StageB] head-only warmup enabled: first {hw} epochs freeze GCN + NodeFeatureEncoder")

    for epoch in range(args.epochs_stageB):
        model.train()
        optB.zero_grad()

        head_warmup = hw > 0 and epoch < hw
        _apply_gcn_stageb_requires_grad(
            model,
            freeze_node=freeze_node,
            train_gcn_stageb=train_gcn_stageb,
            head_only_warmup=head_warmup,
        )
        if head_warmup and epoch + 1 == hw:
            print(f"[StageB] epoch {epoch + 1}: last head-only epoch; next epoch joint backbone training")

        backbone_needs_grad = (train_gcn_stageb or (not freeze_node)) and (not head_warmup)
        if not backbone_needs_grad:
            with torch.no_grad():
                model.node_encoder.eval()
                h0_tr = model.node_encoder(exp_tr, cds_tensor, chunk_size=64)
                h_tr = h0_tr
                for layer in model.gcn_layers:
                    h_tr = layer(h_tr, src=tr_src, dst=tr_dst, edge_norm=tr_norm)
        else:
            if freeze_node:
                model.node_encoder.eval()
                with torch.no_grad():
                    h0_tr = model.node_encoder(exp_tr, cds_tensor, chunk_size=64)
            else:
                model.node_encoder.train()
                h0_tr = model.node_encoder(exp_tr, cds_tensor, chunk_size=64)
            h_tr = h0_tr
            for layer in model.gcn_layers:
                h_tr = layer(h_tr, src=tr_src, dst=tr_dst, edge_norm=tr_norm)

        logitsA_tr = model.stage_a_logits(h_tr, tf_tr, tg_tr)

        reg_tf = tf_tr[reg_tr]
        reg_tg = tg_tr[reg_tr]
        reg_logitsA = logitsA_tr[reg_tr]
        nB = len(yB_tr)
        edge_bs = max(256, args.batch_size * 4)
        losses = []
        for st in range(0, nB, edge_bs):
            ed = min(nB, st + edge_bs)
            logitsB_chunk = model.stage_b_logits(
                h_tr,
                reg_tf[st:ed],
                reg_tg[st:ed],
                reg_logitsA[st:ed],
            )
            ce_chunk = F.cross_entropy(logitsB_chunk, yB_tr[st:ed], reduction="none")
            wt_chunk = torch.where(
                yB_tr[st:ed] == 0,
                torch.full_like(ce_chunk, args.stage_b_weight_repress, dtype=torch.float),
                torch.full_like(ce_chunk, args.stage_b_weight_activate, dtype=torch.float),
            )
            if args.stage_b_loss == "focal":
                pt = torch.exp(-ce_chunk)
                per_sample_loss_chunk = ((1.0 - pt) ** args.stage_b_focal_gamma) * ce_chunk * wt_chunk
            else:
                per_sample_loss_chunk = ce_chunk * wt_chunk
            losses.append(per_sample_loss_chunk)
        per_sample_loss = torch.cat(losses, dim=0)
        eff_ohem = float(args.ohem_frac_stageB)
        if args.gcn_ohem_warmup_epochs > 0 and epoch < args.gcn_ohem_warmup_epochs:
            eff_ohem = 1.0
        if eff_ohem < 1.0:
            k = max(1, int(len(per_sample_loss) * eff_ohem))
            lossB = torch.topk(per_sample_loss, k=k, largest=True).values.mean()
        else:
            lossB = per_sample_loss.mean()
        lossB.backward()
        _gcn_maybe_clip_grad(model, args.gcn_max_grad_norm)
        optB.step()

        model.eval()
        with torch.no_grad():
            h0_va = model.node_encoder(exp_va, cds_tensor, chunk_size=64)
            h_va = h0_va
            for layer in model.gcn_layers:
                h_va = layer(h_va, src=va_src, dst=va_dst, edge_norm=va_norm)
            logitsA_va = model.stage_a_logits(h_va, tf_va, tg_va)
            reg_tf_va = tf_va[reg_va]
            reg_tg_va = tg_va[reg_va]
            reg_logitsA_va = logitsA_va[reg_va]

            pred_chunks = []
            val_b_ce_sum = 0.0
            n_bva = len(yB_va)
            w_rep_b = float(args.stage_b_weight_repress)
            w_act_b = float(args.stage_b_weight_activate)
            edge_bs = max(256, args.batch_size * 4)
            for st in range(0, n_bva, edge_bs):
                ed = min(n_bva, st + edge_bs)
                logitsB_chunk = model.stage_b_logits(
                    h_va,
                    reg_tf_va[st:ed],
                    reg_tg_va[st:ed],
                    reg_logitsA_va[st:ed],
                )
                pred_chunks.append(logitsB_chunk.argmax(dim=1))
                ce_v = F.cross_entropy(logitsB_chunk, yB_va[st:ed], reduction="none")
                wt_v = torch.where(
                    yB_va[st:ed] == 0,
                    ce_v.new_full((), w_rep_b),
                    ce_v.new_full((), w_act_b),
                )
                val_b_ce_sum += float((ce_v * wt_v).sum().item())
            pred_va = torch.cat(pred_chunks, dim=0)
            val_acc = (pred_va == yB_va).float().mean().item()
            ybva_np = yB_va.detach().cpu().numpy()
            pbva_np = pred_va.detach().cpu().numpy()
            val_bal = _safe_balanced_accuracy(ybva_np, pbva_np)
            val_mf1 = _binary_macro_f1(ybva_np, pbva_np)
            val_mcc = _safe_binary_mcc(ybva_np, pbva_np)
            val_b_ce = val_b_ce_sum / max(n_bva, 1)
        ohem_note = ""
        if args.ohem_frac_stageB < 1.0 and epoch < args.gcn_ohem_warmup_epochs and args.gcn_ohem_warmup_epochs > 0:
            ohem_note = " [OHEM off:warmup]"
        if head_warmup:
            ohem_note += " [head-only]"
        keyB = (val_mf1, val_mcc, -val_b_ce)
        print(
            f"[StageB] Epoch {epoch+1}/{args.epochs_stageB} | val_acc={val_acc:.4f} | val_balanced_acc={val_bal:.4f} | "
            f"val_macro_f1={val_mf1:.4f} | val_mcc={val_mcc:.4f} | val_wce={val_b_ce:.4f} | train_loss={float(lossB.item()):.4f}"
            f"{ohem_note}"
        )
        if keyB > bestB_key:
            bestB_key = keyB
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "stageB_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("[StageB] early stop")
                break

    model.load_state_dict(torch.load(os.path.join(run_dir, "stageB_best.pth"), map_location=device))

    def _predict_split(ds, exp_all, src, dst, norm, tf_idx, tg_idx, y012, out_csv):
        model.eval()
        with torch.no_grad():
            h = model.encode_nodes(exp_all, cds_tensor, src, dst, norm)
            logitsA = model.stage_a_logits(h, tf_idx, tg_idx)
            pA = torch.softmax(logitsA, dim=1)

            probB = torch.zeros((len(y012), 2), dtype=torch.float32, device=device)
            reg = y012 > 0
            if reg.any():
                logitsB = model.stage_b_logits(h, tf_idx[reg], tg_idx[reg], logitsA[reg])
                probB[reg] = torch.softmax(logitsB, dim=1)

            prob3 = torch.zeros((len(y012), 3), dtype=torch.float32, device=device)
            prob3[:, 0] = pA[:, 0]
            prob3[:, 1] = pA[:, 1] * probB[:, 0]
            prob3[:, 2] = pA[:, 1] * probB[:, 1]

        out = {
            "TF": [gene_list[i] for i, _, _ in ds.samples],
            "Target": [gene_list[j] for _, j, _ in ds.samples],
            "True_Label": y012.detach().cpu().numpy().tolist(),
            "Prob_NoReg": prob3[:, 0].detach().cpu().numpy().tolist(),
            "Prob_Repress": prob3[:, 1].detach().cpu().numpy().tolist(),
            "Prob_Activate": prob3[:, 2].detach().cpu().numpy().tolist(),
        }
        import pandas as pd

        pd.DataFrame(out).to_csv(out_csv, index=False)

    val_csv = os.path.join(run_dir, "val_prediction.csv")
    test_csv = os.path.join(run_dir, "test_prediction.csv")
    _predict_split(ds_val, exp_va, va_src, va_dst, va_norm, tf_va, tg_va, y012_va, val_csv)
    _predict_split(ds_test, exp_te, te_src, te_dst, te_norm, tf_te, tg_te, y012_te, test_csv)

    import pandas as pd

    val_df = pd.read_csv(val_csv)
    y_true_val = val_df["True_Label"].values.astype(int)
    prob_val = val_df[["Prob_NoReg", "Prob_Repress", "Prob_Activate"]].values.astype(np.float32)
    # 层级决策：先判断是否 Reg（Repress+Activate vs NoReg），再在 Reg 内判断 Repress vs Activate
    prob_reg_val = prob_val[:, 1] + prob_val[:, 2]
    no_reg_mask = prob_val[:, 0] >= prob_reg_val
    y_pred_val = np.empty_like(y_true_val, dtype=np.int64)
    y_pred_val[no_reg_mask] = 0
    reg_mask = ~no_reg_mask
    y_pred_val[reg_mask] = np.where(
        prob_val[reg_mask, 1] >= prob_val[reg_mask, 2],
        1,  # Repress
        2,  # Activate
    )
    metrics_val = calculate_metrics(y_true_val, y_pred_val, prob_val)
    with open(os.path.join(run_dir, "val_metrics_argmax.json"), "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(metrics_val), f, ensure_ascii=False, indent=2)

    test_df = pd.read_csv(test_csv)
    y_true_test = test_df["True_Label"].values.astype(int)
    prob_test = test_df[["Prob_NoReg", "Prob_Repress", "Prob_Activate"]].values.astype(np.float32)
    prob_reg_test = prob_test[:, 1] + prob_test[:, 2]
    no_reg_mask = prob_test[:, 0] >= prob_reg_test
    y_pred_test = np.empty_like(y_true_test, dtype=np.int64)
    y_pred_test[no_reg_mask] = 0
    reg_mask = ~no_reg_mask
    y_pred_test[reg_mask] = np.where(
        prob_test[reg_mask, 1] >= prob_test[reg_mask, 2],
        1,  # Repress
        2,  # Activate
    )
    metrics_test = calculate_metrics(y_true_test, y_pred_test, prob_test)
    with open(os.path.join(run_dir, "test_metrics_argmax.json"), "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(metrics_test), f, ensure_ascii=False, indent=2)

    if args.gcn_auto_tune_threshold:
        min_act = args.gcn_tune_min_activate_accuracy
        if min_act is not None and min_act < 0:
            min_act = None
        tune_kw = dict(
            objective=args.gcn_tune_objective,
            grid_min=args.gcn_tune_grid_min,
            grid_max=args.gcn_tune_grid_max,
            grid_step=args.gcn_tune_grid_step,
            min_repress_precision=args.gcn_tune_min_repress_precision,
            min_activate_accuracy=min_act,
        )
        try:
            run_threshold_tune(val_csv, test_csv, **tune_kw)
        except RuntimeError as e:
            print(f"[threshold] 约束下无可行阈值: {e}")
            print(
                "[threshold] 改用无约束搜索（min_repress_precision=0, min_activate_accuracy=None），"
                "仍写入 threshold_tuned_metrics.json。"
            )
            run_threshold_tune(
                val_csv,
                test_csv,
                objective=args.gcn_tune_objective,
                grid_min=args.gcn_tune_grid_min,
                grid_max=args.gcn_tune_grid_max,
                grid_step=args.gcn_tune_grid_step,
                min_repress_precision=0.0,
                min_activate_accuracy=None,
            )

    print(f"[Done] run_dir={run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Gene-disjoint + unseen bulk + two-stage GRN classifier (TCGA-LUAD).")

    parser.add_argument(
        "--exp-csv",
        required=True,
        help="参考表达 CSV：第一列为 gene。未指定 --reference-gene-list-csv 时，用它作为锚点基因列表（与原先 GSE+TCGA 流程一致）。",
    )
    parser.add_argument("--grn-csv", required=True, help="Known GRN CSV.")
    parser.add_argument("--grn-tf-col", default="TF", help="TF symbol column in GRN CSV.")
    parser.add_argument("--grn-target-col", default="target", help="Target symbol column in GRN CSV.")
    parser.add_argument("--grn-type-col", default="Re_type", help="Relation type column in GRN CSV.")
    parser.add_argument("--cds-file", default=None, help="CDS feature CSV (legacy row-aligned mode).")
    parser.add_argument("--tss-file", default=None, help="TSS feature CSV (legacy row-aligned mode).")
    parser.add_argument(
        "--cds-gene-feature-csv",
        default=None,
        help="CDS gene feature CSV (first col gene, rest features).",
    )
    parser.add_argument(
        "--tss-gene-feature-csv",
        default=None,
        help="TSS gene feature CSV (first col gene, rest features).",
    )
    parser.add_argument(
        "--ppi-csv",
        required=True,
        help="PPI 边表 CSV（from,to）；可用 build_ppi_csv_from_string 从 STRING 生成。",
    )

    parser.add_argument(
        "--tcga-expression-csv",
        "--tcga-aligned-csv",
        dest="tcga_expression_csv",
        required=True,
        help=(
            "TCGA/外部 bulk 表达矩阵路径：行=gene symbol，列=样本；第一列基因名或列名 Gene。"
            "可直接使用原始 FPKM symbol 表（如 TCGA-LUSC_FPKM_Symbol.csv），无需先跑 prepare 生成 processed 文件。"
        ),
    )
    parser.add_argument(
        "--reference-gene-list-csv",
        default=None,
        help="可选：锚点基因列表（第一列 gene）。默认与未指定时，使用 --exp-csv 的第一列；仅保留 TCGA 矩阵中存在的基因且不补零。",
    )
    parser.add_argument(
        "--anchor-gene-order",
        choices=["expression", "reference"],
        default="expression",
        help="锚点后基因顺序。expression=与历史 E5 口径一致（按表达文件原顺序过滤）；reference=按参考基因列表顺序。",
    )
    parser.add_argument("--output-base-dir", default="experiments")
    parser.add_argument("--experiment-name", default="gene_disjoint_unseen_bulk_twostage_tcga")
    parser.add_argument("--no-timestamp", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-stageA", type=float, default=1e-3)
    parser.add_argument("--lr-stageB", type=float, default=1e-3)
    parser.add_argument("--epochs-stageA", type=int, default=50)
    parser.add_argument("--epochs-stageB", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--neg-sample-ratio", type=float, default=1.0)

    parser.add_argument("--ohem-frac-stageB", type=float, default=1.0)

    # Split ratios: genes 6:2:2, bulk columns 5:2:3
    parser.add_argument("--gene-split", default="6,2,2")
    parser.add_argument(
        "--gene-split-mode",
        choices=["random", "weighted_regulated"],
        default="random",
        help="random: 与原先一致；weighted_regulated: 在固定基因比例下最大化 w_rep*块内抑制边 + w_act*块内激活边。",
    )
    parser.add_argument("--gene-split-weight-repress", type=float, default=1.0)
    parser.add_argument("--gene-split-weight-activate", type=float, default=0.25)
    parser.add_argument("--gene-split-opt-steps", type=int, default=30000)
    parser.add_argument("--gene-split-opt-starts", type=int, default=3)
    parser.add_argument(
        "--gene-split-balance-gamma",
        type=float,
        default=0.0,
        help="仅 weighted_regulated：目标 += gamma * min(各块内受监管边数)，抬高 val/test 块内正边下限，减轻 train 样本远大于 val/test。0 关闭。",
    )
    parser.add_argument(
        "--gene-split-assignment-json",
        default=None,
        help="若路径存在则加载 train/val/test 基因列表并忽略随机/加权划分；加权划分完成后写入 run_dir/gene_split_assignment.json；若指定路径不存在也会在优化后写入该路径。",
    )
    parser.add_argument("--bulk-split", default="5,2,3")

    # Expression transform: log2(x + offset)
    parser.add_argument("--log-offset", type=float, default=1.0)

    # Model hyperparams
    parser.add_argument("--emb-dim", type=int, default=160)
    parser.add_argument("--expr-d-model", type=int, default=128)
    parser.add_argument("--expr-pooled-len", type=int, default=512)
    parser.add_argument("--expr-nhead", type=int, default=4)
    parser.add_argument("--expr-num-layers", type=int, default=2)
    parser.add_argument("--expr-dim-ff", type=int, default=256)
    parser.add_argument("--expr-dropout", type=float, default=0.1)

    parser.add_argument(
        "--fusion-variant",
        choices=["legacy", "e7", "e7_stageB_noppi", "e7_stageB_ppi_cascade", "e7_stageB_ppi_gcn"],
        default="legacy",
        help=(
            "legacy: 原 FusionModel（含 PPI）；"
            "e7: Stage-A/Stage-B 均无 PPI；"
            "e7_stageB_noppi: Stage-A 保留 PPI，Stage-B 用无 PPI 级联门控；"
            "e7_stageB_ppi_cascade: Stage-A 保留 PPI，Stage-B 级联门控且保留 PPI（node-level）；"
            "e7_stageB_ppi_gcn: 全图 PPI-GCN 共享节点编码器 + 两阶段级联。"
        ),
    )
    parser.add_argument(
        "--stage-b-loss",
        choices=["ce", "focal"],
        default="focal",
        help="仅 fusion-variant=e7 的 Stage-B：加权 CE 或 focal（均在样本上乘 class weight）。",
    )
    parser.add_argument("--stage-b-focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--stage-b-weight-repress",
        type=float,
        default=2.0,
        help="Stage-B 中 Repress(类0) 的样本权重乘子。",
    )
    parser.add_argument(
        "--stage-b-weight-activate",
        type=float,
        default=1.0,
        help="Stage-B 中 Activate(类1) 的样本权重乘子。",
    )

    # E7.2 PPI-GCN：路线 B（解冻 / Stage-B 训 GCN）与路线 A（训练结束阈值搜索，口径对齐 E5）
    parser.add_argument(
        "--gcn-unfreeze-node-encoder",
        action="store_true",
        help="仅 fusion-variant=e7_stageB_ppi_gcn：训练 NodeFeatureEncoder（显存占用显著上升）。",
    )
    parser.add_argument(
        "--gcn-stageb-train-gcn",
        action="store_true",
        help="仅 e7_stageB_ppi_gcn：Stage-B 不冻结 GCN，与 stage_b_head（及可选节点编码器）联合训练。",
    )
    parser.add_argument(
        "--gcn-node-encoder-lr-mult",
        type=float,
        default=0.1,
        help="节点编码器相对 Stage-A/B 基础学习率的乘子（param group）。",
    )
    parser.add_argument(
        "--gcn-auto-tune-threshold",
        action="store_true",
        help="e7_stageB_ppi_gcn：训练结束后在 val 上搜索阈值并写 threshold_tuned_metrics.json（同 tune_repress_threshold）。",
    )
    parser.add_argument(
        "--gcn-tune-objective",
        default="repress_recall",
        choices=["repress_f1", "repress_recall"],
        help="gcn-auto-tune-threshold 时的优化目标（默认与报告 E5：repress_recall）。",
    )
    parser.add_argument("--gcn-tune-grid-min", type=float, default=0.0)
    parser.add_argument("--gcn-tune-grid-max", type=float, default=0.99)
    parser.add_argument("--gcn-tune-grid-step", type=float, default=0.01)
    parser.add_argument(
        "--gcn-tune-min-repress-precision",
        type=float,
        default=0.35,
        help="阈值搜索约束，与报告 E5 默认一致。",
    )
    parser.add_argument(
        "--gcn-tune-min-activate-accuracy",
        type=float,
        default=0.70,
        help="阈值搜索约束；设为负数则关闭该约束。",
    )
    parser.add_argument(
        "--gcn-max-grad-norm",
        type=float,
        default=2.0,
        help="PPI-GCN 两阶段：按范数裁剪梯度，减轻解冻编码器+GCN 时的震荡；0 关闭。",
    )
    parser.add_argument(
        "--gcn-ohem-warmup-epochs",
        type=int,
        default=25,
        help="PPI-GCN Stage-B：前若干 epoch 使用全量样本损失（等价关闭 OHEM），再恢复 --ohem-frac-stageB；0 表示全程按 ohem 设置。",
    )
    parser.add_argument(
        "--gcn-stageb-head-warmup-epochs",
        type=int,
        default=0,
        help=(
            "PPI-GCN Stage-B：前若干 epoch 仅训练 stage_b_head（GCN 与 NodeFeatureEncoder 不参与反传），"
            "在固定 h 上先学 Repress/Activate，再按原设置联合训练；0 关闭。建议不稳时试 8～15。"
        ),
    )

    args = parser.parse_args()
    if args.cds_file is None and args.cds_gene_feature_csv is None:
        raise ValueError("Either --cds-file or --cds-gene-feature-csv must be provided.")
    stageA_noppi = args.fusion_variant == "e7"
    stageB_cascade = args.fusion_variant in ["e7", "e7_stageB_noppi", "e7_stageB_ppi_cascade", "e7_stageB_ppi_gcn"]
    stageB_noppi = args.fusion_variant in ["e7", "e7_stageB_noppi"]
    use_ppi_gcn = args.fusion_variant == "e7_stageB_ppi_gcn"

    set_seed(args.seed)
    os.makedirs(args.output_base_dir, exist_ok=True)
    if args.no_timestamp:
        run_dir = os.path.join(args.output_base_dir, args.experiment_name)
    else:
        run_dir = os.path.join(args.output_base_dir, args.experiment_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) expression anchor（不补零）
    import pandas as pd

    expr_df = pd.read_csv(args.tcga_expression_csv)
    gene_col = "Gene" if "Gene" in expr_df.columns else expr_df.columns[0]
    expr_df[gene_col] = expr_df[gene_col].astype(str).str.strip()
    expr_df = expr_df[expr_df[gene_col] != ""]
    expr_df = expr_df.drop_duplicates(subset=[gene_col], keep="first").set_index(gene_col)
    expr_df = expr_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    ref_csv = args.reference_gene_list_csv or args.exp_csv
    ref = pd.read_csv(ref_csv, usecols=[0]).iloc[:, 0].astype(str).str.strip().tolist()
    ref_map = {g: i for i, g in enumerate(ref)}
    expr_order = expr_df.index.tolist()
    expr_index_set = set(expr_order)
    if args.anchor_gene_order == "reference":
        kept = [g for g in ref if g in expr_index_set]
    else:
        ref_set = set(ref_map.keys())
        kept = [g for g in expr_order if g in ref_set]

    # 可选：按 gene-level 特征文件进一步对齐基因集合
    cds_gene_feat = _load_gene_feature_csv(args.cds_gene_feature_csv) if args.cds_gene_feature_csv else None
    tss_gene_feat = _load_gene_feature_csv(args.tss_gene_feature_csv) if args.tss_gene_feature_csv else None
    if cds_gene_feat is not None:
        feat_genes = set(cds_gene_feat.index.tolist())
        if tss_gene_feat is not None:
            feat_genes &= set(tss_gene_feat.index.tolist())
        kept = [g for g in kept if g in feat_genes]

    expr_df = expr_df.loc[kept]
    ref_indices = [ref_map[g] for g in kept]
    print(
        f"[twostage] expression anchor genes={len(kept)} from {ref_csv!r} "
        f"(no fill-zero, order={args.anchor_gene_order})"
    )

    gene_list = expr_df.index.tolist()
    raw_expression = expr_df.values.astype(np.float32)
    num_genes, num_samples = raw_expression.shape

    # 2) GRN matrix
    known_GRN_df = pd.read_csv(args.grn_csv)
    tf_col, tg_col, tp_col = resolve_grn_table_columns(
        known_GRN_df, args.grn_tf_col, args.grn_target_col, args.grn_type_col
    )
    if tf_col is None or tg_col is None or tp_col is None:
        raise ValueError(
            f"GRN columns not found. got columns={list(known_GRN_df.columns)}; "
            f"expect tf={args.grn_tf_col}, target={args.grn_target_col}, type={args.grn_type_col} "
            "or fallback source_symbol/target_symbol/relation_subtype."
        )
    known_GRN_df = known_GRN_df[
        known_GRN_df[tf_col].astype(str).isin(set(gene_list))
        & known_GRN_df[tg_col].astype(str).isin(set(gene_list))
    ].copy()

    # 标签清洗：同一 TF-Target 对若同时出现 activator 与 repressor，视为冲突并丢弃
    # （否则 get_GRN 会因为重复行覆盖顺序不同引入噪声）
    known_GRN_df = known_GRN_df.copy()
    rel = known_GRN_df[tp_col].astype(str).str.strip().str.lower()
    rel = rel.replace(
        {
            "activation": "activator",
            "activate": "activator",
            "positive": "activator",
            "upregulation": "activator",
            "inhibition": "repressor",
            "inhibit": "repressor",
            "repression": "repressor",
            "negative": "repressor",
            "downregulation": "repressor",
        }
    )
    known_GRN_df["Re_type"] = rel
    known_GRN_df = known_GRN_df[known_GRN_df["Re_type"].isin(["activator", "repressor"])].copy()

    grp = known_GRN_df.groupby([tf_col, tg_col])["Re_type"].apply(lambda s: set(s.tolist()))
    conflict_pairs = grp[grp.apply(lambda ss: ("activator" in ss) and ("repressor" in ss))].index
    if len(conflict_pairs) > 0:
        conflict_set = set(conflict_pairs.tolist())
        mask_keep = []
        for tf, tgt in zip(known_GRN_df[tf_col].astype(str).values, known_GRN_df[tg_col].astype(str).values):
            mask_keep.append((tf, tgt) not in conflict_set)
        known_GRN_df = known_GRN_df.loc[mask_keep]

    tf_list = known_GRN_df[tf_col].astype(str).values
    target_list = known_GRN_df[tg_col].astype(str).values
    re_type_list = known_GRN_df["Re_type"].values
    known_GRN = (tf_list, target_list, re_type_list)
    GRN_matrix = get_GRN(known_GRN, gene_list)

    # 3) sequence features（支持 gene-level 对齐：CDS 与可选 TSS 拼接）
    if cds_gene_feat is not None or tss_gene_feat is not None:
        feat_blocks = []
        if cds_gene_feat is not None:
            feat_blocks.append(cds_gene_feat.loc[gene_list].values.astype(np.float32))
        else:
            cds_all = pd.read_csv(args.cds_file, header=None, sep=",", skiprows=1).values.astype(np.float32)
            if ref_indices:
                max_idx = max(ref_indices)
                if max_idx >= len(cds_all):
                    raise ValueError(f"CDS行数不足：需要索引到 {max_idx}，但仅有 {len(cds_all)} 行")
            feat_blocks.append(cds_all[ref_indices])

        if tss_gene_feat is not None:
            feat_blocks.append(tss_gene_feat.loc[gene_list].values.astype(np.float32))
        elif args.tss_file:
            tss_all = pd.read_csv(args.tss_file, header=None, sep=",", skiprows=1).values.astype(np.float32)
            if ref_indices:
                max_idx = max(ref_indices)
                if max_idx >= len(tss_all):
                    raise ValueError(f"TSS行数不足：需要索引到 {max_idx}，但仅有 {len(tss_all)} 行")
            feat_blocks.append(tss_all[ref_indices])

        cds_features = np.concatenate(feat_blocks, axis=1)
    else:
        cds_all = pd.read_csv(args.cds_file, header=None, sep=",", skiprows=1).values.astype(np.float32)
        if ref_indices:
            max_idx = max(ref_indices)
            if max_idx >= len(cds_all):
                raise ValueError(f"CDS行数不足：需要索引到 {max_idx}，但仅有 {len(cds_all)} 行")
        cds_features = cds_all[ref_indices]
    if len(cds_features) != len(gene_list):
        raise ValueError(f"序列特征与表达基因数量不一致：{len(cds_features)} vs {len(gene_list)}")
    print(f"[twostage] sequence feature dim={cds_features.shape[1]}")
    args.seq_dim = int(cds_features.shape[1])

    # 4) split genes
    g1, g2, g3 = [int(x.strip()) for x in args.gene_split.split(",")]
    a, b, c = split3(num_genes, g1, g2, g3)
    assign_path = args.gene_split_assignment_json
    if assign_path and os.path.isfile(assign_path):
        g_tr, g_va, g_te, split_meta = load_gene_split_json(assign_path, gene_list, (a, b, c))
        print(f"[gene-split] loaded assignment json: {assign_path} | meta keys={list(split_meta.keys())}")
    elif args.gene_split_mode == "weighted_regulated":
        tf_i, tg_i, is_rep = regulated_edges_from_grn_matrix(GRN_matrix)
        assignment, opt_meta = optimize_gene_split_weighted(
            num_genes=num_genes,
            sizes=(a, b, c),
            tf_i=tf_i,
            tg_i=tg_i,
            is_repress=is_rep,
            w_repress=float(args.gene_split_weight_repress),
            w_activate=float(args.gene_split_weight_activate),
            seed=int(args.seed),
            steps=int(args.gene_split_opt_steps),
            n_random_starts=int(args.gene_split_opt_starts),
            balance_gamma=float(args.gene_split_balance_gamma),
        )
        g_tr, g_va, g_te = assignment_to_gene_lists(assignment)
        out_assign = os.path.join(run_dir, "gene_split_assignment.json")
        save_gene_split_json(
            out_assign,
            gene_list,
            assignment,
            (a, b, c),
            args.gene_split,
            extra=opt_meta,
        )
        print(f"[gene-split] weighted_regulated | {opt_meta} | saved {out_assign}")
        if assign_path and not os.path.isfile(assign_path):
            save_gene_split_json(
                assign_path,
                gene_list,
                assignment,
                (a, b, c),
                args.gene_split,
                extra=opt_meta,
            )
            print(f"[gene-split] also saved {assign_path}")
    else:
        perm = np.random.RandomState(args.seed).permutation(num_genes)
        g_tr = perm[:a].tolist()
        g_va = perm[a : a + b].tolist()
        g_te = perm[a + b :].tolist()

    # 5) split bulk columns (unseen samples)
    c1, c2, c3 = [int(x.strip()) for x in args.bulk_split.split(",")]
    ca, cb, cc = split3(num_samples, c1, c2, c3)
    col_tr = slice(0, ca)
    col_va = slice(ca, ca + cb)
    col_te = slice(ca + cb, num_samples)

    # 6) build datasets
    def make_exp_tensor(cols: slice) -> torch.Tensor:
        x = raw_expression[:, cols]
        return log_expression_tensor(x, offset=args.log_offset)

    exp_tr = make_exp_tensor(col_tr)
    exp_va = make_exp_tensor(col_va)
    exp_te = make_exp_tensor(col_te)

    ds_train = GRNGeneSubsetDataset(
        exp_data=exp_tr,
        GRN_matrix=GRN_matrix,
        cds_file=args.cds_file,
        cds_features=cds_features,
        gene_list=gene_list,
        ppi_file=args.ppi_csv,
        ppi_extractor=None,
        allowed_gene_indices=g_tr,
        neg_sample_ratio=args.neg_sample_ratio,
        device=None,
        seed=args.seed,
        use_ppi=not stageA_noppi,
    )
    ds_val = GRNGeneSubsetDataset(
        exp_data=exp_va,
        GRN_matrix=GRN_matrix,
        cds_file=args.cds_file,
        cds_features=cds_features,
        gene_list=gene_list,
        ppi_file=args.ppi_csv,
        ppi_extractor=ds_train.ppi_extractor,
        allowed_gene_indices=g_va,
        neg_sample_ratio=args.neg_sample_ratio,
        device=None,
        seed=args.seed + 1,
        use_ppi=not stageA_noppi,
    )
    ds_test = GRNGeneSubsetDataset(
        exp_data=exp_te,
        GRN_matrix=GRN_matrix,
        cds_file=args.cds_file,
        cds_features=cds_features,
        gene_list=gene_list,
        ppi_file=args.ppi_csv,
        ppi_extractor=ds_train.ppi_extractor,
        allowed_gene_indices=g_te,
        neg_sample_ratio=args.neg_sample_ratio,
        device=None,
        seed=args.seed + 2,
        use_ppi=not stageA_noppi,
    )
    print(
        f"[pair-count] train={len(ds_train.samples)} "
        f"(pos={len(ds_train.positive_pairs)}, neg={len(ds_train.negative_pairs)}) | "
        f"val={len(ds_val.samples)} (pos={len(ds_val.positive_pairs)}, neg={len(ds_val.negative_pairs)}) | "
        f"test={len(ds_test.samples)} (pos={len(ds_test.positive_pairs)}, neg={len(ds_test.negative_pairs)})"
    )

    if not stageA_noppi:
        # PPI inductive features：训练/验证/测试严格按“训练基因可见，测试基因不可相互聚合”
        ds_train.ppi_extractor.set_train_mask(g_tr)

    if use_ppi_gcn:
        run_twostage_gcn_shared(
            args=args,
            run_dir=run_dir,
            device=device,
            gene_list=gene_list,
            ds_train=ds_train,
            ds_val=ds_val,
            ds_test=ds_test,
            g_tr=g_tr,
            g_va=g_va,
            g_te=g_te,
        )
        return

    # 7) Stage A: NoReg vs Reg
    if stageA_noppi:
        modelA = FusionModelNoPPI(
            seq_dim=args.seq_dim,
            emb_dim=args.emb_dim,
            num_classes=2,
            expr_d_model=args.expr_d_model,
            expr_pooled_len=args.expr_pooled_len,
            expr_nhead=args.expr_nhead,
            expr_num_layers=args.expr_num_layers,
            expr_dim_ff=args.expr_dim_ff,
            expr_dropout=args.expr_dropout,
        ).to(device)
    else:
        modelA = FusionModel(
            seq_dim=args.seq_dim,
            emb_dim=args.emb_dim,
            num_classes=2,
            expr_d_model=args.expr_d_model,
            expr_pooled_len=args.expr_pooled_len,
            expr_nhead=args.expr_nhead,
            expr_num_layers=args.expr_num_layers,
            expr_dim_ff=args.expr_dim_ff,
            expr_dropout=args.expr_dropout,
        ).to(device)

    # Stage A labels
    yA_train = _build_binary_labels_stageA([lab for _, _, lab in ds_train.samples])
    samplerA = _weighted_sampler_from_binary_labels(yA_train, seed=args.seed)
    loaderA_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=samplerA)
    loaderA_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

    optA = torch.optim.Adam(modelA.parameters(), lr=args.lr_stageA)

    bestA = -1.0
    patience_counter = 0
    for epoch in range(args.epochs_stageA):
        train_one_epoch_stageA(modelA, loaderA_train, optA, device=device, stageA_noppi=stageA_noppi)
        met = evaluate_stageA(
            modelA,
            loaderA_val,
            device=device,
            ppi_extractor=ds_train.ppi_extractor if not stageA_noppi else None,
            train_genes=g_tr,
            eval_genes=g_va,
            stageA_noppi=stageA_noppi,
        )
        val_acc = met["acc"]
        print(f"[StageA] Epoch {epoch+1}/{args.epochs_stageA} | val_acc={val_acc:.4f}")

        if val_acc > bestA:
            bestA = val_acc
            patience_counter = 0
            torch.save(modelA.state_dict(), os.path.join(run_dir, "stageA_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("[StageA] early stop")
                break

    modelA.load_state_dict(torch.load(os.path.join(run_dir, "stageA_best.pth"), map_location=device))

    # 8) Stage B: Repress vs Activate on regulated pairs
    if stageB_cascade:
        if stageB_noppi:
            modelB = CascadeStageBModel(
                seq_dim=args.seq_dim,
                emb_dim=args.emb_dim,
                num_classes=2,
                expr_d_model=args.expr_d_model,
                expr_pooled_len=args.expr_pooled_len,
                expr_nhead=args.expr_nhead,
                expr_num_layers=args.expr_num_layers,
                expr_dim_ff=args.expr_dim_ff,
                expr_dropout=args.expr_dropout,
            ).to(device)
        else:
            modelB = CascadeStageBModelWithPPI(
                seq_dim=args.seq_dim,
                emb_dim=args.emb_dim,
                num_classes=2,
                expr_d_model=args.expr_d_model,
                expr_pooled_len=args.expr_pooled_len,
                expr_nhead=args.expr_nhead,
                expr_num_layers=args.expr_num_layers,
                expr_dim_ff=args.expr_dim_ff,
                expr_dropout=args.expr_dropout,
            ).to(device)
    else:
        modelB = FusionModel(
            seq_dim=args.seq_dim,
            emb_dim=args.emb_dim,
            num_classes=2,
            expr_d_model=args.expr_d_model,
            expr_pooled_len=args.expr_pooled_len,
            expr_nhead=args.expr_nhead,
            expr_num_layers=args.expr_num_layers,
            expr_dim_ff=args.expr_dim_ff,
            expr_dropout=args.expr_dropout,
        ).to(device)

    reg_indices_train = [idx for idx, (_, _, lab) in enumerate(ds_train.samples) if lab > 0]
    ds_train_reg = Subset(ds_train, reg_indices_train)
    yB_train = _build_binary_labels_stageB([ds_train.samples[idx][2] for idx in reg_indices_train])
    samplerB = _weighted_sampler_from_binary_labels(yB_train, seed=args.seed + 7)
    loaderB_train = DataLoader(ds_train_reg, batch_size=args.batch_size, sampler=samplerB)

    reg_indices_val = [idx for idx, (_, _, lab) in enumerate(ds_val.samples) if lab > 0]
    ds_val_reg = Subset(ds_val, reg_indices_val)
    loaderB_val = DataLoader(ds_val_reg, batch_size=args.batch_size, shuffle=False)

    optB = torch.optim.Adam(modelB.parameters(), lr=args.lr_stageB)

    bestB = -1.0
    patience_counter = 0
    for epoch in range(args.epochs_stageB):
        train_one_epoch_stageB(
            modelB,
            loaderB_train,
            optB,
            device=device,
            ohem_frac=args.ohem_frac_stageB,
            stageB_noppi=stageB_noppi,
            stageB_cascade=stageB_cascade,
            stageA_noppi=stageA_noppi,
            model_a=modelA if stageB_cascade else None,
            stage_b_loss=args.stage_b_loss,
            focal_gamma=args.stage_b_focal_gamma,
            w_repress=args.stage_b_weight_repress,
            w_activate=args.stage_b_weight_activate,
        )
        met = evaluate_stageB(
            modelB,
            loaderB_val,
            device=device,
            ppi_extractor=ds_train.ppi_extractor if not stageA_noppi else None,
            train_genes=g_tr,
            eval_genes=g_va,
            stageB_noppi=stageB_noppi,
            stageB_cascade=stageB_cascade,
            stageA_noppi=stageA_noppi,
            model_a=modelA if stageB_cascade else None,
        )
        val_acc = met["acc"]
        print(f"[StageB] Epoch {epoch+1}/{args.epochs_stageB} | val_acc={val_acc:.4f}")

        if val_acc > bestB:
            bestB = val_acc
            patience_counter = 0
            torch.save(modelB.state_dict(), os.path.join(run_dir, "stageB_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("[StageB] early stop")
                break

    modelB.load_state_dict(torch.load(os.path.join(run_dir, "stageB_best.pth"), map_location=device))

    # 9) Predict combined probs for val/test, save CSV
    def save_predictions(ds: GRNGeneSubsetDataset, csv_path: str):
        eval_genes = ds.allowed_gene_indices
        gene_pairs, y_true, prob_3 = predict_combined_probs(
            modelA,
            modelB,
            ds,
            gene_list,
            device=device,
            train_genes=g_tr,
            eval_genes=eval_genes,
            stageB_noppi=stageB_noppi,
            stageB_cascade=stageB_cascade,
            stageA_noppi=stageA_noppi,
        )
        tf_names = [p[0] for p in gene_pairs]
        target_names = [p[1] for p in gene_pairs]
        out = {
            "TF": tf_names,
            "Target": target_names,
            "True_Label": y_true.tolist(),
            "Prob_NoReg": prob_3[:, 0].tolist(),
            "Prob_Repress": prob_3[:, 1].tolist(),
            "Prob_Activate": prob_3[:, 2].tolist(),
        }
        import pandas as pd

        pd.DataFrame(out).to_csv(csv_path, index=False)
        return out

    val_csv = os.path.join(run_dir, "val_prediction.csv")
    test_csv = os.path.join(run_dir, "test_prediction.csv")
    print("[Predict] val_prediction.csv ...")
    save_predictions(ds_val, val_csv)
    print("[Predict] test_prediction.csv ...")
    save_predictions(ds_test, test_csv)

    # quick overall metrics (argmax without threshold)
    import pandas as pd

    val_df = pd.read_csv(val_csv)
    y_true_val = val_df["True_Label"].values.astype(int)
    prob_val = val_df[["Prob_NoReg", "Prob_Repress", "Prob_Activate"]].values.astype(np.float32)
    y_pred_val = prob_val.argmax(axis=1)
    metrics_val = calculate_metrics(y_true_val, y_pred_val, prob_val)
    with open(os.path.join(run_dir, "val_metrics_argmax.json"), "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(metrics_val), f, ensure_ascii=False, indent=2)

    test_df = pd.read_csv(test_csv)
    y_true_test = test_df["True_Label"].values.astype(int)
    prob_test = test_df[["Prob_NoReg", "Prob_Repress", "Prob_Activate"]].values.astype(np.float32)
    y_pred_test = prob_test.argmax(axis=1)
    metrics_test = calculate_metrics(y_true_test, y_pred_test, prob_test)
    with open(os.path.join(run_dir, "test_metrics_argmax.json"), "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(metrics_test), f, ensure_ascii=False, indent=2)

    print(f"[Done] run_dir={run_dir}")


if __name__ == "__main__":
    main()

