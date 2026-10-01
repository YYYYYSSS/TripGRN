from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader

from grn.data.de_cartesian_pairs import build_ordered_pairs_packed
from grn.data.kegg_khop_pairs import build_index_pairs_from_kegg_khop
from grn.data.preprocessing import (
    get_GRN,
    log2_minmax_per_gene_fit,
    log2_minmax_per_gene_transform,
    normalize_gene_symbol_series,
    resolve_grn_table_columns,
)
from grn.models.fusion_model import FusionModelAblation
from grn.scripts.train_unseen_samples_no_cv import (
    PairSubsetColumnDataset,
    _build_pair_samples,
    _forward_logits,
    _load_gene_feature_csv,
    _prepare_batch,
    set_seed,
    split3,
)


CLASS_NAMES = ("NoReg", "Repress", "Activate")


def _ts_log(msg: str) -> None:
    """Stdout heartbeat for long runs (conda may still buffer unless PYTHONUNBUFFERED=1 / python -u)."""
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [case study] {msg}", flush=True)


def _autocast_bf16(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _label_slug(s: str) -> str:
    """CSV/列名用：将展示标签中的连字符转为下划线。"""
    return str(s).strip().replace("-", "_")


EDGE_COLOR_MAP = {
    "mBrain_gain_regulation": "#e74c3c",
    "mBrain_loss_regulation": "#3498db",
    "repression_to_activation": "#f39c12",
    "activation_to_repression": "#9b59b6",
    "other_class_switch": "#7f8c8d",
    "same_class_prob_shift": "#95a5a6",
}


def _predict_with_names(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    gene_list: List[str],
    progress_desc: str = "predict",
    show_progress: bool = True,
    use_amp_bf16: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    model.eval()
    y_true, y_pred = [], []
    probs = []
    tf_names, tg_names = [], []
    total_batches = len(loader)
    start_ts = time.time()
    progress_every = max(1, total_batches // 100) if total_batches > 0 else 1
    if show_progress and total_batches > 0:
        print(
            f"[{progress_desc}] batches={total_batches}, batch_size={loader.batch_size}, "
            f"amp_bf16={use_amp_bf16}, starting…",
            flush=True,
        )
    with torch.inference_mode():
        for bi, batch in enumerate(loader, start=1):
            x = _prepare_batch(batch, device)
            with _autocast_bf16(device, use_amp_bf16):
                logits = _forward_logits(model, x)
            prob = torch.softmax(logits.float(), dim=1).cpu().numpy()
            pred = np.argmax(prob, axis=1).astype(np.int64)
            y_true.extend(batch["label"].cpu().numpy().astype(np.int64).tolist())
            y_pred.extend(pred.tolist())
            probs.append(prob)
            ti = batch["tf_idx"].cpu().numpy().astype(np.int64, copy=False)
            tj = batch["target_idx"].cpu().numpy().astype(np.int64, copy=False)
            tf_names.extend(gene_list[int(i)] for i in ti)
            tg_names.extend(gene_list[int(j)] for j in tj)
            if show_progress and total_batches > 0 and (bi == 1 or bi % progress_every == 0 or bi == total_batches):
                elapsed = time.time() - start_ts
                pct = 100.0 * bi / total_batches
                sys.stdout.write(
                    f"\r[{progress_desc}] batch {bi}/{total_batches} ({pct:5.1f}%), pairs={len(y_true)}, elapsed={elapsed/60.0:.1f}m"
                )
                sys.stdout.flush()
    if show_progress and total_batches > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        np.vstack(probs) if probs else np.zeros((0, 3), dtype=np.float32),
        tf_names,
        tg_names,
    )


def _count_pred_classes(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    n_classes: int = 3,
    progress_desc: str = "count",
    show_progress: bool = True,
    use_amp_bf16: bool = False,
) -> Tuple[int, np.ndarray]:
    """Argmax class histogram only (no pair names / softmax dump)."""
    model.eval()
    counts = np.zeros(int(n_classes), dtype=np.int64)
    n_seen = 0
    total_batches = len(loader)
    start_ts = time.time()
    progress_every = max(1, total_batches // 100) if total_batches > 0 else 1
    if show_progress and total_batches > 0:
        print(
            f"[{progress_desc}] batches={total_batches}, batch_size={loader.batch_size}, "
            f"amp_bf16={use_amp_bf16}, starting…",
            flush=True,
        )
    with torch.inference_mode():
        for bi, batch in enumerate(loader, start=1):
            x = _prepare_batch(batch, device)
            with _autocast_bf16(device, use_amp_bf16):
                logits = _forward_logits(model, x)
            pred = torch.argmax(logits, dim=1)
            counts += torch.bincount(pred, minlength=int(n_classes)).detach().cpu().numpy().astype(np.int64)
            n_seen += int(pred.numel())
            if show_progress and total_batches > 0 and (bi == 1 or bi % progress_every == 0 or bi == total_batches):
                elapsed = time.time() - start_ts
                pct = 100.0 * bi / total_batches
                sys.stdout.write(
                    f"\r[{progress_desc}] batch {bi}/{total_batches} ({pct:5.1f}%), "
                    f"pairs={n_seen}, elapsed={elapsed/60.0:.1f}m"
                )
                sys.stdout.flush()
    if show_progress and total_batches > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return n_seen, counts


def _ppi_matrix_from_dataset(ds: PairSubsetColumnDataset) -> torch.Tensor:
    if ds.ppi_node_features is not None:
        return torch.as_tensor(ds.ppi_node_features, dtype=torch.float32)
    if ds.ppi_extractor is None:
        n_g = int(ds.exp_data.shape[0]) if hasattr(ds.exp_data, "shape") else int(len(ds.seq_features_t))
        return torch.zeros((n_g, int(ds.ppi_feat_dim)), dtype=torch.float32)
    feat = ds.ppi_extractor.extract_node_features()
    return feat.detach().cpu().float()


def _count_pred_classes_cached(
    model: torch.nn.Module,
    ds: PairSubsetColumnDataset,
    device: torch.device,
    *,
    n_classes: int = 3,
    encode_batch_size: int = 32,
    pair_batch_size: int = 65536,
    progress_desc: str = "count-cached",
    show_progress: bool = True,
    use_amp_bf16: bool = False,
) -> Tuple[int, np.ndarray]:
    """Argmax class counts with one encoder pass per gene, then fusion over pairs."""
    model.eval()
    pairs = np.ascontiguousarray(ds._pair_array[:, :2], dtype=np.int64)
    n_pairs = int(pairs.shape[0])
    if n_pairs == 0:
        z = np.zeros(int(n_classes), dtype=np.int64)
        return 0, z, {
            "on_kegg": z.copy(),
            "off_kegg": z.copy(),
            "n_true": z.copy(),
            "repress_tf": np.zeros(0, dtype=np.int64),
            "activate_tf": np.zeros(0, dtype=np.int64),
        }

    exp = ds.exp_data if torch.is_tensor(ds.exp_data) else torch.as_tensor(ds.exp_data)
    exp = exp.detach().cpu().contiguous().float()
    seq = ds.seq_features_t.detach().cpu().contiguous().float()
    ppi = _ppi_matrix_from_dataset(ds).contiguous()
    n_genes = int(exp.shape[0])
    emb_dim = int(model.emb_dim)
    uniq = np.unique(pairs.reshape(-1))
    encode_bs = max(1, int(encode_batch_size))
    pair_bs = max(1, int(pair_batch_size))
    use_expr = bool(getattr(model, "use_expression", True))
    use_cds = bool(getattr(model, "use_cds", True))
    use_ppi = bool(getattr(model, "use_ppi", True))

    def _empty() -> torch.Tensor:
        return torch.empty((n_genes, emb_dim), device=device, dtype=torch.float32)

    cache_exp_tf = _empty() if use_expr else None
    cache_exp_tg = _empty() if use_expr else None
    cache_seq_tf = _empty() if use_cds else None
    cache_seq_tg = _empty() if use_cds else None
    cache_ppi_tf = _empty() if use_ppi else None
    cache_ppi_tg = _empty() if use_ppi else None

    start_ts = time.time()
    n_enc = int(len(uniq))
    n_enc_batches = (n_enc + encode_bs - 1) // encode_bs
    if show_progress:
        print(
            f"[{progress_desc}] encode genes={n_enc}/{n_genes}, encode_bs={encode_bs}, "
            f"pairs={n_pairs}, pair_bs={pair_bs}, amp_bf16={use_amp_bf16}",
            flush=True,
        )
    with torch.inference_mode():
        for bi, s in enumerate(range(0, n_enc, encode_bs), start=1):
            idx_np = uniq[s : s + encode_bs]
            idx_cpu = torch.from_numpy(idx_np.astype(np.int64))
            idx_dev = idx_cpu.to(device, non_blocking=True)
            with _autocast_bf16(device, use_amp_bf16):
                if cache_exp_tf is not None:
                    x_exp = exp.index_select(0, idx_cpu).to(device, non_blocking=True)
                    e_tf = model.exp_encoder_tf(x_exp)
                    e_tg = model.exp_encoder_target(x_exp)
                if cache_seq_tf is not None:
                    x_seq = seq.index_select(0, idx_cpu).to(device, non_blocking=True)
                    s_tf = model.seq_branch_tf(x_seq)
                    s_tg = model.seq_branch_target(x_seq)
                if cache_ppi_tf is not None:
                    x_ppi = ppi.index_select(0, idx_cpu).to(device, non_blocking=True)
                    p_tf = model.ppi_branch_tf(x_ppi)
                    p_tg = model.ppi_branch_target(x_ppi)
            if cache_exp_tf is not None:
                cache_exp_tf[idx_dev] = e_tf.float()
                cache_exp_tg[idx_dev] = e_tg.float()
            if cache_seq_tf is not None:
                cache_seq_tf[idx_dev] = s_tf.float()
                cache_seq_tg[idx_dev] = s_tg.float()
            if cache_ppi_tf is not None:
                cache_ppi_tf[idx_dev] = p_tf.float()
                cache_ppi_tg[idx_dev] = p_tg.float()
            if show_progress and (bi == 1 or bi == n_enc_batches or bi % max(1, n_enc_batches // 10) == 0):
                elapsed = time.time() - start_ts
                print(
                    f"[{progress_desc}] encode {bi}/{n_enc_batches} ({100.0 * bi / n_enc_batches:.1f}%), "
                    f"elapsed={elapsed / 60.0:.2f}m",
                    flush=True,
                )

        counts = np.zeros(int(n_classes), dtype=np.int64)
        counts_kegg = np.zeros(int(n_classes), dtype=np.int64)
        counts_off = np.zeros(int(n_classes), dtype=np.int64)
        true_all = np.ascontiguousarray(ds._pair_array[:, 2], dtype=np.int64)
        n_true = np.bincount(np.clip(true_all, 0, 2), minlength=3).astype(np.int64)
        rep_tf = np.zeros(n_genes, dtype=np.int64)
        act_tf = np.zeros(n_genes, dtype=np.int64)
        n_pair_batches = (n_pairs + pair_bs - 1) // pair_bs
        for bi, s in enumerate(range(0, n_pairs, pair_bs), start=1):
            sl = slice(s, min(n_pairs, s + pair_bs))
            tf_i = torch.from_numpy(pairs[sl, 0]).to(device)
            tg_i = torch.from_numpy(pairs[sl, 1]).to(device)
            parts = []
            if cache_exp_tf is not None:
                parts.append(cache_exp_tf.index_select(0, tf_i))
                parts.append(cache_exp_tg.index_select(0, tg_i))
            if cache_seq_tf is not None:
                parts.append(cache_seq_tf.index_select(0, tf_i))
                parts.append(cache_seq_tg.index_select(0, tg_i))
            if cache_ppi_tf is not None:
                parts.append(cache_ppi_tf.index_select(0, tf_i))
                parts.append(cache_ppi_tg.index_select(0, tg_i))
            with _autocast_bf16(device, use_amp_bf16):
                features = torch.cat(parts, dim=1)
                combined = model.fc_fusion(features)
                logits = model.classifier(combined.unsqueeze(1))
                pred = torch.argmax(logits, dim=1)
            pred_np = pred.detach().cpu().numpy().astype(np.int64)
            y_np = true_all[sl]
            tf_np = pairs[sl, 0]
            counts += np.bincount(pred_np, minlength=int(n_classes)).astype(np.int64)
            kegg_m = y_np != 0
            if kegg_m.any():
                counts_kegg += np.bincount(pred_np[kegg_m], minlength=int(n_classes)).astype(np.int64)
            if (~kegg_m).any():
                counts_off += np.bincount(pred_np[~kegg_m], minlength=int(n_classes)).astype(np.int64)
            rep_m = pred_np == 1
            act_m = pred_np == 2
            if rep_m.any():
                rep_tf += np.bincount(tf_np[rep_m], minlength=n_genes)
            if act_m.any():
                act_tf += np.bincount(tf_np[act_m], minlength=n_genes)
            if show_progress and (bi == 1 or bi == n_pair_batches or bi % max(1, n_pair_batches // 20) == 0):
                elapsed = time.time() - start_ts
                print(
                    f"[{progress_desc}] pairs {bi}/{n_pair_batches} ({100.0 * bi / n_pair_batches:.1f}%), "
                    f"elapsed={elapsed / 60.0:.2f}m",
                    flush=True,
                )
    extra = {
        "on_kegg": counts_kegg,
        "off_kegg": counts_off,
        "n_true": n_true,
        "repress_tf": rep_tf,
        "activate_tf": act_tf,
    }
    return n_pairs, counts, extra


def _load_expr_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    gcol = df.columns[0]
    df[gcol] = normalize_gene_symbol_series(df[gcol])
    df = df.drop_duplicates(subset=[gcol], keep="first").set_index(gcol)
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _npy_ncols_only(values_path: str) -> int:
    raw = np.load(str(values_path), mmap_mode="r")
    return int(raw.shape[1])


def _column_range_for_split_part(n_cols: int, sample_split: str, part: str) -> Tuple[int, int]:
    """与 train_unseen / finetune 中 split3 + 列切片一致；区间为左闭右开 [start, end)。"""
    w1, w2, w3 = [int(x.strip()) for x in sample_split.split(",")]
    c_tr, c_va, c_te = split3(int(n_cols), w1, w2, w3)
    if part == "train":
        return 0, c_tr
    if part == "val":
        return c_tr, c_tr + c_va
    if part == "test":
        return c_tr + c_va, int(n_cols)
    raise ValueError(f"unknown split part: {part}")


def _load_expr_df_from_npy(
    values_path: str,
    genes_path: str,
    *,
    col_range: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """与 wide_expression_csv_to_npy 输出一致：values (G×C) float32 memmap + genes object 数组。

    col_range: 若为 (start, end)，仅加载列 [start:end)，与微调管线列划分对齐时可缩小内存。
    """
    vp = Path(values_path)
    gp = Path(genes_path)
    if not vp.is_file() or not gp.is_file():
        raise FileNotFoundError(f"npy missing: {vp} / {gp}")
    _ts_log(f"npy open: values={vp.name} genes={gp.name}")
    t0 = time.monotonic()
    g_arr = np.load(str(gp), allow_pickle=True)
    raw = np.load(str(vp), mmap_mode="r")
    genes = normalize_gene_symbol_series(pd.Series([str(x) for x in g_arr.tolist()])).tolist()
    if len(genes) != raw.shape[0]:
        raise ValueError(f"gene count {len(genes)} != matrix rows {raw.shape[0]}")
    seen: set[str] = set()
    row_idx: List[int] = []
    row_names: List[str] = []
    for i, g in enumerate(genes):
        if g in seen:
            continue
        seen.add(g)
        row_idx.append(i)
        row_names.append(g)
    n_cols_full = int(raw.shape[1])
    c0, c1 = 0, n_cols_full
    if col_range is not None:
        c0 = max(0, int(col_range[0]))
        c1 = min(n_cols_full, int(col_range[1]))
        if c0 >= c1:
            raise ValueError(f"empty column slice [{c0}, {c1}) for n_cols={n_cols_full}")
        _ts_log(f"npy column slice (finetune-aligned): [{c0}, {c1}) of {n_cols_full} cols")
    n_g, n_c = len(row_idx), c1 - c0
    _ts_log(
        f"npy densify start: genes={n_g} cells={n_c} (~{n_g * n_c * 4 / (1024**3):.2f} GiB float32) — can take hours on slow disk/CPU"
    )
    t1 = time.monotonic()
    mat = np.asarray(raw[row_idx, c0:c1], dtype=np.float32)
    _ts_log(f"npy densify done in {(time.monotonic() - t1) / 60.0:.2f} min; total load so far {(time.monotonic() - t0) / 60.0:.2f} min")
    return pd.DataFrame(mat, index=row_names)


def _save_network(
    edges: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
    hub_top_n: int,
    max_nodes_plot: int,
    plot_title: Optional[str] = None,
    plot_group_a_display: str = "non-mBrain",
    plot_group_b_display: str = "mBrain",
) -> Dict[str, Any]:
    edges_path = tables_dir / "differential_edges_used.csv"
    edges.to_csv(edges_path, index=False, encoding="utf-8-sig")
    if "edge_category" in edges.columns:
        edges.to_csv(tables_dir / "differential_edges_annotated.csv", index=False, encoding="utf-8-sig")
    G = nx.Graph()
    for _, r in edges.iterrows():
        u, v, w = str(r["TF"]), str(r["Target"]), float(r["weight"])
        if G.has_edge(u, v):
            G[u][v]["weight"] = max(float(G[u][v]["weight"]), w)
        else:
            G.add_edge(u, v, weight=w)
        if "edge_category" in r:
            G[u][v]["edge_category"] = str(r["edge_category"])
    deg = dict(G.degree())
    wdeg = {n: float(sum(G[n][m]["weight"] for m in G.neighbors(n))) for n in G.nodes()}
    btw = nx.betweenness_centrality(G, weight=None, normalized=True, seed=42) if G.number_of_nodes() else {}
    hubs = pd.DataFrame(
        {
            "gene": list(G.nodes()),
            "degree": [int(deg[n]) for n in G.nodes()],
            "weighted_degree": [wdeg[n] for n in G.nodes()],
            "betweenness": [float(btw.get(n, 0.0)) for n in G.nodes()],
        }
    )
    if not hubs.empty:
        hubs["hub_score"] = hubs["degree"].rank(pct=True) + hubs["betweenness"].rank(pct=True)
        hubs = hubs.sort_values("hub_score", ascending=False).reset_index(drop=True)
    hubs_path = tables_dir / "hub_genes.csv"
    hubs.to_csv(hubs_path, index=False, encoding="utf-8-sig")
    hub_set = set(hubs.head(min(hub_top_n, len(hubs)))["gene"].astype(str).tolist()) if not hubs.empty else set()

    if G.number_of_nodes() > 0:
        Gp = G
        if G.number_of_nodes() > max_nodes_plot:
            keep = set(hub_set)
            for n in sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True):
                if len(keep) >= max_nodes_plot:
                    break
                keep.add(n)
            Gp = G.subgraph(keep).copy()
        pos = nx.spring_layout(Gp, seed=42, k=1.2 / max(np.sqrt(max(Gp.number_of_nodes(), 1)), 1))
        eweights = [float(Gp[u][v].get("weight", 1.0)) for u, v in Gp.edges()]
        emax = max(eweights) if eweights else 1.0
        widths = [1.0 + 4.0 * (x / emax) for x in eweights] if eweights else 1.0
        if "edge_category" in edges.columns:
            edge_colors = [EDGE_COLOR_MAP.get(str(Gp[u][v].get("edge_category", "")), "#7f8c8d") for u, v in Gp.edges()]
        else:
            edge_colors = "#7f8c8d"
        fig, ax = plt.subplots(figsize=(12, 10))
        nx.draw_networkx_edges(Gp, pos, ax=ax, alpha=0.45, width=widths, edge_color=edge_colors)
        nodes = list(Gp.nodes())
        colors = ["#e74c3c" if n in hub_set else "#3498db" for n in nodes]
        sizes = [120 + 40 * Gp.degree(n) for n in nodes]
        nx.draw_networkx_nodes(Gp, pos, ax=ax, node_size=sizes, node_color=colors, alpha=0.9)
        labels = {n: n for n in nodes if n in hub_set}
        nx.draw_networkx_labels(Gp, pos, labels=labels, ax=ax, font_size=8)
        default_title = (
            f"Step1 model {plot_group_a_display} vs {plot_group_b_display} "
            f"differential network ({G.number_of_nodes()} genes, {G.number_of_edges()} edges)"
        )
        ax.set_title(plot_title or default_title)
        if "edge_category" in edges.columns:
            cat_counts = edges["edge_category"].value_counts().to_dict()
            handles = []
            labels_leg = []
            for k in [
                "mBrain_gain_regulation",
                "mBrain_loss_regulation",
                "repression_to_activation",
                "activation_to_repression",
                "other_class_switch",
                "same_class_prob_shift",
            ]:
                if cat_counts.get(k, 0) <= 0:
                    continue
                handles.append(Line2D([0], [0], color=EDGE_COLOR_MAP[k], lw=2.5))
                labels_leg.append(f"{k} (n={cat_counts[k]})")
            if handles:
                ax.legend(handles, labels_leg, loc="upper left", frameon=False, fontsize=8)
            (tables_dir / "edge_category_counts.json").write_text(json.dumps(cat_counts, ensure_ascii=False, indent=2), encoding="utf-8")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(figures_dir / "network_plot.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "empty graph", ha="center", va="center")
        ax.axis("off")
        fig.savefig(figures_dir / "network_plot.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    (tables_dir / "r_hub_gene_symbols.txt").write_text(
        "\n".join(hubs.head(min(hub_top_n, len(hubs)))["gene"].astype(str).tolist()) + "\n",
        encoding="utf-8",
    )
    all_nodes_txt = "\n".join(sorted(str(n) for n in G.nodes())) + ("\n" if G.number_of_nodes() else "")
    (tables_dir / "r_network_all_gene_symbols.txt").write_text(all_nodes_txt, encoding="utf-8")
    (tables_dir / "r_enrich_gene_symbols.txt").write_text(all_nodes_txt, encoding="utf-8")
    return {"n_nodes": int(G.number_of_nodes()), "n_edges": int(G.number_of_edges())}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Use step1 trained model on nonbrain/brain expressions, export differential network with figures/tables split."
    )
    p.add_argument("--step1-run-dir", required=True)
    p.add_argument(
        "--nonbrain-model-path",
        default=None,
        help="non-mBrain 侧权重路径；默认使用 step1-run-dir/best_model_fold0.pth（应用管线可与微调输出目录对准）。",
    )
    p.add_argument(
        "--brain-model-path",
        default=None,
        help="mBrain 侧权重路径；默认同上。",
    )
    p.add_argument(
        "--expression-column-split-part",
        choices=["all", "train", "val", "test"],
        default="all",
        help="仅 npy 输入：按 expression-sample-split 切细胞列，与 finetune 管线 split3 一致；test 对应约 70%% 发现/评测列。",
    )
    p.add_argument(
        "--expression-sample-split",
        default="1,2,7",
        metavar="W1,W2,W3",
        help="列划分比例 train:val:test，须与微调 --sample-split 一致（默认 1,2,7）。",
    )
    p.add_argument(
        "--nonbrain-exp-csv",
        default=None,
        help="non-mBrain 宽大表达 CSV（首列 gene）；与 --nonbrain-exp-npy-* 二选一。",
    )
    p.add_argument(
        "--nonbrain-exp-npy-values",
        default=None,
        help="non-mBrain 表达矩阵 .npy（float32, G×C），与 --nonbrain-exp-npy-genes 成对使用；避免整表 read_csv。",
    )
    p.add_argument("--nonbrain-exp-npy-genes", default=None, help="与 values.npy 行顺序一致的基因名 .npy（object）。")
    p.add_argument("--brain-exp-csv", default=None, help="mBrain 宽大表达 CSV；与 brain npy 二选一。")
    p.add_argument("--brain-exp-npy-values", default=None, help="mBrain 表达 values.npy（float32, G×C）。")
    p.add_argument("--brain-exp-npy-genes", default=None, help="mBrain 表达 genes.npy（object）。")
    p.add_argument("--output-dir", default="experiments/metastasis_case_step1_model")
    p.add_argument("--prob-diff-topk", type=int, default=200)
    p.add_argument(
        "--prob-diff-threshold",
        type=float,
        default=0.5,
        help="差异边：两域 softmax 概率向量 L∞ 差（max_prob_diff）须严格大于该阈值（默认 0.5）。",
    )
    p.add_argument(
        "--edge-mode",
        choices=["auto", "pred_diff", "prob_threshold", "prob_top", "union"],
        default="prob_threshold",
        help="prob_threshold：max_prob_diff > --prob-diff-threshold；pred_diff：仅 argmax 类别不一致。",
    )
    p.add_argument("--min-prob-diff", type=float, default=0.0, help="prob_top 模式下导出边的最小 max_prob_diff。")
    p.add_argument("--hub-top-n", type=int, default=25)
    p.add_argument("--max-nodes-plot", type=int, default=80)
    p.add_argument("--norm-fit-domain", choices=["concat", "nonbrain", "brain", "each"], default="concat")
    p.add_argument(
        "--network-plot-title",
        default=None,
        help="网络图标题；默认使用 group-a vs group-b 文案。",
    )
    p.add_argument(
        "--group-a-label",
        default="non-mBrain",
        help="第一侧表达（原 nonbrain CSV）在图表与导出列中的展示名，默认 non-mBrain。",
    )
    p.add_argument(
        "--group-b-label",
        default="mBrain",
        help="第二侧表达（原 brain CSV）在图表与导出列中的展示名，默认 mBrain。",
    )
    p.add_argument(
        "--kegg-k-hops",
        type=int,
        default=None,
        help="若指定（如 3）：仅用 KEGG 边构无向图，GRN source 的 k 跳内 (s,t) 为正样本，并按 run_config 的 neg_sample_ratio 采负样本；不用 PPI 扩图。省略则仍用 GRN 矩阵内边 + 随机负样本。",
    )
    p.add_argument(
        "--neg-sample-ratio",
        type=float,
        default=None,
        help="覆盖 run_config.neg_sample_ratio：仅在 GRN/k-hop 预设对模式下生效（笛卡尔模式忽略）。"
        "设为 0 则只保留 KEGG 正边；默认 None 表示与训练相同（通常 1.0）。",
    )
    p.add_argument(
        "--kegg-neg-tf-from-sources-only",
        action="store_true",
        help="k 跳模式下负样本仅随机 GRN source 作 TF 端（可选）。",
    )
    p.add_argument(
        "--de-cartesian-csv",
        default=None,
        help="全基因或其它 DE 表路径；若指定则按 p 值筛基因后对筛得集合做有向笛卡尔积 (A,B)+(B,A) 建对，且勿与 --kegg-k-hops 同用。",
    )
    p.add_argument("--de-gene-col", default="gene", help="DE 表中基因名列名")
    p.add_argument("--de-pvalue-col", default="pvalue_wilcox", help="用于筛选的 p 值列名（与 --de-pvalue-max 联用）")
    p.add_argument(
        "--de-use-all-rows",
        action="store_true",
        help="DE 笛卡尔模式下使用 CSV 全部基因行，不再按 p 值阈值筛选。",
    )
    p.add_argument(
        "--de-pvalue-max",
        type=float,
        default=0.01,
        help="保留 p < 该阈值的基因（严格小于）；在未使用 padj/logFC 组合筛选时生效；默认 0.01",
    )
    p.add_argument("--de-padj-col", default="p_val_adj", help="adjusted p 列名；与 --de-padj-max 联用")
    p.add_argument(
        "--de-padj-max",
        type=float,
        default=None,
        help="若指定：保留 padj 严格小于该阈值（并与 --de-logfc-min 等条件 AND）",
    )
    p.add_argument("--de-logfc-col", default="avg_log2FC", help="logFC 列名；与 --de-logfc-min 联用")
    p.add_argument(
        "--de-logfc-min",
        type=float,
        default=None,
        help="若指定：保留 logFC 严格大于该阈值（单侧，如 mBrain 上调）",
    )
    p.add_argument("--no-progress", action="store_true", help="关闭 Step1 批次进度显示")
    p.add_argument(
        "--inference-batch-size",
        type=int,
        default=None,
        help="推理 batch 大小；默认用 step1 run_config.json 的 batch_size。大规模笛卡尔对时可设大些（显存允许）以提高 GPU 利用率。",
    )
    p.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        help="DataLoader worker 数。Windows 上大表达矩阵时建议 0（避免多进程复制整表占内存）。",
    )
    p.add_argument(
        "--no-inference-amp-bf16",
        action="store_true",
        help="关闭推理阶段 bf16 autocast（默认跟随 step1 run_config 的 amp_bf16，且需 GPU 支持 bf16）。",
    )
    p.add_argument(
        "--minmax-col-chunk",
        type=int,
        default=8192,
        help="log2-minmax 按列分块（与整表数值一致），块间打印进度；0=单次向量化（无块进度、略快）。",
    )
    p.add_argument(
        "--max-expression-cols",
        type=int,
        default=0,
        help="若 >0 且表达列数超过该值：按 seed 子采样细胞列，与 train_unseen --max-expression-cols 对齐（推荐 10000）。",
    )
    p.add_argument(
        "--expression-col-seed",
        type=int,
        default=42,
        help="--max-expression-cols 子采样随机种子。",
    )
    p.add_argument(
        "--count-pred-classes-only",
        action="store_true",
        help="只统计笛卡尔积 argmax 类别计数（NoReg/Repress/Activate），不重写差异网络表/图。",
    )
    args = p.parse_args()

    _ts_log("=== Step1 case study: startup (set PYTHONUNBUFFERED=1 or python -u for live logs) ===")
    out_dir = Path(args.output_dir).resolve()
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    _mm_kw: Dict[str, Any] = {}
    if int(args.minmax_col_chunk) > 0:
        _mm_kw = {"col_chunk": int(args.minmax_col_chunk), "progress_callback": _ts_log}

    cfg_path = Path(args.step1_run_dir) / "run_config.json"
    model_path_nb = Path(args.nonbrain_model_path) if args.nonbrain_model_path else Path(args.step1_run_dir) / "best_model_fold0.pth"
    model_path_br = Path(args.brain_model_path) if args.brain_model_path else Path(args.step1_run_dir) / "best_model_fold0.pth"
    if not cfg_path.is_file():
        raise SystemExit(f"missing run_config under {args.step1_run_dir}")
    if not model_path_nb.is_file():
        raise SystemExit(f"missing non-mBrain model: {model_path_nb}")
    if not model_path_br.is_file():
        raise SystemExit(f"missing mBrain model: {model_path_br}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    _ts_log(f"loaded run_config from {cfg_path}")

    use_ppi = not bool(cfg.get("no_ppi", False))
    use_cds = not bool(cfg.get("no_cds", False))
    use_expr = not bool(cfg.get("no_expression", False))
    if use_ppi and not cfg.get("ppi_csv"):
        raise SystemExit("step1 config requires ppi_csv")
    if use_cds and not (cfg.get("cds_gene_feature_csv") or cfg.get("tss_gene_feature_csv")):
        raise SystemExit("step1 config requires cds/tss feature csv")

    nb_npy = bool(args.nonbrain_exp_npy_values and args.nonbrain_exp_npy_genes)
    br_npy = bool(args.brain_exp_npy_values and args.brain_exp_npy_genes)
    nb_csv = bool(args.nonbrain_exp_csv)
    br_csv = bool(args.brain_exp_csv)
    if nb_npy == nb_csv:
        raise SystemExit("non-mBrain: specify exactly one of (--nonbrain-exp-csv) OR (--nonbrain-exp-npy-values + --nonbrain-exp-npy-genes)")
    if br_npy == br_csv:
        raise SystemExit("mBrain: specify exactly one of (--brain-exp-csv) OR (--brain-exp-npy-values + --brain-exp-npy-genes)")
    if str(args.expression_column_split_part) != "all" and (not nb_npy or not br_npy):
        raise SystemExit("--expression-column-split-part 非 all 时必须两侧均使用 npy 表达输入")

    nb_col_range: Optional[Tuple[int, int]] = None
    br_col_range: Optional[Tuple[int, int]] = None
    if str(args.expression_column_split_part) != "all":
        ss = str(args.expression_sample_split).strip()
        n_nb = _npy_ncols_only(str(args.nonbrain_exp_npy_values))
        n_br = _npy_ncols_only(str(args.brain_exp_npy_values))
        nb_col_range = _column_range_for_split_part(n_nb, ss, str(args.expression_column_split_part))
        br_col_range = _column_range_for_split_part(n_br, ss, str(args.expression_column_split_part))
        _ts_log(
            f"expression column subset: part={args.expression_column_split_part} sample_split={ss} "
            f"non-mBrain [{nb_col_range[0]}, {nb_col_range[1]}) n_cols={n_nb} | "
            f"mBrain [{br_col_range[0]}, {br_col_range[1]}) n_cols={n_br}"
        )

    if nb_npy:
        _ts_log(f"load non-mBrain expr npy: {args.nonbrain_exp_npy_values}")
        nb_df = _load_expr_df_from_npy(
            str(args.nonbrain_exp_npy_values), str(args.nonbrain_exp_npy_genes), col_range=nb_col_range
        )
    else:
        _ts_log(f"load non-mBrain expr csv (may be very slow): {args.nonbrain_exp_csv}")
        t_nb = time.monotonic()
        nb_df = _load_expr_df(str(args.nonbrain_exp_csv))
        _ts_log(f"non-mBrain CSV load done in {(time.monotonic() - t_nb) / 60.0:.2f} min; shape={nb_df.shape}")
    if br_npy:
        _ts_log(f"load mBrain expr npy: {args.brain_exp_npy_values}")
        br_df = _load_expr_df_from_npy(
            str(args.brain_exp_npy_values), str(args.brain_exp_npy_genes), col_range=br_col_range
        )
    else:
        _ts_log(f"load mBrain expr csv (may be very slow): {args.brain_exp_csv}")
        t_br = time.monotonic()
        br_df = _load_expr_df(str(args.brain_exp_csv))
        _ts_log(f"mBrain CSV load done in {(time.monotonic() - t_br) / 60.0:.2f} min; shape={br_df.shape}")
    genes = set(nb_df.index.tolist()) & set(br_df.index.tolist())
    _ts_log(f"gene intersection raw: {len(genes)} genes (nb rows={len(nb_df)}, br rows={len(br_df)})")

    cds_feat = _load_gene_feature_csv(cfg["cds_gene_feature_csv"]) if cfg.get("cds_gene_feature_csv") else None
    tss_feat = _load_gene_feature_csv(cfg["tss_gene_feature_csv"]) if cfg.get("tss_gene_feature_csv") else None
    if use_cds:
        if cds_feat is not None:
            genes &= set(cds_feat.index.tolist())
        if tss_feat is not None:
            genes &= set(tss_feat.index.tolist())
    gene_list = [g for g in nb_df.index.tolist() if g in genes]
    if len(gene_list) == 0:
        raise SystemExit("no shared genes between nonbrain/brain/features")
    _ts_log(f"subset to panel genes: n={len(gene_list)}")
    nb_df = nb_df.loc[gene_list]
    br_df = br_df.loc[gene_list]

    _ts_log(f"read GRN/edges: {cfg['grn_csv']}")
    t_grn = time.monotonic()
    grn_df = pd.read_csv(cfg["grn_csv"])
    tf_col, tg_col, tp_col = resolve_grn_table_columns(
        grn_df,
        str(cfg.get("grn_tf_col", "TF")),
        str(cfg.get("grn_target_col", "target")),
        str(cfg.get("grn_type_col", "Re_type")),
    )
    if tf_col is None or tg_col is None or tp_col is None:
        raise SystemExit("grn columns not found")
    grn_df[tf_col] = normalize_gene_symbol_series(grn_df[tf_col])
    grn_df[tg_col] = normalize_gene_symbol_series(grn_df[tg_col])
    gset = set(gene_list)
    grn_df = grn_df[grn_df[tf_col].isin(gset) & grn_df[tg_col].isin(gset)].copy()
    rel = grn_df[tp_col].astype(str).str.lower().str.strip().replace(
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
    grn_df = grn_df[rel.isin(["activator", "repressor"])].copy()
    rel = rel[grn_df.index]
    grn_matrix = get_GRN((grn_df[tf_col].values, grn_df[tg_col].values, rel.values), gene_list)
    _ts_log(f"GRN matrix built in {(time.monotonic() - t_grn) / 60.0:.2f} min")
    kh_stats = None
    de_stats: Optional[Dict[str, int]] = None
    if args.de_cartesian_csv:
        if args.kegg_k_hops is not None:
            raise SystemExit("请勿同时指定 --de-cartesian-csv 与 --kegg-k-hops")
        de_path = Path(args.de_cartesian_csv)
        if not de_path.is_file():
            raise SystemExit(f"missing DE csv: {de_path}")
        _ts_log(f"read DE / gene list for pairs: {de_path}")
        t_de = time.monotonic()
        de_df = pd.read_csv(de_path)
        _ts_log(f"DE table read in {(time.monotonic() - t_de):.1f}s, rows={len(de_df)}")
        gcol = str(args.de_gene_col)
        pcol = str(args.de_pvalue_col)
        padj_col = str(args.de_padj_col)
        logfc_col = str(args.de_logfc_col)
        if gcol not in de_df.columns:
            raise SystemExit(f"DE csv 缺少列 {gcol}: {list(de_df.columns)}")
        de_df[gcol] = normalize_gene_symbol_series(de_df[gcol].astype(str))
        if bool(args.de_use_all_rows):
            sel = de_df
        else:
            use_padj_logfc = (args.de_padj_max is not None) or (args.de_logfc_min is not None)
            sel = de_df
            if use_padj_logfc:
                if args.de_padj_max is not None:
                    if padj_col not in de_df.columns:
                        raise SystemExit(f"DE csv 缺少列 {padj_col}: {list(de_df.columns)}")
                    sel = sel[sel[padj_col].astype(float) < float(args.de_padj_max)]
                if args.de_logfc_min is not None:
                    if logfc_col not in de_df.columns:
                        raise SystemExit(f"DE csv 缺少列 {logfc_col}: {list(de_df.columns)}")
                    sel = sel[sel[logfc_col].astype(float) > float(args.de_logfc_min)]
            else:
                if pcol not in de_df.columns:
                    raise SystemExit(f"DE csv 缺少列 {pcol}: {list(de_df.columns)}")
                sel = sel[sel[pcol].astype(float) < float(args.de_pvalue_max)]
        genes_de = sel[gcol].astype(str).tolist()
        if len(genes_de) < 2:
            if bool(args.de_use_all_rows):
                raise SystemExit(f"DE 全行模式下基因不足 2 个: n={len(genes_de)}")
            raise SystemExit(f"DE 筛选后基因不足 2 个: n={len(genes_de)}")
        print("[case study] build ordered pair index (packed numpy, avoids millions of Python tuples)…", flush=True)
        all_pairs, de_stats = build_ordered_pairs_packed(gene_list, grn_matrix, genes_de)
        print(
            f"[case study] pairs ready: n_genes={de_stats['n_de_genes_in_panel']}, n_ordered={de_stats['n_ordered_pairs']}",
            flush=True,
        )
        use_padj_logfc = (not bool(args.de_use_all_rows)) and (
            (args.de_padj_max is not None) or (args.de_logfc_min is not None)
        )
        (tables_dir / "de_cartesian_pair_stats.json").write_text(
            json.dumps(
                {
                    "de_csv": str(de_path.resolve()),
                    "de_gene_col": gcol,
                    "de_use_all_rows": bool(args.de_use_all_rows),
                    "de_pvalue_col": (None if bool(args.de_use_all_rows) or use_padj_logfc else pcol),
                    "de_pvalue_max": (
                        None if bool(args.de_use_all_rows) or use_padj_logfc else float(args.de_pvalue_max)
                    ),
                    "de_padj_col": (padj_col if use_padj_logfc and args.de_padj_max is not None else None),
                    "de_padj_max": (float(args.de_padj_max) if use_padj_logfc and args.de_padj_max is not None else None),
                    "de_logfc_col": (logfc_col if use_padj_logfc and args.de_logfc_min is not None else None),
                    "de_logfc_min": (float(args.de_logfc_min) if use_padj_logfc and args.de_logfc_min is not None else None),
                    **{k: int(v) for k, v in de_stats.items()},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        neg_ratio = (
            float(args.neg_sample_ratio)
            if args.neg_sample_ratio is not None
            else float(cfg.get("neg_sample_ratio", 1.0))
        )
        seed_pairs = int(cfg.get("seed", 42))
        if args.kegg_k_hops is not None:
            all_pairs, kh_stats = build_index_pairs_from_kegg_khop(
                gene_list,
                grn_matrix,
                str(cfg["grn_csv"]),
                str(cfg.get("grn_tf_col", "TF")),
                str(cfg.get("grn_target_col", "target")),
                str(cfg.get("grn_type_col", "Re_type")),
                int(args.kegg_k_hops),
                neg_ratio,
                seed_pairs,
                neg_tf_from_sources_only=bool(args.kegg_neg_tf_from_sources_only),
            )
            if kh_stats:
                (tables_dir / "kegg_khop_pair_stats.json").write_text(
                    json.dumps(
                        {
                            "k_hops": int(args.kegg_k_hops),
                            "neg_sample_ratio": float(neg_ratio),
                            **{k: int(v) for k, v in kh_stats.items()},
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        else:
            all_pairs = _build_pair_samples(grn_matrix, neg_ratio, seed_pairs)
            n_pos = int(sum(1 for _, _, lab in all_pairs if int(lab) != 0))
            n_neg = int(len(all_pairs) - n_pos)
            pair_stats = {
                "pair_build": "grn_matrix_neg_sampled",
                "neg_sample_ratio": float(neg_ratio),
                "seed": int(seed_pairs),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_pairs": int(len(all_pairs)),
                "note": "Same construction as train_unseen_samples_no_cv (KEGG positives + seeded negatives).",
            }
            (tables_dir / "grn_pair_stats.json").write_text(
                json.dumps(pair_stats, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            _ts_log(
                f"preset GRN pairs: n_pos={n_pos}, n_neg={n_neg}, n_total={len(all_pairs)} "
                f"(neg_sample_ratio={neg_ratio}, seed={seed_pairs})"
            )
    if len(all_pairs) == 0:
        raise SystemExit("no pairs built (de_cartesian / kegg_khop / grn)")

    _ts_log("build CDS/TSS sequence feature matrix…")
    t_seq = time.monotonic()
    if use_cds:
        blocks = []
        if cds_feat is not None:
            blocks.append(cds_feat.loc[gene_list].values.astype(np.float32))
        if tss_feat is not None:
            blocks.append(tss_feat.loc[gene_list].values.astype(np.float32))
        seq_features = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(gene_list), 20), dtype=np.float32)
    else:
        seq_features = np.zeros((len(gene_list), 20), dtype=np.float32)
    seq_dim = int(seq_features.shape[1])
    _ts_log(f"seq_features ready in {(time.monotonic() - t_seq):.1f}s, dim={seq_dim}")

    ppi_emb_dim = 160
    ppi_extractor = None
    if use_ppi:
        from grn.features.ppi_features import PPIFeatureExtractor

        # PPINet weights are NOT saved in the TripGRN checkpoint. Training extracts features once
        # after set_seed(seed) (torch RNG largely unused before PPI init). Case study must
        # recreate the same features; otherwise argmax collapses toward NoReg on the cartesian dump.
        seed = int(cfg.get("seed", 42))
        step1_dir = Path(str(args.step1_run_dir))
        nb_model_dir = Path(str(args.nonbrain_model_path)).resolve().parent
        ppi_cache_candidates = [
            step1_dir / "ppi_node_features.pt",
            nb_model_dir / "ppi_node_features.pt",
            nb_model_dir / "ppi_node_features_aligned.pt",
        ]
        loaded_ppi = False
        for cache_path in ppi_cache_candidates:
            if not cache_path.is_file():
                continue
            try:
                blob = torch.load(str(cache_path), map_location="cpu")
                cached_genes = [str(g) for g in blob.get("gene_list", [])]
                feats = blob.get("features")
                if feats is None or len(cached_genes) != int(feats.shape[0]):
                    continue
                g2i = {g: i for i, g in enumerate(cached_genes)}
                if not all(g in g2i for g in gene_list):
                    _ts_log(f"PPI cache gene mismatch, skip: {cache_path}")
                    continue
                idx = torch.tensor([g2i[g] for g in gene_list], dtype=torch.long)
                feats_sub = feats.index_select(0, idx).contiguous().float()
                ppi_extractor = PPIFeatureExtractor(cfg["ppi_csv"], gene_list=gene_list, emb_dim=ppi_emb_dim, device=None)
                ppi_extractor._cached_features = feats_sub
                loaded_ppi = True
                _ts_log(f"loaded PPI node features from {cache_path} (reindexed to n={len(gene_list)})")
                break
            except Exception as e:
                _ts_log(f"PPI cache load failed ({cache_path}): {e}")
        if not loaded_ppi:
            print("[case study] init PPI feature extractor (may take a minute)…", flush=True)
            set_seed(seed)
            ppi_extractor = PPIFeatureExtractor(cfg["ppi_csv"], gene_list=gene_list, emb_dim=ppi_emb_dim, device=None)
            _ = ppi_extractor.extract_node_features()
            _ts_log(
                f"PPI features rebuilt with set_seed({seed}); "
                "prefer saving ppi_node_features.pt at train time for exact reproducibility"
            )
        if getattr(ppi_extractor, "_cached_features", None) is not None:
            ppi_extractor._cached_features = ppi_extractor._cached_features.detach().cpu()
            print("[case study] PPI node embeddings moved to CPU (enables DataLoader pin_memory).", flush=True)

    _ts_log("copy expression to float32 numpy (nb_raw, br_raw)…")
    t_cp = time.monotonic()
    nb_raw = nb_df.values.astype(np.float32)
    br_raw = br_df.values.astype(np.float32)
    _ts_log(f"expression numpy copy done in {(time.monotonic() - t_cp) / 60.0:.2f} min; shapes nb={nb_raw.shape} br={br_raw.shape}")
    if args.norm_fit_domain == "each":
        _ts_log("log2-minmax fit: per-side (nonbrain, brain)…")
        t_fit = time.monotonic()
        nb_min, nb_max = log2_minmax_per_gene_fit(nb_raw, offset=1.0, **_mm_kw)
        _ts_log(f"nonbrain minmax fit done in {(time.monotonic() - t_fit) / 60.0:.2f} min")
        t_fit2 = time.monotonic()
        br_min, br_max = log2_minmax_per_gene_fit(br_raw, offset=1.0, **_mm_kw)
        _ts_log(f"brain minmax fit done in {(time.monotonic() - t_fit2) / 60.0:.2f} min")
        _ts_log("log2-minmax transform -> torch tensors…")
        t_tr = time.monotonic()
        exp_nb = log2_minmax_per_gene_transform(nb_raw, nb_min, nb_max, offset=1.0, **_mm_kw)
        exp_br = log2_minmax_per_gene_transform(br_raw, br_min, br_max, offset=1.0, **_mm_kw)
        _ts_log(f"transform done in {(time.monotonic() - t_tr) / 60.0:.2f} min")
    else:
        if args.norm_fit_domain == "concat":
            _ts_log(
                "log2-minmax fit: concat(nonbrain, brain) columns — large RAM pass, may take tens of minutes to hours…"
            )
        t_cat = time.monotonic()
        fit_src = {"nonbrain": nb_raw, "brain": br_raw, "concat": np.concatenate([nb_raw, br_raw], axis=1)}[args.norm_fit_domain]
        _ts_log(f"fit_src built in {(time.monotonic() - t_cat) / 60.0:.2f} min; fit_src.shape={fit_src.shape}")
        t_fit = time.monotonic()
        x_min, x_max = log2_minmax_per_gene_fit(fit_src, offset=1.0, **_mm_kw)
        _ts_log(f"log2-minmax fit done in {(time.monotonic() - t_fit) / 60.0:.2f} min")
        _ts_log("log2-minmax transform nb / br -> torch tensors…")
        t_tr = time.monotonic()
        exp_nb = log2_minmax_per_gene_transform(nb_raw, x_min, x_max, offset=1.0, **_mm_kw)
        exp_br = log2_minmax_per_gene_transform(br_raw, x_min, x_max, offset=1.0, **_mm_kw)
        _ts_log(f"transform done in {(time.monotonic() - t_tr) / 60.0:.2f} min")

    max_expr_cols = int(args.max_expression_cols)
    if max_expr_cols > 0:

        def _subsample_expr_cols(mat: torch.Tensor, *, name: str) -> torch.Tensor:
            t = int(mat.shape[1])
            if t <= max_expr_cols:
                return mat
            rng = np.random.default_rng(int(args.expression_col_seed) + (0 if name == "nb" else 1))
            pick = np.sort(rng.choice(t, size=max_expr_cols, replace=False))
            _ts_log(
                f"subsample {name} expression cols {t} -> {max_expr_cols} "
                f"(seed={args.expression_col_seed})"
            )
            return mat[:, torch.as_tensor(pick, dtype=torch.long)].contiguous()

        if not isinstance(exp_nb, torch.Tensor):
            exp_nb = torch.as_tensor(exp_nb)
        if not isinstance(exp_br, torch.Tensor):
            exp_br = torch.as_tensor(exp_br)
        exp_nb = _subsample_expr_cols(exp_nb, name="nb")
        exp_br = _subsample_expr_cols(exp_br, name="br")

    _ts_log("build PairSubsetColumnDataset (nonbrain + brain)…")
    ds_nb = PairSubsetColumnDataset(
        exp_nb, all_pairs, seq_features, ppi_extractor, device=None, use_ppi=use_ppi, use_expression=use_expr, use_cds=use_cds, ppi_feat_dim=ppi_emb_dim
    )
    ds_br = PairSubsetColumnDataset(
        exp_br, all_pairs, seq_features, ppi_extractor, device=None, use_ppi=use_ppi, use_expression=use_expr, use_cds=use_cds, ppi_feat_dim=ppi_emb_dim
    )
    _ts_log(f"datasets OK; n_pairs={len(ds_nb)}")
    bs = int(args.inference_batch_size) if args.inference_batch_size is not None else int(cfg.get("batch_size", 32))
    if bs < 1:
        raise SystemExit("inference batch_size must be >= 1")
    pin_mem = bool(torch.cuda.is_available())
    dl_kw: Dict[str, Any] = {
        "batch_size": bs,
        "shuffle": False,
        "pin_memory": pin_mem,
        "num_workers": int(args.dataloader_num_workers),
    }
    if int(args.dataloader_num_workers) > 0:
        dl_kw["persistent_workers"] = True
        dl_kw["prefetch_factor"] = 2
    ln = DataLoader(ds_nb, **dl_kw)
    lb = DataLoader(ds_br, **dl_kw)
    _ts_log(
        f"DataLoader batch_size={bs}, num_workers={int(args.dataloader_num_workers)}, pin_memory={pin_mem}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _bf16_fn = getattr(torch.cuda, "is_bf16_supported", None)
    _bf16_ok = bool(_bf16_fn()) if callable(_bf16_fn) and device.type == "cuda" else False
    use_amp_infer = bool(cfg.get("amp_bf16")) and _bf16_ok and not bool(args.no_inference_amp_bf16)
    if bool(cfg.get("amp_bf16")) and device.type == "cuda" and not _bf16_ok:
        print("[case study] warning: bf16 not supported on this GPU; inference runs in fp32.", flush=True)
    if use_amp_infer:
        print("[case study] inference AMP: bfloat16 autocast enabled.", flush=True)

    _ts_log(f"load non-mBrain model -> {device} from {model_path_nb}")
    t_m = time.monotonic()
    model_nb = FusionModelAblation(seq_dim=seq_dim, num_classes=3, use_expression=use_expr, use_cds=use_cds, use_ppi=use_ppi).to(device)
    model_nb.load_state_dict(torch.load(str(model_path_nb), map_location=device))
    _ts_log(f"load mBrain model -> {device} from {model_path_br}")
    model_br = FusionModelAblation(seq_dim=seq_dim, num_classes=3, use_expression=use_expr, use_cds=use_cds, use_ppi=use_ppi).to(device)
    model_br.load_state_dict(torch.load(str(model_path_br), map_location=device))
    _ts_log(f"models ready in {(time.monotonic() - t_m):.1f}s — starting pair-wise inference")

    slug_a = _label_slug(args.group_a_label)
    slug_b = _label_slug(args.group_b_label)
    ca_col = f"pred_{slug_a}"
    cb_col = f"pred_{slug_b}"
    pair_prefix = f"pairs_step1_{slug_a}_vs_{slug_b}"

    if bool(args.count_pred_classes_only):
        n_a, c_a, extra_a = _count_pred_classes_cached(
            model_nb,
            ds_nb,
            device,
            n_classes=len(CLASS_NAMES),
            progress_desc=f"{args.group_a_label} class-count",
            show_progress=not bool(args.no_progress),
            use_amp_bf16=use_amp_infer,
        )
        n_b, c_b, extra_b = _count_pred_classes_cached(
            model_br,
            ds_br,
            device,
            n_classes=len(CLASS_NAMES),
            progress_desc=f"{args.group_b_label} class-count",
            show_progress=not bool(args.no_progress),
            use_amp_bf16=use_amp_infer,
        )
        if int(n_a) != int(n_b):
            raise RuntimeError(f"pair count mismatch: {args.group_a_label}={n_a} vs {args.group_b_label}={n_b}")

        def _pack(n_scope: int, counts: np.ndarray) -> Dict[str, Any]:
            d = {CLASS_NAMES[i]: int(counts[i]) for i in range(len(CLASS_NAMES))}
            n_reg = int(counts[1] + counts[2])
            d["n_regulatory_edges"] = n_reg
            d["pct_of_scope_regulatory"] = (float(n_reg) / float(n_scope)) if n_scope else 0.0
            d["pct_Activate_among_regulatory"] = (float(counts[2]) / float(n_reg)) if n_reg else 0.0
            d["pct_Repress_among_regulatory"] = (float(counts[1]) / float(n_reg)) if n_reg else 0.0
            d["Activate_to_Repress"] = (float(counts[2]) / float(counts[1])) if int(counts[1]) else None
            return d

        def _top_tf(arr: np.ndarray, k: int = 8) -> List[Dict[str, Any]]:
            if arr.size == 0 or int(arr.sum()) == 0:
                return []
            top = np.argsort(arr)[::-1][:k]
            out = []
            tot = int(arr.sum())
            for i in top:
                n = int(arr[i])
                if n <= 0:
                    break
                name = gene_list[int(i)] if int(i) < len(gene_list) else str(int(i))
                out.append({"gene": name, "n": n, "pct_of_class_edges": float(n) / float(tot)})
            return out

        def _side(n: int, counts: np.ndarray, extra: Dict[str, Any]) -> Dict[str, Any]:
            n_kegg = int(extra["n_true"][1] + extra["n_true"][2])
            n_off = int(extra["n_true"][0])
            d = _pack(n, counts)
            d["on_KEGG_labeled_pairs"] = _pack(n_kegg, extra["on_kegg"])
            d["off_KEGG_unlabeled_pairs"] = _pack(n_off, extra["off_kegg"])
            d["top_Repress_TF"] = _top_tf(extra["repress_tf"])
            d["top_Activate_TF"] = _top_tf(extra["activate_tf"])
            return d

        payload = {
            "n_pairs": int(n_a),
            "n_true_NoReg_Repress_Activate": [int(x) for x in extra_a["n_true"].tolist()],
            "class_order": list(CLASS_NAMES),
            slug_a: _side(int(n_a), c_a, extra_a),
            slug_b: _side(int(n_b), c_b, extra_b),
            "max_expression_cols": int(args.max_expression_cols),
            "expression_column_split_part": str(args.expression_column_split_part),
            "expression_sample_split": str(args.expression_sample_split),
            "norm_fit_domain": str(args.norm_fit_domain),
            "inference_batch_size": int(bs),
            "count_method": "gene_encoder_cache",
            "note": "argmax on 3-class logits; regulatory = Repress + Activate. on_KEGG = pair true label 1/2 in GRN matrix; off_KEGG = unlabeled cartesian pairs.",
        }
        out_json = tables_dir / "cartesian_pred_class_counts.json"
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _ts_log(f"cartesian class counts -> {out_json}")
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        _ts_log("=== count-pred-classes-only finished (no tables/figures rewritten) ===")
        return

    yt, p_nb, pr_nb, tfn, tgn = _predict_with_names(
        model_nb,
        ln,
        device,
        gene_list,
        progress_desc=f"{args.group_a_label} inference",
        show_progress=not bool(args.no_progress),
        use_amp_bf16=use_amp_infer,
    )
    yt2, p_br, pr_br, tfn2, tgn2 = _predict_with_names(
        model_br,
        lb,
        device,
        gene_list,
        progress_desc=f"{args.group_b_label} inference",
        show_progress=not bool(args.no_progress),
        use_amp_bf16=use_amp_infer,
    )
    if not (np.array_equal(yt, yt2) and tfn == tfn2 and tgn == tgn2):
        raise RuntimeError("pair alignment mismatch between nonbrain and brain predictions")

    n_total = int(len(yt))
    _ts_log(f"both-side inference finished; n_pairs={n_total}")
    diff_mask = p_nb != p_br
    n_diff = int(diff_mask.sum())
    _ts_log(f"pred argmax diff count={n_diff} — writing pred_diff / prob tables and figures…")
    pred_csv = tables_dir / f"{pair_prefix}_pred_diff.csv"
    pred_summary = tables_dir / f"{pair_prefix}_pred_diff_summary.json"
    col_pred_a, col_pred_b = f"Pred_{slug_a}", f"Pred_{slug_b}"
    col_pred_a_n, col_pred_b_n = f"{col_pred_a}_Name", f"{col_pred_b}_Name"
    pred_cols = {
        "TF": [tfn[i] for i in range(n_total) if diff_mask[i]],
        "Target": [tgn[i] for i in range(n_total) if diff_mask[i]],
        "True_Label": yt[diff_mask],
        "True_Label_Name": [CLASS_NAMES[int(yt[i])] for i in range(n_total) if diff_mask[i]],
        col_pred_a: p_nb[diff_mask],
        col_pred_a_n: [CLASS_NAMES[int(p_nb[i])] for i in range(n_total) if diff_mask[i]],
        col_pred_b: p_br[diff_mask],
        col_pred_b_n: [CLASS_NAMES[int(p_br[i])] for i in range(n_total) if diff_mask[i]],
    }
    for k, name in enumerate(CLASS_NAMES):
        pred_cols[f"P_{slug_a}_{name}"] = pr_nb[diff_mask, k]
        pred_cols[f"P_{slug_b}_{name}"] = pr_br[diff_mask, k]
    pd.DataFrame(pred_cols).to_csv(pred_csv, index=False, encoding="utf-8-sig")
    pred_summary.write_text(
        json.dumps(
            {"n_pairs": n_total, "n_pred_diff": n_diff, "fraction_pred_diff": (n_diff / n_total) if n_total else 0.0, "csv": pred_csv.name},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    abs_diff = np.abs(pr_nb - pr_br)
    max_diff = abs_diff.max(axis=1)
    prob_thr = float(args.prob_diff_threshold)
    prob_mask = max_diff > prob_thr
    n_prob_gt = int(prob_mask.sum())
    thr_slug = str(prob_thr).replace(".", "p")
    prob_thr_csv = tables_dir / f"{pair_prefix}_prob_diff_gt{thr_slug}.csv"
    prob_thr_summary = tables_dir / f"{pair_prefix}_prob_diff_gt{thr_slug}_summary.json"
    _ts_log(f"max_prob_diff > {prob_thr}: n={n_prob_gt} ({(n_prob_gt / n_total) if n_total else 0.0:.4%} of pairs)")
    prob_thr_cols = {
        "TF": [tfn[i] for i in range(n_total) if prob_mask[i]],
        "Target": [tgn[i] for i in range(n_total) if prob_mask[i]],
        "True_Label": [int(yt[i]) for i in range(n_total) if prob_mask[i]],
        "True_Label_Name": [CLASS_NAMES[int(yt[i])] for i in range(n_total) if prob_mask[i]],
        col_pred_a: [int(p_nb[i]) for i in range(n_total) if prob_mask[i]],
        col_pred_b: [int(p_br[i]) for i in range(n_total) if prob_mask[i]],
        col_pred_a_n: [CLASS_NAMES[int(p_nb[i])] for i in range(n_total) if prob_mask[i]],
        col_pred_b_n: [CLASS_NAMES[int(p_br[i])] for i in range(n_total) if prob_mask[i]],
        "max_prob_diff": [float(max_diff[i]) for i in range(n_total) if prob_mask[i]],
    }
    for k, name in enumerate(CLASS_NAMES):
        prob_thr_cols[f"P_{slug_a}_{name}"] = [float(pr_nb[i, k]) for i in range(n_total) if prob_mask[i]]
        prob_thr_cols[f"P_{slug_b}_{name}"] = [float(pr_br[i, k]) for i in range(n_total) if prob_mask[i]]
    pd.DataFrame(prob_thr_cols).to_csv(prob_thr_csv, index=False, encoding="utf-8-sig")
    prob_thr_summary.write_text(
        json.dumps(
            {
                "n_pairs": n_total,
                "prob_diff_threshold": prob_thr,
                "n_prob_diff_gt_threshold": n_prob_gt,
                "fraction_prob_diff_gt_threshold": (n_prob_gt / n_total) if n_total else 0.0,
                "n_pred_diff_argmax": n_diff,
                "csv": prob_thr_csv.name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    topk = min(int(args.prob_diff_topk), n_total)
    top_idx = np.argsort(-max_diff)[:topk]
    prob_csv = tables_dir / f"{pair_prefix}_prob_diff_top{topk}.csv"
    prob_summary = tables_dir / f"{pair_prefix}_prob_diff_summary.json"
    prob_cols = {
        "TF": [tfn[i] for i in top_idx],
        "Target": [tgn[i] for i in top_idx],
        "True_Label": [int(yt[i]) for i in top_idx],
        "True_Label_Name": [CLASS_NAMES[int(yt[i])] for i in top_idx],
        col_pred_a: [int(p_nb[i]) for i in top_idx],
        col_pred_b: [int(p_br[i]) for i in top_idx],
        col_pred_a_n: [CLASS_NAMES[int(p_nb[i])] for i in top_idx],
        col_pred_b_n: [CLASS_NAMES[int(p_br[i])] for i in top_idx],
        "max_prob_diff": [float(max_diff[i]) for i in top_idx],
    }
    for k, name in enumerate(CLASS_NAMES):
        prob_cols[f"P_{slug_a}_{name}"] = [float(pr_nb[i, k]) for i in top_idx]
        prob_cols[f"P_{slug_b}_{name}"] = [float(pr_br[i, k]) for i in top_idx]
    pd.DataFrame(prob_cols).to_csv(prob_csv, index=False, encoding="utf-8-sig")
    prob_summary.write_text(
        json.dumps(
            {
                "n_pairs": n_total,
                "n_pred_diff": n_diff,
                "prob_diff_threshold": prob_thr,
                "n_prob_diff_gt_threshold": n_prob_gt,
                "fraction_prob_diff_gt_threshold": (n_prob_gt / n_total) if n_total else 0.0,
                "topk": topk,
                "max_prob_diff_all_edges": float(max_diff.max()) if n_total else 0.0,
                "mean_max_prob_diff_all_edges": float(max_diff.mean()) if n_total else 0.0,
                "csv": prob_csv.name,
                "prob_diff_gt_csv": prob_thr_csv.name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    pred_edges = pd.DataFrame()
    if n_diff > 0:
        w = abs_diff.max(axis=1)
        pred_edges = pd.DataFrame(
            {"TF": [tfn[i] for i in range(n_total) if diff_mask[i]], "Target": [tgn[i] for i in range(n_total) if diff_mask[i]], "weight": w[diff_mask], "edge_origin": "pred_argmax_diff"}
        )
    prob_top_edges = pd.DataFrame(
        {"TF": [tfn[i] for i in top_idx], "Target": [tgn[i] for i in top_idx], "weight": [float(max_diff[i]) for i in top_idx], "edge_origin": "prob_top_export"}
    )
    if args.min_prob_diff > 0:
        prob_top_edges = prob_top_edges[prob_top_edges["weight"] >= float(args.min_prob_diff)].copy()

    prob_threshold_edges = pd.DataFrame()
    if n_prob_gt > 0:
        prob_threshold_edges = pd.DataFrame(
            {
                "TF": [tfn[i] for i in range(n_total) if prob_mask[i]],
                "Target": [tgn[i] for i in range(n_total) if prob_mask[i]],
                "weight": [float(max_diff[i]) for i in range(n_total) if prob_mask[i]],
                "edge_origin": "prob_diff_gt_threshold",
            }
        )

    edge_mode_used = args.edge_mode
    if args.edge_mode == "auto":
        edge_mode_used = "prob_threshold" if len(prob_threshold_edges) > 0 else "prob_top"
    if edge_mode_used == "pred_diff":
        edges_used = pred_edges.copy()
    elif edge_mode_used == "prob_threshold":
        edges_used = prob_threshold_edges.copy()
    elif edge_mode_used == "prob_top":
        edges_used = prob_top_edges.copy()
    else:
        parts = [df for df in (pred_edges, prob_threshold_edges, prob_top_edges) if len(df)]
        edges_used = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not edges_used.empty:
            edges_used = (
                edges_used.groupby(["TF", "Target"], as_index=False)
                .agg(weight=("weight", "max"), edge_origin=("edge_origin", lambda s: "|".join(sorted(set(s.astype(str))))))
            )
    if edges_used.empty:
        edges_used = pd.DataFrame(columns=["TF", "Target", "weight", "edge_origin"])

    # Add edge categories for visualization/interpretation.
    pair_df = pd.DataFrame(
        {
            "TF": tfn,
            "Target": tgn,
            ca_col: p_nb.astype(int),
            cb_col: p_br.astype(int),
            "max_prob_diff": max_diff.astype(float),
        }
    )
    pair_df = pair_df.drop_duplicates(subset=["TF", "Target"], keep="first")
    edges_used = edges_used.merge(pair_df, on=["TF", "Target"], how="left")
    edges_used[f"{ca_col}_name"] = edges_used[ca_col].map(lambda x: CLASS_NAMES[int(x)] if pd.notna(x) else "")
    edges_used[f"{cb_col}_name"] = edges_used[cb_col].map(lambda x: CLASS_NAMES[int(x)] if pd.notna(x) else "")

    def _cat_row(r: pd.Series) -> str:
        if pd.isna(r[ca_col]) or pd.isna(r[cb_col]):
            return "other_class_switch"
        a = int(r[ca_col])
        b = int(r[cb_col])
        if a == b:
            return "same_class_prob_shift"
        if a == 0 and b != 0:
            return "mBrain_gain_regulation"
        if a != 0 and b == 0:
            return "mBrain_loss_regulation"
        if a == 1 and b == 2:
            return "repression_to_activation"
        if a == 2 and b == 1:
            return "activation_to_repression"
        return "other_class_switch"

    edges_used["edge_category"] = edges_used.apply(_cat_row, axis=1)

    net_stat = _save_network(
        edges_used,
        tables_dir,
        figures_dir,
        int(args.hub_top_n),
        int(args.max_nodes_plot),
        plot_title=args.network_plot_title,
        plot_group_a_display=str(args.group_a_label),
        plot_group_b_display=str(args.group_b_label),
    )
    meta = {
        "step1_run_dir": str(Path(args.step1_run_dir).resolve()),
        "nonbrain_model_path": str(model_path_nb.resolve()),
        "brain_model_path": str(model_path_br.resolve()),
        "expression_column_split_part": str(args.expression_column_split_part),
        "expression_sample_split": str(args.expression_sample_split),
        "nonbrain_column_range_half_open": list(nb_col_range) if nb_col_range else None,
        "brain_column_range_half_open": list(br_col_range) if br_col_range else None,
        "nonbrain_expression_source": "npy" if nb_npy else "csv",
        "nonbrain_exp_csv": str(Path(args.nonbrain_exp_csv).resolve()) if args.nonbrain_exp_csv else None,
        "nonbrain_exp_npy_values": str(Path(args.nonbrain_exp_npy_values).resolve()) if args.nonbrain_exp_npy_values else None,
        "nonbrain_exp_npy_genes": str(Path(args.nonbrain_exp_npy_genes).resolve()) if args.nonbrain_exp_npy_genes else None,
        "brain_expression_source": "npy" if br_npy else "csv",
        "brain_exp_csv": str(Path(args.brain_exp_csv).resolve()) if args.brain_exp_csv else None,
        "brain_exp_npy_values": str(Path(args.brain_exp_npy_values).resolve()) if args.brain_exp_npy_values else None,
        "brain_exp_npy_genes": str(Path(args.brain_exp_npy_genes).resolve()) if args.brain_exp_npy_genes else None,
        "inference_amp_bf16": bool(use_amp_infer),
        "pin_memory_dataloader": bool(pin_mem),
        "group_a_label": str(args.group_a_label),
        "group_b_label": str(args.group_b_label),
        "pair_export_prefix": pair_prefix,
        "network_plot_title": args.network_plot_title,
        "output_dir": str(out_dir),
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
        "pair_build": (
            "de_cartesian"
            if args.de_cartesian_csv
            else (f"kegg_khop_k{int(args.kegg_k_hops)}" if args.kegg_k_hops is not None else "grn_matrix_neg_sampled")
        ),
        "de_cartesian_csv": str(Path(args.de_cartesian_csv).resolve()) if args.de_cartesian_csv else None,
        "de_use_all_rows": bool(args.de_use_all_rows) if args.de_cartesian_csv else None,
        "de_pvalue_col": (
            str(args.de_pvalue_col)
            if (
                args.de_cartesian_csv
                and (not bool(args.de_use_all_rows))
                and (args.de_padj_max is None and args.de_logfc_min is None)
            )
            else None
        ),
        "de_pvalue_max": (
            float(args.de_pvalue_max)
            if (
                args.de_cartesian_csv
                and (not bool(args.de_use_all_rows))
                and (args.de_padj_max is None and args.de_logfc_min is None)
            )
            else None
        ),
        "de_padj_col": (str(args.de_padj_col) if (args.de_cartesian_csv and args.de_padj_max is not None) else None),
        "de_padj_max": (float(args.de_padj_max) if (args.de_cartesian_csv and args.de_padj_max is not None) else None),
        "de_logfc_col": (str(args.de_logfc_col) if (args.de_cartesian_csv and args.de_logfc_min is not None) else None),
        "de_logfc_min": (float(args.de_logfc_min) if (args.de_cartesian_csv and args.de_logfc_min is not None) else None),
        "de_cartesian_pair_stats": de_stats,
        "kegg_k_hops": int(args.kegg_k_hops) if args.kegg_k_hops is not None else None,
        "kegg_khop_pair_stats": kh_stats,
        "edge_mode_requested": args.edge_mode,
        "edge_mode_effective": edge_mode_used,
        "norm_fit_domain": args.norm_fit_domain,
        "minmax_col_chunk": int(args.minmax_col_chunk),
        "inference_batch_size": int(bs),
        "dataloader_num_workers": int(args.dataloader_num_workers),
        "n_shared_genes": len(gene_list),
        "n_pairs": n_total,
        "n_pred_diff": n_diff,
        "prob_diff_threshold": prob_thr,
        "n_prob_diff_gt_threshold": n_prob_gt,
        "prob_topk": topk,
        "n_edges_used": int(len(edges_used)),
        "n_nodes": int(net_stat["n_nodes"]),
    }
    (tables_dir / "case_meta_step1_model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _ts_log(f"=== finished OK -> {out_dir} ===")


if __name__ == "__main__":
    main()
