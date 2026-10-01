"""
脑转移 GSE131907 案例分析：非脑训练模型在「非脑 test 列」与「脑 mBrain 全列」上的预测差异网络。

输入：train_metastasis_nonbrain_brain_eval 运行目录下的
  - pairs_test_nonbrain_vs_mBrain_pred_diff.csv（argmax 不一致的边，可为空）
  - pairs_test_nonbrain_vs_mBrain_prob_diff_top{K}.csv（按类概率 L∞ 差排序的前 K 条边）

输出（默认 run_dir/case_study_brain_nb_diff/；可用 --case-study-out 指定到 experiments 下任意目录）：
  - differential_edges_used.csv：用于构图的边及权重
  - hub_genes.csv：度与介数等 hub 指标
  - network_plot.png：网络示意图（大图为稀疏展示，仅标注部分 hub）
  - enrichment_hubs_gprofiler.json / enrichment_hubs_gprofiler_terms.csv：hub 基因集 g:Profiler 富集
  - enrichment_all_nodes_gprofiler.json / ..._terms.csv：（可选）网络中全部基因的富集

依赖：pandas, matplotlib, networkx；富集使用 urllib 调用 g:Profiler REST（需联网）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


PRED_DIFF_NAME = "pairs_test_nonbrain_vs_mBrain_pred_diff.csv"


def _find_pred_diff_csv(run_dir: Path) -> Path | None:
    """匹配 *pred_diff.csv 且非 summary；多文件时优先 nonbrain_vs_mBrain。"""
    cands = [p for p in run_dir.glob("*pred_diff*.csv") if "summary" not in p.name.lower()]
    if not cands:
        p = run_dir / PRED_DIFF_NAME
        return p if p.is_file() else None
    pref = [p for p in cands if "nonbrain" in p.name.lower() and "mbrain" in p.name.lower()]
    use = pref if pref else cands
    return sorted(use, key=lambda x: x.name)[0]


def _find_prob_diff_gt_csv(run_dir: Path) -> Path | None:
    """优先 tables/differential_edges_used.csv，其次 *prob_diff_gt*.csv。"""
    for sub in ("tables", "."):
        p = run_dir / sub / "differential_edges_used.csv"
        if p.is_file():
            return p
    cands: list[Path] = []
    for sub in ("tables", "."):
        base = run_dir / sub if sub != "." else run_dir
        if base.is_dir():
            cands.extend(base.glob("*prob_diff_gt*.csv"))
    if not cands:
        return None
    pref = [p for p in cands if "summary" not in p.name.lower()]
    pool = pref if pref else cands
    return max(pool, key=lambda p: p.stat().st_mtime)


def _load_prob_threshold_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty or "TF" not in df.columns or "Target" not in df.columns:
        return pd.DataFrame()
    if "weight" in df.columns:
        w = df["weight"].astype(float)
    elif "max_prob_diff" in df.columns:
        w = df["max_prob_diff"].astype(float)
    else:
        w = pd.Series(1.0, index=df.index)
    return pd.DataFrame(
        {
            "TF": df["TF"].astype(str),
            "Target": df["Target"].astype(str),
            "weight": w,
            "edge_origin": "prob_diff_gt_threshold",
        }
    )


def _find_prob_diff_csv(run_dir: Path) -> Path | None:
    cands = sorted(run_dir.glob("*prob_diff_top*.csv"))

    def topk_num(p: Path) -> int:
        m = re.search(r"top(\d+)\.csv$", p.name, re.I)
        return int(m.group(1)) if m else 0

    if not cands:
        return None
    pref = [p for p in cands if "nonbrain" in p.name.lower() and "mbrain" in p.name.lower()]
    pool = pref if pref else cands
    return max(pool, key=lambda p: (topk_num(p), p.stat().st_mtime))


def _load_pred_diff_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty or "TF" not in df.columns:
        return pd.DataFrame()
    wcols = [c for c in df.columns if c.startswith("P_NonBrain_") and c.replace("P_NonBrain_", "P_mBrain_") in df.columns]
    if wcols:
        diffs = []
        for cnb in wcols:
            cmb = cnb.replace("P_NonBrain_", "P_mBrain_")
            diffs.append((df[cnb].astype(float) - df[cmb].astype(float)).abs())
        weight = pd.concat(diffs, axis=1).max(axis=1)
    else:
        weight = pd.Series(1.0, index=df.index)
    out = pd.DataFrame(
        {
            "TF": df["TF"].astype(str),
            "Target": df["Target"].astype(str),
            "weight": weight.astype(float),
            "edge_origin": "pred_argmax_diff",
        }
    )
    if "True_Label_Name" in df.columns:
        out["true_label_name"] = df["True_Label_Name"].astype(str)
    return out


def _load_prob_top_edges(path: Path, min_weight: float, max_rows: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "max_prob_diff" not in df.columns:
        raise ValueError(f"{path}: missing column max_prob_diff")
    df = df[df["max_prob_diff"].astype(float) >= min_weight].copy()
    df = df.sort_values("max_prob_diff", ascending=False)
    if max_rows is not None and max_rows > 0:
        df = df.head(int(max_rows))
    return pd.DataFrame(
        {
            "TF": df["TF"].astype(str),
            "Target": df["Target"].astype(str),
            "weight": df["max_prob_diff"].astype(float),
            "edge_origin": "prob_top_export",
        }
    )


def _merge_edges(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame(columns=["TF", "Target", "weight", "edge_origin"])
    all_e = pd.concat(dfs, ignore_index=True)
    all_e["pair"] = list(zip(all_e["TF"], all_e["Target"]))
    # 同一对有多个来源时保留较大权重，来源合并为列表
    agg = (
        all_e.groupby(["TF", "Target"], as_index=False)
        .agg(weight=("weight", "max"), edge_origin=("edge_origin", lambda s: "|".join(sorted(set(s.astype(str))))))
    )
    return agg


def _build_undirected_graph(edges: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, r in edges.iterrows():
        u, v, w = str(r["TF"]), str(r["Target"]), float(r["weight"])
        if G.has_edge(u, v):
            G[u][v]["weight"] = max(float(G[u][v]["weight"]), w)
        else:
            G.add_edge(u, v, weight=w)
    return G


def _hub_table(G: nx.Graph) -> pd.DataFrame:
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["gene", "degree", "weighted_degree", "betweenness"])
    wdeg = {n: sum(G[n][m]["weight"] for m in G.neighbors(n)) for n in G.nodes()}
    btw = nx.betweenness_centrality(G, weight=None, normalized=True, seed=42)
    deg = dict(G.degree())
    rows = []
    for n in G.nodes():
        rows.append(
            {
                "gene": n,
                "degree": int(deg[n]),
                "weighted_degree": float(wdeg[n]),
                "betweenness": float(btw.get(n, 0.0)),
            }
        )
    tab = pd.DataFrame(rows)
    tab["hub_score"] = tab["degree"].rank(pct=True) + tab["betweenness"].rank(pct=True)
    return tab.sort_values("hub_score", ascending=False).reset_index(drop=True)


def _draw_network(
    G: nx.Graph,
    hub_genes: set[str],
    out_png: Path,
    title: str,
    max_nodes_display: int,
) -> None:
    if G.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "empty graph", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    if G.number_of_nodes() > max_nodes_display:
        nodes_keep = set(hub_genes)
        for n in sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True):
            if len(nodes_keep) >= max_nodes_display:
                break
            nodes_keep.add(n)
        Gs = G.subgraph(nodes_keep).copy()
    else:
        Gs = G

    pos = nx.spring_layout(Gs, seed=42, k=1.2 / max(np.sqrt(Gs.number_of_nodes()), 1))
    w = [float(Gs[u][v].get("weight", 1.0)) for u, v in Gs.edges()]
    w_max = max(w) if w else 1.0
    widths = [1.0 + 4.0 * (x / w_max) for x in w]
    node_list = list(Gs.nodes())
    sizes = []
    colors = []
    for n in node_list:
        deg = Gs.degree(n)
        sizes.append(120 + 40 * deg)
        colors.append("#e74c3c" if n in hub_genes else "#3498db")

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_edges(Gs, pos, ax=ax, alpha=0.35, width=widths, edge_color="#7f8c8d")
    nx.draw_networkx_nodes(Gs, pos, ax=ax, node_size=sizes, node_color=colors, alpha=0.9)
    labels = {n: n for n in node_list if n in hub_genes}
    nx.draw_networkx_labels(Gs, pos, labels=labels, ax=ax, font_size=8)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _gprofiler_enrich(query: list[str], organism: str) -> dict[str, Any]:
    url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
    payload: dict[str, Any] = {
        "query": query,
        "organism": organism,
        "significant": True,
        "sources": ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC", "WP"],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _gprofiler_to_terms_table(res: dict[str, Any]) -> pd.DataFrame:
    rows = res.get("result") or []
    if not rows:
        return pd.DataFrame(columns=["source", "native", "name", "p_value", "intersection_size"])
    rec = []
    for r in rows:
        rec.append(
            {
                "source": r.get("source"),
                "native": r.get("native"),
                "name": r.get("name"),
                "p_value": r.get("p_value"),
                "intersection_size": r.get("intersection_size"),
                "term_size": r.get("term_size"),
            }
        )
    return pd.DataFrame(rec).sort_values("p_value", ascending=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Case study: brain vs non-brain pred-diff network, hubs, enrichment.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="metastasis_nonbrain_brain_eval 实验目录（含 pred_diff / prob_diff CSV）。",
    )
    parser.add_argument(
        "--out-subdir",
        default="case_study_brain_nb_diff",
        help="在 run-dir 下创建的输出子目录名（与 --case-study-out 二选一，后者优先）。",
    )
    parser.add_argument(
        "--case-study-out",
        default=None,
        help="若指定：网络图、边表、hub、R 用基因列表等全部写入该目录（绝对路径或相对当前工作目录），便于统一放到 experiments/ 下。",
    )
    parser.add_argument(
        "--edge-mode",
        choices=["auto", "pred_diff", "prob_threshold", "prob_top", "union"],
        default="prob_threshold",
        help="prob_threshold：differential_edges 或 prob_diff_gt* CSV（max_prob_diff>阈值）；auto 优先 prob_threshold。",
    )
    parser.add_argument("--min-prob-diff", type=float, default=0.0, help="prob_top 边最小 max_prob_diff。")
    parser.add_argument("--max-edges", type=int, default=0, help="prob_top 最多保留边数；0 表示不截断。")
    parser.add_argument("--hub-top-n", type=int, default=25, help="按 hub_score 取前 N 个 hub 基因用于标注与富集。")
    parser.add_argument("--max-nodes-plot", type=int, default=80, help="网络图中最多展示的节点数（过大则取子图）。")
    parser.add_argument("--organism", default="hsapiens", help="g:Profiler organism id。")
    parser.add_argument("--skip-enrichment", action="store_true", help="跳过 g:Profiler（离线或调试）。")
    parser.add_argument(
        "--enrich-all-nodes",
        action="store_true",
        help="额外对网络中全部节点做一次富集（通常较宽；默认仅 hub 集）",
    )
    parser.add_argument(
        "--pred-diff-csv",
        default=None,
        help="手动指定 pred_diff CSV（默认在 run-dir 下自动发现 *pred_diff*.csv）。",
    )
    parser.add_argument(
        "--prob-diff-csv",
        default=None,
        help="手动指定 prob_diff_top*.csv（默认自动发现 *prob_diff_top*.csv，优先 nonbrain*mBrain）。",
    )
    args = parser.parse_args()

    run_dir = Path(os.path.abspath(args.run_dir))
    if not run_dir.is_dir():
        raise SystemExit(f"run-dir not found: {run_dir}")

    if args.case_study_out:
        out_dir = Path(os.path.abspath(args.case_study_out))
    else:
        out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = Path(os.path.abspath(args.pred_diff_csv)) if args.pred_diff_csv else _find_pred_diff_csv(run_dir)
    prob_path = Path(os.path.abspath(args.prob_diff_csv)) if args.prob_diff_csv else _find_prob_diff_csv(run_dir)
    prob_gt_path = _find_prob_diff_gt_csv(run_dir)

    if pred_path is None or not pred_path.is_file():
        pred_edges = pd.DataFrame()
    else:
        pred_edges = _load_pred_diff_edges(pred_path)
    if prob_path is not None and prob_path.is_file():
        prob_edges = _load_prob_top_edges(prob_path, args.min_prob_diff, args.max_edges or None)
    else:
        prob_edges = pd.DataFrame()
    if prob_gt_path is not None and prob_gt_path.is_file():
        prob_threshold_edges = _load_prob_threshold_edges(prob_gt_path)
    else:
        prob_threshold_edges = pd.DataFrame()

    frames: list[pd.DataFrame] = []
    mode = args.edge_mode
    if mode == "auto":
        if len(prob_threshold_edges) > 0:
            frames.append(prob_threshold_edges)
        elif len(pred_edges) > 0:
            frames.append(pred_edges)
        elif len(prob_edges) > 0:
            frames.append(prob_edges)
        else:
            raise SystemExit(
                "无可用边：未找到 differential_edges_used / prob_diff_gt* / pred_diff / prob_diff_top CSV。"
                "请先完成 case study 或 train_metastasis 导出。"
            )
    elif mode == "prob_threshold":
        if len(prob_threshold_edges) == 0:
            raise SystemExit("edge-mode=prob_threshold 但差异边为空（检查 differential_edges_used 或 prob_diff_gt* CSV）。")
        frames.append(prob_threshold_edges)
    elif mode == "pred_diff":
        if len(pred_edges) == 0:
            raise SystemExit("edge-mode=pred_diff 但 pred_diff CSV 无数据行。")
        frames.append(pred_edges)
    elif mode == "prob_top":
        if len(prob_edges) == 0:
            raise SystemExit("edge-mode=prob_top 但 prob 边为空（检查 min-prob-diff 或 CSV）。")
        frames.append(prob_edges)
    else:
        if len(prob_threshold_edges) > 0:
            frames.append(prob_threshold_edges)
        if len(pred_edges) > 0:
            frames.append(pred_edges)
        if len(prob_edges) > 0:
            frames.append(prob_edges)
        if not frames:
            raise SystemExit("union 模式下 prob_threshold / pred_diff / prob_top 均无数据。")

    edges_used = _merge_edges(frames)
    edges_path = out_dir / "differential_edges_used.csv"
    edges_used.to_csv(edges_path, index=False, encoding="utf-8-sig")

    G = _build_undirected_graph(edges_used)
    hubs = _hub_table(G)
    hubs_path = out_dir / "hub_genes.csv"
    hubs.to_csv(hubs_path, index=False, encoding="utf-8-sig")

    topn = min(args.hub_top_n, len(hubs))
    hub_set = set(hubs.head(topn)["gene"].astype(str)) if topn else set()
    hub_gene_list = hubs.head(topn)["gene"].astype(str).tolist() if topn else []

    plot_path = out_dir / "network_plot.png"
    _draw_network(
        G,
        hub_set,
        plot_path,
        title=f"Brain vs non-brain prediction differential network ({G.number_of_nodes()} genes, {G.number_of_edges()} edges)",
        max_nodes_display=args.max_nodes_plot,
    )
    r_hub_path = out_dir / "r_hub_gene_symbols.txt"
    r_net_path = out_dir / "r_network_all_gene_symbols.txt"
    r_hub_path.write_text("\n".join(hub_gene_list) + ("\n" if hub_gene_list else ""), encoding="utf-8")
    r_net_path.write_text("\n".join(sorted(str(n) for n in G.nodes())) + ("\n" if G.number_of_nodes() else ""), encoding="utf-8")
    print(f"[case-study] R gene lists -> {r_hub_path.name}, {r_net_path.name}")

    meta = {
        "run_dir": str(run_dir),
        "case_study_output_dir": str(out_dir),
        "edge_mode_effective": mode,
        "pred_diff_path": str(pred_path) if pred_path is not None and pred_path.is_file() else None,
        "prob_diff_path": str(prob_path) if prob_path is not None and prob_path.is_file() else None,
        "n_edges_used": int(len(edges_used)),
        "n_nodes": int(G.number_of_nodes()),
        "hub_top_n": int(topn),
        "outputs": {
            "edges": str(edges_path),
            "hubs": str(hubs_path),
            "network_plot": str(plot_path),
            "r_hub_gene_symbols": str(r_hub_path),
            "r_network_all_gene_symbols": str(r_net_path),
        },
    }
    with open(out_dir / "case_study_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.skip_enrichment:
        print(f"[case-study] done (skip Python g:Profiler; run Rscript on {r_hub_path.parent}) -> {out_dir}")
        return

    hub_genes = hub_gene_list
    all_nodes = list(G.nodes())

    def _run_one(label: str, genes: list[str]) -> None:
        if len(genes) < 2:
            print(f"[case-study] skip enrichment {label}: too few genes ({len(genes)})")
            return
        try:
            raw = _gprofiler_enrich(genes, args.organism)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"[case-study] g:Profiler failed ({label}): {e}")
            with open(out_dir / f"enrichment_{label}_FAILED.txt", "w", encoding="utf-8") as f:
                f.write(repr(e) + "\n\n")
                f.write("可在本机用 R 做 GO 富集（不访问 g:Profiler）：\n")
                f.write("  Rscript grn/r/metastasis_case_study_enrich_clusterprofiler.R <本目录>\n")
                f.write("需已安装 Bioconductor：clusterProfiler, org.Hs.eg.db\n\n")
                f.write("genes:\n" + "\n".join(genes))
            return
        js_path = out_dir / f"enrichment_{label}_gprofiler.json"
        js_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        tbl = _gprofiler_to_terms_table(raw)
        tbl.to_csv(out_dir / f"enrichment_{label}_gprofiler_terms.csv", index=False, encoding="utf-8-sig")
        print(f"[case-study] enrichment {label}: {len(tbl)} terms -> {js_path.name}")

    _run_one("hubs", hub_genes)
    if args.enrich_all_nodes and len(all_nodes) >= 2:
        _run_one("all_nodes", all_nodes)

    print(f"[case-study] done -> {out_dir}")


if __name__ == "__main__":
    main()
