# -*- coding: utf-8 -*-
"""Update independent case mainline to mBrain-gain Top500 by max_prob_diff.

Writes tables, hub stats, network plot, g:Profiler enrichment + dotplots,
and refreshes INDEPENDENT_CASE_REPORT.md section summary JSON.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "experiments" / "gse131907_independent_tlung_mbrain_seed42" / "case_tlung_vs_mbrain"
TABLES = CASE / "tables"
OUT = CASE / "followup"
REPORT = ROOT / "experiments" / "gse131907_independent_tlung_mbrain_seed42" / "INDEPENDENT_CASE_REPORT.md"
KEGG = ROOT / "data" / "network" / "grn" / "KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv"
TOPK = 500
HUB_FRAC = 0.10


def _hub_gain(gain: pd.DataFrame) -> pd.DataFrame:
    out = gain.groupby("TF").agg(out_degree=("Target", "count"), out_w=("max_prob_diff", "sum"))
    inn = gain.groupby("Target").agg(in_degree=("TF", "count"), in_w=("max_prob_diff", "sum"))
    genes = sorted(set(gain["TF"].astype(str)) | set(gain["Target"].astype(str)))
    rows = []
    n = max(len(gain), 1)
    for g in genes:
        od = int(out.out_degree.get(g, 0)) if g in out.index else 0
        ow = float(out.out_w.get(g, 0.0)) if g in out.index else 0.0
        ind = int(inn.in_degree.get(g, 0)) if g in inn.index else 0
        iw = float(inn.in_w.get(g, 0.0)) if g in inn.index else 0.0
        rows.append(
            {
                "gene": g,
                "out_degree": od,
                "in_degree": ind,
                "degree": od + ind,
                "weighted_degree": ow + iw,
                "in_frac_of_edges": ind / n,
                "out_frac_of_edges": od / n,
            }
        )
    return pd.DataFrame(rows).sort_values(["degree", "weighted_degree"], ascending=False)


def _gprofiler(query: list[str]) -> dict[str, Any]:
    url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
    payload = {
        "query": query,
        "organism": "hsapiens",
        "significant": True,
        "sources": ["GO:BP", "GO:MF", "GO:CC", "KEGG"],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "TripGRN"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _results_to_df(gost: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for r in gost.get("result", []) or []:
        rows.append(
            {
                "source": r.get("source"),
                "native": r.get("native"),
                "name": r.get("name"),
                "p_value": r.get("p_value"),
                "term_size": r.get("term_size"),
                "query_size": r.get("query_size"),
                "intersection_size": r.get("intersection_size"),
            }
        )
    return pd.DataFrame(rows)


def _dotplot(df: pd.DataFrame, source: str, out_png: Path, title: str, n: int = 15) -> None:
    sub = df[df["source"] == source].sort_values("p_value").head(n).copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"no significant {source} terms", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    sub = sub.iloc[::-1]
    sub["neglog10"] = -np.log10(sub["p_value"].astype(float).clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.35 * len(sub))))
    ax.scatter(
        sub["neglog10"],
        range(len(sub)),
        s=40 + 8 * sub["intersection_size"].astype(float),
        c=sub["neglog10"],
        cmap="Reds",
        edgecolors="#333",
        linewidths=0.4,
    )
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["name"].astype(str).str.slice(0, 55))
    ax.set_xlabel("-log10(p)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _draw_network(edges: pd.DataFrame, hub: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    G = nx.DiGraph()
    for _, r in edges.iterrows():
        G.add_edge(str(r["TF"]), str(r["Target"]), weight=float(r["max_prob_diff"]))
    # label top hubs (super + filtered markers)
    supers = set(
        hub.loc[
            (hub["out_frac_of_edges"] >= HUB_FRAC) | (hub["in_frac_of_edges"] >= HUB_FRAC),
            "gene",
        ].astype(str)
    )
    top_label = set(hub.head(12)["gene"].astype(str)) | supers
    # display: keep nodes touching top-degree genes to avoid hairball
    keep = set(top_label)
    for n in sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True):
        if len(keep) >= 120:
            break
        keep.add(n)
    Gs = G.subgraph(keep).copy()
    # undirected layout for readability
    Gu = Gs.to_undirected()
    pos = nx.spring_layout(Gu, seed=42, k=1.4 / max(np.sqrt(Gu.number_of_nodes()), 1))

    fig, ax = plt.subplots(figsize=(11, 9))
    widths = [0.4 + 2.5 * float(Gs[u][v]["weight"]) for u, v in Gs.edges()]
    nx.draw_networkx_edges(
        Gs, pos, ax=ax, alpha=0.25, width=widths, edge_color="#95a5a6", arrows=True, arrowsize=8
    )
    node_list = list(Gs.nodes())
    sizes = [80 + 18 * Gs.degree(n) for n in node_list]
    colors = []
    for n in node_list:
        if n in supers:
            colors.append("#c0392b")
        elif n in set(hub.head(8)["gene"].astype(str)):
            colors.append("#e67e22")
        else:
            colors.append("#2980b9")
    nx.draw_networkx_nodes(Gs, pos, ax=ax, nodelist=node_list, node_size=sizes, node_color=colors, alpha=0.9)
    labels = {n: n for n in node_list if n in top_label}
    nx.draw_networkx_labels(Gs, pos, labels=labels, ax=ax, font_size=7)
    ax.set_title(
        f"mBrain-gain Top{TOPK} differential network "
        f"(n_edges={len(edges)}, display_nodes={Gs.number_of_nodes()})"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    used = TABLES / "differential_edges_used.csv"
    edges_all = pd.read_csv(used)
    gain_all = edges_all[edges_all["edge_category"] == "mBrain_gain_regulation"].copy()
    gain_all = gain_all.sort_values("max_prob_diff", ascending=False)
    # archive full gain
    gain_all.to_csv(OUT / "mBrain_gain_regulation_edges_ALL.csv", index=False)

    gain = gain_all.head(TOPK).copy()
    gain.to_csv(OUT / "mBrain_gain_regulation_edges.csv", index=False)
    gain.to_csv(OUT / f"mBrain_gain_regulation_edges_top{TOPK}.csv", index=False)

    hub = _hub_gain(gain)
    hub.to_csv(OUT / "mBrain_gain_hub_genes.csv", index=False)

    supers = hub[(hub["out_frac_of_edges"] >= HUB_FRAC) | (hub["in_frac_of_edges"] >= HUB_FRAC)]
    supers.to_csv(OUT / "super_hubs_ge10pct.csv", index=False)

    markers = hub[
        (hub["in_frac_of_edges"] <= HUB_FRAC)
        & (hub["out_frac_of_edges"] <= HUB_FRAC)
        & (hub["degree"] >= 5)
    ].copy()
    markers.to_csv(OUT / "marker_candidates_filtered.csv", index=False)

    # novel vs KEGG
    kegg = pd.read_csv(KEGG)
    pairs = set(
        zip(
            kegg["source_symbol"].astype(str).str.upper(),
            kegg["target_symbol"].astype(str).str.upper(),
        )
    )
    g2 = gain.copy()
    g2["in_kegg_directed"] = [
        (str(a).upper(), str(b).upper()) in pairs for a, b in zip(g2["TF"], g2["Target"])
    ]
    novel = g2[~g2["in_kegg_directed"]].sort_values("max_prob_diff", ascending=False)
    novel.to_csv(OUT / "mBrain_gain_edges_not_in_KEGG_directed.csv", index=False)

    genes = sorted(set(gain["TF"].astype(str)) | set(gain["Target"].astype(str)))
    (OUT / "mBrain_gain_top500_genes.txt").write_text("\n".join(genes) + "\n", encoding="utf-8")

    # network
    _draw_network(
        gain,
        hub,
        OUT / "fig_mBrain_gain_top500_network.png",
        OUT / "fig_mBrain_gain_top500_network.pdf",
    )

    # enrichment
    enrich_ok = True
    try:
        gost = _gprofiler(genes)
        (OUT / "enrich_gprofiler_top500.json").write_text(
            json.dumps(gost, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        edf = _results_to_df(gost)
        edf.to_csv(OUT / "enrich_gprofiler_top500_terms.csv", index=False)
        # neural filter
        neu = edf[
            edf["name"]
            .astype(str)
            .str.contains(
                r"neuron|neural|synapse|axon|neuro|Alzheimer|Parkinson|brain|"
                r"Neuroactive|Neurotrophin|dopamin|cholin|glutamat|GABA",
                case=False,
                regex=True,
            )
        ].sort_values("p_value")
        neu.to_csv(OUT / "enrich_gprofiler_top500_neural.csv", index=False)
        _dotplot(
            edf,
            "KEGG",
            OUT / "fig_mBrain_gain_top500_enrich_KEGG.png",
            f"KEGG enrichment (mBrain-gain Top{TOPK} genes, n={len(genes)})",
        )
        _dotplot(
            edf,
            "GO:BP",
            OUT / "fig_mBrain_gain_top500_enrich_GO_BP.png",
            f"GO BP enrichment (mBrain-gain Top{TOPK} genes, n={len(genes)})",
        )
        # also save pdf via same png path siblings
        for png in (
            OUT / "fig_mBrain_gain_top500_enrich_KEGG.png",
            OUT / "fig_mBrain_gain_top500_enrich_GO_BP.png",
        ):
            # re-export pdf quickly by reading is hard; skip if already png
            pass
    except Exception as exc:  # noqa: BLE001
        enrich_ok = False
        (OUT / "enrich_gprofiler_top500_ERROR.txt").write_text(repr(exc), encoding="utf-8")
        edf = pd.DataFrame()
        neu = pd.DataFrame()

    # class counts
    class_counts = Counter(gain["pred_mBrain_name"].astype(str)) if "pred_mBrain_name" in gain.columns else {}

    summary = {
        "mainline_definition": (
            f"mBrain_gain_regulation edges ranked by max_prob_diff, keep Top{TOPK} "
            "(primary=tLung NoReg -> mBrain Activate/Repress)"
        ),
        "n_diff_edges_all": int(len(edges_all)),
        "n_gain_edges_all": int(len(gain_all)),
        "n_gain_edges_top500": int(len(gain)),
        "n_genes_top500": int(len(genes)),
        "max_prob_diff_range": [float(gain["max_prob_diff"].max()), float(gain["max_prob_diff"].min())],
        "mBrain_pred_class_counts": dict(class_counts),
        "n_novel_not_in_kegg": int(len(novel)),
        "super_hubs_ge10pct": supers[["gene", "out_degree", "in_degree", "out_frac_of_edges", "in_frac_of_edges"]].to_dict(
            orient="records"
        ),
        "top_hubs": hub.head(10).to_dict(orient="records"),
        "top_markers_filtered": markers.head(10).to_dict(orient="records"),
        "enrichment_ok": enrich_ok,
        "n_enrich_terms": int(len(edf)) if enrich_ok else 0,
        "n_neural_terms": int(len(neu)) if enrich_ok else 0,
        "top_kegg": edf[edf["source"] == "KEGG"].sort_values("p_value").head(10).to_dict(orient="records")
        if enrich_ok and not edf.empty
        else [],
        "top_neural": neu.head(10).to_dict(orient="records") if enrich_ok and not neu.empty else [],
        "figures": {
            "network_png": "fig_mBrain_gain_top500_network.png",
            "network_pdf": "fig_mBrain_gain_top500_network.pdf",
            "enrich_kegg": "fig_mBrain_gain_top500_enrich_KEGG.png",
            "enrich_go_bp": "fig_mBrain_gain_top500_enrich_GO_BP.png",
        },
    }
    (OUT / "followup_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # rewrite report (main sections)
    top_kegg_lines = []
    for r in summary.get("top_kegg", [])[:8]:
        top_kegg_lines.append(
            f"| {r.get('name','')} | {r.get('p_value'):.2e} | {r.get('intersection_size')} |"
        )
    top_neu_lines = []
    for r in summary.get("top_neural", [])[:8]:
        top_neu_lines.append(
            f"| {r.get('source','')} | {r.get('name','')} | {r.get('p_value'):.2e} |"
        )
    super_lines = []
    for r in summary.get("super_hubs_ge10pct", []):
        super_lines.append(
            f"| {r['gene']} | out={r['out_degree']} ({100*r['out_frac_of_edges']:.1f}%) | "
            f"in={r['in_degree']} ({100*r['in_frac_of_edges']:.1f}%) |"
        )
    marker_lines = []
    for r in summary.get("top_markers_filtered", [])[:5]:
        marker_lines.append(
            f"| {r['gene']} | out={r['out_degree']} | in={r['in_degree']} | deg={r['degree']} |"
        )

    report = f"""# GSE131907 ?????????tLung ?? vs mBrain?????

**?????????????** ?????????? **mBrain-gain Top{TOPK}**?? `max_prob_diff`??

---

## 0. ??

| ? | ?? |
|----|------|
| ?? | **tLung ????** vs **mBrain ???** |
| ?? | ?? **from scratch**???? / fusion-head finetune |
| ????? | `max_prob_diff > 0.5`??????? |
| **????** | `edge_category == mBrain_gain_regulation`?? `max_prob_diff` ? **Top{TOPK}** |
| ????? | ???? NoReg ? ????? Activate/Repress |

???`step07_GSE131907_tLung??_vs_mBrain_????_seed42.bat`  
?????`python -m grn.scripts.update_independent_case_top500_mainline`

---

## 1. ?????????held-out test????

| ? | Accuracy | Macro-AUC | MCC |
|----|----------|-----------|-----|
| **mBrain** | 0.920 | 0.970 | 0.866 |
| **tLung** | 0.929 | 0.973 | 0.881 |

---

## 2. ????????mBrain-gain Top{TOPK}?

| ?? | ? |
|------|-----|
| ????? (`prob_diff>0.5`) | {len(edges_all)} |
| ?? mBrain-gain | {len(gain_all)} |
| **?? Top{TOPK}** | **{len(gain)}** |
| Top{TOPK} ??? | **{len(genes)}** |
| max_prob_diff ?? | {float(gain['max_prob_diff'].max()):.4f} .. {float(gain['max_prob_diff'].min()):.4f} |
| mBrain ??? | {dict(class_counts)} |
| KEGG ????Top{TOPK} ?? | {len(novel)} |

??`followup/mBrain_gain_regulation_edges.csv`?= Top{TOPK}?  
?? gain ???`followup/mBrain_gain_regulation_edges_ALL.csv`

### ??????/?? ?10% Top{TOPK}?

| ?? | ?? | ?? |
|------|------|------|
{chr(10).join(super_lines) if super_lines else '| (none) | | |'}

### ??? marker ????/??? ?10%?degree?5?

| ?? | out | in | deg |
|------|-----|----|-----|
{chr(10).join(marker_lines) if marker_lines else '| (none) | | | |'}

---

## 3. ?????g:Profiler?Top{TOPK} ???

enrichment_ok={enrich_ok}

### Top KEGG

| Term | p | overlap |
|------|---|---------|
{chr(10).join(top_kegg_lines) if top_kegg_lines else '| (failed or empty) | | |'}

### ????????

| Source | Term | p |
|--------|------|---|
{chr(10).join(top_neu_lines) if top_neu_lines else '| (none / failed) | | |'}

??
- `followup/fig_mBrain_gain_top500_network.png`
- `followup/fig_mBrain_gain_top500_enrich_KEGG.png`
- `followup/fig_mBrain_gain_top500_enrich_GO_BP.png`

---

## 4. ?????

???????? **mBrain ????? Top{TOPK}**?????? **MAPK10/ARRB ???????**?????????????? Top{TOPK} ?????????? followup ?? CSV??
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:4000])
    print("OUT", OUT)
    print("REPORT", REPORT)


if __name__ == "__main__":
    main()
