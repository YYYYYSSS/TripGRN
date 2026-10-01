# -*- coding: utf-8 -*-
"""
Careful follow-up on independent tLung vs mBrain case:
  gain network, neural enrichment emphasis, marker filters, novel non-KEGG edges.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "experiments" / "gse131907_independent_tlung_mbrain_seed42" / "case_tlung_vs_mbrain"
TABLES = CASE / "tables"
OUT = CASE / "followup"
KEGG = ROOT / "data" / "network" / "grn" / "KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv"

NEURAL_RE = re.compile(
    r"neuron|neural|synapse|synaptic|axon|dendrit|glial|astrocyte|oligodend|"
    r"brain|neuro|glutamat|GABA|dopamin|serotonin|neurotrans|neurodegen|"
    r"Alzheimer|Parkinson|amyotrophic|Prion|nervous|CNS|cerebell|hippocamp|"
    r"myelin|neuromusc|Neurotrophin|Axon guidance|forebrain|Neuroactive",
    re.I,
)


def _hub_gain(gain: pd.DataFrame) -> pd.DataFrame:
    out = gain.groupby("TF").agg(out_degree=("Target", "count"), out_w=("max_prob_diff", "sum"))
    inn = gain.groupby("Target").agg(in_degree=("TF", "count"), in_w=("max_prob_diff", "sum"))
    genes = sorted(set(gain["TF"].astype(str)) | set(gain["Target"].astype(str)))
    rows = []
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
                "in_frac_of_edges": ind / max(len(gain), 1),
                "out_frac_of_edges": od / max(len(gain), 1),
            }
        )
    return pd.DataFrame(rows).sort_values(["degree", "weighted_degree"], ascending=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    used_path = TABLES / "differential_edges_used.csv"
    if not used_path.is_file():
        raise SystemExit(f"missing {used_path}; run case study first")

    edges = pd.read_csv(used_path)
    # gain: mBrain acquired regulation (label may still say mBrain_gain_*)
    if "edge_category" not in edges.columns:
        raise SystemExit("differential_edges_used.csv missing edge_category")
    gain_all = edges[edges["edge_category"] == "mBrain_gain_regulation"].copy()
    gain_all = gain_all.sort_values("max_prob_diff", ascending=False)
    gain_all.to_csv(OUT / "mBrain_gain_regulation_edges_ALL.csv", index=False)
    # Mainline: Top500 by max_prob_diff (see update_independent_case_top500_mainline.py)
    topk = 500
    gain = gain_all.head(topk).copy()
    gain.to_csv(OUT / "mBrain_gain_regulation_edges.csv", index=False)
    gain.to_csv(OUT / f"mBrain_gain_regulation_edges_top{topk}.csv", index=False)

    hub = _hub_gain(gain)
    hub.to_csv(OUT / "mBrain_gain_hub_genes.csv", index=False)

    # Marker filter: reject extreme sinks (>10% edges) or sources (>10%); prefer balanced hubs
    markers = hub[
        (hub["in_frac_of_edges"] <= 0.10)
        & (hub["out_frac_of_edges"] <= 0.10)
        & (hub["degree"] >= 5)
    ].copy()
    markers.to_csv(OUT / "marker_candidates_filtered.csv", index=False)

    # Neural enrich from tables if present
    neural_rows = []
    for name in ("enrich_R_GO_BP.csv", "enrich_R_GO_CC.csv", "enrich_R_GO_MF.csv", "enrich_R_KEGG.csv"):
        p = TABLES / name
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        if "Description" not in df.columns:
            continue
        m = df["Description"].astype(str).str.contains(NEURAL_RE)
        sub = df[m].copy()
        if "p.adjust" in sub.columns:
            sub = sub.sort_values("p.adjust")
        sub.insert(0, "source_table", name)
        neural_rows.append(sub)
        sub.head(30).to_csv(OUT / f"neural_{name}", index=False)
    if neural_rows:
        all_n = pd.concat(neural_rows, ignore_index=True)
        all_n.to_csv(OUT / "neural_enrichment_all.csv", index=False)

    # Novel edges vs KEGG directed
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

    # Diverse high-conf novel (cap per target to avoid single-gene sinks)
    picks = []
    seen_tgt = {}
    for _, r in novel.iterrows():
        t = str(r["Target"])
        seen_tgt[t] = seen_tgt.get(t, 0) + 1
        if seen_tgt[t] > 3:
            continue
        picks.append(r)
        if len(picks) >= 40:
            break
    pd.DataFrame(picks).to_csv(OUT / "novel_edges_diverse_top40.csv", index=False)

    summary = {
        "case_dir": str(CASE),
        "n_diff_edges": int(len(edges)),
        "n_gain_edges": int(len(gain)),
        "n_gain_genes": int(hub.shape[0]),
        "n_novel_not_in_kegg": int(len(novel)),
        "edge_category_counts": edges["edge_category"].value_counts().to_dict(),
        "top_hubs_raw": hub.head(10).to_dict(orient="records"),
        "top_markers_filtered": markers.head(10).to_dict(orient="records"),
        "filter_note": (
            "Markers require degree>=5 and in/out edge fraction <=10% of gain network "
            "to reject extreme sinks/sources (e.g. UCN-like artifacts)."
        ),
        "design": {
            "domains": "tLung_primary vs mBrain",
            "training": "from_scratch independent (no transfer/finetune)",
            "sample_split": "5,2,3",
            "case_expression_part": "test",
        },
    }
    (OUT / "followup_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("OUT", OUT)


if __name__ == "__main__":
    main()
