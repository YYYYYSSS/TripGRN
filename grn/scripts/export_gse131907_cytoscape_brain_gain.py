# -*- coding: utf-8 -*-
"""Export GSE131907 mBrain-gain network as Cytoscape tables (GSE123902-compatible)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CASE = (
    ROOT
    / "experiments"
    / "gse131907_transfer_from_gse253718_epi_seed2"
    / "case_tlung_vs_mbrain_kegg_khop_k1"
)
FOLLOW = CASE / "followup"
OUT = CASE / "cytoscape"
TOP3 = ("EPHA2", "WNT7B", "PLCB3")


def main() -> None:
    edges_in = FOLLOW / "mBrain_gain_regulation_edges_top500.csv"
    gain = pd.read_csv(edges_in)
    gain = gain[gain["edge_category"] == "mBrain_gain_regulation"].copy()

    # --- edges (match GSE123902 brain_gain_network_edges.csv) ---
    edges = pd.DataFrame(
        {
            "shared name": [
                f"{tf} ({reg}) -> {tgt}"
                for tf, reg, tgt in zip(
                    gain["TF"].astype(str),
                    gain["pred_mBrain_name"].astype(str),
                    gain["Target"].astype(str),
                )
            ],
            "source": gain["TF"].astype(str),
            "target": gain["Target"].astype(str),
            "regulation_type": gain["pred_mBrain_name"].astype(str),
            "primary_class": gain["pred_tLung_primary_name"].astype(str),
            "brain_met_weight": gain["weight"].astype(float),
            "max_prob_diff": gain["max_prob_diff"].astype(float),
            "edge_category": gain["edge_category"].astype(str),
            "edge_origin": gain["edge_origin"].astype(str),
        }
    )
    edges = edges.sort_values(["source", "target", "regulation_type"]).reset_index(drop=True)

    # --- nodes with degree annotations for manual Cytoscape styling ---
    genes = sorted(set(edges["source"]) | set(edges["target"]))
    out_deg = edges.groupby("source").size().to_dict()
    out_act = (
        edges[edges["regulation_type"] == "Activate"].groupby("source").size().to_dict()
    )
    out_rep = (
        edges[edges["regulation_type"] == "Repress"].groupby("source").size().to_dict()
    )
    in_deg = edges.groupby("target").size().to_dict()
    regs = set(edges["source"])
    tgts = set(edges["target"])
    top3 = set(TOP3)

    rows = []
    for g in genes:
        is_reg = g in regs
        is_tgt = g in tgts
        if is_reg and is_tgt:
            role = "regulator_and_target"
            gene_label = "both"
        elif is_reg:
            role = "regulator"
            gene_label = "TF"
        else:
            role = "target"
            gene_label = "Target"
        # Top-3 hubs always labeled TF for Cytoscape styling
        if g in top3:
            gene_label = "TF"
        rows.append(
            {
                # Cytoscape default node key is "shared name"; avoid a separate
                # "name" column (reserved / conflicts -> "Copied 0 rows").
                "shared name": g,
                "out_degree": int(out_deg.get(g, 0)),
                "out_activate": int(out_act.get(g, 0)),
                "out_repress": int(out_rep.get(g, 0)),
                "in_degree": int(in_deg.get(g, 0)),
                "is_regulator": int(is_reg),
                "is_target": int(is_tgt),
                "node_role": role,
                "is_top3_hub": int(g in top3),
                "gene_label": gene_label,
            }
        )
    nodes = pd.DataFrame(rows).sort_values(
        ["out_degree", "shared name"], ascending=[False, True]
    ).reset_index(drop=True)

    OUT.mkdir(parents=True, exist_ok=True)
    # Mirror GSE123902 folder naming for convenience
    nested = OUT / "cytoscape input"
    nested.mkdir(parents=True, exist_ok=True)

    for dest in (OUT, nested):
        edges.to_csv(dest / "brain_gain_network_edges.csv", index=False)
        nodes.to_csv(dest / "brain_gain_network_nodes.csv", index=False)

    # Compact hub preview for manual layout
    hubs = nodes[nodes["out_degree"] > 0].sort_values(
        ["out_degree", "out_activate"], ascending=False
    )
    hubs.to_csv(OUT / "brain_gain_network_hubs_by_outdegree.csv", index=False)

    print(f"[ok] edges={len(edges)} nodes={len(nodes)}")
    print(f"[ok] wrote -> {OUT}")
    print(f"[ok] also  -> {nested}")
    print("[top out-degree hubs]")
    print(hubs.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
