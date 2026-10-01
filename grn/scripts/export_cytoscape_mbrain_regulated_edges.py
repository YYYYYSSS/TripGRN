#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export Cytoscape inputs: differential edges with pred_mBrain_name != NoReg."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SRC = Path(
    r"E:\project\project_GRN\experiments\gse131907_application_step4_case_nonmbrain_vs_mbrain"
    r"\tables\differential_edges_used.csv"
)
OUT_DIR = Path(
    r"E:\project\project_GRN\experiments\gse131907_application_step4_case_nonmbrain_vs_mbrain"
    r"\tables\cytoscape input"
)

REG_MAP = {"Activate": "activation", "Repress": "repression"}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)
    sub = df.loc[df["pred_mBrain_name"].astype(str).str.strip() != "NoReg"].copy()
    sub["pred_mBrain_name"] = sub["pred_mBrain_name"].astype(str).str.strip()
    sub["pred_non_mBrain_name"] = sub["pred_non_mBrain_name"].astype(str).str.strip()
    sub["regulation"] = sub["pred_mBrain_name"].map(REG_MAP).fillna(sub["pred_mBrain_name"])

    edges = pd.DataFrame(
        {
            "source": sub["TF"].astype(str),
            "target": sub["Target"].astype(str),
            "interaction": sub["regulation"],
            "regulation": sub["regulation"],
            "pred_mBrain_name": sub["pred_mBrain_name"],
            "pred_non_mBrain_name": sub["pred_non_mBrain_name"],
            "edge_category": sub["edge_category"],
            "weight": sub["weight"].astype(float),
            "max_prob_diff": sub["max_prob_diff"].astype(float),
            "edge_origin": sub["edge_origin"],
            "TF": sub["TF"].astype(str),
            "Target": sub["Target"].astype(str),
        }
    )
    edges = edges.drop_duplicates(subset=["source", "target", "interaction"], keep="first")

    genes = sorted(set(edges["source"]).union(set(edges["target"])))
    out_deg = edges.groupby("source").size()
    in_deg = edges.groupby("target").size()
    w_out = edges.groupby("source")["weight"].sum()
    w_in = edges.groupby("target")["weight"].sum()
    act_out = edges.loc[edges["regulation"] == "activation"].groupby("source").size()
    rep_out = edges.loc[edges["regulation"] == "repression"].groupby("source").size()

    nodes = pd.DataFrame({"shared_name": genes})
    nodes["node_type"] = "gene"
    nodes["out_degree"] = nodes["shared_name"].map(out_deg).fillna(0).astype(int)
    nodes["in_degree"] = nodes["shared_name"].map(in_deg).fillna(0).astype(int)
    nodes["degree"] = nodes["out_degree"] + nodes["in_degree"]
    nodes["weighted_out_degree"] = nodes["shared_name"].map(w_out).fillna(0.0)
    nodes["weighted_in_degree"] = nodes["shared_name"].map(w_in).fillna(0.0)
    nodes["n_activation_as_TF"] = nodes["shared_name"].map(act_out).fillna(0).astype(int)
    nodes["n_repression_as_TF"] = nodes["shared_name"].map(rep_out).fillna(0).astype(int)
    nodes = nodes.sort_values(["degree", "shared_name"], ascending=[False, True])

    edges_path = OUT_DIR / "cytoscape_edges_mBrain_regulated.tsv"
    nodes_path = OUT_DIR / "cytoscape_nodes_mBrain_regulated.tsv"
    edges.to_csv(edges_path, sep="\t", index=False, encoding="utf-8")
    nodes.to_csv(nodes_path, sep="\t", index=False, encoding="utf-8")

    reg_counts = edges["regulation"].value_counts().to_string()
    cat_counts = edges["edge_category"].value_counts().to_string()
    readme = OUT_DIR / "README_cytoscape_import.txt"
    readme.write_text(
        "\n".join(
            [
                "Cytoscape import — mBrain regulated edges (pred_mBrain_name != NoReg)",
                "=" * 70,
                f"Source: {SRC}",
                "Filter: pred_mBrain_name is Activate or Repress (exclude NoReg)",
                "",
                "Files:",
                f"  - cytoscape_edges_mBrain_regulated.tsv  ({len(edges)} directed edges)",
                f"  - cytoscape_nodes_mBrain_regulated.tsv  ({len(nodes)} genes)",
                "",
                "Edge counts by regulation (mBrain prediction):",
                reg_counts,
                "",
                "Edge counts by edge_category:",
                cat_counts,
                "",
                "Import steps (Cytoscape 3.x):",
                "1. File -> Import -> Network from File...",
                "   - Select: cytoscape_edges_mBrain_regulated.tsv",
                "   - Source Node: source",
                "   - Target Node: target",
                "   - Interaction Type: interaction   (activation / repression)",
                "   - Directed network: source -> target (TF -> Target)",
                "   - Edge attributes: regulation, pred_mBrain_name, pred_non_mBrain_name,",
                "     edge_category, weight, max_prob_diff",
                "",
                "2. Style by regulation class:",
                "   - Style -> Edge -> Stroke Color / Line Type / Target Arrow Shape",
                "   - Discrete mapping on column 'interaction' (or 'regulation'):",
                "       activation  -> e.g. green / solid / arrow",
                "       repression  -> e.g. red / dashed / T-bar",
                "",
                "3. Optional node table:",
                "   File -> Import -> Table from File... -> Node Table Columns",
                "   - Key: shared_name",
                "   - Attributes: degree, out_degree, in_degree, n_activation_as_TF, n_repression_as_TF",
                "",
                "Notes:",
                "- Directed: TF (source) -> Target",
                "- interaction/regulation = mBrain predicted class (activation or repression)",
                "- edge_category = differential category vs non-mBrain",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[ok] edges={len(edges)} nodes={len(nodes)}")
    print(f"[ok] regulation counts:\n{reg_counts}")
    print(f"[ok] -> {edges_path}")
    print(f"[ok] -> {nodes_path}")
    print(f"[ok] -> {readme}")


if __name__ == "__main__":
    main()
