"""
Export differential GRN edges to Cytoscape-friendly TSV tables (nodes + edges with attributes).

Reads:
  tables/differential_edges_used.csv  (or --edges-csv)
  optional tables/hub_genes.csv

Writes under <case-dir>/tables/cytoscape/:
  cytoscape_nodes.tsv
  cytoscape_edges.tsv
  cytoscape_import_README.txt

Import in Cytoscape: File -> Import -> Network from File (edges TSV, source/target columns)
then Import Table -> Nodes (key = shared_name).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _pred_name(code) -> str:
    m = {0: "NoReg", 1: "Repress", 2: "Activate", "0": "NoReg", "1": "Repress", "2": "Activate"}
    return m.get(code, str(code))


def main() -> None:
    p = argparse.ArgumentParser(description="Export case-study differential edges for Cytoscape.")
    p.add_argument(
        "--case-dir",
        type=Path,
        default=Path("experiments/gse131907_application_step4_case_nonmbrain_vs_mbrain"),
    )
    p.add_argument("--edges-csv", type=Path, default=None, help="default: <case-dir>/tables/differential_edges_used.csv")
    p.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="If >0, export only top-K edges by weight (0 = all rows in edges csv)",
    )
    args = p.parse_args()

    case_dir = args.case_dir.resolve()
    tab = case_dir / "tables"
    edges_path = Path(args.edges_csv).resolve() if args.edges_csv else tab / "differential_edges_used.csv"
    if not edges_path.is_file():
        raise SystemExit(f"Missing edges: {edges_path}")

    edges = pd.read_csv(edges_path)
    if args.top_k and args.top_k > 0 and "weight" in edges.columns:
        edges = edges.sort_values("weight", ascending=False).head(int(args.top_k)).copy()

    req = {"TF", "Target"}
    if not req.issubset(edges.columns):
        raise SystemExit(f"edges csv must contain {req}, got {list(edges.columns)[:12]}...")

    edges = edges.copy()
    edges["source"] = edges["TF"].astype(str)
    edges["target"] = edges["Target"].astype(str)
    edges["interaction"] = edges.apply(
        lambda r: f"{r.get('pred_non_mBrain_name', _pred_name(r.get('pred_non_mBrain', '')))}"
        f"->{r.get('pred_mBrain_name', _pred_name(r.get('pred_mBrain', '')))}",
        axis=1,
    )
    if "weight" not in edges.columns:
        edges["weight"] = 1.0

    edge_cols = [
        "source",
        "target",
        "interaction",
        "weight",
        "edge_category",
        "edge_origin",
        "pred_non_mBrain",
        "pred_mBrain",
        "pred_non_mBrain_name",
        "pred_mBrain_name",
        "max_prob_diff",
        "TF",
        "Target",
    ]
    edge_cols = [c for c in edge_cols if c in edges.columns]
    edges_out = edges[edge_cols].copy()

    genes = sorted(set(edges["source"]) | set(edges["target"]))
    hubs_path = tab / "hub_genes.csv"
    hub_df = pd.read_csv(hubs_path) if hubs_path.is_file() else pd.DataFrame()
    nodes = pd.DataFrame({"shared_name": genes, "node_type": "gene"})
    if not hub_df.empty and "gene" in hub_df.columns:
        hub_set = set(hub_df["gene"].astype(str))
        nodes["is_hub"] = nodes["shared_name"].isin(hub_set)
        nodes = nodes.merge(
            hub_df.rename(columns={"gene": "shared_name"}),
            on="shared_name",
            how="left",
        )
    else:
        nodes["is_hub"] = False

    out_dir = tab / "cytoscape"
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = out_dir / "cytoscape_nodes.tsv"
    edges_path_out = out_dir / "cytoscape_edges.tsv"
    nodes.to_csv(nodes_path, sep="\t", index=False, encoding="utf-8")
    edges_out.to_csv(edges_path_out, sep="\t", index=False, encoding="utf-8")

    readme = f"""Cytoscape import (GSE131907 case differential GRN)
============================================
Edges: {edges_path_out.name} ({len(edges_out)} edges)
Nodes: {nodes_path.name} ({len(nodes)} genes)

Recommended import:
1. Cytoscape -> File -> Import -> Network from File...
   - Source: column 'source'
   - Target: column 'target'
   - Interaction type: column 'interaction' (optional)
   - Edge attributes: weight, edge_category, pred_non_mBrain_name, pred_mBrain_name, max_prob_diff
2. File -> Import -> Table from File... -> Node Table
   - Key column: shared_name (map to node name)

edge_category legend:
  mBrain_gain_regulation, mBrain_loss_regulation,
  repression_to_activation, activation_to_repression,
  other_class_switch, same_class_prob_shift

Source edges file: {edges_path}
"""
    (out_dir / "cytoscape_import_README.txt").write_text(readme, encoding="utf-8")
    print(f"[cytoscape] nodes={len(nodes)} edges={len(edges_out)}")
    print(f"[cytoscape] wrote {out_dir}")


if __name__ == "__main__":
    main()
