# -*- coding: utf-8 -*-
"""Export 1-hop Cytoscape tables for top-3 out-degree hubs (EPHA2/WNT7B/PLCB3)."""
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
CYTO = CASE / "cytoscape"
TOP3 = ("EPHA2", "WNT7B", "PLCB3")


def _nodes_from_edges(edges: pd.DataFrame, hub: str) -> pd.DataFrame:
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

    rows = []
    for g in genes:
        is_reg = g in regs
        is_tgt = g in tgts
        if is_reg and is_tgt:
            role = "regulator_and_target"
        elif is_reg:
            role = "regulator"
        else:
            role = "target"
        # relative to hub: hub / direct_target / direct_regulator
        if g == hub:
            hub_role = "hub"
        elif (edges["source"] == hub).any() and g in set(edges.loc[edges["source"] == hub, "target"]):
            hub_role = "direct_target"
        elif (edges["target"] == hub).any() and g in set(edges.loc[edges["target"] == hub, "source"]):
            hub_role = "direct_regulator"
        else:
            hub_role = "other"
        rows.append(
            {
                "name": g,
                "out_degree": int(out_deg.get(g, 0)),
                "out_activate": int(out_act.get(g, 0)),
                "out_repress": int(out_rep.get(g, 0)),
                "in_degree": int(in_deg.get(g, 0)),
                "is_regulator": bool(is_reg),
                "is_target": bool(is_tgt),
                "node_role": role,
                "is_top3_hub": bool(g == hub),
                "hub_center": hub,
                "relation_to_hub": hub_role,
            }
        )
    df = pd.DataFrame(rows)
    order = {"hub": 0, "direct_target": 1, "direct_regulator": 2, "other": 3}
    df["_ord"] = df["relation_to_hub"].map(order)
    return df.sort_values(["_ord", "out_degree", "name"], ascending=[True, False, True]).drop(
        columns=["_ord"]
    ).reset_index(drop=True)


def export_hub(edges_all: pd.DataFrame, hub: str, out_dir: Path) -> None:
    # Direct connections only: edges touching the hub (out + in)
    sub = edges_all[(edges_all["source"] == hub) | (edges_all["target"] == hub)].copy()
    sub = sub.sort_values(["source", "target"]).reset_index(drop=True)
    nodes = _nodes_from_edges(sub, hub)

    out_dir.mkdir(parents=True, exist_ok=True)
    # outgoing-only (hub as regulator) � main "influence" view
    out_only = edges_all[edges_all["source"] == hub].copy().reset_index(drop=True)
    nodes_out = _nodes_from_edges(out_only, hub)

    sub.to_csv(out_dir / f"{hub}_direct_edges.csv", index=False)
    nodes.to_csv(out_dir / f"{hub}_direct_nodes.csv", index=False)
    out_only.to_csv(out_dir / f"{hub}_outgoing_edges.csv", index=False)
    nodes_out.to_csv(out_dir / f"{hub}_outgoing_nodes.csv", index=False)

    print(
        f"[{hub}] 1-hop edges={len(sub)} "
        f"(out={(sub['source']==hub).sum()}, in={(sub['target']==hub).sum()}); "
        f"nodes={len(nodes)} -> {out_dir}"
    )


def main() -> None:
    edges = pd.read_csv(CYTO / "brain_gain_network_edges.csv")
    root = CYTO / "top3_direct"
    root.mkdir(parents=True, exist_ok=True)
    for hub in TOP3:
        export_hub(edges, hub, root / hub)
    print(f"[done] {root}")


if __name__ == "__main__":
    main()
