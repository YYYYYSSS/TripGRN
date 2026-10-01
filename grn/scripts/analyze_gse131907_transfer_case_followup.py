# -*- coding: utf-8
"""Follow-up for GSE131907 transfer case: stats, Top500 export, Enrichr, figures, Cytoscape."""
from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from scipy.stats import binomtest, chi2_contingency, fisher_exact

LIBS = ["KEGG_2021_Human", "GO_Biological_Process_2023"]
COL_A = "pred_tLung_primary_name"
COL_B = "pred_mBrain_name"
KEGG_PRIORITY = [
    "Pathways in cancer",
    "PI3K-Akt signaling pathway",
    "MAPK signaling pathway",
    "Chemokine signaling pathway",
    "Ras signaling pathway",
    "Neurotrophin signaling pathway",
    "Axon guidance",
    "Cholinergic synapse",
    "Dopaminergic synapse",
    "Alzheimer disease",
    "Parkinson disease",
    "Neuroactive ligand-receptor interaction",
    "Calcium signaling pathway",
]


def enrichr(genes: set[str], desc: str) -> dict[str, list[dict]]:
    gl = sorted(g for g in genes if g and g != "nan")
    if len(gl) < 5:
        return {}
    r = requests.post(
        "https://maayanlab.cloud/Enrichr/addList",
        files={"list": (None, "\n".join(gl)), "description": (None, desc)},
        timeout=120,
    )
    r.raise_for_status()
    uid = r.json()["userListId"]
    out: dict[str, list[dict]] = {}
    for lib in LIBS:
        url = "https://maayanlab.cloud/Enrichr/enrich?" + urllib.parse.urlencode(
            {"userListId": uid, "backgroundType": lib}
        )
        r2 = requests.get(url, timeout=120)
        r2.raise_for_status()
        rows = []
        for item in r2.json().get(lib, []):
            genes_hit = item[5] if isinstance(item[5], list) else []
            rows.append(
                {
                    "name": item[1],
                    "p_value": float(item[2]),
                    "adjusted_p": float(item[6]),
                    "intersection_size": len(genes_hit),
                    "genes": ",".join(genes_hit) if genes_hit else "",
                }
            )
        rows.sort(key=lambda x: x["adjusted_p"])
        out[lib] = rows
        time.sleep(0.25)
    return out


def _font(size: int, bold: bool = False):
    for p in [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _short_go(name: str) -> str:
    name = re.sub(r"\s*\(GO:\d+\)\s*$", "", str(name)).strip()
    return name[:59] + "..." if len(name) > 62 else name


def draw_barplot_png(df: pd.DataFrame, out_png: Path, title: str, subtitle: str, name_fn=None) -> None:
    name_fn = name_fn or (lambda x: str(x))
    rows = [
        {
            "name": name_fn(r["name"]),
            "nlp": -math.log10(max(float(r["adjusted_p"]), 1e-300)),
            "n": int(r["intersection_size"]) if pd.notna(r.get("intersection_size")) else 0,
        }
        for _, r in df.iterrows()
    ]
    left, plot_w, top, row_h, bottom = 420, 520, 90, 36, 70
    W, H = left + plot_w + 90, top + row_h * max(len(rows), 1) + bottom
    img = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    dr.text((28, 22), title, fill=(25, 25, 25), font=_font(22, True))
    dr.text((28, 52), subtitle, fill=(90, 90, 90), font=_font(12))
    xmax = max((r["nlp"] for r in rows), default=1.0) * 1.08
    x0 = left
    for i, r in enumerate(rows):
        y = top + i * row_h + row_h / 2
        dr.text((20, y - 8), r["name"], fill=(30, 30, 30), font=_font(13))
        bw = plot_w * (r["nlp"] / xmax)
        t = min(1.0, r["nlp"] / xmax)
        rgb = (
            int(210 + (20 - 210) * t),
            int(230 + (90 - 230) * t),
            int(240 + (120 - 240) * t),
        )
        dr.rounded_rectangle([x0, y - 10, x0 + bw, y + 10], radius=5, fill=rgb)
        dr.text((x0 + bw + 8, y - 7), f"n={r['n']}", fill=(70, 70, 70), font=_font(11))
    img.save(out_png, dpi=(300, 300))


def draw_barplot_pdf(df: pd.DataFrame, out_pdf: Path, title: str, subtitle: str, name_fn=None) -> None:
    name_fn = name_fn or (lambda x: str(x))
    rows = [
        {
            "name": name_fn(r["name"]),
            "nlp": -math.log10(max(float(r["adjusted_p"]), 1e-300)),
            "n": int(r["intersection_size"]) if pd.notna(r.get("intersection_size")) else 0,
        }
        for _, r in df.iterrows()
    ]
    left, plot_w, top, row_h, bottom = 300, 280, 55, 22, 45
    W, H = left + plot_w + 70, top + row_h * max(len(rows), 1) + bottom
    c = canvas.Canvas(str(out_pdf), pagesize=(W, H))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(18, H - 22, title)
    c.setFont("Helvetica", 8)
    c.drawString(18, H - 36, subtitle)
    xmax = max((r["nlp"] for r in rows), default=1.0) * 1.08
    y_base = H - top
    for i, r in enumerate(rows):
        y = y_base - i * row_h - row_h / 2
        c.drawRightString(left - 8, y - 2, r["name"][:58])
        bw = plot_w * (r["nlp"] / xmax)
        c.roundRect(left, y - 6, bw, 12, 3, fill=1, stroke=0)
        c.drawString(left + bw + 5, y - 2, f"n={r['n']}")
    c.showPage()
    c.save()


def export_cytoscape(edges: pd.DataFrame, out_dir: Path, top_k: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = edges.copy()
    if top_k > 0:
        df = df.sort_values("max_prob_diff", ascending=False).head(top_k)
    df["source"] = df["TF"].astype(str)
    df["target"] = df["Target"].astype(str)
    df["interaction"] = df[COL_A].astype(str) + "->" + df[COL_B].astype(str)
    cols = [
        c
        for c in [
            "source",
            "target",
            "interaction",
            "weight",
            "edge_category",
            COL_A,
            COL_B,
            "max_prob_diff",
            "TF",
            "Target",
        ]
        if c in df.columns
    ]
    df[cols].to_csv(out_dir / "cytoscape_edges.tsv", sep="\t", index=False, encoding="utf-8")
    genes = sorted(set(df["source"]) | set(df["target"]))
    pd.DataFrame({"shared_name": genes, "node_type": "gene"}).to_csv(
        out_dir / "cytoscape_nodes.tsv", sep="\t", index=False, encoding="utf-8"
    )
    # SIF for Top500 gain
    gain = df[df["edge_category"] == "mBrain_gain_regulation"].copy()
    gain["reg"] = gain[COL_B].map({"Activate": "activation", "Repress": "repression"}).fillna("unknown")
    sif_lines = [f"{r.TF}\t{r.reg}\t{r.Target}" for r in gain.itertuples(index=False)]
    (out_dir / "mBrain_gain_top500.sif").write_text("\n".join(sif_lines) + "\n", encoding="utf-8")


def draw_network(df: pd.DataFrame, out_pdf: Path, out_png: Path | None) -> None:
    from collections import Counter

    tfs = sorted(set(df["TF"].astype(str)))
    targets = sorted(set(df["Target"].astype(str)))
    out_deg = Counter(df["TF"].astype(str))
    h = max(1100, 14 * max(len(tfs), len(targets)) + 120)
    w = 1400
    left_x, right_x, margin_y = 160, w - 180, 70
    pos = {}
    for i, g in enumerate(tfs):
        pos[("TF", g)] = (left_x, margin_y + (h - 2 * margin_y) * (1 - i / max(len(tfs) - 1, 1)))
    for i, g in enumerate(targets):
        pos[("TG", g)] = (right_x, margin_y + (h - 2 * margin_y) * (1 - i / max(len(targets) - 1, 1)))

    c = canvas.Canvas(str(out_pdf), pagesize=(w, h))
    c.setFont("Helvetica-Bold", 13)
    n_act = int((df[COL_B] == "Activate").sum())
    n_rep = int((df[COL_B] == "Repress").sum())
    c.drawString(28, h - 28, f"mBrain-gain network ({len(df)} edges, {len(tfs)} TFs, {len(targets)} targets)")
    c.setFont("Helvetica", 9)
    c.drawString(28, h - 44, f"Activate n={n_act}; Repress n={n_rep}. Red=Activate, blue blunt=Repress.")

    for _, r in df.iterrows():
        tf, tg, cls = str(r["TF"]), str(r["Target"]), str(r[COL_B])
        x1, y1 = pos[("TF", tf)]
        x2, y2 = pos[("TG", tg)]
        ang = math.atan2(y2 - y1, x2 - x1)
        pad = 8.0
        x1s, y1s = x1 + pad * math.cos(ang), y1 + pad * math.sin(ang)
        x2s, y2s = x2 - pad * math.cos(ang), y2 - pad * math.sin(ang)
        if cls == "Activate":
            c.setStrokeColor(colors.Color(0.75, 0.15, 0.15))
            c.setLineWidth(0.9)
            c.line(x1s, y1s, x2s, y2s)
        else:
            c.setStrokeColor(colors.Color(0.15, 0.35, 0.70))
            c.setLineWidth(0.35)
            c.line(x1s, y1s, x2s, y2s)

    label_tfs = set(list(tfs)[:25]) | {g for g, _ in out_deg.most_common(12)}
    for g in tfs:
        x, y = pos[("TF", g)]
        c.setFillColor(colors.Color(0.90, 0.45, 0.10))
        c.circle(x, y, 4, fill=1, stroke=0)
        if g in label_tfs:
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 6.5)
            c.drawRightString(x - 6, y - 2, g)
    for g in targets:
        x, y = pos[("TG", g)]
        c.setFillColor(colors.Color(0.20, 0.55, 0.75))
        c.circle(x, y, 3.5, fill=1, stroke=0)
    c.showPage()
    c.save()

    if out_png is not None:
        try:
            import pypdfium2 as pdfium

            doc = pdfium.PdfDocument(str(out_pdf))
            doc[0].render(scale=1.5).to_pil().save(out_png)
        except Exception as exc:
            print(f"[followup] png from pdf skip: {exc}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp-root",
        type=Path,
        default=Path("experiments/gse131907_transfer_from_gse253718_epi_seed2"),
    )
    ap.add_argument("--top-k", type=int, default=500)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    exp_root = (root / args.exp_root).resolve() if not args.exp_root.is_absolute() else args.exp_root
    case = exp_root / "case_tlung_vs_mbrain_kegg_khop_k1"
    tab = case / "tables"
    out = case / "followup"
    fig = case / "figures"
    cyto = case / "cytoscape"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    cyto.mkdir(parents=True, exist_ok=True)

    edges = pd.read_csv(tab / "differential_edges_used.csv")
    order = ["NoReg", "Activate", "Repress"]
    ct = (
        pd.crosstab(edges[COL_A], edges[COL_B])
        .reindex(index=order, columns=order, fill_value=0)
        .astype(int)
    )
    chi2, chi2_p, dof, _ = chi2_contingency(ct.values)
    ga, gr = int(ct.loc["NoReg", "Activate"]), int(ct.loc["NoReg", "Repress"])
    la, lr = int(ct.loc["Activate", "NoReg"]), int(ct.loc["Repress", "NoReg"])
    bp = float(binomtest(ga, max(ga + gr, 1), 0.5).pvalue)
    od_gl, p_gl = fisher_exact([[ga, gr], [la, lr]])
    N = int(ct.values.sum())
    prim_nr, brain_act = int(ct.loc["NoReg"].sum()), int(ct["Activate"].sum())
    _, p_na = fisher_exact([[ga, prim_nr - ga], [brain_act - ga, N - prim_nr - (brain_act - ga)]])

    gain_all = edges[edges["edge_category"] == "mBrain_gain_regulation"].sort_values(
        "max_prob_diff", ascending=False
    )
    topk = int(args.top_k)
    gain = gain_all.head(topk).copy()
    genes_top = set(gain["TF"].astype(str)) | set(gain["Target"].astype(str))

    kegg = pd.read_csv(root / "data/network/grn/KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv")
    pairs = set(
        zip(kegg["source_symbol"].astype(str).str.upper(), kegg["target_symbol"].astype(str).str.upper())
    )
    novel_n = sum(
        1 for a, b in zip(gain["TF"], gain["Target"]) if (str(a).upper(), str(b).upper()) not in pairs
    )

    deg_path = exp_root / "deg.csv"
    de = pd.read_csv(deg_path) if deg_path.is_file() else pd.DataFrame()
    fov = {}
    if not de.empty:
        gain_genes = set(gain_all["TF"].astype(str)) | set(gain_all["Target"].astype(str))
        deg = set(de.loc[de["padj_BH"] < 0.05, "gene"].astype(str))
        uni = set(de["gene"].astype(str))
        q, d = gain_genes & uni, deg & uni
        a, b, c = len(q & d), len(q - d), len(d - q)
        e = len(uni - q - d)
        odds, p = fisher_exact([[a, b], [c, e]])
        fov = {"n_overlap": a, "OR": float(odds), "p_value": float(p)}

    out_deg = gain.groupby("TF").size().sort_values(ascending=False)
    top3 = list(out_deg.head(3).index.astype(str))

    stats = {
        "matrix_counts": ct.to_dict(),
        "chi2_p": float(chi2_p),
        "gain_Activate": ga,
        "gain_Repress": gr,
        "binom_p": bp,
        "fisher_gain_vs_loss_p": float(p_gl),
        "fisher_gain_vs_loss_OR": float(od_gl),
        "fisher_NoRegAct_p": float(p_na),
        "n_diff_edges": int(len(edges)),
        "n_gain_all": int(len(gain_all)),
        "n_gain_topk": int(len(gain)),
        "n_genes_topk": int(len(genes_top)),
        "n_novel_kegg_directed_topk": int(novel_n),
        "top3_hubs": top3,
        "deg_fisher_gain_all": fov,
    }
    (out / "transition_matrix_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    ct.to_csv(out / "transition_matrix_counts.csv")
    gain.to_csv(out / f"mBrain_gain_regulation_edges_top{topk}.csv", index=False)
    (out / f"mBrain_gain_top{topk}_genes.txt").write_text("\n".join(sorted(genes_top)), encoding="utf-8")

    print("[followup] Enrichr Top500 genes ...", flush=True)
    enr = enrichr(genes_top, f"gse131907_transfer_top{topk}")
    for lib, rows in enr.items():
        pd.DataFrame(rows).to_csv(out / f"enrichr_top{topk}_{lib}.csv", index=False)

    kegg_df = pd.DataFrame(enr.get("KEGG_2021_Human", []))
    go_df = pd.DataFrame(enr.get("GO_Biological_Process_2023", []))
    if not kegg_df.empty:
        sig = kegg_df[kegg_df["adjusted_p"] < 0.05]
        picked = []
        seen = set()
        for term in KEGG_PRIORITY:
            hit = sig[sig["name"] == term]
            if len(hit):
                picked.append(hit.iloc[0])
                seen.add(term)
            if len(picked) >= 12:
                break
        for _, row in sig.iterrows():
            if row["name"] in seen:
                continue
            picked.append(row)
            if len(picked) >= 12:
                break
        kplot = pd.DataFrame(picked).sort_values("adjusted_p")
        sub = f"mBrain-gain Top{len(gain)} genes (n={len(genes_top)}), Enrichr"
        draw_barplot_png(kplot, fig / "fig_mBrain_gain_top500_enrich_KEGG.png", "KEGG pathway enrichment", sub)
        draw_barplot_pdf(kplot, fig / "fig_mBrain_gain_top500_enrich_KEGG.pdf", "KEGG pathway enrichment", sub)
    if not go_df.empty:
        gplot = go_df[go_df["adjusted_p"] < 0.05].sort_values("adjusted_p").head(12).sort_values("adjusted_p")
        sub = f"mBrain-gain Top{len(gain)} genes (n={len(genes_top)}), Enrichr"
        draw_barplot_png(
            gplot,
            fig / "fig_mBrain_gain_top500_enrich_GO_BP.png",
            "GO Biological Process enrichment",
            sub,
            name_fn=_short_go,
        )
        draw_barplot_pdf(
            gplot,
            fig / "fig_mBrain_gain_top500_enrich_GO_BP.pdf",
            "GO Biological Process enrichment",
            sub,
            name_fn=_short_go,
        )

    draw_network(
        gain,
        fig / "fig_mBrain_gain_top500_network.pdf",
        fig / "fig_mBrain_gain_top500_network.png",
    )
    export_cytoscape(gain, cyto, top_k=0)

    summary = {
        "exp_root": str(exp_root),
        "figures": str(fig),
        "cytoscape": str(cyto),
        "stats": stats,
        "finetune_test": {
            d: json.loads((exp_root / d / "unseen_samples_test_metrics.json").read_text(encoding="utf-8"))
            for d in ("tlung", "mbrain")
            if (exp_root / d / "unseen_samples_test_metrics.json").is_file()
        },
    }
    (out / "followup_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    print(f"[followup] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
