# -*- coding: utf-8 -*-
"""Manuscript figures for GSE131907 transfer case (Fig. 6A-F), PDF output."""
from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, RegularPolygon
from reportlab.lib import colors
from reportlab.pdfgen import canvas

try:
    from matplotlib_venn import venn2

    HAS_VENN = True
except ImportError:
    HAS_VENN = False

# Match GSE123902 enrichment PDF style (article/_ms_audit/_export_enrichment_pdfs.py)
SKIP_ENRICH_TERMS = (
    "Shigellosis",
    "Yersinia",
    "Legionellosis",
    "Viral carcinogenesis",
    "Bacterial invasion",
    "Colorectal cancer",
    "Pancreatic cancer",
    "Non-alcoholic fatty liver",
    "Viral protein",
)

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/gse131907_transfer_from_gse253718_epi_seed2"
CASE = EXP / "case_tlung_vs_mbrain_kegg_khop_k1"
FOLLOW = CASE / "followup"
FIG = CASE / "figures"
TOP3 = {"EPHA2", "WNT7B", "PLCB3"}

# Align with manuscript Results (Fig. 6C–D): neural/synaptic + metastasis signaling.
# Exact names must match Enrichr; plot order is by adjusted_p (asc).
KEGG_ACT_TERMS = [
    # Named in manuscript
    "Neurotrophin signaling pathway",
    "Axon guidance",
    "Cholinergic synapse",
    "Dopaminergic synapse",
    "Calcium signaling pathway",
    "Long-term depression",
    "PI3K-Akt signaling pathway",
    "Rap1 signaling pathway",
    "MAPK signaling pathway",
    "Chemokine signaling pathway",
    # Closely related brain / metastasis context to fill Top15
    "Glutamatergic synapse",
    "Wnt signaling pathway",
    "Regulation of actin cytoskeleton",
    "Focal adhesion",
    "Neuroactive ligand-receptor interaction",
]

GO_ACT_TERMS = [
    # Named in manuscript
    "Generation Of Neurons (GO:0048699)",
    "Neuron Differentiation (GO:0030182)",
    "Axon Guidance (GO:0007411)",
    "Neuron Projection Guidance (GO:0097485)",
    "Axonogenesis (GO:0007409)",
    "Negative Regulation Of Neuron Death (GO:1901215)",
    "Ephrin Receptor Signaling Pathway (GO:0048013)",
    "Wnt Signaling Pathway (GO:0016055)",
    # Microenvironmental / metastasis programs implied by the same paragraph
    "Regulation Of Cell Migration (GO:0030334)",
    "Positive Regulation Of Cell Migration (GO:0030335)",
    "Positive Regulation Of Epithelial Cell Migration (GO:0010634)",
    "Positive Regulation Of Cell Motility (GO:2000147)",
    "Regulation Of Cell Adhesion (GO:0030155)",
    "Neurotrophin Signaling Pathway (GO:0038179)",
    "Nervous System Development (GO:0007399)",
]


def _short_go(name: str) -> str:
    return re.sub(r"\s*\(GO:\d+\)\s*$", "", str(name)).strip()


def enrichr(genes: list[str], desc: str) -> dict[str, list[dict]]:
    if len(genes) < 5:
        return {}
    r = requests.post(
        "https://maayanlab.cloud/Enrichr/addList",
        files={"list": (None, "\n".join(genes)), "description": (None, desc)},
        timeout=120,
    )
    r.raise_for_status()
    uid = r.json()["userListId"]
    out: dict[str, list[dict]] = {}
    for lib in ["KEGG_2021_Human", "GO_Biological_Process_2023"]:
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
                    "adjusted_p": float(item[6]),
                    "intersection_size": len(genes_hit),
                }
            )
        rows.sort(key=lambda x: x["adjusted_p"])
        out[lib] = rows
        time.sleep(0.3)
    return out


def hub_cluster_layout(
    edges: pd.DataFrame, min_out: int = 1, top3: set[str] | None = None
) -> tuple[dict[str, tuple[float, float]], pd.DataFrame, set[str]]:
    top3 = top3 or set()
    grp = edges.groupby("TF")["Target"].apply(lambda s: sorted(set(s.astype(str)))).to_dict()
    regs = [
        tf
        for tf, tgts in sorted(grp.items(), key=lambda x: (-len(x[1]), x[0]))
        if len(tgts) >= min_out or tf in top3
    ]
    for tf in top3:
        if tf in grp and tf not in regs:
            regs.append(tf)
    regs = sorted(set(regs), key=lambda tf: (-len(grp.get(tf, [])), tf))

    sub = edges[edges["TF"].isin(regs)].copy()
    assigned: set[str] = set()
    cluster_targets: dict[str, list[str]] = {}
    for tf in regs:
        tgts = [t for t in grp[tf] if t not in assigned and t not in regs]
        cluster_targets[tf] = tgts
        assigned.update(tgts)

    n_regs = len(regs)
    cols = max(1, math.ceil(math.sqrt(n_regs * 1.35)))
    rows = max(1, math.ceil(n_regs / cols))
    cell_w, cell_h = 1.0, 1.0
    pos: dict[str, tuple[float, float]] = {}

    for idx, tf in enumerate(regs):
        row, col = divmod(idx, cols)
        cx = col * cell_w + cell_w / 2
        cy = (rows - 1 - row) * cell_h + cell_h / 2
        pos[tf] = (cx, cy)

    for idx, tf in enumerate(regs):
        cx, cy = pos[tf]
        tgts = cluster_targets[tf]
        r_orbit = min(0.42, 0.12 + 0.016 * len(tgts))
        for j, tgt in enumerate(tgts):
            if tgt in pos:
                continue
            ang = -math.pi / 2 + j * (2 * math.pi / max(len(tgts), 1))
            pos[tgt] = (cx + r_orbit * math.cos(ang), cy + r_orbit * math.sin(ang))

    return pos, sub, set(regs)


def _edge_endpoints(
    x0: float, y0: float, x1: float, y1: float, r0: float, r1: float
) -> tuple[float, float, float, float]:
    dx, dy = x1 - x0, y1 - y0
    dist = max(math.hypot(dx, dy), 1e-9)
    ux, uy = dx / dist, dy / dist
    return x0 + ux * r0, y0 + uy * r0, x1 - ux * r1, y1 - uy * r1


def draw_cluster_network(
    edges: pd.DataFrame,
    out_pdf: Path,
    *,
    min_out: int = 4,
    title: str = "mBrain-gain regulatory network (hub-cluster layout)",
    label_targets: bool = True,
    label_tf_min_out: int = 1,
) -> None:
    pos, sub, sources = hub_cluster_layout(edges, min_out=min_out, top3=TOP3)
    if sub.empty:
        raise RuntimeError("No edges to plot")

    out_deg = sub.groupby("TF").size().to_dict()
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = 0.55
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad

    n_regs = len(sources)
    scale = max(0.28, min(1.0, 10.0 / math.sqrt(max(n_regs, 1))))
    fig_w = max(14.0, (xmax - xmin) * 7.5)
    fig_h = max(10.0, (ymax - ymin) * 7.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")

    r_tf, r_top3, r_hex = 0.055 * scale, 0.068 * scale, 0.038 * scale

    for row in sub.itertuples(index=False):
        u, v = str(row.TF), str(row.Target)
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        rs = r_top3 if u in TOP3 else r_tf
        rt = r_top3 if v in TOP3 else (r_tf if v in sources else r_hex)
        xs0, ys0, xs1, ys1 = _edge_endpoints(x0, y0, x1, y1, rs, rt)
        is_act = str(row.pred_mBrain_name) == "Activate"
        color = "#B03A2E" if is_act else "#2471A3"
        lw = max(0.35, 1.0 * scale) if is_act else max(0.3, 0.9 * scale)
        if is_act:
            arr = FancyArrowPatch(
                (xs0, ys0),
                (xs1, ys1),
                arrowstyle="-|>",
                mutation_scale=max(4, 10 * scale),
                linewidth=lw,
                color=color,
                alpha=0.65,
                zorder=1,
            )
            ax.add_patch(arr)
        else:
            ax.plot([xs0, xs1], [ys0, ys1], color=color, lw=lw, alpha=0.65, zorder=1)
            dx, dy = xs1 - xs0, ys1 - ys0
            d = max(math.hypot(dx, dy), 1e-9)
            px, py = -dy / d, dx / d
            bar = 0.028 * scale
            ax.plot(
                [xs1 - px * bar, xs1 + px * bar],
                [ys1 - py * bar, ys1 + py * bar],
                color=color,
                lw=max(1.2, 2.4 * scale),
                solid_capstyle="butt",
                zorder=2,
            )

    tf_label_fs = max(3.5, 7 * scale)
    top_label_fs = max(5.0, 8 * scale)
    tgt_label_fs = max(3.0, 4.5 * scale)

    for n, (x, y) in pos.items():
        if n in sources:
            is_top = n in TOP3
            r = r_top3 if is_top else r_tf
            fill = "#E74C3C" if is_top else "#F39C12"
            edge = "#922B21" if is_top else "#AF601A"
            circ = Circle((x, y), r, facecolor=fill, edgecolor=edge, linewidth=2 if is_top else 1.2, zorder=4)
            ax.add_patch(circ)
            if is_top or out_deg.get(n, 0) >= label_tf_min_out:
                fs = top_label_fs if is_top else tf_label_fs
                fw = "bold" if is_top else "normal"
                ax.text(x, y + r + 0.02, n, ha="center", va="bottom", fontsize=fs, fontweight=fw, zorder=5)
        else:
            hexagon = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=r_hex,
                orientation=0,
                facecolor="#EBF5FB",
                edgecolor="#5D6D7E",
                linewidth=0.8,
                zorder=3,
            )
            ax.add_patch(hexagon)
            if label_targets and len(n) <= 8:
                ax.text(x, y - r_hex - 0.018, n, ha="center", va="top", fontsize=tgt_label_fs, color="#34495E", zorder=5)

    n_act = int((sub["pred_mBrain_name"] == "Activate").sum())
    n_rep = int((sub["pred_mBrain_name"] == "Repress").sum())
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    note = (
        f"Edges {len(sub)}/{len(edges)} | regulators {len(sources)}"
        + (f" (out-degree >= {min_out})" if min_out > 1 else " (all)")
        + f" | Activate {n_act}, Repress {n_rep}"
    )
    ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=8, color="#566573")
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E74C3C", markersize=10, label="Top-3 TF (circle)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#F39C12", markersize=8, label="Other TF (circle)"),
        Line2D([0], [0], marker="h", color="w", markerfacecolor="#EBF5FB", markeredgecolor="#5D6D7E", markersize=10, label="Target (hexagon)"),
        Line2D([0], [0], color="#B03A2E", lw=1.5, label="Activating (arrow)"),
        Line2D([0], [0], color="#2471A3", lw=1.5, label="Repressive (blunt arrow)"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8, framealpha=0.95)

    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out_pdf}")


def draw_deg_overlap(out_pdf: Path) -> None:
    stats = json.loads((FOLLOW / "transition_matrix_stats.json").read_text(encoding="utf-8"))
    fov = stats["deg_fisher_gain_all"]
    de = pd.read_csv(EXP / "deg.csv")
    deg = set(de.loc[de["padj_BH"] < 0.05, "gene"].astype(str))
    net = set((FOLLOW / "mBrain_gain_top500_genes.txt").read_text(encoding="utf-8").strip().split("\n"))
    a_only = len(net - deg)
    b_only = len(deg - net)
    overlap = len(net & deg)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if HAS_VENN:
        v = venn2(
            subsets=(a_only, b_only, overlap),
            set_labels=("Network genes", "DEG (padj<0.05)"),
            ax=ax,
            set_colors=("#5DADE2", "#F5B041"),
            alpha=0.55,
        )
        for t in v.set_labels:
            if t is not None:
                t.set_fontsize(11)
                t.set_fontweight("bold")
        for t in v.subset_labels:
            if t is not None:
                t.set_fontsize(10)
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        c1 = Circle((0.38, 0.52), 0.28, facecolor="#5DADE2", edgecolor="#2E86C1", alpha=0.45, lw=1.5)
        c2 = Circle((0.62, 0.52), 0.28, facecolor="#F5B041", edgecolor="#D68910", alpha=0.45, lw=1.5)
        ax.add_patch(c1)
        ax.add_patch(c2)
        ax.text(0.28, 0.52, f"{a_only:,}", ha="center", va="center", fontsize=11)
        ax.text(0.50, 0.52, f"{overlap:,}", ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(0.72, 0.52, f"{b_only:,}", ha="center", va="center", fontsize=11)
        ax.text(0.28, 0.82, "Network genes", ha="center", fontsize=11, fontweight="bold")
        ax.text(0.72, 0.82, "DEG (padj<0.05)", ha="center", fontsize=11, fontweight="bold")

    ax.set_title("Network genes vs DEG overlap (Fig. 6B)", fontweight="bold", fontsize=12, pad=14)
    ax.text(
        0.5,
        -0.08,
        f"Network n={len(net):,}; DEG n={len(deg):,}; overlap={overlap:,}\n"
        f"Fisher's exact test: OR = {fov['OR']:.2f}, P = {fov['p_value']:.2e}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out_pdf}")


def _pick_terms(df: pd.DataFrame, wanted: list[str], name_fn=None) -> pd.DataFrame:
    """Pick curated terms that are significant; return sorted by adjusted_p (asc)."""
    name_fn = name_fn or (lambda x: x)
    rows = []
    seen = set()
    sig = df[df["adjusted_p"] < 0.05].copy()
    for term in wanted:
        hit = sig[sig["name"] == term]
        if len(hit):
            r = hit.iloc[0].copy()
            r["label"] = name_fn(r["name"])
            rows.append(r)
            seen.add(term)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("adjusted_p", ascending=True).reset_index(drop=True)


def _pval_rgb(t: float) -> tuple[float, float, float]:
    t = max(0.0, min(1.0, t))
    return (1.0, (245 - 170 * t) / 255.0, (245 - 205 * t) / 255.0)


def _truncate_label(text: str, max_chars: int = 58) -> str:
    text = str(text)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _prepare_enrich_plot_df(df: pd.DataFrame, total_genes: int, top_n: int = 15) -> pd.DataFrame:
    """Build GeneRatio / -log10(adj.P) table in GSE123902 enrichment-PDF format."""
    if df.empty or total_genes <= 0:
        return pd.DataFrame()
    d = df.copy()
    if "label" not in d.columns:
        d["label"] = d["name"].map(_short_go)
    picked = []
    for _, r in d.iterrows():
        name = str(r["label"])
        if not name or any(s.lower() in name.lower() for s in SKIP_ENRICH_TERMS):
            continue
        padj = float(r["adjusted_p"])
        if padj >= 0.05:
            continue
        n = int(r.get("intersection_size") or 0)
        if n <= 0:
            continue
        picked.append(
            {
                "name": name,
                "adjusted_p": padj,
                "intersection_size": n,
                "gene_ratio": n / float(total_genes),
                "neglog10_padj": -math.log10(max(padj, 1e-300)),
            }
        )
        if len(picked) >= top_n:
            break
    out = pd.DataFrame(picked)
    if out.empty:
        return out
    # Y-axis: small Count (point size) at top → large at bottom (clean size gradient).
    return out.sort_values(
        ["intersection_size", "neglog10_padj"], ascending=[True, True]
    ).reset_index(drop=True)


def draw_enrich_dotplot(
    df: pd.DataFrame,
    out_pdf: Path,
    title: str,
    total_genes: int,
    top_n: int = 15,
) -> None:
    """Vector enrichment bubble plot matching GSE123902 ReportLab style.

    GeneRatio on x; point size = Count; color = -log10(adj. P).
    """
    plot_df = _prepare_enrich_plot_df(df, total_genes, top_n=top_n)
    if plot_df.empty:
        print(f"[skip empty enrich] {out_pdf.name}")
        return

    n = len(plot_df)
    row_h = 28.0
    top = 68.0
    bottom_pad = 36.0
    page_w = 760.0
    page_h = top + n * row_h + bottom_pad
    left_label = 18.0
    plot_left = 200.0
    plot_right = 520.0
    cbar_left = 548.0
    cbar_w = 10.0

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_pdf), pagesize=(page_w, page_h))

    c.setFont("Helvetica-Bold", 12)
    c.drawString(18, page_h - 32, title)
    c.setFont("Helvetica", 9)
    c.drawString(plot_left, page_h - 48, "GeneRatio")

    max_ratio = max(float(plot_df["gene_ratio"].max()) * 1.08, 0.05)
    max_n = max(int(plot_df["intersection_size"].max()), 1)
    min_n = int(plot_df["intersection_size"].min())
    max_neg = max(float(plot_df["neglog10_padj"].max()), 1.0)
    plot_bottom = bottom_pad
    plot_top = page_h - top

    for tick in np.linspace(0, max_ratio, 6):
        x = plot_left + (plot_right - plot_left) * tick / max_ratio
        c.setStrokeColor(colors.HexColor("#dddddd"))
        c.line(x, plot_bottom, x, plot_top)
        c.setFillColor(colors.HexColor("#555555"))
        c.setFont("Helvetica", 7)
        c.drawCentredString(x, plot_bottom - 12, f"{tick:.2f}")

    for i, r in enumerate(plot_df.itertuples(index=False)):
        y = plot_top - (i + 0.5) * row_h
        c.setStrokeColor(colors.HexColor("#f0f0f0"))
        c.line(plot_left, y, plot_right, y)
        c.setFillColor(colors.HexColor("#222222"))
        c.setFont("Helvetica", 8.5)
        c.drawString(left_label, y - 3.5, _truncate_label(r.name))

        x = plot_left + (plot_right - plot_left) * float(r.gene_ratio) / max_ratio
        radius = 4.0 + 8.0 * int(r.intersection_size) / max_n
        rgb = _pval_rgb(float(r.neglog10_padj) / max_neg)
        c.setFillColorRGB(*rgb)
        c.setStrokeColor(colors.HexColor("#888888"))
        c.setLineWidth(0.6)
        c.circle(x, y, radius, fill=1, stroke=1)

    cbar_bottom = plot_bottom + 0.38 * (plot_top - plot_bottom)
    cbar_height = 0.52 * (plot_top - plot_bottom)
    steps = 60
    for s in range(steps):
        yy0 = cbar_bottom + cbar_height * s / steps
        yy1 = cbar_bottom + cbar_height * (s + 1) / steps
        t = 1.0 - s / max(steps - 1, 1)
        rgb = _pval_rgb(t)
        c.setFillColorRGB(*rgb)
        c.setStrokeColorRGB(*rgb)
        c.rect(cbar_left, yy0, cbar_w, yy1 - yy0, fill=1, stroke=0)
    c.setStrokeColor(colors.HexColor("#666666"))
    c.setLineWidth(0.7)
    c.rect(cbar_left, cbar_bottom, cbar_w, cbar_height, fill=0, stroke=1)
    c.setFillColor(colors.HexColor("#555555"))
    c.setFont("Helvetica", 7)
    c.drawString(cbar_left + cbar_w + 3, cbar_bottom + cbar_height - 4, f"{max_neg:.1f}")
    c.drawString(cbar_left + cbar_w + 3, cbar_bottom - 2, "0")
    c.saveState()
    c.translate(cbar_left - 10, cbar_bottom + cbar_height / 2)
    c.rotate(90)
    c.setFont("Helvetica", 7)
    c.drawCentredString(0, 0, "-log10(adj. P)")
    c.restoreState()

    legend_counts = sorted({min_n, int(np.median(plot_df["intersection_size"])), max_n})
    count_title_y = cbar_bottom - 16
    c.setFillColor(colors.HexColor("#333333"))
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(cbar_left - 4, count_title_y, "Count")
    leg_y = count_title_y - 16
    for ref in legend_counts:
        rr = 2.5 + 4.5 * ref / max_n
        c.setFillColor(colors.HexColor("#d45a2a"))
        c.setStrokeColor(colors.HexColor("#888888"))
        c.setLineWidth(0.5)
        c.circle(cbar_left + 6, leg_y, rr, fill=1, stroke=1)
        c.setFillColor(colors.HexColor("#333333"))
        c.setFont("Helvetica", 7)
        c.drawString(cbar_left + 16, leg_y - 2.5, str(ref))
        leg_y -= 14

    c.showPage()
    c.save()
    print(f"[fig] {out_pdf}")


def load_or_run_act_enrich() -> dict:
    cache = FOLLOW / "enrichr_activate_gain_genes.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    gain = pd.read_csv(FOLLOW / "mBrain_gain_regulation_edges_top500.csv")
    act = gain[gain["pred_mBrain_name"] == "Activate"]
    genes = sorted(set(act["TF"].astype(str)) | set(act["Target"].astype(str)))
    data = enrichr(genes, "gse131907_activate_gain")
    cache.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def load_or_run_rep_enrich() -> dict:
    cache = FOLLOW / "enrichr_repress_gain_genes.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    gain = pd.read_csv(FOLLOW / "mBrain_gain_regulation_edges_top500.csv")
    rep = gain[gain["pred_mBrain_name"] == "Repress"]
    genes = sorted(set(rep["TF"].astype(str)) | set(rep["Target"].astype(str)))
    data = enrichr(genes, "gse131907_repress_gain")
    cache.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def draw_all_network_figures(gain: pd.DataFrame) -> None:
    """Hub-cluster network PDFs: schematic + full combined / activate / repress."""
    act = gain[gain["pred_mBrain_name"] == "Activate"].copy()
    rep = gain[gain["pred_mBrain_name"] == "Repress"].copy()

    jobs = [
        (gain, FIG / "fig6B_mBrain_gain_network_cluster.pdf", 1, "mBrain-gain network (Fig. 6B, full hub-cluster layout)", False, 3),
        (gain, FIG / "fig_mBrain_gain_network_full.pdf", 1, "mBrain-gain network (full, all regulators)", False, 3),
        (act, FIG / "fig_mBrain_gain_activate_network_full.pdf", 1, "Activating mBrain-gain network (full)", False, 3),
        (rep, FIG / "fig_mBrain_gain_repress_network_full.pdf", 1, "Repressive mBrain-gain network (full)", True, 1),
    ]
    for edges, out_pdf, min_out, title, label_tgt, tf_min in jobs:
        draw_cluster_network(
            edges,
            out_pdf,
            min_out=min_out,
            title=title,
            label_targets=label_tgt,
            label_tf_min_out=tf_min,
        )


def draw_supplementary_figures(gain: pd.DataFrame, *, skip_networks: bool = False) -> None:
    if not skip_networks:
        draw_all_network_figures(gain)

    # Fig. 6B Venn (network vs DEG) retired; UMAP is Fig. 6A (see gse131907_findallmarkers_deg_and_umap.py)

    act_edges = gain[gain["pred_mBrain_name"] == "Activate"]
    rep_edges = gain[gain["pred_mBrain_name"] == "Repress"]
    act_genes = sorted(set(act_edges["TF"].astype(str)) | set(act_edges["Target"].astype(str)))
    rep_genes = sorted(set(rep_edges["TF"].astype(str)) | set(rep_edges["Target"].astype(str)))
    (FOLLOW / "activate_gain_genes.txt").write_text("\n".join(act_genes), encoding="utf-8")

    act_enrich = load_or_run_act_enrich()
    k_act_df = pd.DataFrame(act_enrich.get("KEGG_2021_Human", []))
    g_act_df = pd.DataFrame(act_enrich.get("GO_Biological_Process_2023", []))
    k_act = _pick_terms(k_act_df, KEGG_ACT_TERMS)
    g_act = _pick_terms(g_act_df, GO_ACT_TERMS, _short_go)
    draw_enrich_dotplot(
        k_act,
        FIG / "fig6C_activate_gain_KEGG.pdf",
        "Activating mBrain-gain genes: KEGG enrichment",
        len(act_genes),
    )
    draw_enrich_dotplot(
        g_act,
        FIG / "fig6D_activate_gain_GO_BP.pdf",
        "Activating mBrain-gain genes: GO BP enrichment",
        len(act_genes),
    )

    rep_enrich = load_or_run_rep_enrich()
    k_rep = pd.DataFrame(rep_enrich.get("KEGG_2021_Human", []))
    g_rep = pd.DataFrame(rep_enrich.get("GO_Biological_Process_2023", []))
    if not k_rep.empty:
        k_rep["label"] = k_rep["name"].astype(str)
    if not g_rep.empty:
        g_rep["label"] = g_rep["name"].map(_short_go)
    # Prefer significance order for repress (same as GSE123902 top-N pick)
    if not k_rep.empty:
        k_rep = k_rep.sort_values("adjusted_p")
    if not g_rep.empty:
        g_rep = g_rep.sort_values("adjusted_p")
    draw_enrich_dotplot(
        k_rep,
        FIG / "fig6E_repress_gain_KEGG.pdf",
        "Repressive mBrain-gain genes: KEGG enrichment",
        len(rep_genes),
    )
    draw_enrich_dotplot(
        g_rep,
        FIG / "fig6F_repress_gain_GO_BP.pdf",
        "Repressive mBrain-gain genes: GO BP enrichment",
        len(rep_genes),
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-networks", action="store_true", help="Only redraw overlap/enrichment figures")
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.family"] = "sans-serif"

    gain = pd.read_csv(FOLLOW / "mBrain_gain_regulation_edges_top500.csv")
    draw_supplementary_figures(gain, skip_networks=args.skip_networks)

    print(f"[done] figures -> {FIG}")


if __name__ == "__main__":
    main()
