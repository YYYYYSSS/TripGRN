# -*- coding: utf-8 -*-
"""GSE131907 epithelial tLung vs mBrain: FindAllMarkers-style DE + UMAP (Fig. 6A)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import fisher_exact


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "gse131907_transfer_from_gse253718_epi_seed2"
CASE = EXP / "case_tlung_vs_mbrain_kegg_khop_k1"
FOLLOW = CASE / "followup"
FIG = CASE / "figures"
NPY = ROOT / "data" / "exp" / "GSE131907" / "precomputed_tlung_mbrain_epi_npy"


def _build_adata() -> "sc.AnnData":
    genes = np.load(NPY / "genes.npy", allow_pickle=True).astype(str)
    xt = np.load(NPY / "tlung_values_log_normalized.npy", mmap_mode="r")
    xb = np.load(NPY / "mbrain_values_log_normalized.npy", mmap_mode="r")
    xt = np.asarray(xt.T, dtype=np.float32)
    xb = np.asarray(xb.T, dtype=np.float32)
    X = np.vstack([xt, xb])
    labels = np.array(["tLung"] * xt.shape[0] + ["mBrain"] * xb.shape[0], dtype=object)
    adata = sc.AnnData(X)
    adata.var_names = pd.Index(genes, name="gene")
    adata.obs["compartment"] = pd.Categorical(labels)
    adata.obs_names = [f"c{i}" for i in range(adata.n_obs)]
    return adata


def run_findallmarkers(
    adata: "sc.AnnData",
    *,
    min_pct: float = 0.1,
    logfc_threshold: float = 0.25,
) -> pd.DataFrame:
    sc.tl.rank_genes_groups(
        adata,
        groupby="compartment",
        groups=["mBrain"],
        reference="tLung",
        method="wilcoxon",
        pts=True,
        use_raw=False,
        key_added="de_mbrain_vs_tlung",
    )
    df = sc.get.rank_genes_groups_df(adata, group="mBrain", key="de_mbrain_vs_tlung")
    df = df.rename(
        columns={
            "names": "gene",
            "scores": "score",
            "logfoldchanges": "avg_log2FC",
            "pvals": "pvalue_wilcox",
            "pvals_adj": "padj_BH",
            "pct_nz_group": "pct_mBrain",
        }
    )
    # scanpy stores both group fractions in uns['...']['pts'] (genes x compartments)
    pts = adata.uns.get("de_mbrain_vs_tlung", {}).get("pts")
    if pts is None:
        pts = adata.uns.get("rank_genes_groups", {}).get("pts")
    if pts is not None and "mBrain" in pts.columns and "tLung" in pts.columns:
        df["pct_mBrain"] = df["gene"].map(pts["mBrain"]).astype(float)
        df["pct_tLung"] = df["gene"].map(pts["tLung"]).astype(float)
    elif "pct_mBrain" not in df.columns:
        df["pct_mBrain"] = np.nan
        df["pct_tLung"] = np.nan
    else:
        df["pct_tLung"] = np.nan
    df["max_pct"] = df[["pct_mBrain", "pct_tLung"]].max(axis=1)
    df["abs_log2FC"] = df["avg_log2FC"].abs()
    keep = (df["abs_log2FC"] >= float(logfc_threshold)) & (
        df["max_pct"].fillna(0.0) >= float(min_pct)
    )
    out = df.loc[keep].copy().sort_values("padj_BH", ascending=True)
    return out.reset_index(drop=True)


def fisher_network_overlap(deg_genes: set[str], background_n: int) -> dict:
    net_path = FOLLOW / "mBrain_gain_top500_genes.txt"
    if not net_path.is_file():
        return {}
    net = {g.strip() for g in net_path.read_text(encoding="utf-8").splitlines() if g.strip()}
    a = len(net & deg_genes)
    b = len(net - deg_genes)
    c = len(deg_genes - net)
    d = max(0, int(background_n) - a - b - c)
    odds, p = fisher_exact([[a, b], [c, d]])
    return {
        "n_network": len(net),
        "n_deg": len(deg_genes),
        "n_overlap": a,
        "OR": float(odds),
        "p_value": float(p),
        "background_n": int(background_n),
    }


def run_umap(adata: "sc.AnnData", *, max_cells: int, seed: int, n_hvg: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Cell subset first (if requested), without duplicating the full gene matrix.
    if max_cells > 0 and adata.n_obs > max_cells:
        idx = []
        for lab in ["tLung", "mBrain"]:
            ii = np.where(adata.obs["compartment"].to_numpy() == lab)[0]
            n = min(len(ii), max_cells // 2)
            idx.append(rng.choice(ii, size=n, replace=False))
        idx = np.concatenate(idx)
        ad_view = adata[idx]
    else:
        ad_view = adata

    X = ad_view.X
    if hasattr(X, "A"):
        X = X.A
    # Variance over genes without an extra full AnnData.copy() of all genes.
    X32 = np.asarray(X, dtype=np.float32)
    gene_var = np.var(X32, axis=0)
    n_keep = min(int(n_hvg), int((gene_var > 0).sum()), X32.shape[1])
    top = np.argsort(gene_var)[::-1][:n_keep]
    del X32

    ad = ad_view[:, top].copy()
    sc.pp.scale(ad, max_value=10, zero_center=True)
    n_comps = min(50, ad.n_vars - 1, max(ad.n_obs - 1, 2))
    sc.tl.pca(ad, n_comps=n_comps, svd_solver="arpack")
    n_pcs = min(30, ad.obsm["X_pca"].shape[1])
    sc.pp.neighbors(ad, n_neighbors=15, n_pcs=n_pcs)
    sc.tl.umap(ad, min_dist=0.3, random_state=seed)
    xy = ad.obsm["X_umap"]
    return pd.DataFrame(
        {
            "cell_idx": ad.obs_names.astype(str),
            "compartment": ad.obs["compartment"].astype(str).values,
            "umap1": xy[:, 0],
            "umap2": xy[:, 1],
        }
    )


def draw_umap(coords: pd.DataFrame, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    colors = {"tLung": "#5DADE2", "mBrain": "#E74C3C"}
    for lab in ["tLung", "mBrain"]:
        sub = coords[coords["compartment"] == lab]
        ax.scatter(
            sub["umap1"],
            sub["umap2"],
            s=3,
            c=colors[lab],
            alpha=0.55,
            linewidths=0,
            label="%s (n=%s)" % (lab, format(len(sub), ",")),
            rasterized=True,
        )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(
        "GSE131907 epithelial cells: tLung vs mBrain (Fig. 6A)",
        fontweight="bold",
        fontsize=11,
    )
    ax.legend(markerscale=4, frameon=True, fontsize=9, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[umap]", out_pdf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pct", type=float, default=0.1)
    ap.add_argument("--logfc-threshold", type=float, default=0.25)
    ap.add_argument("--sig-logfc", type=float, default=0.5)
    ap.add_argument("--padj", type=float, default=0.05)
    ap.add_argument(
        "--max-umap-cells",
        type=int,
        default=0,
        help="Cap total cells for UMAP (balanced tLung/mBrain). 0 = use all cells (no downsampling).",
    )
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--skip-umap", action="store_true")
    ap.add_argument("--umap-only", action="store_true", help="Skip DEG; only recompute/draw UMAP")
    args = ap.parse_args()

    EXP.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams["pdf.fonttype"] = 42

    print("[de] loading epithelial tLung/mBrain matrices...")
    adata = _build_adata()
    print("[de] AnnData %d cells x %d genes" % (adata.n_obs, adata.n_vars))

    if not args.umap_only:
        markers = run_findallmarkers(
            adata, min_pct=float(args.min_pct), logfc_threshold=float(args.logfc_threshold)
        )
        markers_path = EXP / "deg_findallmarkers.csv"
        markers.to_csv(markers_path, index=False, encoding="utf-8-sig")
        print("[de] FindAllMarkers-filtered candidates -> %s (n=%d)" % (markers_path, len(markers)))

        summary = {"n_genes_tested_background": int(adata.n_vars), "min_pct": args.min_pct}
        for thr in [0.25, 0.5, 1.0]:
            sub = markers[(markers["padj_BH"] < args.padj) & (markers["abs_log2FC"] >= thr)]
            up = int((sub["avg_log2FC"] > 0).sum())
            down = int((sub["avg_log2FC"] < 0).sum())
            key = "padj<%s_absLog2FC>=%s" % (args.padj, thr)
            summary[key] = {"n": int(len(sub)), "up_mBrain": up, "down_mBrain": down}
            print("[de] %s: n=%d (up=%d, down=%d)" % (key, len(sub), up, down))

        sig = markers[
            (markers["padj_BH"] < float(args.padj)) & (markers["abs_log2FC"] >= float(args.sig_logfc))
        ].copy()
        deg_out = sig[["gene", "avg_log2FC", "pct_mBrain", "pct_tLung", "pvalue_wilcox", "padj_BH"]]
        deg_path = EXP / "deg.csv"
        deg_out.to_csv(deg_path, index=False, encoding="utf-8-sig")
        print("[de] reported DEGs -> %s (n=%d; |log2FC|>=%s)" % (deg_path, len(deg_out), args.sig_logfc))

        fov = fisher_network_overlap(set(deg_out["gene"].astype(str)), background_n=int(adata.n_vars))
        summary["deg_fisher_gain_all"] = fov
        summary["sig_logfc"] = args.sig_logfc
        summary["padj"] = args.padj
        if fov:
            print(
                "[de] Fisher network vs DEG: overlap=%d, OR=%.3f, P=%.2e"
                % (fov["n_overlap"], fov["OR"], fov["p_value"])
            )

        stats_path = FOLLOW / "transition_matrix_stats.json"
        if stats_path.is_file() and fov:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            stats["deg_fisher_gain_all"] = fov
            stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            print("[de] updated", stats_path)

        fs = FOLLOW / "followup_summary.json"
        if fs.is_file() and fov:
            payload = json.loads(fs.read_text(encoding="utf-8"))
            if "stats" in payload:
                payload["stats"]["deg_fisher_gain_all"] = fov
                fs.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        summ_path = EXP / "deg_findallmarkers_summary.json"
        summ_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.skip_umap:
        print(
            "[umap] computing on %s cells (max_umap_cells=%s)..."
            % (adata.n_obs, args.max_umap_cells if args.max_umap_cells > 0 else "all")
        )
        coords = run_umap(
            adata, max_cells=int(args.max_umap_cells), seed=int(args.seed), n_hvg=int(args.n_hvg)
        )
        coords_path = EXP / "umap_tlung_mbrain_epi_coords.csv"
        coords.to_csv(coords_path, index=False, encoding="utf-8-sig")
        print("[umap] coords -> %s (n=%d)" % (coords_path, len(coords)))
        print(coords["compartment"].value_counts().to_string())
        draw_umap(coords, FIG / "fig6A_tlung_mbrain_epi_umap.pdf")

    print("[done]")


if __name__ == "__main__":
    main()
