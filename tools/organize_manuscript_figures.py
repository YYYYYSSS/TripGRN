#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild figure/ tree to match manuscript_main.docx panel definitions."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grn.data.pancancer_paths import PANCANCER_GSE_IDS
from grn.scripts.plot_step01_pancancer_method_comparison import (
    METHOD_COLORS_3CLASS_AUC_AUPRC,
    METHOD_COLORS_BINARY,
    METHODS_BINARY,
    _configure_matplotlib,
    _load_3class_fold_table,
    _save_figure,
)
from grn.scripts.plot_step01_trrust_metrics import (
    draw_macro_auc_boxplot_pdf,
    draw_threeclass_auc_boxplot_pdf,
    _ylim_step,
)


def _put(dst: Path, fig: str, panel: str, src: Path, name: Optional[str] = None) -> bool:
    if not src.is_file():
        print(f"[MISS] {src}")
        return False
    out = dst / fig / panel
    out.mkdir(parents=True, exist_ok=True)
    name = name or src.name
    shutil.copy2(src, out / name)
    png = src.with_suffix(".png")
    if png.is_file():
        shutil.copy2(png, out / f"{Path(name).stem}.png")
    print(f"[OK] {fig}/{panel}")
    return True


def _rm_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _load_macro_auc_folds(summary: Path) -> pd.DataFrame:
    """5-method fold-level macro_auc (matches 投稿 Figure 2A/C)."""
    files = {
        "TripGRN": "tripgrn_10cv_fold_metrics.csv",
        "DeepFGRN": "deepfgrn_10cv_fold_metrics.csv",
        "DeepTGI": "deeptgi_10cv_fold_metrics.csv",
        "scMGATGRN": "scmgatgrn_10cv_fold_metrics.csv",
        "GENELink": "genelink_10cv_fold_metrics.csv",
    }
    rows: List[Dict] = []
    for method, fname in files.items():
        path = summary / fname
        if not path.is_file():
            print(f"[warn] missing {path}")
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append(
                {
                    "gse_id": r["gse_id"],
                    "method": method,
                    "fold": int(r["fold"]),
                    "macro_auc": float(r["macro_auc"]),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No macro_auc folds under {summary}")
    return pd.DataFrame(rows)


def _find_legacy_figure1(leg: Path) -> Optional[Path]:
    """Locate 投稿 Figure_1.tif or 机制图 PDF under article/GRN修图."""
    art = leg / "article"
    if not art.is_dir():
        return None
    # Prefer 投稿 TIF, then 原图 PDF
    cands: List[Path] = []
    for p in art.rglob("*"):
        if not p.is_file():
            continue
        n = p.name
        if n == "Figure_1.tif" or n.startswith("Figure 1"):
            cands.append(p)
    # Prefer 投稿图 / Figure_1.tif
    for p in cands:
        if "投稿" in str(p) and p.suffix.lower() == ".tif":
            return p
    for p in cands:
        if p.suffix.lower() == ".tif":
            return p
    for p in cands:
        if p.suffix.lower() == ".pdf":
            return p
    return None


def _write_readme(dst: Path) -> None:
    (dst / "README.md").write_text(
        """# Figures aligned to manuscript_main.docx

```bat
python tools\\organize_manuscript_figures.py --only f2,s,f3
```

## Figure 1 — leave alone (framework schematic)

## Figure 2 (pan-cancer = refactor fig01) — READY
- 2a/2c Macro-AUROC (5 methods, KEGG / DoRothEA)
- 2b/2d three-class OvR AUROC (Trip vs Deep)
- S1–S4 radars

## Figure 3 (ablation + cell type)
| panel | status |
|-------|--------|
| 3a UMAP | LEGACY epithelial (+ 投稿 composite for full facets) |
| 3b per-class ROC | LEGACY PDF (fig02 ROC not re-exported) |
| 3c metrics heatmap | NEW fig02 |
| 3d cell-type macro-AUROC bar | NEW refactor 10CV |
| 3e cell-type heatmap | NEW refactor 10CV |

## Figure 4 — deferred
""",
        encoding="utf-8",
    )


def build_figure1(dst: Path, leg: Path) -> None:
    out = dst / "figure 1" / "figure 1a"
    _rm_tree(dst / "figure 1")
    out.mkdir(parents=True, exist_ok=True)
    src = _find_legacy_figure1(leg)
    if src is None:
        (out / "MISSING_schematic.txt").write_text(
            "Fig.1 = TripGRN multimodal framework schematic.\n"
            "Expected: legacy article/GRN修图/投稿图/Figure_1.tif\n",
            encoding="utf-8",
        )
        print("[MISS] Figure 1 schematic")
        return
    dst_name = f"figure_1a_framework{src.suffix.lower()}"
    shutil.copy2(src, out / dst_name)
    # Also keep a second copy if both tif+pdf exist nearby
    sib_pdf = list(src.parent.parent.rglob("Figure 1*.pdf"))
    for p in sib_pdf[:1]:
        if p.resolve() != src.resolve():
            shutil.copy2(p, out / "figure_1a_framework_mechanism.pdf")
            print(f"[OK] figure 1/figure 1a (+pdf {p.name})")
            break
    print(f"[OK] figure 1/figure 1a <- {src}")


def _pdf_to_png(pdf: Path) -> None:
    try:
        import fitz

        doc = fitz.open(pdf)
        pix = doc[0].get_pixmap(dpi=180)
        pix.save(str(pdf.with_suffix(".png")))
        doc.close()
    except Exception:
        pass


def build_figure2(dst: Path) -> None:
    _rm_tree(dst / "figure 2")
    for exp, net, panel_macro, panel_3 in [
        ("kegg_epithelial_10cv", "KEGG", "figure 2a", "figure 2b"),
        ("dorothea_abc_epithelial_10cv", "DoRothEA", "figure 2c", "figure 2d"),
    ]:
        summary = ROOT / "fig01_pancancer_benchmark" / "experiments" / exp / "summary"
        macro_df = _load_macro_auc_folds(summary)
        out_m = dst / "figure 2"
        out_m.mkdir(parents=True, exist_ok=True)
        stem = panel_macro.replace(" ", "_")
        ymin, ymax = _ylim_step(macro_df["macro_auc"].to_numpy(dtype=float), step=0.1)
        macro_pdf = out_m / f"{stem}_macro_AUROC_boxplot.pdf"
        draw_macro_auc_boxplot_pdf(
            macro_df,
            methods=METHODS_BINARY,
            gse_ids=list(PANCANCER_GSE_IDS),
            method_colors=METHOD_COLORS_BINARY,
            out_path=macro_pdf,
            network_label=net,
            ymin=ymin,
            ymax=ymax,
            value_col="macro_auc",
            title="",
            ylabel="AUROC",
        )
        _pdf_to_png(macro_pdf)

        three_df = _load_3class_fold_table(summary)
        out_3 = dst / "figure 2"
        out_3.mkdir(parents=True, exist_ok=True)
        stem3 = panel_3.replace(" ", "_")
        ymin3, ymax3 = _ylim_step(three_df["auc"].to_numpy(dtype=float), step=0.1)
        three_pdf = out_3 / f"{stem3}_threeclass_AUROC_boxplot.pdf"
        draw_threeclass_auc_boxplot_pdf(
            three_df,
            gse_ids=list(PANCANCER_GSE_IDS),
            method_colors=METHOD_COLORS_3CLASS_AUC_AUPRC,
            out_path=three_pdf,
            network_label=net,
            ymin=ymin3,
            ymax=ymax3,
            value_col="auc",
            ylabel="AUROC",
            title="",
        )
        _pdf_to_png(three_pdf)
        print(f"[plot] Figure 2 {net}")


def build_supplements(dst: Path) -> None:
    for name in ("figure S1", "figure S2", "figure S3", "figure S4"):
        _rm_tree(dst / name)
    f1_sup = ROOT / "fig01_pancancer_benchmark" / "figures" / "supplement"
    radar_maps = [
        (
            "figure S1",
            "S1_KEGG_binary",
            [
                ("radar_auprc.pdf", "figure S1a", "figure_S1a_AUPRC.pdf"),
                ("radar_mcc.pdf", "figure S1b", "figure_S1b_MCC.pdf"),
                ("radar_accuracy.pdf", "figure S1c", "figure_S1c_accuracy.pdf"),
                ("radar_precision.pdf", "figure S1d", "figure_S1d_precision.pdf"),
                ("radar_recall.pdf", "figure S1e", "figure_S1e_recall.pdf"),
                ("radar_f1.pdf", "figure S1f", "figure_S1f_F1.pdf"),
                ("radar_auc.pdf", "figure S1g", "figure_S1g_AUROC.pdf"),
            ],
        ),
        (
            "figure S2",
            "S2_DoRothEA_binary",
            [
                ("radar_auprc.pdf", "figure S2a", "figure_S2a_AUPRC.pdf"),
                ("radar_mcc.pdf", "figure S2b", "figure_S2b_MCC.pdf"),
                ("radar_accuracy.pdf", "figure S2c", "figure_S2c_accuracy.pdf"),
                ("radar_precision.pdf", "figure S2d", "figure_S2d_precision.pdf"),
                ("radar_recall.pdf", "figure S2e", "figure_S2e_recall.pdf"),
                ("radar_f1.pdf", "figure S2f", "figure_S2f_F1.pdf"),
                ("radar_auc.pdf", "figure S2g", "figure_S2g_AUROC.pdf"),
            ],
        ),
    ]
    for fig_name, sub, items in radar_maps:
        for src_name, panel, dst_name in items:
            _put(dst, fig_name, "", f1_sup / sub / src_name, dst_name)

    # S3/S4: per-class faceted boxplots (NoReg / Activation / Repression), Fig.2B/D style
    s3s4_maps = [
        (
            "figure S3",
            "S3_KEGG_3class",
            [
                ("boxplot_3class_accuracy.pdf", "figure S3a", "figure_S3a_accuracy_byclass.pdf"),
                ("boxplot_3class_precision.pdf", "figure S3b", "figure_S3b_precision_byclass.pdf"),
                ("boxplot_3class_recall.pdf", "figure S3c", "figure_S3c_recall_byclass.pdf"),
                ("boxplot_3class_auprc.pdf", "figure S3d", "figure_S3d_AUPRC_byclass.pdf"),
                ("boxplot_3class_auroc.pdf", "figure S3e", "figure_S3e_AUROC_byclass.pdf"),
            ],
        ),
        (
            "figure S4",
            "S4_DoRothEA_3class",
            [
                ("boxplot_3class_accuracy.pdf", "figure S4a", "figure_S4a_accuracy_byclass.pdf"),
                ("boxplot_3class_precision.pdf", "figure S4b", "figure_S4b_precision_byclass.pdf"),
                ("boxplot_3class_recall.pdf", "figure S4c", "figure_S4c_recall_byclass.pdf"),
                ("boxplot_3class_auprc.pdf", "figure S4d", "figure_S4d_AUPRC_byclass.pdf"),
                ("boxplot_3class_auroc.pdf", "figure S4e", "figure_S4e_AUROC_byclass.pdf"),
            ],
        ),
    ]
    for fig_name, sub, items in s3s4_maps:
        for src_name, panel, dst_name in items:
            _put(dst, fig_name, "", f1_sup / sub / src_name, dst_name)


def _copy_if(src: Path, dst_file: Path) -> bool:
    if not src.is_file():
        return False
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_file)
    png = src.with_suffix(".png")
    if png.is_file():
        shutil.copy2(png, dst_file.with_suffix(".png"))
    return True


def build_figure3(dst: Path, leg: Path) -> None:
    """Assemble Fig.3: 3C=new fig02; 3D/3E=refactor celltype if present else LEGACY; 3A/3B reuse prior."""
    _rm_tree(dst / "figure 3")
    f3 = dst / "figure 3"
    f3.mkdir(parents=True, exist_ok=True)

    for p in (leg / "article").rglob("Figure_3.tif"):
        if "投稿" in str(p) or "投稿图" in str(p.parent):
            _copy_if(p, f3 / "_reference_投稿_Figure_3.tif")
            print(f"[OK] figure 3/_reference <- {p}")
            break

    # ---- 3A UMAP (reuse prior) ----
    out3a = f3 / "figure 3a"
    out3a.mkdir(parents=True, exist_ok=True)
    umap = (
        leg
        / "experiments/old/gse253718_ablation_step1_seed42/plots_R_ablation_seed42/fig_ablation_umap_cells_split.pdf"
    )
    if _copy_if(umap, out3a / "figure_3a_UMAP_train_val_test_LEGACY.pdf"):
        print(f"[OK] figure 3/figure 3a <- {umap.name}")
        (out3a / "NOTE.txt").write_text(
            "Fig.3A = SingleR cell-type UMAP + per-type train/val/test facets.\n"
            "Standalone: epithelial train/val/test UMAP (legacy ablation).\n"
            "Full faceted panel: crop A from _reference_投稿_Figure_3.tif.\n",
            encoding="utf-8",
        )
    else:
        (out3a / "MISSING_UMAP.txt").write_text("UMAP not found.\n", encoding="utf-8")
        print("[MISS] figure 3a UMAP")

    # ---- 3B per-class ROC: prefer NEW fig02, else LEGACY ----
    out3b = f3 / "figure 3b"
    out3b.mkdir(parents=True, exist_ok=True)
    new_abl_fig = (
        ROOT
        / "fig02_modality_ablation/experiments/gse253718_ablation_epithelial_10cv_seed42/figures"
    )
    leg_abl_fig = leg / "experiments/gse253718_ablation_epithelial_10cv_seed42/figures"
    roc_pc_new = new_abl_fig / "ablation_perclass_roc_expr_exprcds_all.pdf"
    roc_mac_new = new_abl_fig / "ablation_macro_roc_expr_exprcds_all.pdf"
    roc_pc_leg = leg_abl_fig / "ablation_perclass_roc_expr_exprcds_all.pdf"
    roc_mac_leg = leg_abl_fig / "ablation_macro_roc_expr_exprcds_all.pdf"
    if _copy_if(roc_pc_new, out3b / "figure_3b_ablation_perclass_ROC.pdf"):
        print("[OK] figure 3/figure 3b perclass ROC (NEW fig02)")
        (out3b / "NOTE.txt").write_text("NEW fig02 per-class OvR ROC (expr / expr+cds / all).\n", encoding="utf-8")
    elif _copy_if(roc_mac_new, out3b / "figure_3b_ablation_macro_ROC.pdf"):
        print("[OK] figure 3/figure 3b macro ROC (NEW fig02)")
        (out3b / "NOTE.txt").write_text("NEW fig02 macro ROC (per-class not found).\n", encoding="utf-8")
    elif _copy_if(roc_pc_leg, out3b / "figure_3b_ablation_perclass_ROC_LEGACY.pdf"):
        print("[OK] figure 3/figure 3b perclass ROC (LEGACY)")
        (out3b / "NOTE.txt").write_text(
            "LEGACY per-class ROC. Re-run:\n"
            "  python -m grn.scripts.plot_gse253718_ablation_heatmap_auc --skip-heatmap\n",
            encoding="utf-8",
        )
    elif _copy_if(roc_mac_leg, out3b / "figure_3b_ablation_macro_ROC_LEGACY.pdf"):
        print("[OK] figure 3/figure 3b macro ROC (LEGACY)")

    # ---- 3C ablation metrics heatmap (NEW fig02) ----
    ok3c = _put(
        dst,
        "figure 3",
        "figure 3c",
        ROOT / "fig02_modality_ablation/figures/main/Fig3C_ablation_overall_heatmap.pdf",
        "figure_3c_ablation_overall_metrics_heatmap.pdf",
    )
    if not ok3c:
        for p in (ROOT / "fig02_modality_ablation/figures").rglob("*overall*heatmap*.pdf"):
            _put(dst, "figure 3", "figure 3c", p, "figure_3c_ablation_overall_metrics_heatmap.pdf")
            break

    # ---- 3D / 3E cell-type: prefer refactor summary ----
    new_csv = (
        ROOT / "experiments/gse253718_celltype_bytype_seed42/summary/tripgrn_10cv_type_summary.csv"
    )
    leg_csv = leg / "experiments/gse253718_celltype_bytype_seed42/summary/tripgrn_10cv_type_summary.csv"
    new_fig = ROOT / "experiments/gse253718_celltype_bytype_seed42/figures"
    if new_csv.is_file():
        ct_csv, ct_fig, src_tag = new_csv, new_fig, "NEW"
        note = (
            "Source: experiments/gse253718_celltype_bytype_seed42 (refactor, VAE+gene-disjoint).\n"
            "Fig.3D manuscript wants macro ROC curves; bar is Adobe-ready stand-in.\n"
            "Crop ROC panel D from _reference_投稿_Figure_3.tif if needed for layout.\n"
        )
    else:
        ct_csv, ct_fig, src_tag = leg_csv, leg / "experiments/gse253718_celltype_bytype_seed42/figures", "LEGACY"
        note = "FALLBACK LEGACY cell-type summary (refactor summary missing).\n"

    ct = pd.read_csv(ct_csv)
    out3d = f3 / "figure 3d"
    out3d.mkdir(parents=True, exist_ok=True)
    roc_bytype = ct_fig / "celltype_macro_roc_bytype.pdf"
    if _copy_if(roc_bytype, out3d / "figure_3d_celltype_macro_ROC.pdf"):
        print(f"[OK] figure 3/figure 3d ROC ({src_tag})")
        (out3d / "NOTE.txt").write_text(
            note + "Primary panel: macro-average ROC by cell type.\n",
            encoding="utf-8",
        )
    else:
        order = ct.sort_values("macro_auc_mean", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(order))
        y = order["macro_auc_mean"].to_numpy(dtype=float)
        yerr = order["macro_auc_std"].to_numpy(dtype=float)
        ax.bar(x, y, yerr=yerr, color="#4C78A8", alpha=0.85, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(order["type_label"], rotation=30, ha="right")
        ax.set_ylabel("Macro-AUROC")
        ax.set_title(f"Fig.3D TripGRN macro-AUROC by cell type (10CV mean±SD) [{src_tag}]")
        y0 = max(0.0, float(np.nanmin(y - yerr)) - 0.05)
        y1 = min(1.0, float(np.nanmax(y + yerr)) + 0.03)
        ax.set_ylim(y0, y1)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        _save_figure(fig, out3d / "figure_3d_celltype_macro_AUROC_bar.png")
        plt.close(fig)
        (out3d / "NOTE.txt").write_text(
            note
            + "Bar stand-in until ROC ready:\n"
            "  python -m grn.scripts.plot_gse253718_celltype_heatmap_auc --skip-heatmap --no-skip-roc\n",
            encoding="utf-8",
        )
        print(f"[OK] figure 3/figure 3d bar ({src_tag})")

    out3e = f3 / "figure 3e"
    out3e.mkdir(parents=True, exist_ok=True)
    cols = {
        "macro_auprc_mean": "Macro-AUPRC",
        "mcc_mean": "MCC",
        "accuracy_mean": "Accuracy",
        "precision_macro_mean": "Macro precision",
        "recall_macro_mean": "Macro recall",
        "f1_macro_mean": "Macro F1",
    }
    mat = ct.set_index("type_label")[list(cols)].rename(columns=cols)
    vmin = float(np.floor(np.nanmin(mat.to_numpy()) * 10) / 10)
    vmax = float(np.ceil(np.nanmax(mat.to_numpy()) * 10) / 10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Fig.3E cell-type metrics heatmap [{src_tag}]")
    fig.tight_layout()
    _save_figure(fig, out3e / "figure_3e_celltype_metrics_heatmap.png")
    plt.close(fig)
    zpdf = ct_fig / "celltype_metrics_heatmap_zscore_column.pdf"
    if zpdf.is_file():
        shutil.copy2(zpdf, out3e / "figure_3e_celltype_metrics_heatmap_zscore.pdf")
    (out3e / "NOTE.txt").write_text(note, encoding="utf-8")
    print(f"[OK] figure 3/figure 3e heatmap ({src_tag})")


def build_figure4(dst: Path) -> None:
    _rm_tree(dst / "figure 4")
    f3 = ROOT / "fig03_bulk_cross_cancer" / "figures" / "main"
    for panel, src, name in [
        ("figure 4a", "bulk_10cv_bar_macro_auc.pdf", "figure_4a_macro_AUROC.pdf"),
        ("figure 4b", "bulk_10cv_bar_macro_auprc.pdf", "figure_4b_macro_AUPRC.pdf"),
        ("figure 4c", "bulk_10cv_bar_mcc.pdf", "figure_4c_MCC.pdf"),
        ("figure 4d", "bulk_10cv_bar_accuracy.pdf", "figure_4d_accuracy.pdf"),
        ("figure 4e", "bulk_10cv_bar_precision_macro.pdf", "figure_4e_macro_precision.pdf"),
        ("figure 4f", "bulk_10cv_bar_f1_macro.pdf", "figure_4f_macro_F1.pdf"),
    ]:
        _put(dst, "figure 4", panel, f3 / src, name)
    _put(
        dst,
        "figure 4",
        "figure 4_extra_recall_precision",
        f3 / "bulk_10cv_bar_recall_line_precision.pdf",
        "figure_4_extra_recall_precision.pdf",
    )
    for panel, msg in [
        ("figure 4g", "Fig.4G = transfer OvR ROC (GSE253718→GSE189357). fig04 not run."),
        ("figure 4h", "Fig.4H = transfer metrics radar. fig04 not run."),
    ]:
        d = dst / "figure 4" / panel
        d.mkdir(parents=True, exist_ok=True)
        (d / "MISSING.txt").write_text(msg + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Organize manuscript figure/ tree")
    ap.add_argument(
        "--only",
        default="all",
        help="Comma list: f1,f2,s,f3,f4,all (default all). Example: f1,f2,s",
    )
    args = ap.parse_args()
    wanted = {x.strip().lower() for x in str(args.only).split(",") if x.strip()}
    # Aliases: "1"/"2"/"3"/"4" -> f1/f2/f3/f4
    alias = {"1": "f1", "2": "f2", "3": "f3", "4": "f4", "fig1": "f1", "fig2": "f2", "fig3": "f3", "fig4": "f4"}
    wanted = {alias.get(x, x) for x in wanted}
    if "all" in wanted:
        wanted = {"f1", "f2", "s", "f3", "f4"}

    if wanted & {"f3"}:
        _configure_matplotlib()
    dst = ROOT / "figure"
    dst.mkdir(parents=True, exist_ok=True)
    leg = Path((ROOT / "LEGACY_ROOT.txt").read_text(encoding="utf-8").strip())

    if "f1" in wanted:
        build_figure1(dst, leg)
    if "f2" in wanted:
        build_figure2(dst)
    if "s" in wanted:
        build_supplements(dst)
    if "f3" in wanted:
        build_figure3(dst, leg)
    if "f4" in wanted:
        build_figure4(dst)

    _write_readme(dst)
    n_pdf = len(list(dst.rglob("*.pdf")))
    print(f"[done] {dst} pdfs={n_pdf} only={sorted(wanted)}")
    for p in sorted(x for x in dst.iterdir() if x.is_dir()):
        panels = sorted(c.name for c in p.iterdir() if c.is_dir())
        print(f"  {p.name}: {', '.join(panels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
