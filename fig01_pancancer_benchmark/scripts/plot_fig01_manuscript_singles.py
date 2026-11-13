#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.1 (manuscript Figure 2 + S1–S4) — single-panel PDFs for Adobe compositing.

Uses reportlab for boxplots (matplotlib savefig hard-crashes on this Windows env).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grn.data.pancancer_paths import PANCANCER_GSE_IDS  # noqa: E402
from grn.scripts.export_binary_and_threeclass_summary_tables import export_one  # noqa: E402
from grn.scripts.plot_step01_pancancer_method_comparison import (  # noqa: E402
    METHOD_COLORS_3CLASS_AUC_AUPRC,
    METHOD_COLORS_BINARY,
    METHODS_BINARY,
    _load_3class_fold_table,
)
from grn.scripts.plot_step01_trrust_metrics import (  # noqa: E402
    draw_macro_auc_boxplot_pdf,
    draw_threeclass_auc_boxplot_pdf,
    _ylim_step,
)

NETWORKS = (
    ("kegg_epithelial_10cv", "KEGG", "A", "B", "S1", "S3"),
    ("dorothea_abc_epithelial_10cv", "DoRothEA", "C", "D", "S2", "S4"),
)

S3_S4_PERCLASS_SPECS = (
    ("accuracy", "Accuracy (OvR)", "boxplot_3class_accuracy"),
    ("precision", "Precision", "boxplot_3class_precision"),
    ("recall", "Recall", "boxplot_3class_recall"),
    ("auprc", "AUPRC (OvR)", "boxplot_3class_auprc"),
    ("auc", "AUROC (OvR)", "boxplot_3class_auroc"),
)


def _fig01_exp(root: Path, exp: str) -> Path:
    return root / "fig01_pancancer_benchmark" / "experiments" / exp


def _pdf_to_png(pdf: Path) -> None:
    """Best-effort PDF→PNG for previews; ignore if renderer unavailable."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf)
        page = doc[0]
        pix = page.get_pixmap(dpi=180)
        pix.save(str(pdf.with_suffix(".png")))
        doc.close()
    except Exception:
        pass


def _load_macro_auc_folds(summary: Path) -> pd.DataFrame:
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


def _plot_main_boxplots(summary: Path, out_main: Path, network: str, panel_macro: str, panel_3: str) -> None:
    out_main.mkdir(parents=True, exist_ok=True)
    gse_ids = list(PANCANCER_GSE_IDS)
    macro_df = _load_macro_auc_folds(summary)
    three_df = _load_3class_fold_table(summary)

    ymin, ymax = _ylim_step(macro_df["macro_auc"].to_numpy(dtype=float), step=0.1)
    macro_pdf = out_main / f"Fig2{panel_macro}_{network}_macro_auroc_boxplot.pdf"
    draw_macro_auc_boxplot_pdf(
        macro_df,
        methods=METHODS_BINARY,
        gse_ids=gse_ids,
        method_colors=METHOD_COLORS_BINARY,
        out_path=macro_pdf,
        network_label=network,
        ymin=ymin,
        ymax=ymax,
        value_col="macro_auc",
        title=f"{network}: Macro-AUROC (10-fold CV)",
        ylabel="Macro-AUROC",
    )
    _pdf_to_png(macro_pdf)

    ymin3, ymax3 = _ylim_step(three_df["auc"].to_numpy(dtype=float), step=0.1)
    three_pdf = out_main / f"Fig2{panel_3}_{network}_threeclass_auroc_boxplot.pdf"
    draw_threeclass_auc_boxplot_pdf(
        three_df,
        gse_ids=gse_ids,
        method_colors=METHOD_COLORS_3CLASS_AUC_AUPRC,
        out_path=three_pdf,
        network_label=network,
        ymin=ymin3,
        ymax=ymax3,
        value_col="auc",
        ylabel="AUROC",
        title=f"{network}: OvR AUROC by class (TripGRN vs DeepFGRN)",
    )
    _pdf_to_png(three_pdf)

    ymin_a, ymax_a = _ylim_step(three_df["auprc"].to_numpy(dtype=float), step=0.1)
    auprc_pdf = out_main / f"{network}_threeclass_auprc_boxplot.pdf"
    draw_threeclass_auc_boxplot_pdf(
        three_df,
        gse_ids=gse_ids,
        method_colors=METHOD_COLORS_3CLASS_AUC_AUPRC,
        out_path=auprc_pdf,
        network_label=network,
        ymin=ymin_a,
        ymax=ymax_a,
        value_col="auprc",
        ylabel="AUPRC",
        title=f"{network}: OvR AUPRC by class (TripGRN vs DeepFGRN)",
    )
    _pdf_to_png(auprc_pdf)
    print(f"[fig01-main] {network} -> {out_main}")


def _plot_supplement_binary_radars(
    summary: Path,
    out_bin: Path,
    network_label: str,
) -> None:
    from grn.scripts.plot_step01_binary_threeclass_radars import (
        BINARY_RADAR_SPECS,
        METHOD_COLORS_BINARY as RADAR_COLORS,
        METHODS_BINARY as RADAR_METHODS,
        _plot_radar_set,
    )

    out_bin.mkdir(parents=True, exist_ok=True)
    bin_path = summary / "binary_five_method_fold_metrics.csv"
    if not bin_path.is_file():
        export_one(summary)
    bin_df = pd.read_csv(bin_path)
    gse_ids = [g for g in PANCANCER_GSE_IDS if g in set(bin_df["gse_id"])]
    _plot_radar_set(
        bin_df,
        specs=BINARY_RADAR_SPECS,
        methods=RADAR_METHODS,
        method_colors=RADAR_COLORS,
        gse_ids=gse_ids,
        out_dir=out_bin,
        network_label=network_label,
        combined_csv_name="radar_binary_all_metrics_gse_means.csv",
    )
    print(f"[fig01-supp] {network_label} binary radars -> {out_bin}")


def _plot_supplement_3class_boxplots(
    summary: Path,
    out_3: Path,
    network: str,
) -> None:
    out_3.mkdir(parents=True, exist_ok=True)
    three_df = _load_3class_fold_table(summary)
    gse_ids = list(PANCANCER_GSE_IDS)
    for value_col, ylabel, stem in S3_S4_PERCLASS_SPECS:
        ymin, ymax = _ylim_step(three_df[value_col].to_numpy(dtype=float), step=0.1)
        pdf = out_3 / f"{stem}.pdf"
        draw_threeclass_auc_boxplot_pdf(
            three_df,
            gse_ids=gse_ids,
            method_colors=METHOD_COLORS_3CLASS_AUC_AUPRC,
            out_path=pdf,
            network_label=network,
            ymin=ymin,
            ymax=ymax,
            value_col=value_col,
            ylabel=ylabel,
            title="",
        )
        _pdf_to_png(pdf)
    print(f"[fig01-supp] {network} 3-class per-class boxplots -> {out_3}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fig.1 manuscript single-panel PDFs")
    ap.add_argument("--only-network", choices=("KEGG", "DoRothEA", "ALL"), default="ALL")
    ap.add_argument(
        "--only",
        choices=("all", "export", "boxplots", "radars", "s3s4"),
        default="all",
        help="boxplots=Fig2 A-D; radars=S1/S2 binary; s3s4=S3/S4 per-class boxplots.",
    )
    args = ap.parse_args()

    root = ROOT
    fig_root = root / "fig01_pancancer_benchmark" / "figures"
    do_export = args.only in ("all", "export", "boxplots", "radars", "s3s4")
    do_box = args.only in ("all", "boxplots")
    do_radar = args.only in ("all", "radars")
    do_s3s4 = args.only in ("all", "boxplots", "s3s4")
    if args.only == "export":
        do_box = do_radar = do_s3s4 = False

    for exp, net, pA, pB, s_bin, s_3 in NETWORKS:
        if args.only_network != "ALL" and args.only_network != net:
            continue
        summary = _fig01_exp(root, exp) / "summary"
        if not summary.is_dir():
            raise FileNotFoundError(summary)
        if do_export:
            print(f"==== export tables {exp} ====")
            export_one(summary)
        if do_box:
            _plot_main_boxplots(summary, fig_root / "main", net, pA, pB)
        if do_radar:
            _plot_supplement_binary_radars(
                summary,
                fig_root / "supplement" / f"{s_bin}_{net}_binary",
                network_label=net if net != "DoRothEA" else "DoRothEA A-C",
            )
        if do_s3s4:
            _plot_supplement_3class_boxplots(
                summary,
                fig_root / "supplement" / f"{s_3}_{net}_3class",
                network=net,
            )

    print(f"[fig01] done ({args.only}) -> {fig_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
