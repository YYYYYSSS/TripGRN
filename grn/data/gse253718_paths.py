# -*- coding: utf-8 -*-
"""GSE253718 ablation experiment paths (10-fold CV)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_ABLATION_EXPERIMENT = "gse253718_ablation_epithelial_10cv_seed42"

# (stage_dir, display_label, extra CLI flags for train_unseen_samples_no_cv)
ABLATION_STAGES: List[Tuple[str, str, List[str]]] = [
    ("01_expr_only", "expr", ["--no-cds", "--no-ppi"]),
    ("02_expr_cds", "expr+cds", ["--no-ppi"]),
    ("03_expr_ppi", "expr+ppi", ["--no-cds"]),
    ("04_expr_cds_ppi", "all", []),
    ("05_cds_only", "cds", ["--no-expression", "--no-ppi"]),
    ("06_ppi_only", "ppi", ["--no-expression", "--no-cds"]),
    ("07_cds_ppi", "cds+ppi", ["--no-expression"]),
]


def grn_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def gse253718_layout(grn_root: Path | None = None) -> Dict[str, Path]:
    root = Path(grn_root or grn_root_from_here()).resolve()
    full_prefix = root / "data" / "exp" / "GSE253718_exp_counts"
    epi_prefix = root / "data" / "exp" / "GSE253718_epithelial_counts"
    abl_base = root / "experiments" / DEFAULT_ABLATION_EXPERIMENT
    return {
        "grn_root": root,
        "exp_values_full": full_prefix.with_name(full_prefix.name + "_values.npy"),
        "exp_genes_full": full_prefix.with_name(full_prefix.name + "_genes.npy"),
        "exp_values": epi_prefix.with_name(epi_prefix.name + "_values.npy"),
        "exp_genes": epi_prefix.with_name(epi_prefix.name + "_genes.npy"),
        "exp_cells": epi_prefix.with_name(epi_prefix.name + "_cells.npy"),
        "exp_csv": root / "data" / "exp" / "GSE253718_exp_counts.csv",
        "epithelial_dir": root / "data" / "exp" / "GSE253718_epithelial",
        "cv_splits": root / "data" / "exp" / "GSE253718_epithelial_cv_splits_10fold.json",
        "grn_csv": root / "data" / "network" / "grn" / "KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv",
        "cds_gene": root / "data" / "CDS" / "features" / "gene_shared" / "cds_gene_feature_k4.csv",
        "ppi_csv": root / "data" / "network" / "ppi" / "ppi_string_v12_human_score900.csv",
        "experiment_base": abl_base,
        "summary_dir": abl_base / "summary",
        "figures_dir": abl_base / "figures",
    }


def ablation_stage_dir(stage_id: str, fold_idx: int, grn_root: Path | None = None) -> Path:
    layout = gse253718_layout(grn_root)
    return layout["experiment_base"] / stage_id / f"fold_{int(fold_idx):02d}"
