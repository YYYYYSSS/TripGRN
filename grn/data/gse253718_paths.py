# -*- coding: utf-8 -*-
"""GSE253718 modality-ablation paths (Fig.2) — prefer fig02_modality_ablation/."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_ABLATION_EXPERIMENT = "gse253718_ablation_epithelial_10cv_seed42"
FIG02_DIRNAME = "fig02_modality_ablation"

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


def fig02_root(grn_root: Path | None = None) -> Path:
    return Path(grn_root or grn_root_from_here()).resolve() / FIG02_DIRNAME


def gse253718_layout(grn_root: Path | None = None) -> Dict[str, Path]:
    """Resolve expression / GRN / PPI / experiment dirs (fig02 first, then shared / legacy shim)."""
    root = Path(grn_root or grn_root_from_here()).resolve()
    fig02 = fig02_root(root)
    epi = fig02 / "data" / "expression" / "GSE253718_epithelial"
    shared_grn = root / "shared" / "data" / "network" / "grn"
    shared_ppi = root / "shared" / "data" / "network" / "ppi"
    shared_cds = root / "shared" / "data" / "features" / "cds"
    abl_base = fig02 / "experiments" / DEFAULT_ABLATION_EXPERIMENT

    def _pick(*cands: Path) -> Path:
        for p in cands:
            if p.is_file() or p.is_dir():
                return p.resolve()
        return cands[0].resolve()

    return {
        "grn_root": root,
        "fig02": fig02,
        "exp_values": _pick(epi / "values.npy", root / "data" / "exp" / "GSE253718_epithelial_counts_values.npy"),
        "exp_genes": _pick(epi / "genes.npy", root / "data" / "exp" / "GSE253718_epithelial_counts_genes.npy"),
        "exp_cells": _pick(epi / "cells.npy", root / "data" / "exp" / "GSE253718_epithelial_counts_cells.npy"),
        "cv_splits": _pick(
            fig02 / "data" / "splits" / "cv_splits_10fold.json",
            root / "data" / "exp" / "GSE253718_epithelial_cv_splits_10fold.json",
        ),
        "grn_csv": _pick(
            fig02 / "data" / "network" / "grn" / "KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv",
            shared_grn / "KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv",
        ),
        "cds_gene": _pick(
            fig02 / "data" / "features" / "cds" / "cds_gene_feature_k4.csv",
            shared_cds / "cds_gene_feature_k4.csv",
        ),
        # Fig.2 default PPI = raw STRING; nokegg kept for optional pilot (--ppi-nokegg).
        "ppi_csv": _pick(
            fig02 / "data" / "network" / "ppi" / "ppi_string_v12_human_score900_nokegg.csv",
            shared_ppi / "ppi_string_v12_human_score900_nokegg.csv",
            fig02 / "data" / "network" / "ppi" / "ppi_string_v12_human_score900.csv",
            shared_ppi / "ppi_string_v12_human_score900.csv",
        ),
        "ppi_csv_raw": _pick(
            fig02 / "data" / "network" / "ppi" / "ppi_string_v12_human_score900.csv",
            shared_ppi / "ppi_string_v12_human_score900.csv",
        ),
        "experiment_base": abl_base,
        "summary_dir": abl_base / "summary",
        "figures_dir": abl_base / "figures",
    }


def ablation_stage_dir(stage_id: str, fold_idx: int, grn_root: Path | None = None) -> Path:
    layout = gse253718_layout(grn_root)
    return layout["experiment_base"] / stage_id / f"fold_{int(fold_idx):02d}"
