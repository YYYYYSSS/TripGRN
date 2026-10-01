# -*- coding: utf-8 -*-
"""GSE131907 step06 cell-type TripGRN paths (per annotation Cell_type, 10CV + fixed split)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_STEP06_EXPERIMENT = "gse131907_celltype_step06_bytype_seed42"

# Annotation Cell_type labels (GSE131907). Undetermined excluded.
ANNOTATION_CELL_TYPES: Tuple[str, ...] = (
    "Epithelial cells",
    "T lymphocytes",
    "B lymphocytes",
    "NK cells",
    "Myeloid cells",
    "Fibroblasts",
    "Endothelial cells",
    "MAST cells",
    "Oligodendrocytes",
)


def slugify_cell_type(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    return s.strip("_")


# (type_id, display_label) — one entry per annotation Cell_type
CELL_TYPES: List[Tuple[str, str]] = [
    (slugify_cell_type(label), label) for label in ANNOTATION_CELL_TYPES
]

TYPE_ID_TO_LABEL: Dict[str, str] = {tid: lab for tid, lab in CELL_TYPES}
LABEL_TO_TYPE_ID: Dict[str, str] = {lab: tid for tid, lab in CELL_TYPES}


def grn_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def gse131907_celltype_layout(grn_root: Path | None = None) -> Dict[str, Path]:
    root = Path(grn_root or grn_root_from_here()).resolve()
    base = root / "data" / "exp" / "GSE131907" / "celltype_step06_bytype"
    exp_base = root / "experiments" / DEFAULT_STEP06_EXPERIMENT
    return {
        "grn_root": root,
        "master_exp_csv": root / "data" / "exp" / "GSE131907" / "GSE131907_exp_counts.csv",
        "cell_annotation_gz": root
        / "data"
        / "exp"
        / "GSE131907"
        / "GSE131907_Lung_Cancer_cell_annotation.txt.gz",
        "celltype_data_dir": base,
        "cv_splits_dir": base / "cv_splits",
        "grn_csv": root / "data" / "network" / "grn" / "KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv",
        "cds_gene": root / "data" / "CDS" / "features" / "gene_shared" / "cds_gene_feature_k4.csv",
        "ppi_csv": root / "data" / "network" / "ppi" / "ppi_string_v12_human_score900.csv",
        "experiment_base": exp_base,
        "cv10_base": exp_base / "tripgrn_10cv",
        "fixed_base": exp_base / "tripgrn_fixed_seed42",
        "grn_infer_base": exp_base / "grn_test_probs",
        "unique_edges_dir": exp_base / "unique_edges",
        "enrichment_dir": exp_base / "enrichment",
        "summary_dir": exp_base / "summary",
        "figures_dir": exp_base / "figures",
    }


def celltype_type_dir(type_id: str, grn_root: Path | None = None) -> Path:
    return gse131907_celltype_layout(grn_root)["celltype_data_dir"] / type_id


def celltype_values_npy(type_id: str, *, log_normalized: bool = True, grn_root: Path | None = None) -> Path:
    td = celltype_type_dir(type_id, grn_root)
    if log_normalized:
        return td / "values_log_normalized.npy"
    return td / "values.npy"


def celltype_genes_npy(type_id: str, grn_root: Path | None = None) -> Path:
    return celltype_type_dir(type_id, grn_root) / "genes.npy"


def celltype_columns_npy(type_id: str, grn_root: Path | None = None) -> Path:
    return celltype_type_dir(type_id, grn_root) / "column_names.npy"


def cv_splits_path(type_id: str, *, fixed: bool = False, grn_root: Path | None = None) -> Path:
    d = gse131907_celltype_layout(grn_root)["cv_splits_dir"]
    if fixed:
        return d / f"{type_id}_fixed_seed42.json"
    return d / f"{type_id}_10fold.json"


def tripgrn_10cv_fold_dir(type_id: str, fold_idx: int, grn_root: Path | None = None) -> Path:
    base = gse131907_celltype_layout(grn_root)["cv10_base"]
    return base / type_id / f"fold_{int(fold_idx):02d}"


def tripgrn_fixed_run_dir(type_id: str, grn_root: Path | None = None) -> Path:
    """Fixed-split run dir. train_unseen_samples_no_cv appends fold_01 when --cv-splits-json is set."""
    return gse131907_celltype_layout(grn_root)["fixed_base"] / type_id / "fold_01"


def grn_prob_csv_path(type_id: str, grn_root: Path | None = None) -> Path:
    return gse131907_celltype_layout(grn_root)["grn_infer_base"] / type_id / "pairs_test_all_probs.csv"


def resolve_type_ids(type_arg: str) -> List[Tuple[str, str]]:
    """Resolve CLI --type ALL|type_id|label -> list of (type_id, label)."""
    fa = str(type_arg).strip()
    if fa.upper() == "ALL":
        return list(CELL_TYPES)
    for tid, lab in CELL_TYPES:
        if fa == tid or fa == lab:
            return [(tid, lab)]
    ids = ", ".join(t[0] for t in CELL_TYPES)
    raise SystemExit(f"Unknown type {type_arg!r}. Use ALL or one of: {ids}")
