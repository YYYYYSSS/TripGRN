# -*- coding: utf-8 -*-
"""GSE253718 cell-type TripGRN paths (SingleR major types, 10CV)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_CELLTYPE_EXPERIMENT = "gse253718_celltype_bytype_seed42"

# Major SingleR pruned labels (n>=~450); NK included per workflow plan.
MAJOR_SINGLER_TYPES: Tuple[str, ...] = (
    "Macrophage",
    "Epithelial_cells",
    "T_cells",
    "B_cell",
    "Fibroblasts",
    "Endothelial_cells",
    "Smooth_muscle_cells",
    "Monocyte",
    "MSC",
    "NK_cell",
)

# Display labels for figures / tables
DISPLAY_LABELS: Dict[str, str] = {
    "Macrophage": "Macrophage",
    "Epithelial_cells": "Epithelial cells",
    "T_cells": "T cells",
    "B_cell": "B cells",
    "Fibroblasts": "Fibroblasts",
    "Endothelial_cells": "Endothelial cells",
    "Smooth_muscle_cells": "Smooth muscle cells",
    "Monocyte": "Monocyte",
    "MSC": "MSC",
    "NK_cell": "NK cells",
}


def slugify_cell_type(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    return s.strip("_")


CELL_TYPES: List[Tuple[str, str]] = [
    (slugify_cell_type(lab), DISPLAY_LABELS.get(lab, lab.replace("_", " "))) for lab in MAJOR_SINGLER_TYPES
]

TYPE_ID_TO_SINGLER: Dict[str, str] = {
    slugify_cell_type(lab): lab for lab in MAJOR_SINGLER_TYPES
}
TYPE_ID_TO_LABEL: Dict[str, str] = {tid: lab for tid, lab in CELL_TYPES}
LABEL_TO_TYPE_ID: Dict[str, str] = {lab: tid for tid, lab in CELL_TYPES}
SINGLER_TO_TYPE_ID: Dict[str, str] = {v: k for k, v in TYPE_ID_TO_SINGLER.items()}


def grn_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def gse253718_celltype_layout(grn_root: Path | None = None) -> Dict[str, Path]:
    root = Path(grn_root or grn_root_from_here()).resolve()
    base = root / "data" / "exp" / "GSE253718" / "celltype_bytype"
    exp_base = root / "experiments" / DEFAULT_CELLTYPE_EXPERIMENT
    return {
        "grn_root": root,
        "master_exp_csv": root / "data" / "exp" / "GSE253718_exp_counts.csv",
        "master_values_npy": root / "data" / "exp" / "GSE253718_exp_values.npy",
        "master_genes_npy": root / "data" / "exp" / "GSE253718_exp_genes.npy",
        "singler_labels_csv": root
        / "data"
        / "exp"
        / "GSE253718_epithelial"
        / "singler"
        / "singler_labels.csv",
        "celltype_data_dir": base,
        "cv_splits_dir": base / "cv_splits",
        "grn_csv": root / "data" / "network" / "grn" / "KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv",
        "cds_gene": root / "data" / "CDS" / "features" / "gene_shared" / "cds_gene_feature_k4.csv",
        "ppi_csv": root / "data" / "network" / "ppi" / "ppi_string_v12_human_score900.csv",
        "experiment_base": exp_base,
        "cv10_base": exp_base / "tripgrn_10cv",
        "summary_dir": exp_base / "summary",
        "figures_dir": exp_base / "figures",
    }


def celltype_type_dir(type_id: str, grn_root: Path | None = None) -> Path:
    return gse253718_celltype_layout(grn_root)["celltype_data_dir"] / type_id


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
    d = gse253718_celltype_layout(grn_root)["cv_splits_dir"]
    if fixed:
        return d / f"{type_id}_fixed_seed42.json"
    return d / f"{type_id}_10fold.json"


def tripgrn_10cv_fold_dir(type_id: str, fold_idx: int, grn_root: Path | None = None) -> Path:
    base = gse253718_celltype_layout(grn_root)["cv10_base"]
    return base / type_id / f"fold_{int(fold_idx):02d}"


def resolve_type_ids(type_arg: str, grn_root: Path | None = None) -> List[Tuple[str, str]]:
    """Resolve CLI --type ALL|type_id|label -> list of (type_id, label).

    If materialize wrote active_cell_types.json, ALL uses that kept subset.
    """
    import json

    fa = str(type_arg).strip()
    active_path = gse253718_celltype_layout(grn_root)["celltype_data_dir"] / "active_cell_types.json"
    active: List[Tuple[str, str]] = list(CELL_TYPES)
    if active_path.is_file():
        rows = json.loads(active_path.read_text(encoding="utf-8"))
        active = [(str(r["type_id"]), str(r["label"])) for r in rows]

    if fa.upper() == "ALL":
        return active
    for tid, lab in active:
        if fa == tid or fa == lab or fa == TYPE_ID_TO_SINGLER.get(tid):
            return [(tid, lab)]
    ids = ", ".join(t[0] for t in active)
    raise SystemExit(f"Unknown type {type_arg!r}. Use ALL or one of: {ids}")

