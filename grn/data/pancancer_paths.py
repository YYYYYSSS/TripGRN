# -*- coding: utf-8 -*-
"""Unified pancancer scRNA-seq path layout (step01 epithelial 10CV)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Formal Fig.1 cohorts (tissue primary / LUAD). GSE145825 is cell-line — excluded.
PANCANCER_GSE_IDS: List[str] = [
    "GSE103322",
    "GSE132465",
    "GSE148071",
    "GSE156632",
    "GSE159115",
    "GSE176031",
    "GSE232237",
    "GSE253718",
    "GSE288479",
]
EXCLUDED_GSE_IDS: List[str] = ["GSE145825"]

# Default gold-standard GRN (KEGG PPrel). Override with env STEP01_GRN_CSV.
DEFAULT_GRN_CSV_NAME = "KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv"
# Default experiment tree under fig01. Override with env STEP01_EXPERIMENTS_NAME.
DEFAULT_EXPERIMENTS_NAME = "kegg_epithelial_10cv"
FIG01_DIRNAME = "fig01_pancancer_benchmark"


def grn_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def fig01_root(grn_root: Optional[Path] = None) -> Path:
    return Path(grn_root or grn_root_from_here()).resolve() / FIG01_DIRNAME


def step01_experiments_name() -> str:
    name = os.environ.get("STEP01_EXPERIMENTS_NAME", "").strip()
    return name or DEFAULT_EXPERIMENTS_NAME


def step01_grn_csv(grn_root: Optional[Path] = None) -> Path:
    """Gold-standard edge CSV for step01 (KEGG or DoRothEA/TRRUST, etc.)."""
    root = Path(grn_root or grn_root_from_here()).resolve()
    env = os.environ.get("STEP01_GRN_CSV", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # Prefer shared/, then fig01 network link, then legacy data/ shim.
    candidates = [
        root / "shared" / "data" / "network" / "grn" / DEFAULT_GRN_CSV_NAME,
        root / "shared" / "data" / "network" / "grn" / "kegg_pprel_activation_inhibition.csv",
        fig01_root(root) / "data" / "network" / "grn" / DEFAULT_GRN_CSV_NAME,
        root / "data" / "network" / "grn" / DEFAULT_GRN_CSV_NAME,
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return candidates[0].resolve()


def scRNA_layout(grn_root: Optional[Path] = None) -> Dict[str, Path]:
    root = Path(grn_root or grn_root_from_here()).resolve()
    fig01 = fig01_root(root)
    # Single source for Fig.1 matrices; data/scRNA remains a temporary shim only.
    base = fig01 / "data" / "expression"
    exp_root = fig01 / "experiments" / step01_experiments_name()
    return {
        "root": base,
        "manifest": base / "pancancer_manifest.json",
        "full": base / "full",
        "epithelial": base / "epithelial",
        "legacy_npy": root / "data" / "exp" / "pancancer_scdata_step5" / "npy",
        "legacy_csv_dir": root / "data" / "exp" / "pancancer",
        "experiments_root": exp_root,
        "experiments_tripgrn": exp_root / "TripGRN",
        "experiments_deepfgrn": exp_root / "DeepFGRN",
        "experiments_deeptgi": exp_root / "DeepTGI",
        "experiments_scmgatgrn": exp_root / "scMGATGRN",
        "experiments_genelink": exp_root / "GENELink",
        "experiments_summary": exp_root / "summary",
        "grn_csv": step01_grn_csv(root),
        "fig01": fig01,
    }


def gse_full_dir(gse_id: str, grn_root: Optional[Path] = None) -> Path:
    return scRNA_layout(grn_root)["full"] / gse_id


def gse_epithelial_dir(gse_id: str, grn_root: Optional[Path] = None) -> Path:
    return scRNA_layout(grn_root)["epithelial"] / gse_id


def gse_npy_triplet(
    tier: str,
    gse_id: str,
    grn_root: Optional[Path] = None,
) -> tuple[Path, Path, Path]:
    """Return (values, genes, cells) paths for tier 'full' or 'epithelial'."""
    layout = scRNA_layout(grn_root)
    base = layout["full" if tier == "full" else "epithelial"] / gse_id
    return base / "values.npy", base / "genes.npy", base / "cells.npy"


def cv_splits_path(gse_id: str, grn_root: Optional[Path] = None) -> Path:
    """Per-GSE 10-fold cell splits.

    Override root with env ``STEP01_CV_SPLITS_ROOT`` (seed-scan etc.) so that
    ``{ROOT}/{gse_id}/cv_splits_10fold.json`` is used instead of the default
    epithelial path. Does not rewrite the shared seed-42 epithelial splits.
    """
    env = os.environ.get("STEP01_CV_SPLITS_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve() / str(gse_id) / "cv_splits_10fold.json"
    return gse_epithelial_dir(gse_id, grn_root) / "cv_splits_10fold.json"


def tripgrn_fold_dir(gse_id: str, fold_idx: int, grn_root: Optional[Path] = None) -> Path:
    layout = scRNA_layout(grn_root)
    return layout["experiments_tripgrn"] / gse_id / f"fold_{int(fold_idx):02d}"


def deepfgrn_fold_dir(gse_id: str, fold_idx: int, grn_root: Optional[Path] = None) -> Path:
    layout = scRNA_layout(grn_root)
    return layout["experiments_deepfgrn"] / gse_id / f"fold_{int(fold_idx):02d}"


def deeptgi_fold_dir(gse_id: str, fold_idx: int, grn_root: Optional[Path] = None) -> Path:
    layout = scRNA_layout(grn_root)
    return layout["experiments_deeptgi"] / gse_id / f"fold_{int(fold_idx):02d}"


def scmgatgrn_fold_dir(gse_id: str, fold_idx: int, grn_root: Optional[Path] = None) -> Path:
    layout = scRNA_layout(grn_root)
    return layout["experiments_scmgatgrn"] / gse_id / f"fold_{int(fold_idx):02d}"


def genelink_fold_dir(gse_id: str, fold_idx: int, grn_root: Optional[Path] = None) -> Path:
    layout = scRNA_layout(grn_root)
    return layout["experiments_genelink"] / gse_id / f"fold_{int(fold_idx):02d}"


def method_fold_dir(
    method: str,
    gse_id: str,
    fold_idx: int,
    grn_root: Optional[Path] = None,
) -> Path:
    key = str(method).strip().lower()
    if key == "tripgrn":
        return tripgrn_fold_dir(gse_id, fold_idx, grn_root)
    if key == "deepfgrn":
        return deepfgrn_fold_dir(gse_id, fold_idx, grn_root)
    if key == "deeptgi":
        return deeptgi_fold_dir(gse_id, fold_idx, grn_root)
    if key in ("scmgatgrn", "scmgat"):
        return scmgatgrn_fold_dir(gse_id, fold_idx, grn_root)
    if key == "genelink":
        return genelink_fold_dir(gse_id, fold_idx, grn_root)
    root = Path(grn_root or grn_root_from_here()).resolve()
    return root / "experiments" / step01_experiments_name() / key / gse_id / f"fold_{int(fold_idx):02d}"


def load_manifest(grn_root: Optional[Path] = None) -> Dict[str, Any]:
    path = scRNA_layout(grn_root)["manifest"]
    if not path.is_file():
        return {"gse_ids": list(PANCANCER_GSE_IDS)}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(payload: Dict[str, Any], grn_root: Optional[Path] = None) -> Path:
    path = scRNA_layout(grn_root)["manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _first_existing(candidates: List[Path]) -> Path:
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return candidates[0].resolve()


def cds_gene_feature_csv(grn_root: Optional[Path] = None) -> Path:
    """CDS k4 gene feature table (prefer shared/, then fig01 / data shim)."""
    root = Path(grn_root or grn_root_from_here()).resolve()
    name = "cds_gene_feature_k4.csv"
    return _first_existing(
        [
            root / "shared" / "data" / "features" / "cds" / name,
            fig01_root(root) / "data" / "features" / "cds" / name,
            root / "data" / "CDS" / "features" / "gene_shared" / name,
            root / "data" / "features" / "cds" / name,
        ]
    )


def resolve_external_project(env_key: str, *rel_parts: str) -> Optional[Path]:
    """Locate sibling baseline repos (DeepTGI / scMGATGRN / GENELink)."""
    env = os.environ.get(env_key, "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p.resolve()
    root = Path(grn_root_from_here())
    candidates = [
        Path(r"E:\project").joinpath(*rel_parts),
        Path("/mnt/e/project").joinpath(*rel_parts),
        root.parent.joinpath(*rel_parts),
    ]
    legacy_txt = root / "LEGACY_ROOT.txt"
    if legacy_txt.is_file():
        try:
            legacy = Path(legacy_txt.read_text(encoding="utf-8").strip())
            candidates.append(legacy.parent.joinpath(*rel_parts))
        except Exception:
            pass
    for path in candidates:
        try:
            if path.is_dir():
                return path.resolve()
        except OSError:
            continue
    return None


def ppi_string_csv(grn_root: Optional[Path] = None) -> Path:
    """STRING PPI edges (prefer shared/)."""
    root = Path(grn_root or grn_root_from_here()).resolve()
    name = "ppi_string_v12_human_score900.csv"
    return _first_existing(
        [
            root / "shared" / "data" / "network" / "ppi" / name,
            fig01_root(root) / "data" / "network" / "ppi" / name,
            root / "data" / "network" / "ppi" / name,
        ]
    )


def ppi_string_nokegg_csv(grn_root: Optional[Path] = None) -> Path:
    """STRING PPI with undirected edges overlapping gold GRN removed (decontaminated)."""
    root = Path(grn_root or grn_root_from_here()).resolve()
    name = "ppi_string_v12_human_score900_nokegg.csv"
    return _first_existing(
        [
            root / "shared" / "data" / "network" / "ppi" / name,
            fig01_root(root) / "data" / "network" / "ppi" / name,
            root / "data" / "network" / "ppi" / name,
        ]
    )
