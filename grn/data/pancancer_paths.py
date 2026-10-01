# -*- coding: utf-8 -*-
"""Unified pancancer scRNA-seq path layout (step01 epithelial 10CV)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

PANCANCER_GSE_IDS: List[str] = [
    "GSE103322",
    "GSE132465",
    "GSE145825",
    "GSE148071",
    "GSE156632",
    "GSE159115",
    "GSE176031",
    "GSE232237",
    "GSE253718",
    "GSE288479",
]

# Default gold-standard GRN (KEGG PPrel). Override with env STEP01_GRN_CSV.
DEFAULT_GRN_CSV_NAME = "KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv"
# Default experiment tree. Override with env STEP01_EXPERIMENTS_NAME (e.g. ..._trrust).
DEFAULT_EXPERIMENTS_NAME = "step01_pancancer_epithelial_10cv"


def grn_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def step01_experiments_name() -> str:
    name = os.environ.get("STEP01_EXPERIMENTS_NAME", "").strip()
    return name or DEFAULT_EXPERIMENTS_NAME


def step01_grn_csv(grn_root: Optional[Path] = None) -> Path:
    """Gold-standard edge CSV for step01 (KEGG or TRRUST, etc.)."""
    root = Path(grn_root or grn_root_from_here()).resolve()
    env = os.environ.get("STEP01_GRN_CSV", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (root / "data" / "network" / "grn" / DEFAULT_GRN_CSV_NAME).resolve()


def scRNA_layout(grn_root: Optional[Path] = None) -> Dict[str, Path]:
    root = Path(grn_root or grn_root_from_here()).resolve()
    base = root / "data" / "scRNA" / "pancancer"
    exp_root = root / "experiments" / step01_experiments_name()
    return {
        "root": base,
        "manifest": base / "manifest.json",
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
