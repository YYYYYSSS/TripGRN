# -*- coding: utf-8 -*-
"""TCGA bulk cross-cancer experiment paths (10-fold CV)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_BULK_EXPERIMENT = "bulk_tcga_10cv_seed42"

# 12 TCGA cancers from historical Step03 bulk runs (experiments/old/Bulk/repro_extval_*)
TCGA_CANCER_IDS: List[str] = [
    "TCGA-BRCA",
    "TCGA-COAD",
    "TCGA-HNSC",
    "TCGA-KICH",
    "TCGA-KIRC",
    "TCGA-KIRP",
    "TCGA-LGG",
    "TCGA-LUAD",
    "TCGA-LUSC",
    "TCGA-PRAD",
    "TCGA-THCA",
    "TCGA-UCEC",
]

OLD_BULK_RUN_ROOT = Path("experiments") / "old" / "Bulk"


def grn_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_cancer_id(cancer_id: str) -> str:
    code = str(cancer_id).strip()
    if not code.upper().startswith("TCGA-"):
        code = f"TCGA-{code}"
    return code


def bulk_layout(grn_root: Optional[Path] = None) -> Dict[str, Path]:
    root = Path(grn_root or grn_root_from_here()).resolve()
    base = root / "experiments" / DEFAULT_BULK_EXPERIMENT
    return {
        "grn_root": root,
        "experiment_base": base,
        "summary_dir": base / "summary",
        "figures_dir": base / "figures",
        "cv_splits_dir": root / "data" / "external_validation" / "cv_splits",
        "external_validation_dir": root / "data" / "external_validation",
        "bulk_exp_dir": root / "data" / "exp" / "12bulk",
        "grn_csv": root / "data" / "network" / "grn" / "KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv",
        "cds_gene": root / "data" / "CDS" / "features" / "gene_shared" / "cds_gene_feature_k4.csv",
        "tss_gene": root / "data" / "CDS" / "features" / "gene_shared" / "tss_gene_feature_k4.csv",
        "ppi_csv": root / "data" / "network" / "ppi" / "ppi_string_v12_human_score900.csv",
    }


def _rewrite_legacy_path(path: Path, grn_root: Path) -> Path:
    """Map historical D:\\project_GRN\\... paths to current grn_root."""
    s = str(path).replace("\\", "/")
    for prefix in (
        "D:/project_GRN/",
        "D:/project/project_GRN/",
        "E:/project_GRN/",
        "E:/project/project_GRN/",
    ):
        if s.lower().startswith(prefix.lower()):
            rel = s[len(prefix) :]
            return (grn_root / rel.replace("/", os.sep)).resolve()
    return path.resolve()


def _candidate_exp_csvs(cancer_id: str, grn_root: Path) -> Sequence[Path]:
    """
    Search order:
      1) data/exp/12bulk/TCGA-XXX_FPKM_Symbol.csv           (project bulk bundle)
      2) data/external_validation/TCGA-XXX_FPKM_Symbol.csv  (historical Step03)
      3) data/exp/TCGA-XXX_FPKM_Symbol.csv                (step01 LUAD step2)
      4) data/external_validation/**/raw/...
      5) TCGA_BULK_DATA_ROOT env override
    """
    code = _normalize_cancer_id(cancer_id)
    stem = f"{code}_FPKM_Symbol.csv"
    underscored = code.replace("-", "_")
    candidates: List[Path] = []

    env_root = os.environ.get("TCGA_BULK_DATA_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root) / stem)

    layout = bulk_layout(grn_root)
    candidates.extend(
        [
            layout["bulk_exp_dir"] / stem,
            layout["external_validation_dir"] / stem,
            grn_root / "data" / "exp" / stem,
            layout["external_validation_dir"] / underscored / "raw" / stem,
            layout["external_validation_dir"] / f"TCGA_{code.split('-', 1)[1]}" / "raw" / stem,
        ]
    )

    old_cfg = grn_root / OLD_BULK_RUN_ROOT / f"repro_extval_{code}_FPKM_Symbol" / "run_config.json"
    if old_cfg.is_file():
        try:
            payload = json.loads(old_cfg.read_text(encoding="utf-8"))
            exp_csv = payload.get("exp_csv")
            if exp_csv:
                candidates.insert(0, _rewrite_legacy_path(Path(str(exp_csv)), grn_root))
        except (json.JSONDecodeError, OSError):
            pass

    # de-dup while preserving order
    seen: set[str] = set()
    out: List[Path] = []
    for c in candidates:
        key = str(c.resolve()).lower()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def bulk_exp_csv(cancer_id: str, grn_root: Optional[Path] = None) -> Path:
    root = Path(grn_root or grn_root_from_here()).resolve()
    for p in _candidate_exp_csvs(cancer_id, root):
        if p.is_file():
            return p
    code = _normalize_cancer_id(cancer_id)
    stem = f"{code}_FPKM_Symbol.csv"
    matches = sorted((root / "data").rglob(stem))
    if matches:
        return matches[0]
    return _candidate_exp_csvs(cancer_id, root)[0]


def discover_all_bulk_exp_csvs(grn_root: Optional[Path] = None) -> Dict[str, Optional[Path]]:
    root = Path(grn_root or grn_root_from_here()).resolve()
    found: Dict[str, Optional[Path]] = {}
    for cancer in TCGA_CANCER_IDS:
        p = bulk_exp_csv(cancer, root)
        found[cancer] = p if p.is_file() else None
    return found


def cv_splits_path(cancer_id: str, grn_root: Optional[Path] = None) -> Path:
    layout = bulk_layout(grn_root)
    code = _normalize_cancer_id(cancer_id)
    return layout["cv_splits_dir"] / f"{code}_cv_splits_10fold.json"


def bulk_fold_dir(cancer_id: str, fold_idx: int, grn_root: Optional[Path] = None) -> Path:
    layout = bulk_layout(grn_root)
    code = _normalize_cancer_id(cancer_id)
    return layout["experiment_base"] / code / f"fold_{int(fold_idx):02d}"


def extract_cancer_from_run_dir(name: str) -> Optional[str]:
    m = re.search(r"(TCGA-[A-Z0-9]+)", name)
    return m.group(1) if m else None
