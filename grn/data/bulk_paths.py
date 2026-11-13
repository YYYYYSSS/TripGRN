# -*- coding: utf-8 -*-
"""TCGA bulk cross-cancer paths (Fig.3) — prefer fig03_bulk_cross_cancer/."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_BULK_EXPERIMENT = "bulk_tcga_10cv_seed42"
FIG03_DIRNAME = "fig03_bulk_cross_cancer"

# 12 TCGA cancers from historical Step03 bulk runs
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


def fig03_root(grn_root: Optional[Path] = None) -> Path:
    return Path(grn_root or grn_root_from_here()).resolve() / FIG03_DIRNAME


def _normalize_cancer_id(cancer_id: str) -> str:
    code = str(cancer_id).strip()
    if not code.upper().startswith("TCGA-"):
        code = f"TCGA-{code}"
    return code


def bulk_layout(grn_root: Optional[Path] = None) -> Dict[str, Path]:
    """Resolve expression / GRN / PPI / experiment dirs (fig03 first, then shared / shim)."""
    root = Path(grn_root or grn_root_from_here()).resolve()
    fig03 = fig03_root(root)
    shared_grn = root / "shared" / "data" / "network" / "grn"
    shared_ppi = root / "shared" / "data" / "network" / "ppi"
    shared_cds = root / "shared" / "data" / "features" / "cds"
    abl_base = fig03 / "experiments" / DEFAULT_BULK_EXPERIMENT

    def _pick(*cands: Path) -> Path:
        for p in cands:
            if p.is_file() or p.is_dir():
                return p.resolve()
        return cands[0].resolve()

    return {
        "grn_root": root,
        "fig03": fig03,
        "experiment_base": abl_base,
        "summary_dir": abl_base / "summary",
        "figures_dir": abl_base / "figures",
        "cv_splits_dir": _pick(
            fig03 / "data" / "splits",
            root / "data" / "external_validation" / "cv_splits",
        ),
        "external_validation_dir": root / "data" / "external_validation",
        "bulk_exp_dir": _pick(
            fig03 / "data" / "expression",
            root / "data" / "exp" / "12bulk",
        ),
        "grn_csv": _pick(
            fig03 / "data" / "network" / "grn" / "KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv",
            shared_grn / "KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv",
            root / "data" / "network" / "grn" / "KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv",
        ),
        "cds_gene": _pick(
            fig03 / "data" / "features" / "cds" / "cds_gene_feature_k4.csv",
            shared_cds / "cds_gene_feature_k4.csv",
            root / "data" / "CDS" / "features" / "gene_shared" / "cds_gene_feature_k4.csv",
        ),
        # Fig.3 original protocol requires TSS; Fig.1/2 mainline does not.
        "tss_gene": _pick(
            fig03 / "data" / "features" / "cds" / "tss_gene_feature_k4.csv",
            shared_cds / "tss_gene_feature_k4.csv",
            root / "data" / "CDS" / "features" / "gene_shared" / "tss_gene_feature_k4.csv",
        ),
        # nokegg optional; Fig.3 original uses ppi_csv_raw
        "ppi_csv": _pick(
            fig03 / "data" / "network" / "ppi" / "ppi_string_v12_human_score900_nokegg.csv",
            shared_ppi / "ppi_string_v12_human_score900_nokegg.csv",
            fig03 / "data" / "network" / "ppi" / "ppi_string_v12_human_score900.csv",
            shared_ppi / "ppi_string_v12_human_score900.csv",
        ),
        "ppi_csv_raw": _pick(
            fig03 / "data" / "network" / "ppi" / "ppi_string_v12_human_score900.csv",
            shared_ppi / "ppi_string_v12_human_score900.csv",
        ),
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
      1) fig03/.../expression/TCGA-XXX_FPKM_Symbol.csv
      2) data/exp/12bulk/...
      3) data/external_validation/...
      4) TCGA_BULK_DATA_ROOT env override
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
            fig03_root(grn_root) / "data" / "expression" / stem,
            grn_root / "data" / "exp" / "12bulk" / stem,
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
    matches = sorted((root / "data").rglob(stem)) if (root / "data").is_dir() else []
    fig_matches = sorted((fig03_root(root) / "data").rglob(stem)) if (fig03_root(root) / "data").is_dir() else []
    for m in fig_matches + matches:
        return m
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
