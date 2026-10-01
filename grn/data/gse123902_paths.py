# -*- coding: utf-8 -*-
"""GSE123902 (Laughney) path helpers and sample catalog."""
from __future__ import annotations

from pathlib import Path

# Site labels used throughout the pipeline
SITE_PRIMARY = "primary"
SITE_BRAIN = "brain"
SITE_BONE = "bone"
SITE_ADRENAL = "adrenal"
SITE_NORMAL = "normal"

# Curated from GEO sample titles / published metadata (Laughney Nat Med 2020)
# Metastasis files are all named *_METASTASIS_*; site is not in the filename.
SAMPLE_CATALOG = {
    "GSM3516662": {"patient": "LX653", "site": SITE_PRIMARY, "file_tag": "PRIMARY_TUMOUR"},
    "GSM3516663": {"patient": "LX661", "site": SITE_PRIMARY, "file_tag": "PRIMARY_TUMOUR"},
    "GSM3516664": {"patient": "LX666", "site": SITE_BONE, "file_tag": "METASTASIS"},
    "GSM3516665": {"patient": "LX675", "site": SITE_PRIMARY, "file_tag": "PRIMARY_TUMOUR"},
    "GSM3516666": {"patient": "LX675", "site": SITE_NORMAL, "file_tag": "NORMAL"},
    "GSM3516667": {"patient": "LX676", "site": SITE_PRIMARY, "file_tag": "PRIMARY_TUMOUR"},
    "GSM3516668": {"patient": "LX255B", "site": SITE_BRAIN, "file_tag": "METASTASIS"},
    "GSM3516669": {"patient": "LX679", "site": SITE_PRIMARY, "file_tag": "PRIMARY_TUMOUR"},
    "GSM3516670": {"patient": "LX680", "site": SITE_PRIMARY, "file_tag": "PRIMARY_TUMOUR"},
    "GSM3516671": {"patient": "LX681", "site": SITE_BRAIN, "file_tag": "METASTASIS"},
    "GSM3516672": {"patient": "LX682", "site": SITE_PRIMARY, "file_tag": "PRIMARY_TUMOUR"},
    "GSM3516673": {"patient": "LX682", "site": SITE_NORMAL, "file_tag": "NORMAL"},
    "GSM3516674": {"patient": "LX684", "site": SITE_PRIMARY, "file_tag": "PRIMARY_TUMOUR"},
    "GSM3516675": {"patient": "LX684", "site": SITE_NORMAL, "file_tag": "NORMAL"},
    "GSM3516676": {"patient": "LX685", "site": SITE_NORMAL, "file_tag": "NORMAL"},
    "GSM3516677": {"patient": "LX699", "site": SITE_ADRENAL, "file_tag": "METASTASIS"},
    "GSM3516678": {"patient": "LX701", "site": SITE_BRAIN, "file_tag": "METASTASIS"},
}


def gse123902_root(project_root: Path) -> Path:
    return project_root / "data" / "exp" / "GSE123902"


def gse123902_layout(project_root: Path) -> dict[str, Path]:
    root = gse123902_root(project_root)
    return {
        "root": root,
        "raw_tar": project_root / "data" / "exp" / "GSE123902_RAW.tar",
        "raw_dir": root / "raw",
        "merged_dir": root / "merged",
        "singler_dir": root / "singler",
        "epithelial_dir": root / "epithelial_primary_brain",
        "npy_dir": root / "precomputed_primary_brain_npy",
        "sample_catalog_json": root / "sample_catalog.json",
        "cell_meta_csv": root / "merged" / "cell_metadata.csv",
        "counts_values": root / "merged" / "counts_values.npy",
        "counts_genes": root / "merged" / "counts_genes.npy",
        "counts_cells": root / "merged" / "counts_cells.npy",
        "merged_report": root / "merged" / "merge_qc_report.json",
    }
