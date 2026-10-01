#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paths aligned with step01_单细胞多方法基准_LUAD_TCGA_方法比对.bat (project_GRN root = ROOT).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def grn_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def step01_layout(grn_root: Optional[Path] = None) -> Dict[str, Path]:
    root = Path(grn_root or grn_root_from_script()).resolve()
    return {
        "pancancer_csv_dir": root / "data" / "exp" / "pancancer",
        "npy_dir": root / "data" / "exp" / "pancancer_scdata_step5" / "npy",
        "kegg_csv": root / "data" / "network" / "grn" / "KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv",
        "grn_csv": root / "data" / "network" / "grn" / "KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv",
        "out_fgrn": root / "experiments" / "deepfgrn_baseline_pancancer",
        "out_tgi": root / "experiments" / "deeptgi_pancancer",
        "out_grn": root / "experiments" / "pancancer_scdata_step5",
        "out_scmgatgrn": root / "experiments" / "scmgatgrn_pancancer",
        "out_genelink": root / "experiments" / "genelink_pancancer",
        "tgi_summary_csv": root / "experiments" / "deeptgi_pancancer" / "cross_cancer_summary.csv",
    }


def list_pancancer_gse_ids(grn_root: Optional[Path] = None) -> List[str]:
    """Union of GSE ids from step01 CSV dir, npy dir, and existing benchmark outputs."""
    layout = step01_layout(grn_root)
    gses: set[str] = set()
    csv_dir = layout["pancancer_csv_dir"]
    if csv_dir.is_dir():
        gses.update(p.stem for p in csv_dir.glob("GSE*.csv"))
    npy_dir = layout["npy_dir"]
    if npy_dir.is_dir():
        gses.update(p.stem.replace("_values", "") for p in npy_dir.glob("*_values.npy"))
    for key in ("out_fgrn", "out_tgi", "out_grn"):
        out = layout[key]
        if out.is_dir():
            gses.update(d.name for d in out.iterdir() if d.is_dir() and d.name.upper().startswith("GSE"))
    return sorted(gses)


def exp_csv_for_gse(gse_id: str, grn_root: Optional[Path] = None) -> Path:
    layout = step01_layout(grn_root)
    return layout["pancancer_csv_dir"] / f"{gse_id}.csv"


def load_run_meta(run_dir: Path) -> Dict[str, Any]:
    meta_path = Path(run_dir) / "run_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing run_meta.json: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def resolve_exp_csv(gse_id: str, grn_root: Optional[Path] = None, run_meta: Optional[Dict[str, Any]] = None) -> Path:
    if run_meta and run_meta.get("exp_csv"):
        mp = Path(str(run_meta["exp_csv"])).resolve()
        if mp.is_file():
            return mp
    p = exp_csv_for_gse(gse_id, grn_root)
    if p.is_file():
        return p.resolve()
    layout = step01_layout(grn_root)
    meta_hint = ""
    if run_meta and run_meta.get("exp_csv"):
        meta_hint = f" run_meta exp_csv={run_meta['exp_csv']!r} also missing;"
    raise FileNotFoundError(
        f"Expression CSV not found for {gse_id}.{meta_hint} "
        f"Expected step01 PANCANCER={layout['pancancer_csv_dir']}/{gse_id}.csv "
        f"or npy fallback under {layout['npy_dir']}/{gse_id}_values.npy ."
    )


def npy_expression_paths(gse_id: str, grn_root: Optional[Path] = None) -> tuple[Path, Path]:
    layout = step01_layout(grn_root)
    npy_dir = layout["npy_dir"]
    return npy_dir / f"{gse_id}_values.npy", npy_dir / f"{gse_id}_genes.npy"


def resolve_kegg_csv(grn_root: Optional[Path] = None, run_meta: Optional[Dict[str, Any]] = None) -> Path:
    layout = step01_layout(grn_root)
    p = layout["kegg_csv"]
    if p.is_file():
        return p.resolve()
    if run_meta and run_meta.get("kegg_csv"):
        mp = Path(str(run_meta["kegg_csv"])).resolve()
        if mp.is_file():
            return mp
    raise FileNotFoundError(f"KEGG CSV not found at step01 path: {p}")


def find_latest_fgrn_run_with_ckpt(out_base: Path, gse_id: str) -> Optional[Path]:
    gse_dir = Path(out_base) / gse_id
    if not gse_dir.is_dir():
        return None
    cands = sorted(
        (p for p in gse_dir.iterdir() if p.is_dir() and (p / "fold_1" / "best_model.keras").is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def list_fgrn_gse_with_checkpoint(out_base: Path) -> List[str]:
    out_base = Path(out_base)
    if not out_base.is_dir():
        return []
    gses: List[str] = []
    for gse_dir in sorted(out_base.glob("GSE*")):
        if find_latest_fgrn_run_with_ckpt(out_base, gse_dir.name) is not None:
            gses.append(gse_dir.name)
    return gses


GRN_CLASS_NAMES = ("无调控", "激活", "抑制")
FGRN_CLASS_NAMES = ("No regulation", "Repression", "Activation")
BINARY_CLASS_NAMES = ("Negative", "Positive")


def list_npy_gse_ids(grn_root: Optional[Path] = None) -> List[str]:
    layout = step01_layout(grn_root)
    npy_dir = layout["npy_dir"]
    if not npy_dir.is_dir():
        return []
    return sorted(p.stem.replace("_values", "") for p in npy_dir.glob("*_values.npy"))


def npy_ready_for_gse(gse_id: str, grn_root: Optional[Path] = None) -> bool:
    v_p, g_p = npy_expression_paths(gse_id, grn_root)
    return v_p.is_file() and g_p.is_file()


def grn_run_dir_for_gse(gse_id: str, grn_root: Optional[Path] = None) -> Path:
    layout = step01_layout(grn_root)
    return layout["out_grn"] / gse_id


def grn_run_has_checkpoint(run_dir: Path) -> bool:
    run_dir = Path(run_dir)
    return (run_dir / "best_model_fold0.pth").is_file() and (run_dir / "run_config.json").is_file()


def find_latest_deeptgi_run(out_base: Path, gse_id: str) -> Optional[Path]:
    gse_dir = Path(out_base) / gse_id
    if not gse_dir.is_dir():
        return None
    cands = sorted(
        (p for p in gse_dir.iterdir() if p.is_dir() and (p / "run_meta.json").is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def deeptgi_run_has_checkpoint(run_dir: Path) -> bool:
    run_dir = Path(run_dir)
    return (run_dir / "fold_1_best_model.pth").is_file()


def export_grn_test_cm_from_metrics(run_dir: Path) -> tuple[Path, Path]:
    """Write confusion_matrix_test.{json,png} from unseen_samples_test_metrics.json."""
    from grn.evaluation.confusion_matrix_io import export_confusion_matrix_artifacts

    run_dir = Path(run_dir)
    metrics_path = run_dir / "unseen_samples_test_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    cm = payload.get("confusion_matrix")
    if cm is None:
        raise ValueError(f"No confusion_matrix in {metrics_path}")
    gse_id = run_dir.name
    return export_confusion_matrix_artifacts(
        run_dir,
        np.asarray(cm, dtype=np.int64),
        title="GRN test confusion matrix",
        class_names=GRN_CLASS_NAMES,
        extra={"method": "GRN", "gse_id": gse_id, "source": str(metrics_path.name)},
    )
