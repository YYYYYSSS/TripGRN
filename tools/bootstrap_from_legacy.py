#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

FIG_META = [
    ("fig01_pancancer_benchmark", "Fig.1", "Pan-cancer epithelial 10CV (KEGG + DoRothEA)"),
    ("fig02_modality_ablation", "Fig.2", "GSE253718 modality ablation"),
    ("fig03_bulk_cross_cancer", "Fig.3", "TCGA bulk cross-cancer"),
    ("fig04_cross_dataset_gse189357", "Fig.4", "Transfer GSE253718 to GSE189357"),
    ("fig05_brain_met_case", "Fig.5", "GSE131907 brain-met application"),
]
PANCANCER_GSE = [
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
TCGA = [
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


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def junction(link: Path, target: Path) -> str:
    target = target.resolve()
    if not target.exists():
        return f"SKIP missing target: {target}"
    if link.exists() or link.is_symlink():
        try:
            if link.resolve() == target:
                return f"OK exists: {link.name}"
        except Exception:
            pass
        if link.is_dir() and not link.is_symlink():
            try:
                next(link.iterdir())
                return f"SKIP non-empty exists: {link}"
            except StopIteration:
                link.rmdir()
            except Exception:
                return f"SKIP exists: {link}"
        else:
            try:
                link.unlink()
            except Exception:
                return f"SKIP exists: {link}"
    ensure_dir(link.parent)
    r = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return f"FAIL {link}: {(r.stderr or r.stdout).strip()}"
    return f"JUNCTION {link.name}"


def copy_small(src: Path, dst: Path) -> str:
    if not src.exists():
        return f"SKIP missing: {src}"
    ensure_dir(dst.parent)
    if dst.exists():
        return f"OK exists: {dst.name}"
    shutil.copy2(src, dst)
    return f"COPY {src.name}"


def link_file(link: Path, target: Path) -> str:
    target = Path(target)
    if not target.exists():
        return f"SKIP missing file: {target}"
    ensure_dir(link.parent)
    if link.exists() or link.is_symlink():
        try:
            if link.resolve() == target.resolve():
                return f"OK exists: {link.name}"
        except Exception:
            pass
        try:
            link.unlink()
        except Exception:
            return f"SKIP exists: {link}"
    try:
        os.link(target, link)
        return f"HARDLINK {link.name}"
    except OSError:
        try:
            link.symlink_to(target)
            return f"SYMLINK {link.name}"
        except OSError:
            shutil.copy2(target, link)
            return f"COPY {link.name}"


def copy_grn(legacy: Path, root: Path) -> None:
    src, dst = legacy / "grn", root / "grn"
    if dst.exists():
        shutil.rmtree(dst)

    def ign(d, names):
        return [
            n
            for n in names
            if n in {"__pycache__", ".pytest_cache", ".mypy_cache"} or n.endswith(".pyc")
        ]

    shutil.copytree(src, dst, ignore=ign)
    print("[grn] copied")


def bootstrap(legacy: Path, root: Path) -> Dict[str, List[str]]:
    log: Dict[str, List[str]] = {}
    shared = root / "shared" / "data"
    ensure_dir(shared / "network" / "grn")
    ensure_dir(shared / "network" / "ppi")
    ensure_dir(shared / "features" / "cds")
    acts: List[str] = []
    pairs = [
        (
            legacy / "data/network/grn/KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv",
            shared / "network/grn/kegg_pprel_activation_inhibition.csv",
        ),
        (
            legacy / "data/network/grn/DoRothEA_human_ABC_activation_inhibition_with_symbols.csv",
            shared / "network/grn/dorothea_abc_activation_inhibition.csv",
        ),
        (
            legacy / "data/network/grn/TRRUST_human_activation_inhibition_with_symbols.csv",
            shared / "network/grn/trrust_activation_inhibition.csv",
        ),
        (
            legacy / "data/network/ppi/ppi_string_v12_human_score900.csv",
            shared / "network/ppi/ppi_string_v12_human_score900.csv",
        ),
        (
            legacy / "data/CDS/features/gene_shared/cds_gene_feature_k4.csv",
            shared / "features/cds/cds_gene_feature_k4.csv",
        ),
    ]
    for s, d in pairs:
        acts.append(copy_small(s, d))
    acts.append(junction(shared / "network/ppi/_string_vendor_dump", legacy / "data/network/ppi"))
    for clean, old in [
        ("kegg_pprel_activation_inhibition.csv", "KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv"),
        ("dorothea_abc_activation_inhibition.csv", "DoRothEA_human_ABC_activation_inhibition_with_symbols.csv"),
        ("trrust_activation_inhibition.csv", "TRRUST_human_activation_inhibition_with_symbols.csv"),
    ]:
        acts.append(link_file(shared / "network/grn" / old, shared / "network/grn" / clean))
    log["shared"] = acts

    f1 = root / "fig01_pancancer_benchmark" / "data"
    acts = []
    for gse in PANCANCER_GSE:
        acts.append(
            junction(
                f1 / "expression/epithelial" / gse,
                legacy / "data/scRNA/pancancer/epithelial" / gse,
            )
        )
        acts.append(
            junction(
                f1 / "expression/full" / gse,
                legacy / "data/scRNA/pancancer/full" / gse,
            )
        )
    acts.append(junction(f1 / "network/grn", shared / "network/grn"))
    acts.append(junction(f1 / "network/ppi", shared / "network/ppi"))
    acts.append(junction(f1 / "features/cds", shared / "features/cds"))
    man = legacy / "data/scRNA/pancancer/manifest.json"
    if man.is_file():
        acts.append(copy_small(man, f1 / "expression/pancancer_manifest.legacy.json"))
    log["fig01"] = acts

    f2 = root / "fig02_modality_ablation" / "data"
    acts = []
    epi = f2 / "expression/GSE253718_epithelial"
    ensure_dir(epi)
    ensure_dir(f2 / "splits")
    for dst, src in [
        ("values.npy", "GSE253718_epithelial_counts_values.npy"),
        ("genes.npy", "GSE253718_epithelial_counts_genes.npy"),
        ("cells.npy", "GSE253718_epithelial_counts_cells.npy"),
    ]:
        acts.append(link_file(epi / dst, legacy / "data/exp" / src))
    acts.append(
        link_file(
            f2 / "splits/cv_splits_10fold.json",
            legacy / "data/exp/GSE253718_epithelial_cv_splits_10fold.json",
        )
    )
    acts.append(junction(f2 / "expression/GSE253718_epithelial_meta", legacy / "data/exp/GSE253718_epithelial"))
    acts.append(junction(f2 / "network/grn", shared / "network/grn"))
    acts.append(junction(f2 / "network/ppi", shared / "network/ppi"))
    acts.append(junction(f2 / "features/cds", shared / "features/cds"))
    acts.append(junction(f2 / "raw/GSE253718_10x", legacy / "data/exp/GSE253718"))
    log["fig02"] = acts

    f3 = root / "fig03_bulk_cross_cancer" / "data"
    acts = []
    ensure_dir(f3 / "expression")
    ensure_dir(f3 / "splits")
    for ca in TCGA:
        acts.append(
            link_file(
                f3 / "expression" / f"{ca}_FPKM_Symbol.csv",
                legacy / "data/exp/12bulk" / f"{ca}_FPKM_Symbol.csv",
            )
        )
        acts.append(
            link_file(
                f3 / "splits" / f"{ca}_cv_splits_10fold.json",
                legacy / "data/external_validation/cv_splits" / f"{ca}_cv_splits_10fold.json",
            )
        )
    acts.append(junction(f3 / "network/grn", shared / "network/grn"))
    acts.append(junction(f3 / "network/ppi", shared / "network/ppi"))
    acts.append(junction(f3 / "features/cds", shared / "features/cds"))
    log["fig03"] = acts

    f4 = root / "fig04_cross_dataset_gse189357" / "data"
    acts = []
    ensure_dir(f4 / "source")
    for name in [
        "GSE253718_exp_values.npy",
        "GSE253718_exp_genes.npy",
        "GSE253718_exp_counts_values.npy",
        "GSE253718_exp_counts_genes.npy",
    ]:
        acts.append(link_file(f4 / "source" / name, legacy / "data/exp" / name))
    acts.append(junction(f4 / "target/GSE189357", legacy / "data/exp/GSE189357"))
    acts.append(junction(f4 / "network/grn", shared / "network/grn"))
    acts.append(junction(f4 / "network/ppi", shared / "network/ppi"))
    acts.append(junction(f4 / "features/cds", shared / "features/cds"))
    log["fig04"] = acts

    f5 = root / "fig05_brain_met_case" / "data"
    acts = []
    acts.append(junction(f5 / "expression/GSE131907", legacy / "data/exp/GSE131907"))
    acts.append(junction(f5 / "derived/GSE131907_other", legacy / "data/other/GSE131907"))
    acts.append(junction(f5 / "network/grn", shared / "network/grn"))
    acts.append(junction(f5 / "network/ppi", shared / "network/ppi"))
    acts.append(junction(f5 / "features/cds", shared / "features/cds"))
    log["fig05"] = acts

    compat = root / "data"
    acts = []
    acts.append(junction(compat / "scRNA/pancancer/epithelial", f1 / "expression/epithelial"))
    acts.append(junction(compat / "scRNA/pancancer/full", f1 / "expression/full"))
    acts.append(junction(compat / "network/grn", shared / "network/grn"))
    acts.append(junction(compat / "network/ppi", shared / "network/ppi"))
    acts.append(junction(compat / "CDS/features/gene_shared", shared / "features/cds"))
    ensure_dir(compat / "exp")
    for name in [
        "GSE253718_epithelial_counts_values.npy",
        "GSE253718_epithelial_counts_genes.npy",
        "GSE253718_epithelial_counts_cells.npy",
        "GSE253718_epithelial_cv_splits_10fold.json",
        "GSE253718_exp_values.npy",
        "GSE253718_exp_genes.npy",
        "GSE253718_exp_counts_values.npy",
        "GSE253718_exp_counts_genes.npy",
    ]:
        acts.append(link_file(compat / "exp" / name, legacy / "data/exp" / name))
    acts.append(junction(compat / "exp/12bulk", legacy / "data/exp/12bulk"))
    acts.append(junction(compat / "exp/GSE189357", legacy / "data/exp/GSE189357"))
    acts.append(junction(compat / "exp/GSE131907", legacy / "data/exp/GSE131907"))
    acts.append(junction(compat / "exp/GSE253718", legacy / "data/exp/GSE253718"))
    acts.append(junction(compat / "exp/GSE253718_epithelial", legacy / "data/exp/GSE253718_epithelial"))
    acts.append(junction(compat / "other/GSE131907", legacy / "data/other/GSE131907"))
    acts.append(
        junction(compat / "external_validation/cv_splits", legacy / "data/external_validation/cv_splits")
    )
    log["compat_data"] = acts
    return log


def write_docs(root: Path) -> None:
    write_text(
        root / "README.md",
        """# TripGRN refactor (by manuscript figure)

Read-only extract from E:/project/project_GRN. Do not modify the legacy repo.

## Layout

- fig01_pancancer_benchmark (Fig.1): pan-cancer epithelial 10CV (KEGG + DoRothEA)
- fig02_modality_ablation (Fig.2): GSE253718 modality ablation
- fig03_bulk_cross_cancer (Fig.3): TCGA bulk cross-cancer
- fig04_cross_dataset_gse189357 (Fig.4): transfer to GSE189357
- fig05_brain_met_case (Fig.5): GSE131907 application case

Each figXX_* has: scripts/, data/, experiments/, figures/

Shared priors: shared/data/
Compat shim for old path helpers: data/ (temporary)

## Clean data rules

Old data/exp was flat and messy. New rules:
1. Expression isolated per figure
2. Uniform names: values.npy / genes.npy / cells.npy / cv_splits_10fold.json
3. No GB copies: junctions/hardlinks via bootstrap
4. Raw 10x only under optional raw/

See docs/DATA_LAYOUT.md and docs/LEGACY_MAP.md.

## Setup

1. Check LEGACY_ROOT.txt
2. python tools/bootstrap_from_legacy.py --root <this_repo>
""",
    )
    write_text(
        root / "docs" / "DATA_LAYOUT.md",
        """# Data layout

shared/data/network|features = priors
figXX_*/data/expression|splits = figure-specific matrices
data/ = compatibility aliases for legacy grn path helpers
""",
    )
    write_text(
        root / "docs" / "LEGACY_MAP.md",
        """# Legacy map

Legacy root: E:/project/project_GRN

Fig.1 epithelial/{GSE} <- data/scRNA/pancancer/epithelial/{GSE}
Fig.2 GSE253718_epithelial/*.npy <- data/exp/GSE253718_epithelial_counts_*.npy
Fig.3 TCGA csv <- data/exp/12bulk
Fig.4 source/target <- data/exp/GSE253718_* and GSE189357
Fig.5 GSE131907 <- data/exp/GSE131907 and data/other/GSE131907

Not migrated: GSE123902, pancancer_scdata_step5, ad-hoc test csvs
""",
    )
    for folder, fig, desc in FIG_META:
        write_text(root / folder / "README.md", f"# {fig} - {desc}\n\nFolders: data/, scripts/, experiments/, figures/\n")
        write_text(root / folder / "data" / "README.md", f"# {fig} data\n\nClean view; large files linked by bootstrap.\n")
    write_text(root / "shared" / "data" / "README.md", "# shared/data\n\nShared GRN/PPI/CDS priors.\n")
    write_text(
        root / ".gitignore",
        """__pycache__/
*.py[cod]
*.npy
*.npz
*.h5ad
*.rds
*.tar
*.zip
**/experiments/**
!**/experiments/.gitkeep
**/figures/**
!**/figures/.gitkeep
.repo_root
.env
""",
    )


def resolve_default_root() -> Path:
    marker = Path(r"C:/Users/14842/AppData/Local/Temp/tripgrn_root.txt")
    if marker.is_file():
        return Path(marker.read_text(encoding="utf-8").strip()).resolve()
    skip = {"$RECYCLE.BIN", "System Volume Information"}
    for x in Path("F:/").iterdir():
        if not x.is_dir() or x.name in skip or x.name.startswith("$"):
            continue
        try:
            kids = list(x.iterdir())
        except Exception:
            continue
        for k in kids:
            if k.is_dir() and "TripGRN" in k.name:
                return k.resolve()
    raise FileNotFoundError("cannot resolve TripGRN refactor root on F:")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args()
    root = Path(args.root).resolve() if args.root else resolve_default_root()
    legacy_txt = root / "LEGACY_ROOT.txt"
    if not legacy_txt.is_file():
        write_text(legacy_txt, "E:\\project\\project_GRN\n")
    legacy = Path(legacy_txt.read_text(encoding="utf-8").strip()).resolve()
    print("[root]", root)
    print("[legacy]", legacy)
    if not legacy.is_dir():
        print("legacy missing", file=sys.stderr)
        return 2
    copy_grn(legacy, root)
    req = legacy / "requirements.txt"
    if req.is_file():
        shutil.copy2(req, root / "requirements.txt")
    for folder, _, _ in FIG_META:
        for sub in ("scripts", "experiments", "figures"):
            ensure_dir(root / folder / sub)
            gk = root / folder / sub / ".gitkeep"
            if not gk.exists():
                gk.write_text("", encoding="utf-8")
    log = bootstrap(legacy, root)
    write_docs(root)
    ensure_dir(root / "tools")
    # install this script into the new repo
    self_path = Path(__file__).resolve()
    dst = root / "tools" / "bootstrap_from_legacy.py"
    if self_path != dst:
        shutil.copy2(self_path, dst)
    out = root / "docs" / "bootstrap_log.json"
    out.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[log]", out)
    for section, rows in log.items():
        bad = [r for r in rows if r.startswith("FAIL") or r.startswith("SKIP missing")]
        print(f"  {section}: {len(rows)} actions, {len(bad)} skip/fail")
        for r in bad[:12]:
            print("   ", r)
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
