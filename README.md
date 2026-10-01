# TripGRN refactor (by manuscript figure)

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
