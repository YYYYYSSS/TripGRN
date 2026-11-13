# AGENTS.md - Cursor working rules (TripGRN refactor)

Follow these rules in this repo.

## Repo identity

- This repo: F:/博士阶段项目/TripGRN重构 (or the path you opened)
- Legacy repo: E:/project/project_GRN - READ ONLY. Do not edit/delete/overwrite.
- Legacy path is in LEGACY_ROOT.txt

## Read first

1. README.md
2. docs/DATA_LAYOUT.md
3. docs/LEGACY_MAP.md
4. This file AGENTS.md

## Dataset exclusion (important)

Do NOT use GSE145825.
It is a prostate cancer CELL LINE cohort (VCaP), not tissue primary samples.
Legacy boxplots/radars already excluded it.

- Exclude GSE145825 from Fig.1 and all formal plots/summaries/pan-means
- Do not include it in mainline pan-cancer formal results (junction may still exist as leftover)
- Keep other GEO tissue cohorts and GSE253718

中文补充：GSE145825 是细胞系数据，正式结果一律不要。

## Working principles

1. Organize by figure folders figXX_*/
2. Keep light: link large matrices; do not git-commit checkpoints / huge npy
3. Single source of truth: summary CSV under figXX_*/experiments/**/summary/
4. Finish Fig.1 first before other figures
5. Prefer pointing grn/data/*_paths.py to figXX_*/data or shared/data; data/ shim is temporary

## Do not

- Modify the legacy repo
- Use GSE145825 in formal results
- Rebuild flat messy data/exp dumps
- Create dual metric files like *_strict.csv
- Merge cell-type step06 into main-figure folders without user confirmation

## Commands

python tools/bootstrap_from_legacy.py

## Recent context

Legacy step01 already has strict binary AUC/AUPRC for Trip/Deep, per-class radars, binary_auc boxplots, and GSE145825 removed from displays. This refactor rebuilds by Fig.1-5.

**This-round Fig.1 prep (see `docs/RUN_PROTOCOL_FIG01.md`):**
- All methods: TripGRN unseen-sample 10CV (5:2:3), shared CV splits
- All genes (no 1500-gene subset); record wall-clock per fold
- Comprehensive metrics once (no `*_strict.csv`); summary under `fig01_*/experiments/**/summary/`
- GSE145825 junctions removed from fig01; formal list = 9 GSEs


## Style

Be concise. Small reproducible changes. Commit only when asked.
