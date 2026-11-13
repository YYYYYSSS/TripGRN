# Fig.3 — TCGA bulk cross-cancer（原始流程）

12 TCGA cohorts × 10-fold unseen-sample CV (5:2:3).

## Protocol

See `docs/RUN_PROTOCOL_FIG03.md`.

Defaults (**no VAE**):

- expression encoder = **conv** (legacy CNN)
- pair split = **no gene-disjoint**
- modalities = expr + CDS + TSS + **raw STRING** PPI

## Layout

```
fig03_bulk_cross_cancer/
  data/expression/TCGA-*_FPKM_Symbol.csv
  data/splits/*_cv_splits_10fold.json
  scripts/run_fig03_bulk_cross_cancer.py
  experiments/bulk_tcga_10cv_seed42/{TCGA-XXX}/fold_XX/
  experiments/bulk_tcga_10cv_seed42/summary/
```

## Quick start

```bat
python fig03_bulk_cross_cancer\scripts\run_fig03_bulk_cross_cancer.py smoke
python fig03_bulk_cross_cancer\scripts\run_fig03_bulk_cross_cancer.py run --cancer ALL --fold ALL
python fig03_bulk_cross_cancer\scripts\run_fig03_bulk_cross_cancer.py summarize
REM or: tools\run_fig03_bulk_cross_cancer.bat
```
