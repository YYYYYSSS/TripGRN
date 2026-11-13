# Fig.2 — GSE253718 modality ablation

Epithelial 10CV modality leave-in / leave-out for TripGRN.

## Protocol

See [`docs/RUN_PROTOCOL_FIG02.md`](../docs/RUN_PROTOCOL_FIG02.md).

Defaults this round (**original**):

- expression **conv** (ResNet/CNN; not VAE)
- **legacy** random pair-split (`--no-gene-disjoint`)
- PPI = **raw STRING** (`ppi_string_v12_human_score900.csv`)
- 7 stages × 10 folds → `experiments/gse253718_ablation_epithelial_10cv_seed42/`

## Quick start

```bat
REM Archive prior VAE pilot if present, then:
python fig02_modality_ablation\scripts\run_fig02_modality_ablation.py smoke
python fig02_modality_ablation\scripts\run_fig02_modality_ablation.py status
python fig02_modality_ablation\scripts\run_fig02_modality_ablation.py run --stage ALL --fold ALL
python fig02_modality_ablation\scripts\run_fig02_modality_ablation.py summarize
REM or
tools\run_fig02_modality_ablation.bat
```

Resume-safe: folds with `unseen_samples_test_metrics.json` are skipped (use `--force` to overwrite).
