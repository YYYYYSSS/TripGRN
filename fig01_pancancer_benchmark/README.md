# Fig.1 - Pan-cancer epithelial 10CV (KEGG + DoRothEA)

Folders: `data/`, `scripts/`, `experiments/`, `figures/`

## This-round protocol

See `docs/RUN_PROTOCOL_FIG01.md`.

- 9 tissue cohorts (GSE145825 excluded)
- Unseen-sample 10CV, cell split 5:2:3, shared `cv_splits_10fold.json`
- All methods: **all genes** + full metrics + wall-clock timing
- Experiments: `experiments/{kegg|dorothea_abc}_epithelial_10cv/`

## Commands (KEGG TripGRN)

```bat
REM path smoke (no training)
python fig01_pancancer_benchmark\scripts\run_fig01_tripgrn_kegg.py smoke

REM one fold (use CUDA env)
python fig01_pancancer_benchmark\scripts\run_fig01_tripgrn_kegg.py run --gse GSE156632 --fold 1

REM all GSE / all folds
python fig01_pancancer_benchmark\scripts\run_fig01_tripgrn_kegg.py run --gse ALL --fold ALL

REM summarize
python fig01_pancancer_benchmark\scripts\run_fig01_tripgrn_kegg.py summarize
```

## Commands (DoRothEA-ABC TripGRN)

Same shared epithelial `cv_splits_10fold.json` as KEGG. Outputs:
`experiments/dorothea_abc_epithelial_10cv/`.

```bat
conda activate project_GRN
python fig01_pancancer_benchmark\scripts\run_fig01_tripgrn_dorothea.py smoke
python fig01_pancancer_benchmark\scripts\run_fig01_tripgrn_dorothea.py run --gse ALL --fold ALL
python fig01_pancancer_benchmark\scripts\run_fig01_tripgrn_dorothea.py summarize
```

## Commands (KEGG DeepFGRN)

Same shared `cv_splits_10fold.json` as TripGRN (5:2:3). DeepFGRN then aligns train/val/test lengths via `subsample_to_common_length` (CorrResNet constraint).

**Windows native TF is CPU-only.** Prefer WSL env `deepfgrn_wsl` (TF 2.21 + CUDA, RTX A4000 verified):

```bat
tools\wsl_run_deepfgrn.bat smoke
tools\wsl_run_deepfgrn.bat run --gse GSE156632 --fold 1
tools\wsl_run_deepfgrn.bat run --gse ALL --fold ALL
tools\wsl_run_deepfgrn.bat summarize
```

Or inside Ubuntu-22.04 WSL:

```bash
source /home/yhr/activate_deepfgrn_wsl.sh
python -u fig01_pancancer_benchmark/scripts/run_fig01_deepfgrn_kegg.py smoke
```

Optional: `export DEEPFGRN_ROOT=/mnt/e/project/project_DeepFGRN` (auto-detected if present via Windows path mapping).

## Commands (DoRothEA-ABC DeepFGRN)

Same shared CV as TripGRN DoRothEA. Outputs:
`experiments/dorothea_abc_epithelial_10cv/DeepFGRN/`.

```bat
tools\wsl_run_deepfgrn_dorothea.bat smoke
tools\wsl_run_deepfgrn_dorothea.bat run --gse ALL --fold ALL
tools\wsl_run_deepfgrn_dorothea.bat summarize
```

Or inside WSL:

```bash
source /home/yhr/activate_deepfgrn_wsl.sh
python -u fig01_pancancer_benchmark/scripts/run_fig01_deepfgrn_dorothea.py smoke
python -u fig01_pancancer_benchmark/scripts/run_fig01_deepfgrn_dorothea.py run --gse ALL --fold ALL
python -u fig01_pancancer_benchmark/scripts/run_fig01_deepfgrn_dorothea.py summarize
```

## Commands (remaining binary methods)

DeepTGI / scMGATGRN / GENELink:

- genes = **current GRN-mapped set** (KEGG / DoRothEA); `--max-genes 0` (no random fill)
- **full CV cells** (no subsample); scMGAT / GENELink zero-pad for GAT `input_dim`
- shared `cv_splits_10fold.json` (5:2:3); 9 tissue GSEs

One-shot after Trip/Deep (both nets) + DeepTGI-KEGG are done (resume-safe: skip finished folds):

```bat
tools\run_fig01_remaining_baselines.bat
```

Log: `fig01_pancancer_benchmark\experiments\run_fig01_remaining_baselines.log`

Manual (same launcher):

```bat
conda activate project_GRN
python fig01_pancancer_benchmark\scripts\run_fig01_binary_baselines.py smoke --method ALL --network kegg
python fig01_pancancer_benchmark\scripts\run_fig01_binary_baselines.py run --method scMGATGRN --network kegg --gse ALL --fold ALL
python fig01_pancancer_benchmark\scripts\run_fig01_binary_baselines.py run --method GENELink --network kegg --gse ALL --fold ALL
python fig01_pancancer_benchmark\scripts\run_fig01_binary_baselines.py run --method DeepTGI --network dorothea --gse ALL --fold ALL
python fig01_pancancer_benchmark\scripts\run_fig01_binary_baselines.py run --method scMGATGRN --network dorothea --gse ALL --fold ALL
python fig01_pancancer_benchmark\scripts\run_fig01_binary_baselines.py run --method GENELink --network dorothea --gse ALL --fold ALL
```