# ?? data ?????

????????? `LEGACY_ROOT.txt`??? `E:\project\project_GRN`??

## ????

| ??? | ??? |
|--------|--------|
| `shared/data/network/grn/kegg_pprel_activation_inhibition.csv` | `data/network/grn/KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv` |
| `shared/data/network/grn/dorothea_abc_activation_inhibition.csv` | `data/network/grn/DoRothEA_human_ABC_activation_inhibition_with_symbols.csv` |
| `shared/data/network/grn/trrust_activation_inhibition.csv` | `data/network/grn/TRRUST_human_activation_inhibition_with_symbols.csv` |
| `shared/data/network/ppi/ppi_string_v12_human_score900.csv` | `data/network/ppi/ppi_string_v12_human_score900.csv` |
| `shared/data/features/cds/cds_gene_feature_k4.csv` | `data/CDS/features/gene_shared/cds_gene_feature_k4.csv` |

## ????

| ? | ? | ? |
|----|----|----|
| Fig.1 | `fig01_.../data/expression/epithelial/{GSE}/` | `data/scRNA/pancancer/epithelial/{GSE}/` |
| Fig.1 | `fig01_.../data/expression/full/{GSE}/` | `data/scRNA/pancancer/full/{GSE}/`?GSE253718 full ????? |
| Fig.2 | `fig02_.../data/expression/GSE253718_epithelial/{values,genes,cells}.npy` | `data/exp/GSE253718_epithelial_counts_*.npy` |
| Fig.2 | `fig02_.../data/splits/cv_splits_10fold.json` | `data/exp/GSE253718_epithelial_cv_splits_10fold.json` |
| Fig.3 | `fig03_.../data/expression/TCGA-*_FPKM_Symbol.csv` | `data/exp/12bulk/` |
| Fig.3 | `fig03_.../data/splits/*_cv_splits_10fold.json` | `data/external_validation/cv_splits/` |
| Fig.4 | `fig04_.../data/source/GSE253718_*.npy` | `data/exp/GSE253718_*.npy` |
| Fig.4 | `fig04_.../data/target/GSE189357/` | `data/exp/GSE189357/` |
| Fig.5 | `fig05_.../data/expression/GSE131907/` | `data/exp/GSE131907/` |
| Fig.5 | `fig05_.../data/derived/GSE131907_other/` | `data/other/GSE131907/` |

## ???????

- GSE123902 ??
- `data/exp/pancancer_scdata_step5/`??? scRNA/pancancer ???
- ??????? bulk ???????? CSV

## 排除数据集

- **GSE145825**：细胞系（VCaP），不进主线图表与 pan-mean。重构仓库 fig01 已去掉该 junction；正式列表见 `fig01_.../data/expression/pancancer_manifest.json`（9 队列）。
