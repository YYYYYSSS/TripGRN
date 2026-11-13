# Fig.3 运行协议 — TCGA bulk cross-cancer（原始流程）

> 12 个 TCGA 癌种 · 10-fold 未见样本（5:2:3）· **不做 VAE / gene-disjoint**。

## 本轮设定（与旧仓 Step03 / bulk 10CV 对齐）

1. **数据**：`fig03_bulk_cross_cancer/data/expression/TCGA-*_FPKM_Symbol.csv` + `data/splits/*_cv_splits_10fold.json`
2. **金标准**：KEGG PPrel（activation / repression）
3. **划分**：每折样本列 train:val:test = 5:2:3；**随机 pair-split**（`--no-gene-disjoint`，基因可跨 split）
4. **表达编码器**：**Conv1d（legacy CNN）**，不是 VAE
5. **PPI**：**原始 STRING**（`ppi_string_v12_human_score900.csv`），不做 nokegg
6. **模态**：expr + **CDS + TSS** + PPI
7. **输出**：`fig03_bulk_cross_cancer/experiments/bulk_tcga_10cv_seed42/{TCGA-XXX}/fold_XX/`  
   Summary：同目录下 `summary/`（唯一真相，不写 `*_strict.csv`）

## 癌种列表（12）

TCGA-BRCA, TCGA-COAD, TCGA-HNSC, TCGA-KICH, TCGA-KIRC, TCGA-KIRP,  
TCGA-LGG, TCGA-LUAD, TCGA-LUSC, TCGA-PRAD, TCGA-THCA, TCGA-UCEC

## 启动

```bat
python fig03_bulk_cross_cancer\scripts\run_fig03_bulk_cross_cancer.py smoke
python fig03_bulk_cross_cancer\scripts\run_fig03_bulk_cross_cancer.py status
python fig03_bulk_cross_cancer\scripts\run_fig03_bulk_cross_cancer.py run --cancer ALL --fold ALL
python fig03_bulk_cross_cancer\scripts\run_fig03_bulk_cross_cancer.py summarize
REM 或
tools\run_fig03_bulk_cross_cancer.bat
```

可选（非默认）：`--expression-encoder vae`、`--gene-disjoint`、`--ppi-nokegg`、`--no-tss`。
