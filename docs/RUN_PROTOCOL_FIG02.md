# Fig.2 运行协议 — GSE253718 模态消融

> 上皮子集 · 10-fold 未见样本（5:2:3）· 与 Fig.1 共用该 GSE 的 cell split 语义。

## 本轮设定（原始流程 / 对齐旧仓消融）

1. **数据**：`fig02_modality_ablation/data/expression/GSE253718_epithelial/` + `data/splits/cv_splits_10fold.json`
2. **金标准**：KEGG PPrel（activation / repression）
3. **划分**：每折细胞列 train:val:test = 5:2:3；**随机 pair-split**（`--no-gene-disjoint`，基因可跨 split）
4. **表达编码器**：**Conv1d / ResNet（legacy CNN）**，不是 VAE
5. **PPI**：**原始 STRING**（`ppi_string_v12_human_score900.csv`），不做 nokegg
6. **七阶段**（与旧仓命名一致）：

| stage | 模态 |
|-------|------|
| 01_expr_only | expr |
| 02_expr_cds | expr+cds |
| 03_expr_ppi | expr+ppi |
| 04_expr_cds_ppi | all |
| 05_cds_only | cds |
| 06_ppi_only | ppi |
| 07_cds_ppi | cds+ppi |

7. **输出**：`fig02_modality_ablation/experiments/gse253718_ablation_epithelial_10cv_seed42/{stage}/fold_XX/`  
   Summary：同目录下 `summary/`（唯一真相，不写 `*_strict.csv`）

> 先前 VAE + gene-disjoint + STRING∖KEGG 的试点结果请先挪到 `experiments/old/`，再重跑，避免 resume 跳过旧折。

## 启动

```bat
REM 0) 归档旧 VAE 试点（如已跑过）
move fig02_modality_ablation\experiments\gse253718_ablation_epithelial_10cv_seed42 fig02_modality_ablation\experiments\old\gse253718_ablation_epithelial_10cv_seed42_vae_gene_disjoint

REM 1) 冒烟 + 全量
python fig02_modality_ablation\scripts\run_fig02_modality_ablation.py smoke
python fig02_modality_ablation\scripts\run_fig02_modality_ablation.py run --stage ALL --fold ALL
python fig02_modality_ablation\scripts\run_fig02_modality_ablation.py summarize

REM 或
tools\run_fig02_modality_ablation.bat
```

可选试点（非默认）：`--expression-encoder vae`、`--gene-disjoint`、`--ppi-nokegg`。  
若目录未归档、需覆盖已有 metrics：加 `--force`。
