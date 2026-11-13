# Fig.1 本轮运行协议（重构重跑）

> 对应旧仓 step01 / 文章 Results·Step01。GSE145825 **不进**正式结果。

## 需求对齐（本轮硬约束）

1. **划分**：所有方法统一用 TripGRN 的 **未见细胞 + 未见基因对** 协议  
   - 10-fold CV；每折细胞列 **train:val:test = 5:2:3** 互不相交  
   - 共用各 GSE 的 `cv_splits_10fold.json`（`pair_seed=42`, `cv_base_seed=42`）  
   - 边/基因对层面分层 held-out（约 64/16/20）；TripGRN 负样本 1:1；DeepFGRN 仍用其固定负例池再 held-out 划分  
   - DeepFGRN：held-out 对后，**dGGAN 构图金标准仅用 train 正边**（架构不变；val/test 边不进嵌入）  
   - **负例**：Trip / Deep 均从正例基因宇宙抽取（source∈正例 TF；target∈正例出现过的基因），数量与正例 **1:1**（Deep 默认 `--num-negative 0`）  
   - DeepFGRN 旧结果归档：`old/DeepFGRN_cell_unseen_same_pairs/`、`old/DeepFGRN_heldout_pairs_fullgraph_emb/`、`old/DeepFGRN_trainpairs_emb_genome_neg/`（train-only 图 + 基因组负例，已停）  
   - 本轮 Deep 需重跑；Trip 正式图仍是旧负例，代码已改，重跑 Trip 后才一致  
2. **基因**：二分类基线（DeepTGI / scMGATGRN / GENELink）收缩到 **当前金标准网络边上映射到的基因**（KEGG≈3.5k / DoRothEA 随网而定）；`max_genes=0` 表示不再随机/方差补足。禁止随机 1500。TripGRN / DeepFGRN 保持既有全表达协议。  
3. **细胞**：TripGRN / DeepTGI / scMGATGRN / GENELink 使用 CV JSON 的 **全部 train/val/test 细胞**，禁止 `subsample_to_common_length`。scMGATGRN / GENELink 因 GAT `input_dim=细胞数`，仅对较短 split **右侧补零** 到 `max(n_train,n_val,n_test)`，不丢细胞。DeepFGRN CorrResNet 仍需等长输入（历史 subsample）。  
4. **计时**：每个 method×GSE×fold 记录 wall-clock（秒），写入 fold metrics / `runtime.json`，供方法间比对。  
5. **指标尽量一次记全**，避免作图时重跑（见下）。

## 方法与输出树

| 方法 | 任务形态 | 实验目录 |
|------|----------|----------|
| TripGRN | 三分类（可派生二分类） | `fig01_.../experiments/{kegg\|dorothea_abc}_epithelial_10cv/TripGRN/` |
| DeepFGRN | 三分类（可派生二分类） | `.../DeepFGRN/` |
| DeepTGI | 二分类为主 | `.../DeepTGI/` |
| scMGATGRN | 二分类为主 | `.../scMGATGRN/` |
| GENELink | 二分类为主 | `.../GENELink/` |

Summary 唯一真相：`fig01_pancancer_benchmark/experiments/**/summary/`（**不要**再写 `*_strict.csv`）。

## 队列（9）

`GSE103322, GSE132465, GSE148071, GSE156632, GSE159115, GSE176031, GSE232237, GSE253718, GSE288479`

排除：`GSE145825`（VCaP 细胞系）。

## 必记指标（文章 + 旧 summary 作图用到的并集）

### 三分类（TripGRN / DeepFGRN；其它方法若可输出三类则同样记）

整体：`accuracy`, `precision`/`precision_macro`, `recall`/`recall_macro`, `f1`/`f1_macro`, `mcc`, `macro_auc`, `auc_ovo`, `auprc_macro`  
逐类（NoReg / Activation / Repression）：`auc_ovr_*`, `auprc_*`, `specificity_*`, `acc_class_*`, 以及 precision/recall/F1  
另存：`confusion_matrix`, `per_class_order`

文章主文常用：**Macro-AUC**、Accuracy、Weighted F1、MCC、AUPRC、逐类 OvR Acc。

### 二分类（五方法可比；score = 1 − P(NoReg) 或方法原生二分类）

`accuracy`, `precision`, `recall`, `f1`, `mcc`, `auc`, `auprc`

作图：binary AUC/AUPRC 箱线、多指标雷达、三分类雷达/箱线。

### 运行元数据（每个 fold）

```json
{
  "wall_time_sec": 0.0,
  "train_time_sec": 0.0,
  "eval_time_sec": 0.0,
  "n_genes_used": 0,
  "gene_policy": "all_genes",
  "n_cells_train": 0,
  "n_cells_val": 0,
  "n_cells_test": 0,
  "sample_split": [5, 2, 3],
  "fold": 1,
  "cv_seed": 42,
  "pair_seed": 42,
  "device": "cuda:0"
}
```

## 显存风险（已知）

旧脚本注释：scMGATGRN / GENELink / DeepTGI 在全基因组上可能 OOM（原默认 `max_genes=1500`）。  
本轮仍默认全基因；若 OOM，只允许在 **不改划分、不随机抽基因** 的前提下做工程侧优化（batch、稀疏、梯度检查点等），并在该 fold 的 `runtime.json` 记录失败原因。不得回退到随机/方差截断 1500 作为正式结果。

## 路径

- 表达：`fig01_pancancer_benchmark/data/expression/epithelial/{GSE}/`  
- 网络/CDS：`shared/data/`（fig01 `data/network|features` 为链接）  
- 代码入口：`grn/data/pancancer_paths.py`（已指向 fig01）  
- Fig.1 启动：`fig01_pancancer_benchmark/scripts/run_fig01_tripgrn_kegg.py`（`smoke` / `run` / `summarize`）
- DeepFGRN 启动：`fig01_pancancer_benchmark/scripts/run_fig01_deepfgrn_kegg.py` / `run_fig01_deepfgrn_dorothea.py`（共用 CV + held-out pairs；CorrResNet 仍 `subsample_to_common_length`；回退旧协议加 `--same-pairs-all-splits`）
- 其余二分类：`fig01_pancancer_benchmark/scripts/run_fig01_binary_baselines.py`（DeepTGI / scMGATGRN / GENELink；全基因 + 全细胞）
