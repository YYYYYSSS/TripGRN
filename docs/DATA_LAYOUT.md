# 路径与数据布局

## 原则

按图组织；避免旧仓 `data/exp` 平铺大文件 + 结果混放。

## 布局

```
shared/data/
  network/grn/     # kegg / dorothea_abc / trrust
  network/ppi/
  features/cds/

figXX_*/data/
  expression/      # 表达（junction/hardlink 到旧仓）
  splits/          # 划分
  network/ -> shared
  features/ -> shared
  raw/             # 可选原始

data/              # 临时 shim（junction）；正式路径优先 figXX + shared
```

## Fig.1 表达

- 正式列表：9 个 GSE（**不含 GSE145825**），见 `fig01_.../data/expression/pancancer_manifest.json`
- 每队列：`values.npy` / `genes.npy` / `cells.npy` / `cv_splits_10fold.json`

## 大文件

- GB 级用 junction / hardlink，勿 git 提交
- 小 CSV/JSON 可放 `shared/`

## Bootstrap

```bat
python tools\bootstrap_from_legacy.py
```

日志：`docs/bootstrap_log.json`

本轮协议：`docs/RUN_PROTOCOL_FIG01.md`
