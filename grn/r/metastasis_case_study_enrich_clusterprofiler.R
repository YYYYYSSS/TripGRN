#!/usr/bin/env Rscript
# 脑转移案例分析：hub 基因 GO/KEGG 富集 + enrichment_summary.json + 点图（dotplot）
#
# 用法：
#   Rscript metastasis_case_study_enrich_clusterprofiler.R <case_study目录> [--network-universe]
#
# 依赖：
#   install.packages(c("jsonlite", "ggplot2"))
#   BiocManager::install(c("clusterProfiler", "org.Hs.eg.db", "enrichplot"))
#
# 说明：新版 enrichplot 不再导出 barplot()，统一用 dotplot()。
# 内网若出现 yulab md5.txt / GitHub 联网警告，本脚本对富集与作图段使用 suppressWarnings 尽量压制。

suppressPackageStartupMessages({
  ok <- requireNamespace("clusterProfiler", quietly = TRUE) &&
    requireNamespace("org.Hs.eg.db", quietly = TRUE)
})
if (!ok) {
  stop(
    "缺少 R 包：clusterProfiler, org.Hs.eg.db\n",
    "  BiocManager::install(c('clusterProfiler', 'org.Hs.eg.db'))\n",
    call. = FALSE
  )
}

argv <- commandArgs(trailingOnly = TRUE)
use_net_uni <- "--network-universe" %in% argv
argv <- argv[argv != "--network-universe"]

if (length(argv) < 1) {
  stop(
    "用法: Rscript metastasis_case_study_enrich_clusterprofiler.R <case_study输出目录> [--network-universe]",
    call. = FALSE
  )
}

case_dir <- normalizePath(argv[1], winslash = "/", mustWork = TRUE)

read_syms <- function(path) {
  x <- readLines(path, warn = FALSE, encoding = "UTF-8")
  x <- trimws(x)
  x <- x[nzchar(x) & !grepl("^#", x)]
  unique(x)
}

net_all_file <- file.path(case_dir, "r_network_all_gene_symbols.txt")
enrich_file <- file.path(case_dir, "r_enrich_gene_symbols.txt")
hub_file <- file.path(case_dir, "r_hub_gene_symbols.txt")

sym_query <- character(0)
if (file.exists(net_all_file)) {
  sym_query <- read_syms(net_all_file)
  message("富集：使用差异网络全部节点 r_network_all_gene_symbols.txt, n=", length(sym_query))
} else if (file.exists(enrich_file)) {
  sym_query <- read_syms(enrich_file)
  message("富集：使用 r_enrich_gene_symbols.txt, n=", length(sym_query))
} else if (file.exists(hub_file)) {
  sym_query <- read_syms(hub_file)
  message("富集：使用 hub 列表 r_hub_gene_symbols.txt, n=", length(sym_query))
}
if (length(sym_query) < 2) {
  stop("查询基因少于 2 个；请确认 case tables 下存在 r_network_all_gene_symbols.txt 或 r_hub_gene_symbols.txt", call. = FALSE)
}

sym_uni <- NULL
if (use_net_uni) {
  netf <- file.path(case_dir, "r_network_all_gene_symbols.txt")
  if (!file.exists(netf)) stop("缺少 ", netf, call. = FALSE)
  sym_uni <- read_syms(netf)
}

suppressPackageStartupMessages({
  library(clusterProfiler)
  library(org.Hs.eg.db)
})

map_sym <- function(syms) {
  suppressMessages(
    bitr(syms, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
  )
}

qmap <- map_sym(sym_query)
if (nrow(qmap) < 2) {
  stop("SYMBOL -> ENTREZID 匹配过少。", call. = FALSE)
}

universe_entrez <- NULL
if (!is.null(sym_uni)) {
  umap <- map_sym(sym_uni)
  universe_entrez <- unique(umap$ENTREZID)
}

out_bp <- file.path(case_dir, "enrich_R_GO_BP.csv")
out_mf <- file.path(case_dir, "enrich_R_GO_MF.csv")
out_cc <- file.path(case_dir, "enrich_R_GO_CC.csv")
out_kegg <- file.path(case_dir, "enrich_R_KEGG.csv")
out_summary <- file.path(case_dir, "enrichment_summary.json")
out_plot_bp <- file.path(case_dir, "enrich_R_GO_BP_top15_dotplot.png")
out_plot_kegg <- file.path(case_dir, "enrich_R_KEGG_top15_dotplot.png")

top_term_rows <- function(df, n = 12) {
  if (is.null(df) || nrow(df) == 0) {
    return(data.frame())
  }
  nn <- min(n, nrow(df))
  df[seq_len(nn), intersect(names(df), c("ID", "Description", "pvalue", "p.adjust", "qvalue", "Count", "GeneRatio")), drop = FALSE]
}

df_rows_to_list <- function(df) {
  if (is.null(df) || nrow(df) == 0) {
    return(list())
  }
  lapply(seq_len(nrow(df)), function(i) as.list(df[i, , drop = FALSE]))
}

run_go <- function(ont, out_csv) {
  eg <- suppressWarnings(
    enrichGO(
      gene = qmap$ENTREZID,
      OrgDb = org.Hs.eg.db,
      keyType = "ENTREZID",
      ont = ont,
      pAdjustMethod = "BH",
      pvalueCutoff = 0.05,
      qvalueCutoff = 0.2,
      readable = TRUE,
      universe = universe_entrez
    )
  )
  df <- as.data.frame(eg)
  if (nrow(df) == 0) {
    message("GO ", ont, ": 无显著项")
    write.csv(df, out_csv, row.names = FALSE, fileEncoding = "UTF-8")
    return(list(n = 0, eg = NULL, df = df))
  }
  write.csv(df, out_csv, row.names = FALSE, fileEncoding = "UTF-8")
  message("GO ", ont, ": ", nrow(df), " 行 -> ", basename(out_csv))
  return(list(n = nrow(df), eg = eg, df = df))
}

save_enrich_dotplot <- function(eg, out_png, title_suffix, n_show = 15) {
  if (is.null(eg)) return(invisible(FALSE))
  dfeg <- tryCatch(as.data.frame(eg), error = function(e) data.frame())
  if (nrow(dfeg) == 0) return(invisible(FALSE))
  if (!requireNamespace("enrichplot", quietly = TRUE) || !requireNamespace("ggplot2", quietly = TRUE)) {
    message("跳过作图（需 enrichplot + ggplot2）: ", basename(out_png))
    return(invisible(FALSE))
  }
  ok <- tryCatch(
    {
      # 新版 enrichplot：用 dotplot(enrichResult)；barplot 已非导出接口
      p <- suppressWarnings(enrichplot::dotplot(eg, showCategory = n_show))
      p <- p + ggplot2::labs(title = title_suffix)
      ggplot2::ggsave(out_png, p, width = 11, height = 8, dpi = 160, limitsize = FALSE)
      message("点图: ", basename(out_png))
      TRUE
    },
    error = function(e) {
      message("作图跳过 ", basename(out_png), ": ", conditionMessage(e))
      FALSE
    }
  )
  invisible(ok)
}

message("case_dir=", case_dir)
message("n_query_genes=", length(sym_query), " mapped_ENTREZID=", nrow(qmap))

r_bp <- run_go("BP", out_bp)
r_mf <- run_go("MF", out_mf)
r_cc <- run_go("CC", out_cc)

kegg_n <- 0L
kdf <- data.frame()
ek_obj <- NULL
kegg_err <- NULL
tryCatch(
  {
    ek_obj <- suppressWarnings(
      enrichKEGG(
        gene = qmap$ENTREZID,
        organism = "hsa",
        pAdjustMethod = "BH",
        pvalueCutoff = 0.05,
        qvalueCutoff = 0.2,
        universe = universe_entrez
      )
    )
    kdf <- as.data.frame(ek_obj)
    kegg_n <- nrow(kdf)
    if (kegg_n == 0) {
      message("KEGG: 无显著项")
    } else {
      message("KEGG: ", kegg_n, " 行")
    }
    write.csv(kdf, out_kegg, row.names = FALSE, fileEncoding = "UTF-8")
  },
  error = function(e) {
    kegg_err <<- conditionMessage(e)
    message("KEGG 跳过: ", kegg_err)
    write.csv(data.frame(), out_kegg, row.names = FALSE, fileEncoding = "UTF-8")
    ek_obj <<- NULL
    kdf <<- data.frame()
    kegg_n <<- 0L
  }
)
kegg_top <- top_term_rows(kdf, 15)

# GO-BP / KEGG 点图
suppressWarnings({
  save_enrich_dotplot(r_bp$eg, out_plot_bp, "GO Biological Process (diff network genes)", 15)
  save_enrich_dotplot(ek_obj, out_plot_kegg, "KEGG pathways (diff network genes)", 15)
})

summ <- list(
  case_dir = case_dir,
  network_universe = use_net_uni,
  n_query_symbols = length(sym_query),
  n_query_entrez = nrow(qmap),
  n_hub_symbols = length(sym_query),
  n_hub_entrez = nrow(qmap),
  plots = list(
    GO_BP_dotplot = basename(out_plot_bp),
    KEGG_dotplot = basename(out_plot_kegg)
  ),
  GO_BP = list(
    n_significant = r_bp$n,
    csv = basename(out_bp),
    top_terms = df_rows_to_list(top_term_rows(r_bp$df, 15))
  ),
  GO_MF = list(
    n_significant = r_mf$n,
    csv = basename(out_mf),
    top_terms = df_rows_to_list(top_term_rows(r_mf$df, 15))
  ),
  GO_CC = list(
    n_significant = r_cc$n,
    csv = basename(out_cc),
    top_terms = df_rows_to_list(top_term_rows(r_cc$df, 15))
  ),
  KEGG = list(
    n_significant = as.integer(kegg_n),
    csv = basename(out_kegg),
    top_terms = df_rows_to_list(kegg_top)
  )
)
if (!is.null(kegg_err)) {
  summ$KEGG$error <- kegg_err
}

if (requireNamespace("jsonlite", quietly = TRUE)) {
  jsonlite::write_json(summ, out_summary, pretty = TRUE, auto_unbox = TRUE)
  message("汇总: ", basename(out_summary))
} else {
  warning("未安装 jsonlite，跳过 enrichment_summary.json； install.packages('jsonlite')")
}

message("完成: enrich_R_*.csv, enrichment_summary.json, enrich_R_GO_BP_top15_dotplot.png, enrich_R_KEGG_top15_dotplot.png")
