#!/usr/bin/env Rscript
# 部分 CellChat 验证：仅用 CellChatDB.human 配体-受体库（不做全数据 CCI 概率推断）。
# 1) 筛出「配体 ∈ 网络基因集」且「受体复合体至少一个亚基 ∈ 网络」的 LR 条目；
# 2) 与 differential_edges_used.csv 的 TF→Target 边做方向一致匹配（配体=TF，靶基因为受体亚基之一）。
#
# 依赖：BiocManager::install("CellChat")
#
# 用法：
#   Rscript metastasis_case_study_cellchat_partial_validation.R <case_study目录>

argv <- commandArgs(trailingOnly = TRUE)
if (length(argv) < 1) {
  stop("用法: Rscript metastasis_case_study_cellchat_partial_validation.R <case_study输出目录>", call. = FALSE)
}

case_dir <- normalizePath(argv[1], winslash = "/", mustWork = TRUE)

read_syms <- function(path) {
  if (!file.exists(path)) return(character(0))
  x <- readLines(path, warn = FALSE, encoding = "UTF-8")
  x <- trimws(x)
  x <- x[nzchar(x) & !grepl("^#", x)]
  unique(x)
}

split_complex <- function(s) {
  u <- trimws(unlist(strsplit(as.character(s), "_", fixed = TRUE)))
  unique(u[nzchar(u)])
}

if (!suppressPackageStartupMessages(requireNamespace("CellChat", quietly = TRUE))) {
  msg <- c(
    "未安装 CellChat，已跳过本步骤。",
    "安装: BiocManager::install('CellChat')",
    "或在 R 4.5: options(BioC_mirror='https://mirrors.tuna.tsinghua.edu.cn/bioconductor')"
  )
  writeLines(msg, file.path(case_dir, "cellchat_partial_SKIP.txt"))
  message(paste(msg, collapse = "\n"))
  quit(save = "no", status = 0)
}

suppressPackageStartupMessages(library(CellChat))
cdb <- CellChat::CellChatDB.human
inter <- cdb$interaction
if (!all(c("ligand", "receptor") %in% colnames(inter))) {
  stop("CellChatDB.human$interaction 缺少 ligand/receptor 列，请检查 CellChat 版本。", call. = FALSE)
}

pathway_col <- if ("pathway_name" %in% colnames(inter)) "pathway_name" else NA_character_
name_col <- if ("interaction_name" %in% colnames(inter)) "interaction_name" else NA_character_

net_genes <- read_syms(file.path(case_dir, "r_network_all_gene_symbols.txt"))
edge_file <- file.path(case_dir, "differential_edges_used.csv")

if (length(net_genes) == 0) {
  stop("缺少 r_network_all_gene_symbols.txt", call. = FALSE)
}

net_set <- net_genes

in_network_lr <- vapply(seq_len(nrow(inter)), function(i) {
  lig <- as.character(inter$ligand[i])
  if (!lig %in% net_set) return(FALSE)
  any(split_complex(inter$receptor[i]) %in% net_set)
}, logical(1))

lr_net <- inter[in_network_lr, , drop = FALSE]
out_lr <- file.path(case_dir, "cellchat_lr_in_network.csv")
utils::write.csv(lr_net, out_lr, row.names = FALSE, fileEncoding = "UTF-8")
message("CellChat LR ∩ 网络基因: ", nrow(lr_net), " -> ", basename(out_lr))

matches <- list()
if (file.exists(edge_file)) {
  edges <- utils::read.csv(edge_file, stringsAsFactors = FALSE, check.names = FALSE)
    if (!all(c("TF", "Target") %in% colnames(edges))) {
      message("警告: differential_edges_used.csv 无 TF/Target 列，跳过边匹配")
    } else {
      for (k in seq_len(nrow(edges))) {
        tf <- as.character(edges$TF[k])
        tg <- as.character(edges$Target[k])
        sub <- inter[as.character(inter$ligand) == tf, , drop = FALSE]
        if (nrow(sub) == 0) next
        for (j in seq_len(nrow(sub))) {
          if (!tg %in% split_complex(sub$receptor[j])) next
          row <- data.frame(
            TF = tf,
            Target = tg,
            interaction_name = if (!is.na(name_col)) as.character(sub[[name_col]][j]) else NA_character_,
            pathway_name = if (!is.na(pathway_col)) as.character(sub[[pathway_col]][j]) else NA_character_,
            ligand = tf,
            receptor = as.character(sub$receptor[j]),
            stringsAsFactors = FALSE
          )
          matches[[length(matches) + 1]] <- row
        }
      }
    }
}

out_m <- file.path(case_dir, "cellchat_lr_matches_differential_edges.csv")
mm <- if (length(matches) > 0) do.call(rbind, matches) else data.frame()
utils::write.csv(mm, out_m, row.names = FALSE, fileEncoding = "UTF-8")
if (nrow(mm) > 0) {
  message("LR 与差异边一致匹配: ", nrow(mm), " -> ", basename(out_m))
} else {
  message("LR 与差异边: 无匹配 -> 空 ", basename(out_m))
}

n_match <- nrow(mm)

# ---------- 出图：通路条数条形图 + 匹配边有向网络（非 CellChat 全量 chord，不依赖表达矩阵）----------
plot_out_bar <- file.path(case_dir, "cellchat_lr_in_network_pathway_barplot.png")
plot_out_net <- file.path(case_dir, "cellchat_lr_matches_network.png")

save_cellchat_partial_plots <- function() {
  ok_gg <- requireNamespace("ggplot2", quietly = TRUE)
  ok_ig <- requireNamespace("igraph", quietly = TRUE)
  if (!ok_gg) {
    message("未安装 ggplot2，跳过 CellChat 配图")
    return(invisible(NULL))
  }
  if (!is.na(pathway_col) && pathway_col %in% names(lr_net) && nrow(lr_net) > 0) {
    tb <- sort(table(lr_net[[pathway_col]]), decreasing = TRUE)
    nn <- min(20L, length(tb))
    if (nn > 0) {
      tb <- tb[seq_len(nn)]
      df <- data.frame(pathway = names(tb), n = as.integer(tb), stringsAsFactors = FALSE)
      df$pathway <- factor(df$pathway, levels = rev(unique(df$pathway)))
      p <- ggplot2::ggplot(df, ggplot2::aes(x = pathway, y = n)) +
        ggplot2::geom_col(fill = "#3498db", width = 0.85) +
        ggplot2::coord_flip() +
        ggplot2::labs(
          title = "CellChat LR in network genes (by pathway_name)",
          subtitle = paste0("n = ", nrow(lr_net), " interactions"),
          x = NULL,
          y = "Count"
        ) +
        ggplot2::theme_minimal(base_size = 11)
      tryCatch(
        {
          ggplot2::ggsave(plot_out_bar, p, width = 10, height = 7.5, dpi = 160, limitsize = FALSE)
          message("图: ", basename(plot_out_bar))
        },
        error = function(e) message("条形图失败: ", conditionMessage(e))
      )
    }
  }
  if (nrow(mm) < 1) {
    message("无 LR-差异边匹配，跳过网络图")
    return(invisible(NULL))
  }
  if (!ok_ig) {
    message("未安装 igraph，跳过匹配网络图（可 install.packages(\"igraph\")）")
    return(invisible(NULL))
  }
  el <- unique(mm[, c("TF", "Target")])
  colnames(el) <- c("from", "to")
  g <- igraph::graph_from_data_frame(el, directed = TRUE)
  vs <- igraph::V(g)$name
  lig_set <- unique(as.character(mm$TF))
  tgt_set <- unique(as.character(mm$Target))
  is_lig <- vs %in% lig_set
  is_tgt <- vs %in% tgt_set
  colv <- ifelse(is_lig & is_tgt, "#9b59b6", ifelse(is_lig, "#e74c3c", "#2ecc71"))
  tryCatch(
    {
      grDevices::png(plot_out_net, width = 2400, height = 2200, res = 220)
      graphics::par(mar = c(0.2, 0.2, 1.2, 0.2))
      set.seed(42)
      lo <- igraph::layout_with_fr(g)
      igraph::plot.igraph(
        g,
        layout = lo,
        vertex.size = 20,
        vertex.label.cex = 0.7,
        vertex.color = grDevices::adjustcolor(colv, alpha.f = 0.9),
        vertex.frame.color = grDevices::grey(0.3),
        edge.arrow.size = 0.35,
        edge.color = grDevices::grey(0.35),
        main = "CellChat LR matches: TF -> Target (ligand -> receptor subunit)"
      )
      graphics::legend(
        "bottomright",
        legend = c("Ligand (TF)", "Target only", "Both roles"),
        fill = c("#e74c3c", "#2ecc71", "#9b59b6"),
        cex = 0.85,
        bty = "n"
      )
      grDevices::dev.off()
      message("图: ", basename(plot_out_net))
    },
    error = function(e) {
      if (grDevices::dev.cur() != 1L) suppressWarnings(grDevices::dev.off())
      message("网络图失败: ", conditionMessage(e))
    }
  )
  invisible(NULL)
}

save_cellchat_partial_plots()

summ <- list(
  case_dir = case_dir,
  cellchat_version = as.character(utils::packageVersion("CellChat")),
  n_lr_interactions_total = nrow(inter),
  n_lr_ligand_in_network_and_receptor_touches_network = nrow(lr_net),
  n_differential_edges_matched_as_ligand_receptor = as.integer(n_match),
  outputs = list(
    lr_in_network_csv = basename(out_lr),
    matches_edges_csv = basename(out_m),
    pathway_barplot_png = basename(plot_out_bar),
    matches_network_png = basename(plot_out_net)
  ),
  note_zh = "验证含义：差异网络边若与 CellChat 文献 LR 一致，提示该 TF-靶关系与细胞间通讯配体-受体假设相容；非因果证明。"
)

if (requireNamespace("jsonlite", quietly = TRUE)) {
  jsonlite::write_json(summ, file.path(case_dir, "cellchat_partial_validation_summary.json"), pretty = TRUE, auto_unbox = TRUE)
  message("汇总: cellchat_partial_validation_summary.json")
}

message("CellChat 部分验证完成。")
