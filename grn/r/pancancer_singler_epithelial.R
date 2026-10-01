#!/usr/bin/env Rscript
# SingleR epithelial subset for pancancer GSE (unified step01 pipeline).
# Usage: Rscript pancancer_singler_epithelial.R <singler_input_dir> <out_dir>
#
# Input dir (from Python): matrix.mtx, genes.tsv, barcodes.tsv  (genes x cells, logcounts)
# Requires: SingleR, celldex, SummarizedExperiment, Matrix, jsonlite

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript pancancer_singler_epithelial.R <singler_input_dir> <out_dir>")
}
input_dir <- args[[1]]
out_dir <- args[[2]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(SingleR)
  library(celldex)
  library(SummarizedExperiment)
  library(Matrix)
  library(jsonlite)
})

mtx_path <- file.path(input_dir, "matrix.mtx")
genes_path <- file.path(input_dir, "genes.tsv")
barcodes_path <- file.path(input_dir, "barcodes.tsv")
for (p in c(mtx_path, genes_path, barcodes_path)) {
  if (!file.exists(p)) {
    stop("Missing required file: ", p)
  }
}

mat <- readMM(mtx_path)
genes <- read.delim(genes_path, header = FALSE, stringsAsFactors = FALSE)[[1]]
cells <- read.delim(barcodes_path, header = FALSE, stringsAsFactors = FALSE)[[1]]

if (nrow(mat) != length(genes)) {
  stop(sprintf("gene count mismatch: matrix rows=%d genes.tsv=%d", nrow(mat), length(genes)))
}
if (ncol(mat) != length(cells)) {
  stop(sprintf("cell count mismatch: matrix cols=%d barcodes.tsv=%d", ncol(mat), length(cells)))
}

rownames(mat) <- genes
colnames(mat) <- cells
sce <- SummarizedExperiment(assays = list(logcounts = mat))

ref <- celldex::HumanPrimaryCellAtlasData()
pred <- SingleR(
  test = sce,
  ref = ref,
  labels = ref$label.main,
  assay.type.test = "logcounts",
  de.method = "wilcox"
)

labels <- pred$labels
pruned <- pred$pruned.labels
pruned[is.na(pruned)] <- labels[is.na(pruned)]

is_epithelial <- grepl(
  "pithel|Keratinocyte|Hepatocyte|Mammary|Ductal|Enterocyte|Cholangiocyte",
  pruned,
  ignore.case = TRUE
)

cell_ids <- colnames(sce)
if (is.null(cell_ids)) cell_ids <- paste0("cell_", seq_len(ncol(sce)))

out_labels <- data.frame(
  cell_id = cell_ids,
  singler_label = labels,
  singler_pruned = pruned,
  is_epithelial = is_epithelial,
  stringsAsFactors = FALSE
)
write.csv(out_labels, file.path(out_dir, "singler_labels.csv"), row.names = FALSE)

if (!is.null(pred$scores)) {
  write.csv(as.data.frame(pred$scores), file.path(out_dir, "singler_scores.csv"))
}

epi_idx <- which(is_epithelial) - 1L  # 0-based for Python
writeLines(as.character(epi_idx), file.path(out_dir, "epithelial_indices_0based.txt"))

report <- list(
  method = "SingleR",
  reference = "HumanPrimaryCellAtlasData",
  input_format = "matrix.mtx+tsv",
  n_cells_total = length(cell_ids),
  n_cells_epithelial = sum(is_epithelial),
  epithelial_fraction = if (length(cell_ids) > 0) sum(is_epithelial) / length(cell_ids) else 0,
  pruned_label_table = as.list(sort(table(pruned), decreasing = TRUE))
)
write(toJSON(report, auto_unbox = TRUE, pretty = TRUE),
      file.path(out_dir, "singler_report.json"))

cat(sprintf(
  "[SingleR] %s: epithelial %d / %d (%.1f%%)\n",
  basename(input_dir),
  sum(is_epithelial), length(cell_ids),
  100 * sum(is_epithelial) / max(length(cell_ids), 1)
))
