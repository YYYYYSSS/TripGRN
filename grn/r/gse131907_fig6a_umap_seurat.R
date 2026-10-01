#!/usr/bin/env Rscript
# Fig. 6A UMAP in Seurat style (R): all epithelial tLung vs mBrain cells, no downsample.
#
# Usage:
#   Rscript grn/r/gse131907_fig6a_umap_seurat.R
#
# Requires: Seurat, ggplot2, rhdf5, jsonlite

suppressPackageStartupMessages({
  library(Seurat)
  library(ggplot2)
  library(rhdf5)
  library(jsonlite)
})

root <- "E:/project/project_GRN"
h5_path <- file.path(root, "experiments/gse131907_transfer_from_gse253718_epi_seed2/umap_input_tlung_mbrain_epi.h5")
exp_dir <- file.path(root, "experiments/gse131907_transfer_from_gse253718_epi_seed2")
fig_dir <- file.path(exp_dir, "case_tlung_vs_mbrain_kegg_khop_k1/figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(h5_path)) {
  stop("Missing HDF5 input: ", h5_path, "\nRun: python -m grn.scripts.export_gse131907_umap_h5", call. = FALSE)
}

message("[load] ", h5_path)
genes <- as.character(h5read(h5_path, "genes"))
compartment <- as.character(h5read(h5_path, "compartment"))
lognorm <- h5read(h5_path, "lognorm")
storage.mode(lognorm) <- "double"
# rhdf5 returns Python row-major arrays transposed relative to genes x cells
if (nrow(lognorm) == length(compartment) && ncol(lognorm) == length(genes)) {
  message("[load] transpose rhdf5 array to genes x cells")
  lognorm <- t(lognorm)
}
stopifnot(nrow(lognorm) == length(genes), ncol(lognorm) == length(compartment))
rownames(lognorm) <- make.unique(genes)
colnames(lognorm) <- paste0(compartment, "_", seq_along(compartment))
message(sprintf("[load] %d genes x %d cells", nrow(lognorm), ncol(lognorm)))
print(table(compartment))

# Approximate counts from log1p-normalized values for vst feature selection
counts_approx <- expm1(lognorm)
counts_approx[counts_approx < 0] <- 0

meta <- data.frame(compartment = compartment, row.names = colnames(lognorm))

message("[seurat] CreateSeuratObject ...")
obj <- CreateSeuratObject(counts = counts_approx, meta.data = meta, min.cells = 0, min.features = 0)
obj[["RNA"]] <- SetAssayData(obj[["RNA"]], layer = "data", new.data = lognorm)

message("[seurat] FindVariableFeatures(vst, 2000) / ScaleData / RunPCA / RunUMAP ...")
set.seed(2)
DefaultAssay(obj) <- "RNA"
obj <- FindVariableFeatures(obj, selection.method = "vst", nfeatures = 2000, verbose = TRUE)
obj <- ScaleData(obj, features = VariableFeatures(obj), verbose = TRUE)
obj <- RunPCA(obj, features = VariableFeatures(obj), npcs = 50, verbose = TRUE)
obj <- RunUMAP(
  obj,
  reduction = "pca",
  dims = 1:30,
  n.neighbors = 30,
  min.dist = 0.3,
  metric = "cosine",
  seed.use = 2,
  verbose = TRUE
)

umap <- Embeddings(obj, reduction = "umap")
coords <- data.frame(
  cell_idx = rownames(umap),
  compartment = as.character(obj$compartment),
  umap1 = umap[, 1],
  umap2 = umap[, 2],
  check.names = FALSE
)
coords_path <- file.path(exp_dir, "umap_tlung_mbrain_epi_coords.csv")
utils::write.csv(coords, coords_path, row.names = FALSE, fileEncoding = "UTF-8")
message("[write] ", coords_path, " n=", nrow(coords))
print(table(coords$compartment))

obj$compartment <- factor(as.character(obj$compartment), levels = c("tLung", "mBrain"))
p <- DimPlot(
  obj,
  reduction = "umap",
  group.by = "compartment",
  pt.size = 0.3,
  shuffle = TRUE,
  seed = 2
) +
  scale_color_manual(
    values = c(tLung = "#4E79A7", mBrain = "#E15759"),
    breaks = c("tLung", "mBrain"),
    labels = c(
      sprintf("tLung (n=%s)", format(sum(coords$compartment == "tLung"), big.mark = ",")),
      sprintf("mBrain (n=%s)", format(sum(coords$compartment == "mBrain"), big.mark = ","))
    )
  ) +
  ggtitle("GSE131907 epithelial cells: tLung vs mBrain (Fig. 6A)") +
  theme_bw(base_size = 12) +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", hjust = 0.5, size = 12),
    legend.title = element_blank(),
    legend.position = "right",
    aspect.ratio = 1
  ) +
  labs(x = "UMAP_1", y = "UMAP_2")

pdf_path <- file.path(fig_dir, "fig6A_tlung_mbrain_epi_umap.pdf")
png_path <- file.path(fig_dir, "fig6A_tlung_mbrain_epi_umap.png")
ggsave(pdf_path, p, width = 6.5, height = 5.5, units = "in")
ggsave(png_path, p, width = 6.5, height = 5.5, units = "in", dpi = 300)
message("[write] ", pdf_path)
message("[write] ", png_path)

meta_out <- list(
  method = "Seurat",
  selection.method = "vst",
  nfeatures = 2000,
  npcs = 50,
  umap_dims = "1:30",
  n.neighbors = 30,
  min.dist = 0.3,
  metric = "cosine",
  seed = 2,
  no_downsample = TRUE,
  n_cells = nrow(coords),
  n_tLung = sum(coords$compartment == "tLung"),
  n_mBrain = sum(coords$compartment == "mBrain"),
  h5 = h5_path
)
write_json(meta_out, file.path(exp_dir, "umap_tlung_mbrain_epi_seurat_meta.json"), auto_unbox = TRUE, pretty = TRUE)
message("[done]")
