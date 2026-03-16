args <- commandArgs(trailingOnly=TRUE)
cell_name <- args[1]
library(Seurat)
library(biomaRt)
obj <- readRDS('Manatee/data/seurat_object_E3.5.rds')
obj <- NormalizeData(obj, normalization.method='LogNormalize', scale.factor=10000, verbose=FALSE)
mat <- as.matrix(LayerData(obj, layer='data'))
if (!(cell_name %in% colnames(mat))) {
  cat('ERROR: cell not found\n')
  quit(status=1)
}
cell_vec <- mat[, cell_name]
mart <- useMart('ensembl', dataset='mmusculus_gene_ensembl')
bm <- getBM(attributes=c('ensembl_gene_id','mgi_symbol'),
            filters='ensembl_gene_id', values=names(cell_vec), mart=mart)
bm <- bm[bm$mgi_symbol != '', ]
bm <- bm[!duplicated(bm$ensembl_gene_id), ]
sym_map <- setNames(bm$mgi_symbol, bm$ensembl_gene_id)
model_genes <- readLines('Manatee/GSE72857/processed/genes.txt')
out <- setNames(rep(0, length(model_genes)), model_genes)
mapped <- sym_map[names(cell_vec)]
keep <- !is.na(mapped)
tmp <- tapply(cell_vec[keep], mapped[keep], mean)
common <- intersect(names(tmp), model_genes)
out[common] <- tmp[common]
write.table(t(out), 'Manatee/data/tmp_cell.csv', quote=FALSE, row.names=FALSE, col.names=FALSE, sep=' ')
