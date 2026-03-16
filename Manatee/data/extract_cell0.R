library(Seurat)
library(biomaRt)
obj <- readRDS('Manatee/data/seurat_object_E3.5.rds')
obj <- NormalizeData(obj, normalization.method='LogNormalize', scale.factor=10000, verbose=FALSE)
mat <- as.matrix(LayerData(obj, layer='data'))
cell_vec <- mat[, 1]
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
write.table(t(out), 'Manatee/data/e35_cell0.csv', quote=FALSE, row.names=FALSE, col.names=FALSE, sep=' ')
cat(colnames(mat)[1], '\n')
