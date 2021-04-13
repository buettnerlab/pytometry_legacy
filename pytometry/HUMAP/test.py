import scanpy as sc
import anndata
import numpy as np
import HUMAP.tl._humap as HUMAP

if __name__ == '__main__':
    filelocation = r"/home/felix/Public/datasets/VBh_converted.h5ad"
    adata = anndata.read_h5ad(filelocation)

    sc.pp.subsample(adata, 0.5)
    adata.X = np.arcsinh(adata.X / 10)

    sc.neighbors.neighbors(adata)
    HUMAP.humap(adata)
    sc.pl.umap(adata)
