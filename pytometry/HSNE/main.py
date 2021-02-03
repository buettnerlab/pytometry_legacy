import scanpy as sc
import anndata
import pl as pl
import tl as tl
import numpy as np


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filelocation = r"/home/felix/Public/datasets/VBh_converted.h5ad"
    adata = anndata.read_h5ad(filelocation)

    imp_channels = [1, 3, 5, 7, 9, 13]
    imp_channels_names = list(adata.var_names.values[imp_channels])

    sc.pp.subsample(adata, 0.05)
    adata.X = np.arcsinh(adata.X / 10)
    sc.pp.neighbors(adata, n_neighbors=20)


    tl.hsne(adata, imp_channel_ind=imp_channels, num_scales=2)
    pl.hsne(adata, channels_to_plot=['FSC-A'], scale_num=0)

