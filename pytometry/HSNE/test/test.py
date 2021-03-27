import scanpy as sc
import anndata
import HSNE.plotting as pl
import HSNE.tools as tl
import numpy as np


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # read file from disk
    filelocation = r"test/datasets/VBh_converted.h5ad"
    adata = anndata.read_h5ad(filelocation)

    # reduce dataset to a set of channels
    imp_channels = [1, 3, 5, 7, 9, 13]
    imp_channels_names = list(adata.var_names.values[imp_channels])
    adata = adata[:, imp_channels]

    # subsample
    sc.pp.subsample(adata, 0.04)

    # normalize
    adata.X = np.arcsinh(adata.X / 10)

    # create knn graph
    sc.pp.neighbors(adata, n_neighbors=20)

    # calculate scales
    tl.hsne(adata, beta=50, beta_thresh=1.5, teta=25, num_scales=1, include_root_object=False, verbose=True)

    # drill in first scale
    # adata.uns["hsne_scales"][1].drill()

    # plot first scale
    pl.hsne(adata, channels_to_plot=['FSC-A'], scale_num=0)

