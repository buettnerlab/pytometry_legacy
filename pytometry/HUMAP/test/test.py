import scanpy as sc
import anndata
import numpy as np
from HUMAP.tools._humap import humap



if __name__ == '__main__':
    # Load Dataset
    print("Loading dataset")
    filelocation = r"datasets/VBh_converted.h5ad"
    adata = anndata.read_h5ad(filelocation)
    print(f"dataset with {len(adata)} cells loaded")

    # subsampling (for slow/low memory computers)
    # sc.pp.subsample(adata, 0.8)

    # removing some channels
    for channel in adata.var_names:
        if channel.endswith("-H") or channel == "Time" or channel == "FSC-Width":
            adata.obs[channel] = adata.X[:, adata.var_names == channel]
            adata = adata[:, adata.var_names != channel]

    # normalizing
    print("Normalizing")
    adata.X = np.arcsinh(adata.X / 10)

    # calc knn
    print("Calculating connectivities")
    sc.pp.neighbors(adata, n_neighbors=20)

    # calculating louvain clustering of original dataset
    print("Calculating louvain clustering")
    sc.tl.louvain(adata, resolution=0.5, key_added="louv_05")

    print("Done!")

    humap(adata, subs_mode="knee")

    # plotting
    if 'umap' not in adata.obsm:
        sc.tl.umap(adata)
    sc.pl.umap(adata, color=adata.var_names.values)
    for s in enumerate(adata.uns['HUMAP_scales']):
        print(f"Scale {s[0]} with {np.shape(s[1].X)[0]} cells")
        sc.pl.umap(s[1], color=adata.var_names.values)  # [0]

