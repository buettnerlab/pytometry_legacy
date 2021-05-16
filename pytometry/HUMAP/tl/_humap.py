from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import scanpy as sc
import numpy as np

import matplotlib.pyplot as plt


def _humap(adata, n_scales: int = 1, factor: float = 0.5, copy: bool = False):
    # check if knn graph has been calculated beforehand
    assert 'connectivities' in adata.obsp.keys(), "No knn graph found. Calculate with scanpy.tl.neighbors(...)"

    scales = list()
    scales.append(adata.copy())

    for s in range(n_scales):
        speichi = scales[-1].copy()
        # sc.pp.neighbors(speichi)
        c = speichi.obsp['connectivities']
        T = c.multiply(csr_matrix(1.0 / np.abs(c).sum(1)))
        D, V = eigs(T.T, which='LM')

        # landmark probabilities
        pi = V[:, 0]
        pi = pi.real
        pi /= pi.sum()

        # indices of landmarks sorted by probability
        lm_ind = pi.argsort()

        speichi = speichi[lm_ind[:int(len(speichi) * factor)]]
        print(f"Calculating scale {s} with {len(speichi)} cells")

        plt.plot(pi[pi.argsort()][::-1])
        plt.show()

        # normalizing
        speichi.X = np.arcsinh(speichi.X / 10)

        # (re)calculate knn
        sc.pp.neighbors(speichi, n_neighbors=20)
        # calculate umap embedding
        sc.tl.umap(speichi)
        scales.append(speichi)

    # removing first element
    scales.pop(0)
    adata.uns['HUMAP_scales'] = scales
    return adata if copy else None

