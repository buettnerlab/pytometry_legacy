import scanpy as sc
import matplotlib.pyplot as plt

from ._utils import findCuttingPoint, calcStationaryDistribution


def humap(adata, n_scales: int = 1,
          subs_mode: str = "knee",
          factor: float = 0.5,
          n_knn: int = 20,
          copy: bool = False,
          verbose: bool = False):

    # check if knn graph has been calculated beforehand
    assert 'connectivities' in adata.obsp.keys(), "No knn graph found. Calculate with scanpy.tl.neighbors(...)"

    if subs_mode == "factor":
        assert 0 < factor < 1, "factor has to be between 0 and 1.0"

    # create list with copy of adata as first element
    scales = list()
    scales.append(adata.copy())

    for s in range(n_scales):
        if verbose: print(f"-----\nCalculating scale {s}")

        # copy last scale
        speichi = scales[-1].copy()

        # get connectivities
        c = speichi.obsp['connectivities']

        # calculate stationary distribution
        pi = calcStationaryDistribution(c)

        # find cutting point for subsampling
        cuttingPoint = findCuttingPoint(pi, mode=subs_mode, factor=factor)

        if verbose: print(f"Cutting data at point {cuttingPoint}")

        # indices of landmarks sorted by probability
        lm_ind = pi.argsort()

        # "cut" data at cutting point (subsampling)
        speichi = speichi[lm_ind[:cuttingPoint]]

        if verbose: print(f"Calculating scale {s} with {speichi.n_obs} cells")

        # plotting
        if verbose:
            plt.plot(pi[pi.argsort()][::-1])
            plt.vlines(cuttingPoint, ymin=0, ymax=max(pi))
            plt.show()

        # (re)calculate connectivities
        sc.pp.neighbors(speichi, n_neighbors=n_knn)

        # calculate umap embedding
        sc.tl.umap(speichi)

        # append calculated scale to list
        scales.append(speichi)

    # remove first element as it is a copy of adata
    scales.pop(0)

    # add list of scales to .uns
    adata.uns['HUMAP_scales'] = scales
    return adata if copy else None


