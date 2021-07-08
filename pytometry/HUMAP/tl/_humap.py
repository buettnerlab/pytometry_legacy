from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import scanpy as sc
import numpy as np

import matplotlib.pyplot as plt


def _humap(adata, n_scales: int = 1,
           subs_mode: str = "knee",
           factor: float = 0.5,
           n_knn: int = 20,
           copy: bool = False,
           verbose: bool = False):

    # check if knn graph has been calculated beforehand
    assert 'connectivities' in adata.obsp.keys(), "No knn graph found. Calculate with scanpy.tl.neighbors(...)"

    if subs_mode == "factor":
        assert factor > 0 and factor < 1, "factor has to be between 0 and 1.0"

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


def calcStationaryDistribution(connectivities):
    """
    Calculates stationary distribution of markov chain.

    Parameters
    ----------
    connectivities
        knn graph of markov chain
    """
    # transition matrix
    T = connectivities.multiply(csr_matrix(1.0 / np.abs(connectivities).sum(1)))
    D, V = eigs(T.T, which='LM')

    # landmark probabilities
    pi = V[:, 0]
    pi = pi.real
    pi /= pi.sum()

    return pi



from kneed import KneeLocator
def findCuttingPoint(eigValues,
                     mode: str = "window",
                     factor: float = 1):
    """


    Parameters
    ----------
    eigValues
        Stationary Distribution
    mode
        - "window":
            Point at the middle of the flattest sliding window with 0.1 times of len(eigValues)
        - "knee":
            Uses KneeLocator of kneed to find the knee inflection point
        - "factor":
            Point position specified by parameter: factor
    factor
        float between 0 and 1 --> 0 being NO data and 1 being ALL data
    """


    pi = eigValues[eigValues.argsort()][::-1]
    cutPoint = len(pi)

    if mode == "window":
        windowSize = (int)(len(pi) * 0.1)
        windowErg = list()
        for step in range(len(pi) - windowSize):
            windowErg.append(np.sum(np.abs(pi[step:step + windowSize] - np.mean(pi[step:step + windowSize]))))

        cutPoint = np.argmax(windowErg == min(windowErg))
    elif mode == "knee":
        kn = KneeLocator(range(len(pi)), pi, S=100, curve='convex', direction='decreasing')
        cutPoint = kn.elbow
    elif mode == "factor":
        cutPoint = int(factor * len(pi))
    else:
        raise ValueError(f"Error in findInflectionPoint: unknown mode \"{mode}\"")

    return cutPoint
