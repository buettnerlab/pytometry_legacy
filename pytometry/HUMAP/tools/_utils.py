from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import numpy as np
from kneed import KneeLocator

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
    cutpoint = len(pi)

    if mode == "window":
        windowSize = (int)(len(pi) * 0.1)
        windowErg = list()
        for step in range(len(pi) - windowSize):
            windowErg.append(np.sum(np.abs(pi[step:step + windowSize] - np.mean(pi[step:step + windowSize]))))

        cutpoint = np.argmax(windowErg == min(windowErg))
    elif mode == "knee":
        kn = KneeLocator(range(len(pi)), pi, S=100, curve='convex', direction='decreasing')
        cutpoint = kn.elbow
    elif mode == "factor":
        cutpoint = int(factor * len(pi))
    else:
        raise ValueError(f"Error in findInflectionPoint: unknown mode \"{mode}\"")

    return cutpoint
