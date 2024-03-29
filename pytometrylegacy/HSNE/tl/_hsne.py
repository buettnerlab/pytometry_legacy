from scipy.stats import entropy
from scipy.special import softmax
import multiprocessing as mp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import vstack, hstack
from sklearn.manifold import TSNE
from sklearn.manifold._t_sne import _joint_probabilities
from sklearn.manifold import _utils

HELPER_VAR = dict()

class _Scale:
    def __init__(self, X=None, T=None, I=None, W=None, P=None, X_hsne=None, lm_ind=None, parent_scale=None):
        self.X = X
        self.T = T
        self.I = I
        self.W = W
        self.P = P
        self.X_hsne = X_hsne
        self.lm_ind = lm_ind
        self.parent_scale = parent_scale

def hsne(adata, imp_channel_ind=None, beta=100, beta_thresh=1.5, teta=50, num_scales=1):
    if imp_channel_ind is None:
        imp_channel_ind = range(len(adata.var_names))
    elif len(imp_channel_ind) == 0:
        imp_channel_ind = range(len(adata.var_names))

    # settings dict for all important setting variables
    settings = {
        'beta': beta,
        'beta_thresh': beta_thresh,
        'teta': teta,
        'imp_channel_ind' : imp_channel_ind
    }
    tsne = tSNE()

    # manage k nearest neighbors
    try:
        adata.obsp['distances']
    except KeyError as e:
        raise Exception("k-nearest-neighbor graph has to be constructed first")
    distances_nn = adata.obsp['distances']

    scale_list = list()

    # Create first scale
    s_root = _Scale(X=adata.X[:,imp_channel_ind], W=1)  # reduced x to imp_channels
    # TODO call by reference issue?

    # nomralizing X
    # s_root.X = s_root.X.T
    # for col in enumerate(s_root.X):
    #     s_root.X[col[0]] = np.arcsinh(col[1] / 10) # TODO arcsinh über ganze matrix
    # s_root.X = s_root.X.T

    print('T')
    s_root.T = _calc_first_T(distances_nn, len(adata.X))
    print('P')
    s_root.P = _calc_P(s_root.T)
    print('X_hsne')
    s_root.X_hsne = tsne.fit_transform(s_root.X, P=s_root.P)
    print('lm_ind')
    s_root.lm_ind = _get_landmarks(s_root.T, settings)

    scale_list.append(s_root)   # appending scale

    for i in range(num_scales):
        print('Scale Number %d:' %i)
        s_prev = scale_list[i]
        s_curr = _Scale(X=s_prev.X[s_prev.lm_ind, :], parent_scale=s_prev)
        print('I')
        s_curr.I = _calc_AoI(s_prev)
        print('W')
        s_curr.W = _calc_Weights(s_curr.I, s_prev.W)
        print('T')
        s_curr.T = _calc_next_T(s_curr.I, s_prev.W)
        print('lm_ind')
        s_curr.lm_ind = _get_landmarks(s_curr.T, settings)
        print('P')
        s_curr.P = _calc_P(s_curr.T)
        print('X_hsne')
        s_curr.X_hsne = tsne.fit_transform(s_curr.X, P=s_curr.P)
        scale_list.append(s_curr)

    adata.uns['hsne_settings'] = settings
    adata.uns['hsne_scales'] = scale_list
    return adata


def _calc_P(T):
    return (T + T.transpose()) / (2 * T.shape[0])

def _calc_Weights(I, W_old):
    if type(W_old) is int: #W_old is None or W_old is 1:
        W_old = np.ones((I.shape[0],))
    W_s = np.array(W_old * I).reshape((I.shape[1]))
    return W_s

def _calc_next_T(I, W):
    num_lm_s_prev, num_lm_s = (I.shape[0],I.shape[1])  # dimensionst of I
    # num_lm_s_old > num_lm_s

    I_t = I.transpose()  # transposed Influence matrix

    global HELPER_VAR
    HELPER_VAR = {'W': W, 'num_lm_s_prev': num_lm_s_prev}

    p = mp.Pool(mp.cpu_count())
    I_with_W = p.map(_helper_method_T_next_mul_W, [it for it in I_t])
    I_with_W = hstack(I_with_W)
    I = I_with_W.T * I
    T_next = p.map(_helper_method_T_next_row_div, enumerate(I))
    p.terminate()
    p.join()

    T_next = vstack(T_next)
    return T_next.tocsr()

def _helper_method_T_next_mul_W(i):
    # load globals
    W = HELPER_VAR['W']
    num_lm_s_prev = HELPER_VAR['num_lm_s_prev']
    return csr_matrix(np.reshape(i.toarray().reshape((num_lm_s_prev,)) * W, (num_lm_s_prev, 1)))

def _helper_method_T_next_row_div(r):
    return r[1] / np.sum(r[1])

def _calc_AoI(scale, min_lm=100):
    n_events = scale.T.shape[0]
    # create state matrix containing all initial states
    init_states = csr_matrix((np.ones(n_events), (range(n_events), range(n_events))))
    # save temp static variables in global for outsourced multiprocessing method
    global HELPER_VAR
    HELPER_VAR = {'lm': scale.lm_ind, 'min_lm': min_lm, 'T': scale.T}
    p = mp.Pool(mp.cpu_count())
    I = p.map(_helper_method_AoI, [s for s in init_states])
    p.terminate()
    p.join()
    I = vstack(I)
    return I

def _helper_method_AoI(state):
    # load globals
    T = HELPER_VAR['T']
    lm = HELPER_VAR['lm']
    reached_lm = np.zeros(len(lm))

    cache = list()  # create empty cache list
    cache.append(state)  # append initial state vector as first element
    state_len = np.shape(state)[1]  # get length of vector once

    # do until minimal landmark-"hit"-count is reached (--> landmarks_left < 0)
    landmarks_left = HELPER_VAR['min_lm']
    while landmarks_left >= 0:
        # erg_random_walk = -1
        step = 1
        while True:
            if len(cache) <= step:
                cache.append(cache[step - 1] * T)
            erg_random_walk = np.random.choice(state_len, p=cache[step].toarray()[0])
            if erg_random_walk in lm:
                reached_lm[lm.index(erg_random_walk)] += 1
                landmarks_left -= 1
                break
            step += 1
    erg = reached_lm / np.sum(reached_lm.data)
    return csr_matrix(erg)

def _get_landmarks(T, settings):

    n_events = T.shape[0]
    proposals = np.zeros(n_events)  # counts how many times point has been reached
    landmarks = list()  # list of landmarks
    global HELPER_VAR
    HELPER_VAR = {'T': T,
                  'teta': settings['teta'],
                  'beta': settings['beta'],
                  'beta_thresh': settings['beta_thresh'],
                  'n_events': n_events}
    init_states = csr_matrix((np.ones(n_events), (range(n_events), range(n_events))))
    p = mp.Pool(mp.cpu_count())
    hit_list = p.map(_helper_method_get_landmarks, [state for state in init_states])
    p.terminate()
    p.join()
    # evaluate results
    for state_hits in hit_list:  # for every states hit_list
        for h in state_hits:  # for every hit in some states hit_list
            proposals[h[0]] += h[1]

    # collect landmarks
    min_beta = settings['beta'] * settings['beta_thresh']
    for prop in enumerate(proposals):
        # if event has been hit min_beta times, it counts as landmark
        if prop[1] > min_beta:
            landmarks.append(prop[0])
    return landmarks

def _helper_method_get_landmarks(state):
    for i in range(HELPER_VAR['teta']):
        state *= HELPER_VAR['T']
    destinations = np.random.choice(range(HELPER_VAR['n_events']), HELPER_VAR['beta'], p=state.toarray()[0])
    hits = np.zeros((HELPER_VAR['n_events']))
    for d in destinations:
        hits[d] += 1
    return [(h[0], h[1]) for h in enumerate(hits) if h[1] > 0]

def _calc_first_T(distances_nn, dim):
    p = mp.Pool(mp.cpu_count())
    probs = p.map(_helper_method_calc_T, [dist.data for dist in distances_nn])
    p.terminate()
    p.join()
    data = []
    for pr in probs:
        data.extend(pr)
    T = csr_matrix((data, distances_nn.indices, distances_nn.indptr), shape=(dim,dim))
    return T


def _helper_method_calc_T(dist):
    d = dist / np.max(dist)
    return softmax((-d ** 2) / _binary_search_sigma(d, len(d)))

def _binary_search_sigma(d, n_neigh):
    # binary search
    sigma = 10  # Start Sigma
    goal = np.log(n_neigh)  # log(k) with k being n_neighbors
    # Do binary search until entropy ~== log(k)
    while True:
        ent = entropy(softmax((-d ** 2) / sigma))
        # check sigma
        if np.isclose(ent, goal):
            return sigma
        if ent > goal:
            sigma *= 0.5
        else:
            sigma /= 0.5



########################################################################################################################

MACHINE_EPSILON = np.finfo(np.double).eps

def _joint_probabilities_nn(distances, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances using just nearest
    neighbors.

    This method is approximately equal to _joint_probabilities. The latter
    is O(N), but limiting the joint probability to nearest neighbors improves
    this substantially to O(uN).

    Parameters
    ----------
    distances : CSR sparse matrix, shape (n_samples, n_samples)
        Distances of samples to its n_neighbors nearest neighbors. All other
        distances are left to zero (and are not materialized in memory).

    desired_perplexity : float
        Desired perplexity of the joint probability distributions.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : csr sparse matrix, shape (n_samples, n_samples)
        Condensed joint probability matrix with only nearest neighbors.
    """
    from time import time
    t0 = time()
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances.sort_indices()
    n_samples = distances.shape[0]
    distances_data = distances.data.reshape(n_samples, -1)
    distances_data = distances_data.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances_data, desired_perplexity, verbose)
    assert np.all(np.isfinite(conditional_P)), \
        "All probabilities should be finite"

    # Symmetrize the joint probability distribution using sparse operations
    P = csr_matrix((conditional_P.ravel(), distances.indices,
                    distances.indptr),
                   shape=(n_samples, n_samples))
    P = P + P.T

    # Normalize the joint probability distribution
    sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
    P /= sum_P

    assert np.all(np.abs(P.data) <= 1.0)
    if verbose >= 2:
        duration = time() - t0
        print("[t-SNE] Computed conditional probabilities in {:.3f}s"
              .format(duration))
    return P

class tSNE(TSNE):
    '''
    Overwritten TSNE class for embedding
    '''

    def fit_transform(self, X, P=None, y=None):
        """Fit X into an embedded space and return that transformed
        output.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.

        y : Ignored

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        embedding = self._fit(X, P)
        self.embedding_ = embedding
        return self.embedding_

    def _fit(self, X, P=None, skip_num_points=0):
        """Private function to fit the model using X as training data."""
        from time import time
        import numpy as np
        # from scipy import linalg
        # from scipy.spatial.distance import pdist
        # from scipy.spatial.distance import squareform
        from scipy.sparse import csr_matrix, issparse
        from sklearn.neighbors import NearestNeighbors
        # from sklearn.base import BaseEstimator
        from sklearn.utils import check_array
        from sklearn.utils import check_random_state
        # from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
        from sklearn.utils.validation import check_non_negative
        from sklearn.decomposition import PCA
        from sklearn.metrics.pairwise import pairwise_distances

        if self.method not in ['barnes_hut', 'exact']:
            raise ValueError("'method' must be 'barnes_hut' or 'exact'")
        if self.angle < 0.0 or self.angle > 1.0:
            raise ValueError("'angle' must be between 0.0 - 1.0")
        if self.method == 'barnes_hut':
            X = check_array(X, accept_sparse=['csr'], ensure_min_samples=2,
                            dtype=[np.float32, np.float64])
        else:
            X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                            dtype=[np.float32, np.float64])
        if self.metric == "precomputed":
            if isinstance(self.init, str) and self.init == 'pca':
                raise ValueError("The parameter init=\"pca\" cannot be "
                                 "used with metric=\"precomputed\".")
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square distance matrix")

            check_non_negative(X, "TSNE.fit(). With metric='precomputed', X "
                                  "should contain positive distances.")

            if self.method == "exact" and issparse(X):
                raise TypeError(
                    'TSNE with method="exact" does not accept sparse '
                    'precomputed distance matrix. Use method="barnes_hut" '
                    'or provide the dense distance matrix.')

        if self.method == 'barnes_hut' and self.n_components > 3:
            raise ValueError("'n_components' should be inferior to 4 for the "
                             "barnes_hut algorithm as it relies on "
                             "quad-tree or oct-tree.")
        random_state = check_random_state(self.random_state)

        if self.early_exaggeration < 1.0:
            raise ValueError("early_exaggeration must be at least 1, but is {}"
                             .format(self.early_exaggeration))

        if self.n_iter < 250:
            raise ValueError("n_iter should be at least 250")

        n_samples = X.shape[0]

        neighbors_nn = None  # TODO Verdacht!
        if self.method == "exact":
            # Retrieve the distance matrix, either using the precomputed one or
            # computing it.
            if self.metric == "precomputed":
                distances = X
            else:
                if self.verbose:
                    print("[t-SNE] Computing pairwise distances...")

                if self.metric == "euclidean":
                    distances = pairwise_distances(X, metric=self.metric,
                                                   squared=True)
                else:
                    distances = pairwise_distances(X, metric=self.metric,
                                                   n_jobs=self.n_jobs)

                if np.any(distances < 0):
                    raise ValueError("All distances should be positive, the "
                                     "metric given is not correct")

            # compute the joint probability distribution for the input space
            if P is None:
                P = _joint_probabilities(distances, self.perplexity, self.verbose)
            assert np.all(np.isfinite(P)), "All probabilities should be finite"
            assert np.all(P >= 0), "All probabilities should be non-negative"
            assert np.all(P <= 1), ("All probabilities should be less "
                                    "or then equal to one")

        else:
            # Compute the number of nearest neighbors to find.
            # LvdM uses 3 * perplexity as the number of neighbors.
            # In the event that we have very small # of points
            # set the neighbors to n - 1.
            n_neighbors = min(n_samples - 1, int(3. * self.perplexity + 1))

            if self.verbose:
                print("[t-SNE] Computing {} nearest neighbors..."
                      .format(n_neighbors))

            # Find the nearest neighbors for every point
            knn = NearestNeighbors(algorithm='auto',
                                   n_jobs=self.n_jobs,
                                   n_neighbors=n_neighbors,
                                   metric=self.metric)
            t0 = time()
            knn.fit(X)
            duration = time() - t0
            if self.verbose:
                print("[t-SNE] Indexed {} samples in {:.3f}s...".format(
                    n_samples, duration))

            t0 = time()
            distances_nn = knn.kneighbors_graph(mode='distance')
            duration = time() - t0
            if self.verbose:
                print("[t-SNE] Computed neighbors for {} samples "
                      "in {:.3f}s...".format(n_samples, duration))

            # Free the memory used by the ball_tree
            del knn

            if self.metric == "euclidean":
                # knn return the euclidean distance but we need it squared
                # to be consistent with the 'exact' method. Note that the
                # the method was derived using the euclidean method as in the
                # input space. Not sure of the implication of using a different
                # metric.
                distances_nn.data **= 2

            # compute the joint probability distribution for the input space
            # if no P is given, calculate P
            if P is None:
                P = _joint_probabilities_nn(distances_nn, self.perplexity, self.verbose)

        if isinstance(self.init, np.ndarray):
            X_embedded = self.init
        elif self.init == 'pca':
            pca = PCA(n_components=self.n_components, svd_solver='randomized',
                      random_state=random_state)
            X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
        elif self.init == 'random':
            # The embedding is initialized with iid samples from Gaussians with
            # standard deviation 1e-4.
            X_embedded = 1e-4 * random_state.randn(
                n_samples, self.n_components).astype(np.float32)
        else:
            raise ValueError("'init' must be 'pca', 'random', or "
                             "a numpy array")

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # degrees_of_freedom = n_components - 1 comes from
        # "Learning a Parametric Embedding by Preserving Local Structure"
        # Laurens van der Maaten, 2009.
        degrees_of_freedom = max(self.n_components - 1, 1)

        return self._tsne(P, degrees_of_freedom, n_samples,
                          X_embedded=X_embedded,
                          neighbors=neighbors_nn,
                          skip_num_points=skip_num_points)

