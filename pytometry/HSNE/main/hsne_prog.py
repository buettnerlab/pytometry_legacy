import scanpy as sc
import anndata
import numpy as np
from sklearn.manifold import _utils
from sklearn.manifold._t_sne import _joint_probabilities
from sklearn.neighbors._unsupervised import NearestNeighbors
from scipy.sparse import csr_matrix, bsr_matrix, lil_matrix, vstack, hstack
from scipy.special import softmax
from sklearn.manifold import TSNE
from scipy.stats import entropy
import matplotlib.pyplot as plt
import time as time
import platform
import multiprocessing as mp
from ..Other.LassoScatterSelector import SelectFromCollection


OS = platform.system()
# OS = 'Windows'
use_mp = OS != 'Windows'  # multiprocessing only available on Unix
HELPER_VAR = dict()


class HSNE:
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

        def calc_embedding(self):
            '''
            Calculates an embedding (X_hsne) for this scale using its X and P, overwriting X_hsne
            '''
            tsne = tSNE()
            self.X_hsne = tsne.fit_transform(self.X, P=self.P)

        # TODO
        def show_selected_lm(self):
            if self.X_hsne is None:
                self.calc_embedding()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(self.X_hsne[:, 0], self.X_hsne[:, 1], c='blue')
            ax.scatter(self.X_hsne[self.lm_ind, 0], self.X_hsne[self.lm_ind, 1], c='red')
            plt.show()

    def __init__(self, adata, imp_channels=None, n_neighbors=20, beta=100, beta_thresh=1.5, teta=50):

        self.load_adata(adata, imp_channels=imp_channels)

        # settings dict for all important setting variables
        self.settings = {
            'n_neighbors': n_neighbors,
            'beta': beta,
            'beta_thresh': beta_thresh,
            'teta': teta
        }

        # normalize X
        self.X = self.adata.X.T
        for col in enumerate(self.adata.X.T):
            self.X[col[0]] = np.arcsinh(col[1] / 10)
        self.X = self.X.T

        # initialize empty list of scales
        self.scales = list()

    def print_status(self):
        print('---STATUS HSNE OBJECT---')
        print('Anndata object loaded: %s' % (True if self.adata is not None else False))
        if self.adata is not None:
            print('  Event count: %d' % np.shape(self.adata.X)[0])
            print('  Total channel count: %d' % np.shape(self.adata.X)[1])
            print('  Important channel count: %d' % len(self.adata.uns['imp_channels']))
        print('%d scales calculated' % (len(self.scales) if self.scales is not None else 0))

    def subsample(self, percentage):
        if self.adata is not None:
            sc.pp.subsample(self.adata, percentage)
            self._set_X()

    def fit(self, scale=1, calc_embedding='last', verbose=False):
        '''
        Calculating a certain amount of scales
        :param scale: scale number or "depth"
        :param calc_embedding: "last"|"all" which embedding should be calculated
        :return: list of scales
        '''
        if verbose: print('Starting HSNE...')
        if verbose: print('Anndata object with %d events' % (np.shape(self.adata.X)[0]))
        if use_mp: print('Will use multiprocessing. Processors available: %d' % (mp.cpu_count()))
        tsne = tSNE()
        # create empty scale slots
        for i in range(scale):
            self.scales.append(self._Scale())

        # first scale
        T = self.calc_T(self.X, n_neighbors=self.settings['n_neighbors'], verbose=verbose)
        X = self.X  # [lm_s]
        lm_ind_s = self.get_landmarks(self.X, T, beta=self.settings['beta'], beta_thresh=self.settings['beta_thresh'], teta=self.settings['teta'], verbose=True)
        # I = self.calc_AoI(lm_s, T, verbose=True) # "first" scale does not need I
        W = 1
        P = self.calc_P(T, range(np.shape(self.X)[0]))
        X_hsne = None
        #if calc_embedding == 'all':
        #    X_hsne = tsne.fit_transform(self.X, P=P)  # .adata
        self.scales[0] = self._Scale(X=X, T=T, W=W, P=P, X_hsne=X_hsne, lm_ind=lm_ind_s)  # removed unnecessary I
        if calc_embedding == 'all':
            self.scales[0].calc_embedding()
        # following scales

        iterscales = iter(range(len(self.scales)))
        next(iterscales)
        for s in iterscales:
            self.scales[s]=self.getNextScale(self.scales[s-1], calc_embedding=(calc_embedding=='all'))


        if calc_embedding == 'last' and self.scales[-1].X_hsne is None:
            # X_hsne = tsne.fit_transform(self.adata.X[lm_s], P=P)
            self.scales[-1].calc_embedding()  #X_hsne = X_hsne

        return self.scales

    def getNextScale(self, scale, calc_embedding=False):
        tsne = tSNE()
        lm_ind_s_prev = scale.lm_ind
        X = self.adata.X[lm_ind_s_prev]
        I = self.calc_AoI(lm_ind_s_prev, scale.T, verbose=True)
        T = self.calc_next_T(I, scale.W)
        W = self.calc_Weights(I, scale.W)
        P = self.calc_P(T, lm_ind_s_prev)
        X_hsne = None
        if calc_embedding:
            X_hsne = tsne.fit_transform(self.adata.X[lm_ind_s_prev], P=P)
        lm_s = self.get_landmarks(X, T, beta=self.settings['n_neighbors'], beta_thresh=self.settings['beta_thresh'], teta=self.settings['n_neighbors'], verbose=True)
        return self._Scale(X=X, T=T, I=I, W=W, P=P, X_hsne=X_hsne, lm_ind=lm_s, parent_scale=scale)


    def drill(self, num=-1, gamma=0.5):
        # NEW TODO implementation not finished yet
        s = self.scales[num]
        s_prev = s.parent_scale#self.scales[num-1]
        o = self.lasso_subset(num=num)

        r = [x for x in range(np.shape(s.I)[0]) if np.sum(s.I[x,o])>gamma]
        T_r = s_prev.T[r,:][:,r]
        P_r = self.calc_P(T_r, r)
        tsne = tSNE()
        x_hsne = tsne.fit_transform(s_prev.X[r], P=P_r)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x_hsne[:, 0], x_hsne[:, 1], c='green')
        plt.show()

        sel_scale = self._Scale(X=s.X[r], T=T_r, I=None, W=s.W[r], X_hsne=x_hsne, P=P_r, parent_scale=s_prev)
        s_next = self.getNextScale(sel_scale)
        print('')



    def lasso_subset(self, num=-1):
        if len(self.scales) == 0:
            print("No scales to show")
            pass
        elif num < -1 or num >= len(self.scales):
            print("No scale with number %d found\nnum must be between 0 and %d" % (num, len(self.scales) - 1))
            pass
        else:
            if self.scales[num].X_hsne is None:
                self.scales[num].calc_embedding()
            fig, ax = plt.subplots()
            pts = ax.scatter(self.scales[num].X_hsne[:, 0], self.scales[num].X_hsne[:, 1])
            ax.set_title("Press enter to accept selected points.")
            selector = SelectFromCollection(ax, pts)
            mut_var = {}

            def accept(event):
                if event.key == "enter":
                    print("Selected points:")
                    print(len(selector.ind))
                    mut_var['points'] = selector.ind
                    selector.disconnect()
                    ax.set_title("")
                    fig.canvas.draw()

            fig.canvas.mpl_connect("key_press_event", accept)
            plt.show()
            return mut_var['points']  # indices of selected points

    def plot(self, scale_num=-1, channels=-1, subplot_rc=(1, 1)):

        if len(self.scales) == 0:
            # no scales calculated
            print('No scale calculated yet')  # TODO add error print
        if channels == -1:
            # default value handler for the channels to be colored
            channels = self.adata.uns['imp_channels']
        if self.scales[scale_num].X_hsne is None:
            # embedding for this scale has not been calculated yet
            self.scales[scale_num].calc_embedding()
        # dynamically add rows and cols for subplots
        num_channels = len(channels)
        r, c = subplot_rc
        if r == -1 or c == -1 or r * c < num_channels:
            while r * c < num_channels:
                if r > c:
                    c += 1
                else:
                    r += 1
        # create subplots
        for channel in enumerate(channels):
            plt.subplot(r, c, channel[0] + 1)
            plt.scatter(self.scales[scale_num].X_hsne[:, 0], self.scales[scale_num].X_hsne[:, 1],
                        c=self.scales[scale_num].X[:, channel[0]],
                        s=(self.scales[scale_num].W ))
            plt.title('Colored by %s' % self.adata.var_names[channel[1]])
            plt.colorbar()
        plt.show()

    def show_selected_lm(self, scale_num):
        '''
        Creates a scatter plot with the selected scale embedding and colored landmarks for the next embedding
        :param scale_num: scale number
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if self.scales[scale_num].X_hsne is None:
            self.scales[scale_num].calc_embedding()
        if self.scales[scale_num].lm_ind is None:
            self.scales[scale_num].lm_ind = self.get_landmarks(self.scales[scale_num].X, self.scales[scale_num].T, beta=self.settings['n_neighbors'], beta_thresh=self.settings['beta_thresh'], teta=self.settings['n_neighbors'], verbose=True)

        ax.scatter(self.scales[scale_num].X_hsne[:, 0], self.scales[scale_num].X_hsne[:, 1], c='green')
        ax.scatter(self.scales[scale_num].X_hsne[self.scales[scale_num].lm_ind, 0],
                   self.scales[scale_num].X_hsne[self.scales[scale_num].lm_ind, 1], c='black')
        plt.title('Scale number %d' % (scale_num if scale_num >= 0 else len(self.scales) - 1))
        plt.show()

    def calc_P(self, T_s, lm_s):
        return (T_s + T_s.transpose()) / (2 * len(lm_s))

    def load_adata(self, adata, imp_channels=None):

        # if adata is a link
        if type(adata)==str:
            filelocation = adata
            adata=anndata.read_h5ad(filelocation)

        if adata is not None:
            if imp_channels is None:
                imp_channels = []
                if 'imp_channels' not in adata.uns:
                    imp_channels = np.array(range(np.shape(adata.X)[1]))
            adata.uns['imp_channels'] = imp_channels

        self.adata = adata
        self._set_X(imp_channels=self.adata.uns['imp_channels'])

    def _set_X(self, adata=None, imp_channels=None):
        if adata is None:
            adata = self.adata
        # normalize X
        self.X = adata.X.T
        for col in enumerate(adata.X.T):
            self.X[col[0]] = np.arcsinh(col[1] / 10)
        self.X = self.X.T
        if imp_channels is not None:
            self.X = self.X[:, imp_channels]


    def _set_imp_channels(self, imp_channels):
        '''
        Sets the important channels
        :param imp_channels: indices of important channels in adata.var_names
        :return: None
        '''
        if len(imp_channels) <= 0:
            imp_channels = np.array(range(np.shape(self.adata.X)[1]))
        self.adata.uns['imp_channels'] = imp_channels

    # TODO edit commentary
    def calc_T(self, X, n_neighbors=25, verbose=False):
        '''
        Creates Markov Chain
        :param
        X:
            data array with dimension (event x channel)
        n_neighbors:
            number of nearest neighbors to be calculated
        returns:
            Transition Matrix T of type csr_matrix (scipy.sparse)
        '''
        t0 = time.time()  # start timer
        if use_mp:
            if verbose: print('calc_T: Calculation of the %d nearest neighbors...' % n_neighbors)

            ## Nearest Neighbors: Version Scanpy
            sc.pp.neighbors(self.adata, n_neighbors=n_neighbors)
            distances_nn = self.adata.obsp['distances']

            if verbose: print('calc_T: Calculating transition matrix...')

            p = mp.Pool(mp.cpu_count())
            probs = p.map(_helper_method_calc_T, [dist.data for dist in distances_nn])
            p.terminate()
            p.join()
            data = []
            for pr in probs:
                data.extend(pr)

            T = csr_matrix((data, distances_nn.indices, distances_nn.indptr), shape=(len(X), len(X)))
        else:
            if verbose: print('calc_T: Calculation of the %d nearest neighbors...' % n_neighbors)
            knn = NearestNeighbors(algorithm='auto', n_neighbors=n_neighbors, metric='euclidean')
            knn.fit(X)
            # Getting distances to NN
            distances_nn = knn.kneighbors_graph(mode='distance')

            # TODO exact composition of sigma
            if verbose: print('calc_T: Calculating transition matrix...')
            sigma = n_neighbors / 3  # --> Hinton Roweiß Stochastic Neighbor Embedding np.round(n_neighbors/3)
            probs = []  # 'data' for new T csr_matrix
            for n in range(len(X)):
                d = distances_nn[n].data / np.max(distances_nn[n].data)  # get scaled distances to NN of point n
                probs.extend(softmax((-d ** 2) / sigma))  # calculate T[n]

            T = csr_matrix((probs, distances_nn.indices, distances_nn.indptr), shape=(len(X), len(X)))
        t1 = time.time()  # end timer
        if verbose: print('calc_T: finished in %s sec' % (t1 - t0))
        return T

    # TODO edit commentary
    def get_landmarks(self, X, T, beta=100, beta_thresh=1.5, teta=50, verbose=False):
        '''
        Determines landmarks using Markov Chain (Transition Matrix T)

        beta:
          number of times a point has to be reached to count as a landmark
        beta_thresh:
          event has to be reached at least beta*beta_thresh times to be accepted as
          a landmark
        teta:
          number of random walk steps

        returns:
          1D array with landmarks (indexes)
          Example:
          return value [123 420 599]
          means adata.X[123 & 420 & 599] are landmarks
        '''
        t0 = time.time()  # start timer
        if use_mp:
            n_events = len(X)  # number of events
            proposals = np.zeros(n_events)  # counts how many times point has been reached
            landmarks = list()  # list of landmarks

            # fill global variable, so multiple processes have access
            # --> prevents redundant arguments
            global HELPER_VAR
            HELPER_VAR = {'T': T,
                          'teta': teta,
                          'beta': beta,
                          'beta_thresh': beta_thresh,
                          'n_events': n_events}

            if verbose: print('get_landmarks: start multiprocessing')
            # create matrix with every initial state (diagonal 1)
            init_states = csr_matrix((np.ones(n_events), (range(n_events), range(n_events))))
            p = mp.Pool(mp.cpu_count())
            hit_list = p.map(_helper_method_get_landmarks, [state for state in init_states])
            p.terminate()
            p.join()
            if verbose: print('get_landmarks: multiprocessing finished')

            # evaluate results
            for state_hits in hit_list:  # for every states hit_list
                for h in state_hits:  # for every hit in some states hit_list
                    proposals[h[0]] += h[1]

            # collect landmarks
            min_beta = beta * beta_thresh
            for prop in enumerate(proposals):
                # if event has been hit min_beta times, it counts as landmark
                if prop[1] > min_beta:
                    landmarks.append(prop[0])
        else:
            n_events = len(X)  # number of events
            proposals = np.zeros(n_events)  # counts how many times point has been reached
            landmarks = list()  # list of landmarks

            # creating T_teta
            if verbose: print('get_landmarks: Creating transition matrix for %d steps' % teta)
            t0 = time.time()
            T_teta = self.calc_T_teta(T, teta)
            if verbose: print('get_landmarks: T_teta: %s' % (time.time() - t0))

            if verbose: print('get_landmarks: Starting random walks...')
            status = 0.01 * n_events  # for progress printing

            init_states = csr_matrix((np.ones(n_events), (range(n_events), range(n_events))))

            # iterate over every event
            for event in range(n_events):
                t0 = time.time()  # timer

                # get state vector
                state_prop = init_states[event]

                # calculate probabilities after teta steps
                state_prop = state_prop.dot(T_teta)
                # for t in range(teta):
                #  state_prop = state_prop.dot(T) # TODO ^50

                # do beta random walks
                destinations = np.random.choice(range(n_events), beta, p=state_prop.toarray()[0])
                for dest in destinations:
                    proposals[dest] += 1

                # print progress
                if verbose and event >= status:
                    print('get_landmarks: %s percent done' % (np.round((status / n_events) * 100)))
                    status += 0.01 * n_events
            if verbose: print('get_landmarks: 100 percent done!')

            if verbose: print('get_landmarks: Evaluating')
            min_beta = beta * beta_thresh
            for prop in enumerate(proposals):
                if prop[1] > min_beta:
                    landmarks.append(prop[0])

            # print('Average time per loop: %s\n'%np.mean(time_arr))
        t1 = time.time()  # end timer

        if verbose: print('get_landmarks: finished in %s sec' % (t1 - t0))
        return landmarks

    def calc_Weights(self, I, W_old=None):
        '''
        Calculates the Weight-Vector for the Landmarks of scale s

        I:
          Area of Influence Matrix I with dimension (n_landmarks_s-1, n_landmarks_s)
        W_old:
          Weight-Vector W with the weights of the landmarks from scale s-1. If this
          is the first scale, it is filled with ones (n_landmarks_s-1,)

        returns:
          Weight-Vector W with the weights if the landmarks from scale s with
          length (n_landmarks_s)
        '''
        if W_old is None or W_old is 1:
            W_old = np.ones((np.shape(I)[0],))
        # W_s = np.array(np.sum((W_old * I.transpose()).transpose(), axis=0))  # W_old * I
        # W_s = W_s.flatten()
        W_s = np.array(W_old * I).reshape((np.shape(I)[1]))
        return W_s

    def calc_T_teta(self, T, teta: int, verbose=True):
        '''
        Calculates transition matrix after teta steps
        :param T: Transition matrix (csr_matrix)
        :param teta: Number of steps
        :return: Transition matrix after teta steps
        '''

        t0 = time.time()  # start timer
        '''
        # TODO: to dense when too full...
        ## Version, where T switches from csr_matrix to dense when to full
        toodense = False
        if type(T) is csr_matrix:
            if T.nnz/(np.shape(T)[0]*np.shape(T)[1]) > 0.75:
                T = T.toarray()
                toodense = True
            else:
                T = bsr_matrix(T)
        T_teta = T.copy()
        for t in range(teta):
            print('Mul nr %d: T_teta is %s'%(t, type(T_teta)))
            T_teta *= T
            if toodense is False:
                if T_teta.nnz/(np.shape(T_teta)[0]*np.shape(T_teta)[1]) > 0.6:
                    print('Switching!!')
                    T_teta = T_teta.toarray()
                    toodense = True
        if toodense is True: T_teta = csr_matrix(T_teta)
        return T_teta
        '''
        # old
        if teta <= 0:
            return T
        T_teta = bsr_matrix(T.copy())
        T = bsr_matrix(T)
        if verbose: print('calc_T_teta: initial %s percent full' % (T_teta.nnz / (np.shape(T)[0] ** 2)) * 100)
        for t in range(teta):
            T_teta *= T
            # print(len(T_teta.data)/(np.shape(T_teta)[0]**2)) # prints density of T_teta
            if verbose: print(
                'calc_T_teta: mul %d --> %s percent full' % (t, (T_teta.nnz / (np.shape(T)[0] ** 2)) * 100))
        t1 = time.time()  # end timer
        if verbose: print(
            'calc_T_teta: finished in %s sec. Filled: %s' % ((t1 - t0), (T_teta.nnz / np.shape(T_teta)[0])))
        return csr_matrix(T_teta)

    def calc_AoI(self, lm, T, min_lm=100, verbose=False):
        '''
        Calculates the Area of Influence matrix

        lm:
          list containing landmarks
        T:
          Transition Matrix (Marcov Chain)
        min_lm:
          Minimal number of landmarks that have to be reached for evaluation

        returns:
          Area of Influence Matrix I with dimension (n_events, n_landmarks)
        '''
        t0 = time.time()  # start timer
        if use_mp:
            from scipy.sparse import vstack
            n_events = np.shape(T)[0]

            # create empty influence matrix
            # I = csr_matrix((n_events, len(lm)))

            # create state matrix containing all initial states
            init_states = csr_matrix((np.ones(n_events), (range(n_events), range(n_events))))
            # save temp static variables in global for outsourced multiprocessing method
            global HELPER_VAR
            HELPER_VAR = {'lm': lm, 'min_lm': min_lm, 'T': T}

            # start random walks
            if verbose: print('calc_AoI: Starting random walks')

            p = mp.Pool(mp.cpu_count())
            # OLD TODO remove
            # I = p.map(_helper_method_AoI_rw, [s for s in init_states])
            # I = p.map(_helper_method_AoI_eval, [r for r in I]) # replaced this I and one line before.. maybe 1 function?
            # NEW
            I = p.map(_helper_method_AoI_rw_new_2, [s for s in init_states])
            p.terminate()
            p.join()

            I = vstack(I)
            # for debugging: next line prints density of I
            # print(len(I.data) / (np.shape(I)[0] ** 2)) # density of I

            # Add fraction of hits to row of current event
            # I[state[0]] = csr_matrix((reached_lm / np.sum(reached_lm)))

        else:
            n_events = np.shape(T)[0]

            # create empty influence matrix
            I = csr_matrix((n_events, len(lm)))

            init_states = csr_matrix((np.ones(n_events), (range(n_events), range(n_events))))
            # start random walks
            if verbose: print('calc_AoI: Starting random walks')

            for state in enumerate(init_states):

                # counter array for reached landmarks
                reached_lm = np.zeros(len(lm))
                state_prob = state[1]

                # do until minimal landmark-"hit"-count is reached
                while np.sum(reached_lm) < min_lm:
                    # print('Depth: %d'%depth)
                    state_prob *= T  # (re)calculate state probability

                    # pick some states for current random-walk length (depth)
                    erg_random_walk = np.random.choice(n_events, min_lm * 2, p=state_prob.toarray()[0])

                    # evaluate picks
                    for erg in erg_random_walk:
                        if erg in lm:
                            reached_lm[lm.index(erg)] += 1  # TODO works in colabs

                # Add fraction of hits to row of current event
                I[state[0]] = csr_matrix((reached_lm / np.sum(reached_lm)))
        t1 = time.time()  # end timer
        if verbose: print('calc_AoI: finished in %s sec' % (t1 - t0))
        return I

    def calc_next_T(self, I, W, verbose=False):
        '''
        Calculates next Transition matrix T
        :param I:   Area of Influence Matrix with dimensions (num_lm_s_prev, num_lm_s)
        :param W:   Weight vector with length (num_lm_s_prev)
        :return:    Transition matrix T_next of type csr_matrix
                    with dimension (num_lm_s, num_lm_s)
        '''
        t0 = time.time()  # start timer
        if use_mp:
            if verbose: print('calc_next_T: Start multiprocessing...')
            num_lm_s_prev, num_lm_s = np.shape(I)  # dimensionst of I
            # num_lm_s_old > num_lm_s

            I_t = I.transpose()  # transposed Influence matrix

            # args = list(zip(enumerate(I_t), W, list(itertools.repeat(num_lm_s_prev))))

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

            # FIXME maybe getting dense --> numpy more efficient

        else:
            num_lm_s_prev, num_lm_s = np.shape(I)  # dimensions of I
            # num_lm_s_old > num_lm_s

            I_t = I.transpose()  # transposed Influence matrix
            I_with_W = I.copy()

            # multiply with W
            for i in enumerate(I_t):
                I_with_W[:, i[0]] = np.reshape(i[1].toarray().reshape((num_lm_s_prev,)) * W, (num_lm_s_prev, 1))

            I = I_with_W.T * I
            T_next = lil_matrix(np.zeros((num_lm_s, num_lm_s)))

            for r in enumerate(I):
                T_next[r[0], :] = r[1] / np.sum(r[1])

        t1 = time.time()  # end timer
        if verbose: print('calc_next_T: finished in %s sec' % (t1 - t0))
        return T_next.tocsr()

    def calc_joint_probs(self, T_s):
        '''
        Calculates joint probabilities
        :param T_s: Transition matrix of scale s
        :return: joint probabilities in csr_matrix P
        '''
        n_lm = np.shape(T_s)[0]
        P = (T_s + T_s.transpose()) / (2 * n_lm)
        return P


def _helper_method_get_landmarks(state):
    for i in range(HELPER_VAR['teta']):
        state *= HELPER_VAR['T']

    destinations = np.random.choice(range(HELPER_VAR['n_events']), HELPER_VAR['beta'], p=state.toarray()[0])
    hits = np.zeros((HELPER_VAR['n_events']))
    for d in destinations:
        hits[d] += 1

    return [(h[0], h[1]) for h in enumerate(hits) if h[1] > 0]


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


# old helper method, maybe a little bit slower
def _helper_method_AoI_rw(state_prob):
    # load globals
    lm = HELPER_VAR['lm']
    min_lm = HELPER_VAR['min_lm']
    T = HELPER_VAR['T']
    reached_lm = np.zeros(len(lm))

    # do until minimal landmark-"hit"-count is reached
    while np.sum(reached_lm) < min_lm:
        state_prob *= T  # (re)calculate state probability
        erg_random_walk = np.random.choice(np.shape(state_prob)[1], min_lm * 2, p=state_prob.toarray()[0])

        # evaluate picks
        for erg in erg_random_walk:
            if erg in lm:
                reached_lm[lm.index(erg)] += 1
    return reached_lm


# new helper method
def _helper_method_AoI_rw_new(state):
    # load globals
    # lm = HELPER_VAR['lm']
    # min_lm = HELPER_VAR['min_lm']
    T = HELPER_VAR['T']
    # tick = time.time()
    reached_lm = np.zeros(len(HELPER_VAR['lm']))

    # do until minimal landmark-"hit"-count is reached
    while np.sum(reached_lm) < HELPER_VAR['min_lm']:
        erg_random_walk = -1
        state_prob = state
        while erg_random_walk not in HELPER_VAR['lm']:
            state_prob *= T  # (re)calculate state probability
            erg_random_walk = np.random.choice(np.shape(state)[1], p=state_prob.toarray()[0])
        reached_lm[HELPER_VAR['lm'].index(erg_random_walk)] += 1
    # tack = time.time()
    erg = reached_lm * np.max(reached_lm.data)
    # print(tack-tick)
    return csr_matrix(erg)


# current fastest AoI method
def _helper_method_AoI_rw_new_2(state):
    # load globals
    T = HELPER_VAR['T']
    lm = HELPER_VAR['lm']
    reached_lm = np.zeros(len(HELPER_VAR['lm']))

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


def _helper_method_AoI_eval(row):
    return csr_matrix((row / np.sum(row)))


def _helper_method_get_landmarks_mul_T(enumer_state):
    return enumer_state[1].dot(HELPER_VAR)


def _helper_method_T_next_mul_W(i):
    # load globals
    W = HELPER_VAR['W']
    num_lm_s_prev = HELPER_VAR['num_lm_s_prev']
    return csr_matrix(np.reshape(i.toarray().reshape((num_lm_s_prev,)) * W, (num_lm_s_prev, 1)))


def _helper_method_T_next_row_div(r):
    return r[1] / np.sum(r[1])


# ----------------------------------------------------------------------------
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

        neighbors_nn = None
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
