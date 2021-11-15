from typing import Dict, Any
import skfuzzy as fuzz
import numpy as np
import scanpy as sc
import anndata as ann
from minisom import MiniSom
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import time



colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 
          'Brown', 'ForestGreen', 'silver', 
          'rosybrown', 'sandybrown',
          'springgreen', 'darkcyan', 'darkviolet']


# Class for clustering Anndata objects
class Cluster:

    k_means_clust: Dict[Any, Any]

    def __init__(self, adata, centers=10):
        self.adata = adata
        self.data = adata.X

        self.fuzzy_cluster = dict()
        self.fuzzy_membership = dict()
        self.fuzzy_centers = dict()
        self.fuzzy_inertia = dict()

        self.k_means_clust = dict()
        self.k_medoids_inertia = dict()
        self.k_means_centers = dict()
        self.kmeans_gaps = dict()

        self.optimal_k = None
        self.silhouette_scores = dict()

        self.community_cluster = None
        self.community_inertia = None

        self.SOM_label = dict()
        self.SOM_optimal_k = None
        self.SOM_centers = dict()
        self.SOM_inertia = dict()

        self.__centers = centers

    def fuzzy_clustering(self, centers=None):
        """
        Method for fuzzy clustering.
        :param centers: Number of centers for clustering.
        :return: Runtime of this algorithm.
        """
        if not centers:
            centers = self.__centers

        alldata = self.data.T
        starttime = time.time()
        for ncenter in range(2, centers + 1):
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                alldata, ncenter, 2, error=0.005, maxiter=1000, init=None)

            self.fuzzy_cluster[str(ncenter)] = u
            self.fuzzy_membership[str(ncenter)] = np.argmax(u, axis=0)
            self.fuzzy_centers[str(ncenter)] = cntr

        endtime = time.time()
        return endtime - starttime

    def k_means_clustering(self, ncenters=None):
        """
        Method for k means clustering.
        :param ncenters: Number of centers for clustering.
        :return: Runtime of this algorithm.
        """
        if ncenters is None:
            ncenters = self.__centers

        starttime = time.time()
        for ncenter in range(2, ncenters + 1):
            kmeans = KMeans(n_clusters=ncenter, random_state=0).fit(self.data)
            self.k_means_clust[str(ncenter)] = kmeans.labels_
            self.k_medoids_inertia[str(ncenter)] = kmeans.inertia_
            self.k_means_centers[str(ncenter)] = kmeans.cluster_centers_

        endtime = time.time()
        return endtime - starttime

    def leiden_clustering(self, data=None, resolution=1):
        """
        Method for Leiden clustering.
        :param data: Data to be used for the Leiden clustering.
        :param resolution: Resolution used for the Leiden clustering.
        :return: Runtime of this algorithm.
        """
        if data is not None:
            adata = ann.AnnData(X=data)

        else:
            adata = self.adata

        starttime = time.time()
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, resolution)
        self.adata = adata

        if data is not None:
            return adata.obs['leiden'].values.codes

        else:
            self.community_cluster = adata.obs['leiden'].values.codes

        endtime = time.time()
        return endtime - starttime

    def som_clustering(self, 
                       ncenter=None, 
                       iterations=100, 
                       sigma=0.3, 
                       learning_rate=0.25, 
                       data=None):
        """
        Method for SOM clustering.
        :param ncenter: Number of centers for clustering.
        :param iterations: Number of iterations for clustering.
        :param sigma: Sigma parameter for SOM clustering.
        :param learning_rate: Learning rate parameter for SOM clustering.
        :param data: Data for SOM algorithm.
        :return:    - Labels for the resulting clustering,
                    - Centers for the resulting clustering,
                    - Runtime
        """
        if not ncenter:
            ncenter = self.__centers

        cluster_index = None
        som = None
        starttime = time.time()
        for ncenters in range(2, int(ncenter)+1):
            som_shape = (1, ncenters)
            som = MiniSom(som_shape[0], som_shape[1], len(self.data[0]), 
                          sigma, learning_rate)
            som.random_weights_init(self.data)

            if data is None:
                data = self.data
            # trains the SOM with given iterations
            som.train_batch(data, iterations)

            # Each neuron represents a cluster
            winner_coordinates = np.array([som.winner(x) for x in data]).T
            # With np.ravel_multi_index we convert the bidimensional 
            # coordinates to a monodimensional index.
            cluster_index = np.ravel_multi_index(winner_coordinates, 
                                                 som_shape)
            self.SOM_label[str(ncenters)] = cluster_index
            self.SOM_centers[str(ncenters)] = som.get_weights()[0]

        endtime = time.time()
        return cluster_index, som.get_weights()[0], endtime - starttime

    def calc_optimal_k(self, nrefs=3):
        """
        Calculates the optimal k using k-means algorithm.
        :param nrefs: Number of reference dispertions to be processed.
        :return: Runtime of this algorithm.
        """
        starttime = time.time()
        gaps = np.zeros((len(range(1, self.__centers)),))
        resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})

        if not self.k_medoids_inertia:
            self.k_means_clustering(self.__centers)

        for gap_index, k in enumerate(range(2, self.__centers + 1)):

            # Holder for reference dispersion results
            ref_disps = np.zeros(nrefs)

            # For n references, generate random sample and perform kmeans 
            # getting resulting dispersion of each loop
            for i in range(nrefs):
                # Create new random reference set
                random_reference = np.random.random_sample(size=self.data.shape)

                # Fit to it
                km = KMeans(k)
                km.fit(random_reference)

                ref_disp = km.inertia_
                ref_disps[i] = ref_disp

            orig_disp = self.k_medoids_inertia[str(k)]
            # Calculate gap statistic
            gap = np.log(np.mean(ref_disps)) - np.log(orig_disp)

            # Assign this loop's gap statistic to gaps
            gaps[gap_index] = gap

            resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, 
                                         ignore_index=True)
        # +2 because center count begins with 2
        self.optimal_k = gaps.argmax() + 2
        self.kmeans_gaps = gaps

        endtime = time.time()
        return endtime - starttime

    def silhouette_score(self, method='kmeans', n_center=None):
        """
        Calculates the silhouette score for given algorithm.
        :param method: The algorithm to calculate silhouette score for.
        :param n_center: Number of centers for clustering.
        :return: Silhouette score for choosen algorithm.
        """
        labels = None

        if method == 'kmeans':
            labels = self.k_means_clust
        elif method == 'fuzzy':
            labels = self.fuzzy_membership
        elif method == 'leiden':
            labels = self.community_cluster
        elif method == 'SOM':
            labels = self.SOM_label
        else:
            print('No clustering done yet!')

        if method == 'leiden':
            scores = dict()
            score = silhouette_score(self.data, labels)
            scores[str(len(np.unique(self.community_cluster)))] = score
            self.silhouette_scores[method] = scores
        elif n_center:  # returns the score directly for specified center
            score = silhouette_score(self.data, labels[str(n_center)])
            return score
        else:  # returns the score for range of center
            scores = dict()
            n_center = self.__centers
            for n in range(2, n_center + 1):
                score = silhouette_score(self.data, labels[str(n)])
                scores[str(n)] = score
            self.silhouette_scores[method] = scores

    def compute_inertia(self, data=None, label=None):
        """
        Computes the inertia to be used in gap statistics.
        :param data: Data for calculation.
        :param label: Labels for given data.
        :return: Inertia for given data and labels.
        """
        if data is None:
            data = self.data
        if label is None:
            return None

        # Divide Labels
        clusters = {}
        n_label = np.unique(label)
        center = {}  
        s = np.zeros(len(n_label))
        
        for i in n_label:
            clusters[i] = np.array(data[label == i, :])

            # Find centers
            center[i] = np.array([clusters[i].mean(axis=0)])

            # Sum up distances
            s[i] = np.sum((clusters[i] - center[i]) ** 2)
        #s = [np.sum((np.array(clusters)[i] - center[i, :]) ** 2) 
        #     for i in range(0, len(center))]
        inertia = np.sum(s)
        return inertia

    def gap_score(self, method, inertia, n_center, nrefs=3):
        """
        Calculates the gap score for a given clustering algorithm.
        :param method: The algorithm to calculate gap score for.
        :param inertia: Precalculated inertia for given algorithm.
        :param n_center: Number of centers for given algorithm.
        :param nrefs: Number of reference dispertions to be processed.
        :return: Calculated gap score.
        """
        # Holder for reference dispersion results
        ref_disps = np.zeros(nrefs)

        # Copy of original AnnData object
        org = self.adata

        # For n references, generate random sample and perform kmeans getting 
        # resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            random_reference = np.random.random_sample(size=self.data.shape)

            if method == 'kmeans':
                km = KMeans(n_center)
                km.fit(random_reference)
                ref_disp = km.inertia_
                ref_disps[i] = ref_disp

            elif method == 'fuzzy':
                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    random_reference.T, n_center, 2, error=0.005, 
                    maxiter=1000, init=None)

                label = np.argmax(u, axis=0)+1
                ref_disp = self.compute_inertia(random_reference, label)
                ref_disps[i] = ref_disp

            elif method == 'leiden':
                if not ref_disps[0]:
                    label = self.leidend_clustering(data=random_reference)
                    ref_disp = self.compute_inertia(random_reference, label)
                    ref_disps[i] = ref_disp

            elif method == 'SOM':
                label, center, runtime = self.som_clustering(n_center, 
                                                 data=random_reference)
                ref_disp = self.compute_inertia(random_reference, label)
                ref_disps[i] = ref_disp

        # Calculate gap statistic
        gap = np.log(np.mean(ref_disps)) - np.log(inertia)
        self.adata = org
        return gap
