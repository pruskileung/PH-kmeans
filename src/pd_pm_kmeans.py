# packages
import sys
import random
from gudhi.wasserstein import wasserstein_distance
from gudhi.wasserstein.barycenter import lagrangian_barycenter
from data_utils.pd_pm_methods import *


def kmeans_plusplus(data: list, n_clusters: int, data_type: str, random_state: int, **kwargs):
    """
    Function performs the k-means++ initialisation algorithm, returning n_clusters number of centroids from the given
    data.
    """
    # set random seed
    random.seed(random_state)
    # initialise list of centroids and randomly select 1st centroid
    centroids = [random.choice(data)]
    # calculate distances from chosen centroid to all other data points
    while len(centroids) < n_clusters:
        dist = []
        for data_point in data:
            d = sys.maxsize
            for j in range(len(centroids)):
                if data_type == 'diagram':
                    temp_dist = wasserstein_distances(data_point, [centroids[j]])[0]
                if data_type == 'measure':
                    temp_dist = wass_dist(data_point, centroids[j], kwargs['dist_mat'])
                d = min(d, temp_dist)
            dist.append(d)
        # select new centroids
        sum_squared_dist = sum([d ** 2 for d in dist])
        probs = [(d ** 2) / sum_squared_dist for d in dist]
        centroids.append(random.choices(data, weights=probs, k=1)[0])
    return centroids


def not_equal(c1, c2, n_clusters, centroid_type):
    """
    Function returns True if two data points are not equal and False if equal. Used in k-means algorithm to compare
    previous centroids to current centroids.
    """
    if centroid_type == 'diagram':
        if c1 == [None] * n_clusters or c2 == [None] * n_clusters:
             not_equal_bool = True
        else:
            c1 = [np.concatenate(c).tolist() for c in c1]
            c2 = [np.concatenate(c).tolist() for c in c2]
            not_equal_bool = all([x != y for x, y in zip(c1, c2)])
    if centroid_type == 'measure':
        not_equal_bool = all([not (np.allclose(x, y)) for x, y in zip(c1, c2)])
    return not_equal_bool


def wasserstein_distances(x, centroids):
    """
    Function calculates wasserstein distance between PD x and all centroids
    considered, split by homology degree. This is so only points of the same
    homology are compared in distance calculation.
    """
    centroid_dists_0 = [wasserstein_distance(x[0], c[0], order=2) for c in centroids]
    centroid_dists_1 = [wasserstein_distance(x[1], c[1], order=2) for c in centroids]
    centroid_dists_2 = [wasserstein_distance(x[2], c[2], order=2) for c in centroids]
    centroid_dists = [sum(d) for d in zip(centroid_dists_0, centroid_dists_1,
                                          centroid_dists_2)]
    return centroid_dists


def get_barycenter(diagrams):
    """
    Function returns the Fréchet mean of a set of diagrams. Each set of subdiagrams,
    split by homology degree are considered separately.
    """
    # initialise lists to store subdiagrams
    diagrams_0 = []
    diagrams_1 = []
    diagrams_2 = []
    # concatenate subdiagrams into single list for comparison
    for diag in diagrams:
        diagrams_0.append(diag[0])
        diagrams_1.append(diag[1])
        diagrams_2.append(diag[2])
    # compute Fréchet mean of subdiagrams
    fmean_0 = lagrangian_barycenter(diagrams_0, init=0, verbose=False)
    fmean_1 = lagrangian_barycenter(diagrams_1, init=0, verbose=False)
    fmean_2 = lagrangian_barycenter(diagrams_2, init=0, verbose=False)
    # return whole diagram
    return [fmean_0, fmean_1, fmean_2]


class PD_KMeans:
    def __init__(self, n_clusters, init='kmeans++', random_state=1245, max_iters=25):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iters = max_iters
        self.random_state = random_state

    def fit(self, diagrams):
        # set parameters
        random.seed(self.random_state)
        iteration = 0
        prev_centroids = [None] * self.n_clusters

        # initialisation step
        if self.init == 'random':
            self.centroids = random.sample(diagrams, self.n_clusters)
        if self.init == 'kmeans++':
            self.centroids = kmeans_plusplus(diagrams,
                                             n_clusters=self.n_clusters,
                                             data_type='diagram',
                                             random_state=self.random_state)

        while not_equal(prev_centroids, self.centroids, n_clusters=self.n_clusters, centroid_type='diagram') \
                and iteration < self.max_iters:
            clusters = [[] for _ in range(self.n_clusters)]
            labels = []
            # assignment step
            for diag in diagrams:
                dists = wasserstein_distances(diag, self.centroids)
                assigned_centroid = np.argmin(dists)
                clusters[assigned_centroid].append(diag)
                labels.append(assigned_centroid)

            # update step
            prev_centroids = self.centroids
            self.centroids = [get_barycenter(cluster) for cluster in clusters]

            # increase iteration
            iteration += 1

        return labels


class PM_KMeans:
    def __init__(self, n_clusters, grid_width, init='kmeans++', random_state=1245, max_iters=25):
        self.n_clusters = n_clusters
        self.grid_width = grid_width
        self.init = init
        self.random_state = random_state
        self.max_iters = max_iters

    def fit(self, measures):
        # set parameters
        random.seed(self.random_state)
        iteration = 0
        labels = []
        prev_centroids = [np.zeros(measures[0].shape) for _ in range(self.n_clusters)]
        grid = mesh_gen(self.grid_width)

        self.mp = dist_mat(grid, 2)

        if self.init == 'random':
            self.centroids = random.sample(measures, self.n_clusters)
        if self.init == 'kmeans++':
            self.centroids = kmeans_plusplus(measures,
                                             n_clusters=self.n_clusters,
                                             data_type='measure',
                                             random_state=self.random_state,
                                             dist_mat = self.mp)

        # initialisation step
        self.centroids = random.sample(measures, self.n_clusters)

        while not_equal(prev_centroids, self.centroids, n_clusters=self.n_clusters, centroid_type='measure')  \
                and iteration < self.max_iters:
            clusters = [[] for _ in range(self.n_clusters)]
            labels = []

            # assignment step
            for mesr in measures:
                dists = [wass_dist(centroid, mesr, self.mp) for centroid in self.centroids]
                assigned_centroid = np.argmin(dists)
                clusters[assigned_centroid].append(mesr)
                labels.append(assigned_centroid)
            if len(set(labels)) != 3:
                break
            # update step
            prev_centroids = self.centroids
            self.centroids = [get_mean_mesr(cluster, float_error=1e-8) for cluster in clusters]

            # increase iteration
            iteration += 1

        return labels
