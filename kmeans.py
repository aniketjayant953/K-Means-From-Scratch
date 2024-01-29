import random
import numpy as np


class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.inertia_ = None

    def fit_predict(self, X):
        random_index = random.sample(range(0, X.shape[0]), self.n_clusters)
        self.centroids = X[random_index]

        for i in range(self.max_iter):
            # assign clusters
            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids

            # move centroids
            self.centroids = self.move_centroids(X, cluster_group)

            # check finish
            if (old_centroids == self.centroids).all():
                break

        cluster_type = np.unique(cluster_group)
        wcss = []

        for type in cluster_type:
            group = X[cluster_group == type]
            for i in group:
                wcss.append(np.linalg.norm(self.centroids[type] - i)**2)

        self.inertia_ = sum(wcss)

        return cluster_group

    def assign_clusters(self, X):
        cluster_group = []
        distances = []

        for row in X:
            for centroids in self.centroids:
                distances.append(np.sqrt(np.dot(row - centroids, row - centroids)))

            min_distances = min(distances)
            index_pos = distances.index(min_distances)
            cluster_group.append(index_pos)
            distances.clear()

        return np.array(cluster_group)

    def move_centroids(self, X, cluster_group):
        new_centroids = []

        cluster_type = np.unique(cluster_group)

        for type in cluster_type:
            new_centroids.append(X[cluster_group == type].mean(axis=0))
        return np.array(new_centroids)
