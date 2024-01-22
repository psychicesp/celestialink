# Global Imports
import numpy as np
from sklearn.cluster import KMeans

# ----------------------------------------
# Main
# ----------------------------------------

def get_cluster_centroids(
        data: np.array
):
    """
    Generates centroids based on a clustering algorithm
    """
    points = data[:, :-1]
    labels = data[:, -1]

    num_clusters = len(np.unique(labels))
    clustering_algo = KMeans(n_clusters=num_clusters)

    clustering_algo.fit(points)

    return clustering_algo.cluster_centers_
