# Global Imports
import numpy as np


# ----------------------------------------
# Support
# ----------------------------------------

def _find_closest_centroid(
        centroids,
        unknown_point,
        labelled=False
) -> dict:
    # Determines which centroid an unknown point is closest to by Euclidean Distance.

    # Points carry labels and they need to be stripped
    if labelled:
        unlabelled_point = unknown_point[:-1]
    else:
        unlabelled_point = unknown_point
    distances = np.linalg.norm(centroids - unlabelled_point, axis=1)
    closest_index = np.argmin(distances)
    closest_centroid = centroids[closest_index]

    return tuple(closest_centroid)

def _num_unique_labels(points):
    labels = points[:,-1]
    return np.unique(labels).size

def _sort_to_centroids(
        unknown_points,
        centroids,
        centroid_dict
) -> dict:
    """
    Iterate through the points, sorting them to their associated centroid
    """
    for unknown_point in unknown_points:
        closest_centroid = _find_closest_centroid(
            centroids,
            unknown_point,
            labelled=True)
        centroid_dict[closest_centroid] = np.vstack((centroid_dict[closest_centroid], unknown_point))    
    return centroid_dict
    

# ----------------------------------------
# Main
# ----------------------------------------

def bin_points_to_centroids(
        centroids,
        unknown_points,
        critical_density=0,
        training = False
) -> dict:
    """
    Takes an iterable of centroids and an array of unknown_points and sorts the unknown points into a dictionary according to their closest centroid by Euclidean distance.
    Also removes centroids with fewer than the critical_density of associated points.

    Args:
        centroids: Iterable of reference points
        unknown_points: Numpy array of points to be associated to its closest reference points.
        critical_density: Smallest acceptable number of points to be associated with a centroid.

    Returns:
        A dictionary where each key is a point from centroids and its corresponding value is a list of all unknown_points closer to it than any other reference point. 
        {
            tuple: np.array([])
        }
    """

    # Checking array size is appropriate
    # TODO Error Handling

    num_dimensions = unknown_points.shape[1]

    sorted_points = {tuple(x): np.empty((0, num_dimensions)) for x in centroids}
    small_centroid_indices = []
    unknown_points_combined = []

    sorted_points = _sort_to_centroids(
        unknown_points,
        centroids,
        sorted_points
    )

    # Removing centroids with fewer than critical_density of points, or with only 1 label
    if training:
        for i, (centroid, points) in enumerate(sorted_points.copy().items()):
            if len(points) < critical_density or _num_unique_labels(points) < 2:
                small_centroid_indices.append(i)
                unknown_points_combined.extend(points)
                del sorted_points[centroid]
        centroids = np.delete(centroids, small_centroid_indices, axis=0)

    sorted_points = _sort_to_centroids(
        unknown_points_combined,
        centroids,
        sorted_points
    )

    # convert lists to arrays
    sorted_points = {
        centroid: np.array(points)
        for centroid, points
        in sorted_points.items()
    }

    return sorted_points
