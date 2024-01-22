# Global Imports
import numpy as np
from sklearn.preprocessing import StandardScaler

# Local Imports
from .tools.binning import bin_points_to_centroids
from .tools.clustering import get_cluster_centroids
from .tools.classifying import train_classifier_dict
from .tools.analyzing import calculate_critical_density

def _embed_indices(
    unknown_points
) -> np.array:
    """
    Adds an additional element to each row of an array corresponing to its index
    so that it can be reassembled after being sorted to centroids 
    """

    # Create the index column
    index_column = np.arange(unknown_points.shape[0])[:, np.newaxis]

    # Concatenate the index column with the original array
    indexed_array = np.hstack((unknown_points, index_column))

    return indexed_array


def _snap_indices_to_labels(
        labels: np.array,
        indices: np.array
) -> np.array:
    # Reshaping
    labels = np.reshape(labels, (-1, 1))
    indices = np.reshape(indices, (-1, 1))

    # Snapping
    indexed_labels = np.hstack((labels, indices))

    return indexed_labels


def _extract_indices_and_sort(
    label_array
) -> np.array:
    """
    Assembles generated labels into their original sequence based on embedded indices
    """
    # Get the indices for sorting
    indices = np.argsort(label_array[:, -1])

    # Sort the array based on the indices
    sorted_array = label_array[indices]

    # Remove the index column
    sorted_array = sorted_array[:, :-1]

    # Need 1D array of labels
    sorted_labels = sorted_array[:, -1].reshape(-1)

    return sorted_labels

# ----------------------------------------
# Main
# ----------------------------------------

class SegmentedSVC:
    """
    Boosts the Support Vector Classifier algorithm by preprocessing it into segments with K-Means clustering.
    This fits on initialization using the 'data' and 'labels' arguments

    Args:
        data: np.array containing unlabelled data, where each row is a datapoint\n
        labels: np.array containing labels correspnding to data\n
        critical_density: Smallest acceptable number of points to be associated with a centroid. 
        Floats will be interpreted as a quantile and integers will be interpreted as a threshold
    """

    def __init__(
            self,
            data: np.array,
            labels: np.array,
            critical_density = 1000
    ):

        # Interpret critical_density
        if isinstance(critical_density, float):
            self.critical_density = calculate_critical_density(labels)
        else:
            self.critical_density = critical_density

        # Scale data and append the labels.  Keep the scaler with the trained object so the same scaler can be used with unknown points.
        # Labels must be attached because of the reshuffling which must happen during hyperplane segmentation
        self.scaler = StandardScaler()
        self.number_of_dimensions = data.shape[1]
        data = self.scaler.fit_transform(data)
        data = np.hstack((data, labels[:, np.newaxis]))

        # Get classifier
        centroids = get_cluster_centroids(data)
        centroid_dict = bin_points_to_centroids(
            centroids,
            data,
            critical_density,
            training = True)

        # Output
        self.classifiers = train_classifier_dict(centroid_dict)
        self.centroids = list(self.classifiers.keys())

    def predict(
            self,
            unknown_points: np.array
        ) -> np.array:
        
        """
        Predicts the classification of the unknown points based on the trained segmented classifier

        Args:
            unknown_points: A numpy array of the unknown points to be predicted
            
        Returns:
            A numpy array of the predicted labels for the points. This matches the unknown_points input by index.
        """


        scaled_points = self.scaler.transform(unknown_points)

        # Batch indexing
        indexed_points = _embed_indices(scaled_points)
        binned_points = bin_points_to_centroids(
            self.centroids,
            indexed_points,
            training = False
            )

        all_indexed_labels = np.empty((0, 2))

        # Run batches
        for centroid, points in binned_points.items():
            
            # ESP: Constantly stripping and re-snapping indices is tedious (and not complex enough to want to modulate) 
            # but it really seems to be the best way to keep track of the original index of each point so output is sorted properly
            stripped_points = points[:, :-1]
            indices = points[:, -1]
            labels = self.classifiers[centroid].predict(stripped_points)
            indexed_labels = _snap_indices_to_labels(labels, indices)
            all_indexed_labels = np.vstack((all_indexed_labels, indexed_labels))

        # Join labels and resort to original orientation
        labels = _extract_indices_and_sort(all_indexed_labels)

        return labels
