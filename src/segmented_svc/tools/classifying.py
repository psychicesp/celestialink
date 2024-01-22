# Global Imports
from sklearn import svm

# ----------------------------------------
# Support
# ----------------------------------------

def _train_classifier(
    labelled_points
):
    # Extracting all columns except the last one
    points = labelled_points[:, :-1]
    labels = labelled_points[:, -1]  # Extracting the last column as an array
    classifier = svm.SVC()
    classifier.fit(points, labels)
    return classifier

# ----------------------------------------
# Main
# ----------------------------------------

def train_classifier_dict(
    centroid_dict
) -> dict:
    """
    Converts a dictionary of points sorted to centroids into classifiers per centroid
    """
    classifier_dict = {}
    for centroid, points in centroid_dict.items():
        classifier = _train_classifier(points)
        classifier_dict[centroid] = classifier

    return classifier_dict
