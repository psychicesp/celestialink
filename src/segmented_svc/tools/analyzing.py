# Standard Library Imports
from collections import Counter

# Global Imports
import numpy as np

# ----------------------------------------
# Support
# ----------------------------------------

def _label_counter(
    labels: np.array      
) -> np.array:
    """
    Counts instances of each label in a way that is more robust than np.bincount
    """
    if np.issubdtype(labels.dtype, np.integer) and np.all(labels > 0):
        count_values = np.bincount(labels)
    elif np.issubdtype(labels.dtype, np.floating) and np.all(labels > 0):
        count_values = np.bincount(np.asarray(labels, dtype=int))
    else:
        counter = Counter(labels)
        count_values = np.array(list(counter.values()))

    return count_values


# ----------------------------------------
# Main
# ----------------------------------------

def calculate_critical_density(
        labels: np.array,
        quantile = 0.2
) -> int:
    """
    Determines the minimum allowable number of points per cluster based on n-th quantile

    args:
        labels: a numpy array containing labels of a classified data set
        quantile: the quantile used as lower bound
    """
    
    label_counts = _label_counter(labels)
    lower_bound = np.quantile(label_counts, quantile)
    return int(lower_bound)
