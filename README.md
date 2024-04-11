![celestialink](images/banner.png)

# Boosting classification algorithms for mapping points in hyperspace

## SegmentedSVC
This works by chopping up the space and training a different SVC per segment.  This performs much faster than using a simple SVC without significant accuracy loss.  The results may also be better than the simple SVC for certain data sets.

Here is an example of its usage
```python
from celestialink import SegmentedSVC

# Separate labels from training data.  A scaler is built into SegmentedSVC so scaling data is unnecessary.
labelled_data = pd.read_csv('labelled_dataset.csv')
labels = labelled_data.pop('labels')

celestia_object = SegmentedSVC(
    data = labelled_data,
    labels = labels
)

unlabelled_data = pd.read_csv('unlabelled_dataset.csv')

predicted_labels = celestia_object.predict(unlabelled_data)
```